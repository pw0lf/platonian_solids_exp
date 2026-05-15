import warnings
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader.mol3d_aug import Mol3d_Aug

parser = argparse.ArgumentParser()
parser.add_argument("--path",        required=True, help="Path prefix for output files (no extension)")
parser.add_argument("--datapath",    required=True)
parser.add_argument("--datasize",    type=int,   default=50_000)
parser.add_argument("--epochs",      type=int,   default=100)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--channels",    type=int,   default=64)
parser.add_argument("--dropout",     type=float, default=0.3)
parser.add_argument("--batch_size",  type=int,   default=64)
parser.add_argument("--k",           type=int,   default=2)
parser.add_argument("--weight_decay",type=float, default=0.1)
parser.add_argument("--drop_edge_p", type=float, default=0.1)
args = parser.parse_args()

# --- DDP setup --------------------------------------------------------------

is_ddp = "LOCAL_RANK" in os.environ
if is_ddp:
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    rank        = dist.get_rank()
    world_size  = dist.get_world_size()
    device      = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
else:
    rank       = 0
    world_size = 1
    device     = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")

def print0(*a, **kw):
    if rank == 0:
        print(*a, **kw)

# --- Data -------------------------------------------------------------------

print0(f"Loading dataset (size={args.datasize}, k={args.k})...")
dataset = Mol3d_Aug(root=args.datapath, size=args.datasize, k=args.k)
print0(f"Loaded {len(dataset)} molecules")

n_train = int(0.6 * len(dataset))
n_val   = int(0.8 * len(dataset))

train_labels = torch.cat([dataset[i][-1] for i in range(n_train)])
label_mean = train_labels.mean()
label_std  = train_labels.std().clamp(min=1e-6)
print0(f"Label mean: {label_mean:.4f}, std: {label_std:.4f}")

print0("Computing feature statistics...")
atom_feats_train = torch.cat([dataset[i][0] for i in range(n_train)])
bond_feats_train = torch.cat([dataset[i][1] for i in range(n_train)])
ring_feats_train = torch.cat([dataset[i][2] for i in range(n_train)])

atom_mean, atom_std = atom_feats_train.mean(0), atom_feats_train.std(0).clamp(min=1e-6)
bond_mean, bond_std = bond_feats_train.mean(0), bond_feats_train.std(0).clamp(min=1e-6)
ring_mean, ring_std = ring_feats_train.mean(0), ring_feats_train.std(0).clamp(min=1e-6)


def sparse_block_diag(sparse_list):
    device_ = sparse_list[0].device
    dtype   = sparse_list[0].dtype
    rows, cols, vals = [], [], []
    row_offset = col_offset = 0
    for S in sparse_list:
        S = S.coalesce()
        i, v = S.indices(), S.values()
        n_rows, n_cols = S.shape
        rows.append(i[0] + row_offset)
        cols.append(i[1] + col_offset)
        vals.append(v)
        row_offset += n_rows
        col_offset += n_cols
    indices = torch.stack([torch.cat(rows), torch.cat(cols)])
    return torch.sparse_coo_tensor(indices, torch.cat(vals),
                                   size=(row_offset, col_offset),
                                   device=device_, dtype=dtype)


def collate(batch):
    atom_feat, bond_feat, ring_feat, \
    icd01, icd02, icd12, \
    icd03, icd13, icd23, icd34, \
    adj00, adj11, adj22, adj33, hlgap = zip(*batch)
    labels = (torch.cat(hlgap, dim=0) - label_mean) / label_std
    return (
        (torch.cat(atom_feat, dim=0) - atom_mean) / atom_std,
        (torch.cat(bond_feat, dim=0) - bond_mean) / bond_std,
        (torch.cat(ring_feat, dim=0) - ring_mean) / ring_std,
        sparse_block_diag(icd01),
        sparse_block_diag(icd02),
        sparse_block_diag(icd12),
        sparse_block_diag(icd03),
        sparse_block_diag(icd13),
        sparse_block_diag(icd23),
        sparse_block_diag(icd34),
        sparse_block_diag(adj00),
        sparse_block_diag(adj11),
        sparse_block_diag(adj22),
        sparse_block_diag(adj33),
        labels,
    )


train_subset = Subset(dataset, range(n_train))
val_subset   = Subset(dataset, range(n_train, n_val))
test_subset  = Subset(dataset, range(n_val, len(dataset)))

if is_ddp:
    train_sampler = DistributedSampler(train_subset, shuffle=True)
    val_sampler   = DistributedSampler(val_subset,   shuffle=False)
    train_loader  = DataLoader(train_subset, batch_size=args.batch_size,
                               sampler=train_sampler, collate_fn=collate)
    val_loader    = DataLoader(val_subset,   batch_size=args.batch_size,
                               sampler=val_sampler,   collate_fn=collate)
else:
    train_sampler = None
    train_loader  = DataLoader(train_subset, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate)
    val_loader    = DataLoader(val_subset, batch_size=args.batch_size,
                               collate_fn=collate)

# test only on rank 0
if rank == 0:
    test_loader = DataLoader(test_subset, batch_size=args.batch_size,
                             collate_fn=collate)

print0(f"Train: {n_train} | Val: {n_val - n_train} | Test: {len(dataset) - n_val}")
print0(f"Device: {device}  |  world_size: {world_size}")

# --- Augmentation helpers ---------------------------------------------------

def drop_edge(S, p):
    if p == 0.0 or not S.is_sparse:
        return S
    S = S.coalesce()
    mask = torch.rand(S._nnz(), device=S.device) > p
    return torch.sparse_coo_tensor(
        S.indices()[:, mask], S.values()[mask], S.shape
    ).coalesce()


def random_rotation(atom_feat, dev):
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=dev))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    out = atom_feat.clone()
    out[:, 8:11] = atom_feat[:, 8:11] @ Q.T
    return out

# --- Model ------------------------------------------------------------------

def ssp(x):
    return torch.log(0.5 * torch.exp(x) + 0.5)


class CCConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, normalize=False, update_func=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.xavier_uniform_(self.W)
        self.normalize = normalize
        self.bn = nn.BatchNorm1d(out_channels)
        self.batch_norm = batch_norm
        self.update_func = update_func

    def _activate(self, x):
        match self.update_func:
            case "sigmoid":   return torch.sigmoid(x)
            case "relu":      return F.relu(x)
            case "leakyrelu": return F.leaky_relu(x)
            case "ssp":       return ssp(x)
            case _:           return x

    def forward(self, x, neighborhood):
        agg = neighborhood @ (x @ self.W.T)
        if self.normalize:
            deg = neighborhood.sum(dim=1).to_dense().clamp(min=1).unsqueeze(1)
            agg = agg / deg
        return self._activate(self.bn(agg) if self.batch_norm else agg)


class CCAttLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, update_func=None):
        super().__init__()
        self.Ws  = nn.Parameter(torch.empty(out_channels, in_channels))
        self.Wt  = nn.Parameter(torch.empty(in_channels, out_channels))
        self.a_s = nn.Parameter(torch.empty(out_channels))
        self.a_t = nn.Parameter(torch.empty(in_channels))
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm1d(out_channels)
        self.update_func = update_func
        nn.init.xavier_uniform_(self.Ws)
        nn.init.xavier_uniform_(self.Wt)
        nn.init.normal_(self.a_s, std=0.1)
        nn.init.normal_(self.a_t, std=0.1)

    def _activate(self, x):
        match self.update_func:
            case "sigmoid":   return torch.sigmoid(x)
            case "relu":      return F.relu(x)
            case "leakyrelu": return F.leaky_relu(x)
            case "ssp":       return ssp(x)
            case _:           return x

    def forward(self, x, neighborhood, x_target):
        neighborhood = neighborhood.coalesce()
        Z_s = x        @ self.Ws.T
        Z_t = x_target @ self.Wt.T
        rows, cols = neighborhood.indices()
        e = F.leaky_relu((Z_s[cols] @ self.a_s) + (Z_t[rows] @ self.a_t), negative_slope=0.2)
        e_max = torch.zeros(x_target.shape[0], device=x.device)
        e_max.scatter_reduce_(0, rows, e, reduce='amax', include_self=True)
        e_exp = torch.exp(e - e_max[rows])
        e_sum = torch.zeros(x_target.shape[0], device=x.device).scatter_add_(0, rows, e_exp)
        att = e_exp / (e_sum[rows] + 1e-8)
        weighted = att.unsqueeze(1) * Z_s[cols]
        out = torch.zeros(x_target.shape[0], Z_s.shape[1], device=x.device)
        out.scatter_add_(0, rows.unsqueeze(1).expand_as(weighted), weighted)
        return self._activate(self.bn(out) if self.batch_norm else out)


ATOM_DIM = 11
BOND_DIM = 8
RING_DIM = 6

ch = args.channels

model_kwargs = dict(
    channels_rk0=ch, channels_rk1=ch, channels_rk2=ch, channels_rk3=ch,
    size_hidden_layer1=ch, size_hidden_layer2=ch // 2, output_channels=1,
    lr=args.lr, batch_norm=True, gradient_clipping=True, normalize=False,
    epochs=args.epochs, update_func="leakyrelu", dropout=args.dropout,
)


class Simple_Att_TNN(nn.Module):
    def __init__(self, channels_rk0, channels_rk1, channels_rk2, channels_rk3,
                 size_hidden_layer1, size_hidden_layer2, output_channels,
                 lr, batch_norm, gradient_clipping, normalize, epochs,
                 update_func="leakyrelu", dropout=0.0):
        super().__init__()
        self.lr = lr
        self.gradient_clipping = gradient_clipping
        self.epochs = epochs
        self.dropout = nn.Dropout(dropout)

        self.proj_0 = nn.Linear(ATOM_DIM, channels_rk0)
        self.proj_1 = nn.Linear(BOND_DIM, channels_rk1)
        self.proj_2 = nn.Linear(RING_DIM, channels_rk2)

        self.conv_0_to_3 = CCConvLayer(channels_rk0, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_3 = CCConvLayer(channels_rk1, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)
        self.conv_2_to_3 = CCConvLayer(channels_rk2, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)

        self.att_0_to_0 = CCAttLayer(channels_rk0, channels_rk0, update_func=update_func, batch_norm=batch_norm)
        self.att_0_to_1 = CCAttLayer(channels_rk0, channels_rk1, update_func=update_func, batch_norm=batch_norm)
        self.att_0_to_2 = CCAttLayer(channels_rk0, channels_rk2, update_func=update_func, batch_norm=batch_norm)
        self.att_0_to_3 = CCAttLayer(channels_rk0, channels_rk3, update_func=update_func, batch_norm=batch_norm)

        self.att_1_to_0 = CCAttLayer(channels_rk1, channels_rk0, update_func=update_func, batch_norm=batch_norm)
        self.att_1_to_1 = CCAttLayer(channels_rk1, channels_rk1, update_func=update_func, batch_norm=batch_norm)
        self.att_1_to_2 = CCAttLayer(channels_rk1, channels_rk2, update_func=update_func, batch_norm=batch_norm)
        self.att_1_to_3 = CCAttLayer(channels_rk1, channels_rk3, update_func=update_func, batch_norm=batch_norm)

        self.att_2_to_0 = CCAttLayer(channels_rk2, channels_rk0, update_func=update_func, batch_norm=batch_norm)
        self.att_2_to_1 = CCAttLayer(channels_rk2, channels_rk1, update_func=update_func, batch_norm=batch_norm)
        self.att_2_to_2 = CCAttLayer(channels_rk2, channels_rk2, update_func=update_func, batch_norm=batch_norm)
        self.att_2_to_3 = CCAttLayer(channels_rk2, channels_rk3, update_func=update_func, batch_norm=batch_norm)

        self.att_3_to_0 = CCAttLayer(channels_rk3, channels_rk0, update_func=update_func, batch_norm=batch_norm)
        self.att_3_to_1 = CCAttLayer(channels_rk3, channels_rk1, update_func=update_func, batch_norm=batch_norm)
        self.att_3_to_2 = CCAttLayer(channels_rk3, channels_rk2, update_func=update_func, batch_norm=batch_norm)
        self.att_3_to_3 = CCAttLayer(channels_rk3, channels_rk3, update_func=update_func, batch_norm=batch_norm)

        self.conv_0_to_4 = CCConvLayer(channels_rk0, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_4 = CCConvLayer(channels_rk1, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)
        self.conv_2_to_4 = CCConvLayer(channels_rk2, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)
        self.conv_3_to_4 = CCConvLayer(channels_rk3, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)

        self.ln_out = nn.LayerNorm(channels_rk3)
        self.fc1 = nn.Linear(channels_rk3, size_hidden_layer1)
        self.fc2 = nn.Linear(size_hidden_layer1, size_hidden_layer2)
        self.fc3 = nn.Linear(size_hidden_layer2, output_channels)

    def forward(self, atom, bond, ring,
                icd01, icd02, icd12,
                icd03, icd13, icd23, icd34,
                adj00, adj11, adj22, adj33):
        x_0 = self.proj_0(atom.float())
        x_1 = self.proj_1(bond.float())
        x_2 = self.proj_2(ring.float())
        x_3 = (self.conv_0_to_3(x_0, icd03.T)
              + self.conv_1_to_3(x_1, icd13.T)
              + self.conv_2_to_3(x_2, icd23.T))

        x_0_new = (self.att_0_to_0(x_0, adj00,   x_0)
                 + self.att_1_to_0(x_1, icd01,   x_0)
                 + self.att_2_to_0(x_2, icd02,   x_0)
                 + self.att_3_to_0(x_3, icd03,   x_0) + x_0)

        x_1_new = (self.att_0_to_1(x_0, icd01.T, x_1)
                 + self.att_1_to_1(x_1, adj11,   x_1)
                 + self.att_2_to_1(x_2, icd12,   x_1)
                 + self.att_3_to_1(x_3, icd13,   x_1) + x_1)

        x_2_new = (self.att_0_to_2(x_0, icd02.T, x_2)
                 + self.att_1_to_2(x_1, icd12.T, x_2)
                 + self.att_2_to_2(x_2, adj22,   x_2)
                 + self.att_3_to_2(x_3, icd23,   x_2) + x_2)

        x_3_new = (self.att_0_to_3(x_0, icd03.T, x_3)
                 + self.att_1_to_3(x_1, icd13.T, x_3)
                 + self.att_2_to_3(x_2, icd23.T, x_3)
                 + self.att_3_to_3(x_3, adj33,   x_3) + x_3)

        x_0, x_1, x_2, x_3 = x_0_new, x_1_new, x_2_new, x_3_new

        x_4 = (icd34.T @ self.conv_0_to_4(x_0, icd03.T)
              + icd34.T @ self.conv_1_to_4(x_1, icd13.T)
              + icd34.T @ self.conv_2_to_4(x_2, icd23.T)
              + self.conv_3_to_4(x_3, icd34.T))

        x_out = self.dropout(F.leaky_relu(self.fc1(self.ln_out(x_4))))
        x_out = self.dropout(F.leaky_relu(self.fc2(x_out)))
        return self.fc3(x_out)


# --- Training ---------------------------------------------------------------

raw_model = Simple_Att_TNN(**model_kwargs).to(device)
model = DDP(raw_model, device_ids=[local_rank]) if is_ddp else raw_model

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.MSELoss()

components = list(dict.fromkeys(n.split('.')[0] for n, _ in raw_model.named_parameters()))

train_losses, val_losses, lrs = [], [], []
epoch_times = []
grad_norms_per_component = {c: [] for c in components}

for epoch in range(args.epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    t0 = time.time()
    total_loss = 0
    component_accum = {c: 0.0 for c in components}

    for batch in train_loader:
        atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, icd34, adj00, adj11, adj22, adj33, hlgap = [
            b.to(device) for b in batch
        ]
        atom  = random_rotation(atom, device)
        icd01 = drop_edge(icd01, args.drop_edge_p)
        icd02 = drop_edge(icd02, args.drop_edge_p)
        icd12 = drop_edge(icd12, args.drop_edge_p)
        icd03 = drop_edge(icd03, args.drop_edge_p)
        icd13 = drop_edge(icd13, args.drop_edge_p)
        icd23 = drop_edge(icd23, args.drop_edge_p)
        adj00 = drop_edge(adj00, args.drop_edge_p)
        adj11 = drop_edge(adj11, args.drop_edge_p)
        adj22 = drop_edge(adj22, args.drop_edge_p)
        adj33 = drop_edge(adj33, args.drop_edge_p)

        optimizer.zero_grad()
        out = model(atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, icd34, adj00, adj11, adj22, adj33).squeeze(-1)
        loss = criterion(out, hlgap.squeeze(-1))
        loss.backward()

        if rank == 0:
            for c in components:
                grads = [p.grad.flatten() for n, p in raw_model.named_parameters()
                         if n.split('.')[0] == c and p.grad is not None]
                if grads:
                    component_accum[c] += torch.cat(grads).norm().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    # average train loss across ranks
    train_loss_tensor = torch.tensor(total_loss / len(train_loader), device=device)
    if is_ddp:
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
    train_losses.append(train_loss_tensor.item())

    for c in components:
        grad_norms_per_component[c].append(component_accum[c] / len(train_loader))

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, icd34, adj00, adj11, adj22, adj33, hlgap = [
                b.to(device) for b in batch
            ]
            out = model(atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, icd34, adj00, adj11, adj22, adj33).squeeze(-1)
            val_loss += criterion(out, hlgap.squeeze(-1)).item()

    val_loss_tensor = torch.tensor(val_loss / len(val_loader), device=device)
    if is_ddp:
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
    val_losses.append(val_loss_tensor.item())

    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

    epoch_time = time.time() - t0
    epoch_times.append(epoch_time)
    print0(f"Epoch {epoch+1:3d}/{args.epochs}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}  lr={lrs[-1]:.2e}  ({epoch_time:.1f}s)")

# --- Test (rank 0 only) -----------------------------------------------------

if rank == 0:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, icd34, adj00, adj11, adj22, adj33, hlgap = [
                b.to(device) for b in batch
            ]
            out = model(atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, icd34, adj00, adj11, adj22, adj33).squeeze(-1)
            all_preds.append(out.cpu())
            all_targets.append(hlgap.squeeze(-1).cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mae_norm = (preds - targets).abs().mean().item()
    mae_ev   = mae_norm * label_std.item()
    print(f"Test MAE (normalized): {mae_norm:.4f}")
    print(f"Test MAE (eV):         {mae_ev:.4f}")

    # --- Save results -------------------------------------------------------

    results = {
        "model_kwargs":             model_kwargs,
        "datasize":                 args.datasize,
        "k":                        args.k,
        "world_size":               world_size,
        "weight_decay":             args.weight_decay,
        "drop_edge_p":              args.drop_edge_p,
        "n_train":                  n_train,
        "n_val":                    n_val - n_train,
        "n_test":                   len(dataset) - n_val,
        "label_mean":               label_mean.item(),
        "label_std":                label_std.item(),
        "train_losses":             train_losses,
        "val_losses":               val_losses,
        "lrs":                      lrs,
        "epoch_times":              epoch_times,
        "grad_norms_per_component": grad_norms_per_component,
        "test_mae_norm":            mae_norm,
        "test_mae_ev":              mae_ev,
    }

    Path(args.path).parent.mkdir(parents=True, exist_ok=True)

    json_path = args.path + ".json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    # --- Plots --------------------------------------------------------------

    colors = plt.cm.tab20([i / len(components) for i in range(len(components))])
    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(16, 5))

    ax_loss.plot(train_losses, label="train loss")
    ax_loss.plot(val_losses,   label="val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE loss (normalized)")
    ax_loss.set_title(f"Loss  —  test MAE={mae_ev:.4f} eV")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    for (c, norms), color in zip(grad_norms_per_component.items(), colors):
        ax_grad.plot(norms, label=c, color=color)
    ax_grad.set_xlabel("Epoch")
    ax_grad.set_ylabel("L2 norm (pre-clip)")
    ax_grad.set_title("Gradient norm per component")
    ax_grad.legend(fontsize=7, ncol=2)
    ax_grad.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = args.path + ".png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

if is_ddp:
    dist.destroy_process_group()
