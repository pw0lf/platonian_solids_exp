import warnings
warnings.filterwarnings("ignore")

import time
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

from data_loader.mol3d_aug import Mol3d_Aug

parser = argparse.ArgumentParser()
parser.add_argument("--path",      required=True, help="Path prefix for output files (no extension)")
parser.add_argument("--datapath",  required=True)
parser.add_argument("--datasize",  type=int, default=50_000)
parser.add_argument("--epochs",    type=int, default=100)
parser.add_argument("--lr",        type=float, default=1e-4)
parser.add_argument("--channels",  type=int, default=64)
parser.add_argument("--dropout",   type=float, default=0.1)
parser.add_argument("--batch_size",type=int, default=64)
args = parser.parse_args()

# --- Data -------------------------------------------------------------------

print(f"Loading dataset (size={args.datasize})...")
dataset = Mol3d_Aug(root=args.datapath, size=args.datasize)
print(f"Loaded {len(dataset)} molecules")

n_train = int(0.6 * len(dataset))
n_val   = int(0.8 * len(dataset))

train_labels = torch.cat([dataset[i][-1] for i in range(n_train)])
label_mean = train_labels.mean()
label_std  = train_labels.std().clamp(min=1e-6)
print(f"Label mean: {label_mean:.4f}, std: {label_std:.4f}")

print("Computing feature statistics...")
atom_feats_train = torch.cat([dataset[i][0] for i in range(n_train)])
bond_feats_train = torch.cat([dataset[i][1] for i in range(n_train)])
ring_feats_train = torch.cat([dataset[i][2] for i in range(n_train)])

atom_mean, atom_std = atom_feats_train.mean(0), atom_feats_train.std(0).clamp(min=1e-6)
bond_mean, bond_std = bond_feats_train.mean(0), bond_feats_train.std(0).clamp(min=1e-6)
ring_mean, ring_std = ring_feats_train.mean(0), ring_feats_train.std(0).clamp(min=1e-6)


def sparse_block_diag(sparse_list):
    device = sparse_list[0].device
    dtype  = sparse_list[0].dtype
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
                                   device=device, dtype=dtype)


def collate(batch):
    atom_feat, bond_feat, ring_feat, icd01, icd02, icd12, icd03, icd13, icd23, hlgap = zip(*batch)
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
        labels,
    )


train_loader = DataLoader(Subset(dataset, range(n_train)),
                          batch_size=args.batch_size, shuffle=False, collate_fn=collate)
val_loader   = DataLoader(Subset(dataset, range(n_train, n_val)),
                          batch_size=args.batch_size, collate_fn=collate)
test_loader  = DataLoader(Subset(dataset, range(n_val, len(dataset))),
                          batch_size=args.batch_size, collate_fn=collate)

print(f"Train: {n_train} | Val: {n_val - n_train} | Test: {len(dataset) - n_val}")

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
    lr=args.lr, batch_norm=False, gradient_clipping=True, normalize=True,
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

        self.conv_0_to_1 = CCConvLayer(channels_rk0, channels_rk1, update_func=update_func, batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_2 = CCConvLayer(channels_rk1, channels_rk2, update_func=update_func, batch_norm=batch_norm, normalize=normalize)

        self.att_0_to_1 = CCAttLayer(channels_rk0, channels_rk1, update_func=update_func, batch_norm=batch_norm)
        self.att_1_to_2 = CCAttLayer(channels_rk1, channels_rk2, update_func=update_func, batch_norm=batch_norm)

        self.conv_0_to_3 = CCConvLayer(channels_rk0, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_3 = CCConvLayer(channels_rk1, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)
        self.conv_2_to_3 = CCConvLayer(channels_rk2, channels_rk3, update_func=update_func, batch_norm=batch_norm, normalize=normalize)

        self.ln_x3 = nn.LayerNorm(channels_rk3)
        self.fc1 = nn.Linear(channels_rk3, size_hidden_layer1)
        self.fc2 = nn.Linear(size_hidden_layer1, size_hidden_layer2)
        self.fc3 = nn.Linear(size_hidden_layer2, output_channels)

    def forward(self, atom, bond, ring, icd01, icd02, icd03, icd12, icd13, icd23):
        x_0 = self.proj_0(atom.float())
        x_1 = self.proj_1(bond.float()) + self.conv_0_to_1(x_0, icd01.T)
        x_2 = self.proj_2(ring.float()) + self.conv_1_to_2(x_1, icd12.T)
        x_1 = self.att_0_to_1(x_0, icd01.T, x_1)
        x_2 = self.att_1_to_2(x_1, icd12.T, x_2)
        x_3 = (self.conv_0_to_3(x_0, icd03.T)
              + self.conv_1_to_3(x_1, icd13.T)
              + self.conv_2_to_3(x_2, icd23.T))
        x_out = self.dropout(F.leaky_relu(self.fc1(self.ln_x3(x_3))))
        x_out = self.dropout(F.leaky_relu(self.fc2(x_out)))
        return self.fc3(x_out)


# --- Training ---------------------------------------------------------------

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"Device: {device}")

model = Simple_Att_TNN(**model_kwargs).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr)
criterion = nn.MSELoss()

components = list(dict.fromkeys(n.split('.')[0] for n, _ in model.named_parameters()))

train_losses, val_losses = [], []
epoch_times = []
grad_norms_per_component = {c: [] for c in components}

for epoch in range(model.epochs):
    model.train()
    t0 = time.time()
    total_loss = 0
    component_accum = {c: 0.0 for c in components}

    for batch in train_loader:
        atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, hlgap = [
            b.to(device) for b in batch
        ]
        optimizer.zero_grad()
        out = model(atom, bond, ring, icd01, icd02, icd03, icd12, icd13, icd23).squeeze(-1)
        loss = criterion(out, hlgap.squeeze(-1))
        loss.backward()

        for c in components:
            grads = [p.grad.flatten() for n, p in model.named_parameters()
                     if n.split('.')[0] == c and p.grad is not None]
            if grads:
                component_accum[c] += torch.cat(grads).norm().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    n_batches = len(train_loader)
    train_losses.append(total_loss / n_batches)
    for c in components:
        grad_norms_per_component[c].append(component_accum[c] / n_batches)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, hlgap = [
                b.to(device) for b in batch
            ]
            out = model(atom, bond, ring, icd01, icd02, icd03, icd12, icd13, icd23).squeeze(-1)
            val_loss += criterion(out, hlgap.squeeze(-1)).item()
    val_losses.append(val_loss / len(val_loader))

    epoch_time = time.time() - t0
    epoch_times.append(epoch_time)
    print(f"Epoch {epoch+1:3d}/{model.epochs}  train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}  ({epoch_time:.1f}s)")

# --- Test -------------------------------------------------------------------

model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in test_loader:
        atom, bond, ring, icd01, icd02, icd12, icd03, icd13, icd23, hlgap = [
            b.to(device) for b in batch
        ]
        out = model(atom, bond, ring, icd01, icd02, icd03, icd12, icd13, icd23).squeeze(-1)
        all_preds.append(out.cpu())
        all_targets.append(hlgap.squeeze(-1).cpu())

preds   = torch.cat(all_preds)
targets = torch.cat(all_targets)
mae_norm = (preds - targets).abs().mean().item()
mae_ev   = mae_norm * label_std.item()
print(f"Test MAE (normalized): {mae_norm:.4f}")
print(f"Test MAE (eV):         {mae_ev:.4f}")

# --- Save results -----------------------------------------------------------

results = {
    "model_kwargs":              model_kwargs,
    "datasize":                  args.datasize,
    "n_train":                   n_train,
    "n_val":                     n_val - n_train,
    "n_test":                    len(dataset) - n_val,
    "label_mean":                label_mean.item(),
    "label_std":                 label_std.item(),
    "train_losses":              train_losses,
    "val_losses":                val_losses,
    "epoch_times":               epoch_times,
    "grad_norms_per_component":  grad_norms_per_component,
    "test_mae_norm":             mae_norm,
    "test_mae_ev":               mae_ev,
}

json_path = args.path + ".json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {json_path}")

# --- Plots ------------------------------------------------------------------

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
