import time
import json
import argparse
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from data_loader.mol3d import Mol3d_KHopLifting

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="Path to save results JSON")
args = parser.parse_args()

# ── Model ─────────────────────────────────────────────────────────────────────

class CCConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 update_func: Literal["sigmoid", "relu", "leakyrelu", None] = None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_channels, in_channels, dtype=torch.float32))
        nn.init.xavier_uniform_(self.W)
        self.bn = nn.BatchNorm1d(out_channels) # Batchnorm
        self.update_func = update_func

    def _activate(self, x):
        match self.update_func:
            case "sigmoid":   return torch.sigmoid(x)
            case "relu":      return F.relu(x)
            case "leakyrelu": return F.leaky_relu(x)
            case _:           return x

    def forward(self, x, neighborhood):
        x_proj = x @ self.W.T
        agg = neighborhood @ x_proj
        deg = neighborhood.sum(dim=1).to_dense().clamp(min=1).unsqueeze(1) # Normalizing over the neighbors
        agg = agg / deg
        return self._activate(self.bn(agg))

    
class CCAttLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 update_func: Literal["sigmoid", "relu", "leakyrelu", None] = None):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.update_func  = update_func

        self.Ws  = nn.Parameter(torch.empty(out_channels, in_channels,  dtype=torch.float32))
        self.Wt  = nn.Parameter(torch.empty(in_channels,  out_channels, dtype=torch.float32))
        self.a_s = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
        self.a_t = nn.Parameter(torch.empty(in_channels,  dtype=torch.float32))
        # No W_res: it was a shortcut that let the model bypass attention.
        # All molecules have ≥1 rank-2 cell (filtered in data_loader), so
        # every target cell has at least one incoming neighbor.
        self.bn = nn.BatchNorm1d(out_channels)

        nn.init.xavier_uniform_(self.Ws)
        nn.init.xavier_uniform_(self.Wt)
        nn.init.xavier_uniform_(self.a_s.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_t.unsqueeze(0))

    def _activate(self, x):
        match self.update_func:
            case "sigmoid":   return torch.sigmoid(x)
            case "relu":      return F.relu(x)
            case "leakyrelu": return F.leaky_relu(x)
            case _:           return x

    def forward(self, x, neighborhood, x_target):
        neighborhood = neighborhood.coalesce()
        Z_s = x        @ self.Ws.T   # (N_src, out_ch)
        Z_t = x_target @ self.Wt.T   # (N_tgt, in_ch)

        rows, cols = neighborhood.indices()   # rows=tgt, cols=src
        e = (Z_s[cols] @ self.a_s) + (Z_t[rows] @ self.a_t)
        e = F.leaky_relu(e, negative_slope=0.2)

        e_max = torch.zeros(x_target.shape[0], device=x.device)
        e_max.scatter_reduce_(0, rows, e, reduce='amax', include_self=True)
        e_exp = torch.exp(e - e_max[rows])
        e_sum = torch.zeros(x_target.shape[0], device=x.device)
        e_sum.scatter_add_(0, rows, e_exp)
        att = e_exp / (e_sum[rows] + 1e-8)

        weighted = att.unsqueeze(1) * Z_s[cols]
        out = torch.zeros(x_target.shape[0], Z_s.shape[1], device=x.device)
        out.scatter_add_(0, rows.unsqueeze(1).expand_as(weighted), weighted)

        return self._activate(self.bn(out))
    
class TNN(nn.Module):
    def __init__(self, in_channels, channels_rk0 ,channels_rk1, channels_rk2, channels_rk3,
                 size_hidden_layer1, size_hidden_layer2, output_channels):
        super().__init__()

        self.proj = nn.Linear(in_channels, channels_rk0)

        self.conv_0_to_1 = CCConvLayer(channels_rk0, channels_rk1, update_func="relu")
        self.conv_1_to_2 = CCConvLayer(channels_rk1,  channels_rk2, update_func="relu")
        self.conv_0_to_2 = CCConvLayer(channels_rk0, channels_rk2, update_func="relu")

        self.att_0_to_1  = CCAttLayer(channels_rk0, channels_rk1, update_func="relu")
        self.att_1_to_2  = CCAttLayer(channels_rk1,  channels_rk2, update_func="relu")
        self.att_0_to_2  = CCAttLayer(channels_rk0,  channels_rk2, update_func="relu")

        self.att_2_to_0 = CCAttLayer(channels_rk2, channels_rk0, update_func="relu")
        self.att_2_to_1 = CCAttLayer(channels_rk2, channels_rk1, update_func="relu")
        self.att_1_to_0 = CCAttLayer(channels_rk1, channels_rk0, update_func="relu")

        self.conv_0_to_3 = CCConvLayer(channels_rk0, channels_rk3, update_func="relu")
        self.conv_1_to_3 = CCConvLayer(channels_rk1, channels_rk3, update_func="relu")
        self.conv_2_to_3 = CCConvLayer(channels_rk2, channels_rk3, update_func="relu")

        self.fc1 = nn.Linear(channels_rk3, size_hidden_layer1)
        self.fc2 = nn.Linear(size_hidden_layer1, size_hidden_layer2)
        self.fc3 = nn.Linear(size_hidden_layer2, output_channels)

    def forward(self, x, icd01, icd02, icd03, icd12, icd13, icd23):

        x = x.to(torch.float32)

        x_0 = self.proj(x)
        x_1 = self.conv_0_to_1(x_0, icd01.T)
        x_2 = (self.conv_0_to_2(x_0, icd02.T) + self.conv_1_to_2(x_1,icd12.T))/2

        x_1_new = (self.att_0_to_1(x_0,icd01.T,x_1) + x_1)/2
        x_2_new = (self.att_0_to_2(x_0, icd02.T, x_2) + self.att_1_to_2(x_1,icd12.T,x_2) + x_2)/3

        x_0_new = (self.att_1_to_0(x_1_new,icd01,x_0) + self.att_2_to_0(x_2_new, icd02, x_0) + x_0)/3
        x_1_new = (self.att_2_to_1(x_2_new, icd12, x_1_new) + x_1_new)/2

        x_3 = (self.conv_0_to_3(x_0_new, icd03) + self.conv_1_to_3(x_1_new, icd13) + self.conv_2_to_3(x_2_new, icd23))/3

        x_out = torch.flatten(x_3,start_dim=0)
        x_out = F.relu(self.fc1(x_out))
        x_out = F.relu(self.fc2(x_out))
        x_out = self.fc3(x_out)

        return x_out


# ── Gradient diagnostics ──────────────────────────────────────────────────────

def log_grad_norms(model):
    groups = {"proj": [], "conv_W": [], "conv_bn": [], "att_Ws_Wt": [], "att_a": [], "att_bn": [], "fc": []}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        norm = p.grad.norm().item()
        if name.startswith("proj"):
            groups["proj"].append(norm)
        elif "conv" in name and "bn" in name:
            groups["conv_bn"].append(norm)
        elif "conv" in name:
            groups["conv_W"].append(norm)
        elif "att" in name and ("Ws" in name or "Wt" in name):
            groups["att_Ws_Wt"].append(norm)
        elif "att" in name and ("a_s" in name or "a_t" in name):
            groups["att_a"].append(norm)
        elif "att" in name and "bn" in name:
            groups["att_bn"].append(norm)
        elif "fc" in name:
            groups["fc"].append(norm)
    parts = [f"{k}={sum(v)/len(v):.2e}" for k, v in groups.items() if v]
    print(f"  grad norms (avg) | {' | '.join(parts)}")

# ── Collation helpers ─────────────────────────────────────────────────────────

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
    node_feat, icd01, icd02,icd03, icd12,icd13, icd23, homolumogap = zip(*batch)
    return (
        torch.cat(node_feat, dim=0),   # x_0: atomic numbers
        sparse_block_diag(icd01),
        sparse_block_diag(icd02),
        sparse_block_diag(icd03),
        sparse_block_diag(icd12),
        sparse_block_diag(icd13),
        sparse_block_diag(icd23),
        torch.cat(homolumogap, dim=0),
    )

# ── Dataset ───────────────────────────────────────────────────────────────────

dataset = Mol3d_KHopLifting(root="data/data/raw", size=100000)
print(len(dataset))

batch_size = 64
train_dataloader = DataLoader(Subset(dataset, range(60000)),
                              batch_size=batch_size, shuffle=True, collate_fn=collate)
val_dataloader   = DataLoader(Subset(dataset, range(60000, 80000)),
                              batch_size=batch_size, collate_fn=collate)
test_dataloader  = DataLoader(Subset(dataset, range(80000, len(dataset))),
                              batch_size=batch_size, collate_fn=collate)

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(device)

# ── Model init ────────────────────────────────────────────────────────────────

model = TNN(4,64 ,64, 128, 256, 128, 64, 1)
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.MSELoss()
model.to(device)

def run_model(x, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3 ,icd_2_3):
    return model(x, icd_0_1,icd_0_2,icd_0_3, icd_1_2, icd_1_3 , icd_2_3).squeeze(-1)

def eval_val():
    val_mae = 0.0
    model.eval()
    with torch.no_grad():
        for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in val_dataloader:
            x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
            output = run_model(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3 ,icd_2_3)
            val_mae += (output - hlgap).abs().mean().item()
    val_mae /= len(val_dataloader)
    model.train()
    return val_mae


results = {"epochs": []}

# ── Training ──────────────────────────────────────────────────────────────────

for epoch in range(30):
    total_loss = 0
    collect_stats = (epoch + 1) % 5 == 0
    all_preds, all_targets = [], []
    model.train()
    t0 = time.time()

    for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in train_dataloader:
        x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
        optimizer.zero_grad()
        output = run_model(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3 ,icd_2_3)
        loss = criterion(output, hlgap.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if collect_stats:
            all_preds.append(output.detach().cpu())
            all_targets.append(hlgap.cpu())

    epoch_time = time.time() - t0
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1:2d}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

    entry = {
        "epoch": epoch + 1, "train_loss": avg_loss, "epoch_time_s": epoch_time,
        "val_mae": None, "pred_mean": None, "pred_std": None,
        "target_mean": None, "target_std": None,
    }

    if collect_stats:
        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        entry["pred_mean"]   = all_preds.mean().item()
        entry["pred_std"]    = all_preds.std().item()
        entry["target_mean"] = all_targets.mean().item()
        entry["target_std"]  = all_targets.std().item()
        print(f"  pred  mean={entry['pred_mean']:.4f} std={entry['pred_std']:.4f} | "
              f"target mean={entry['target_mean']:.4f} std={entry['target_std']:.4f}")
        log_grad_norms(model)

    if epoch == 0 or collect_stats:
        val_mae = eval_val()
        entry["val_mae"] = val_mae
        print(f"  Val MAE: {val_mae:.6f}  LR: {optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step(val_mae)

    results["epochs"].append(entry)

# ── Test ──────────────────────────────────────────────────────────────────────

mae = 0.0
model.eval()
with torch.no_grad():
    for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in test_dataloader:
        x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
        output = run_model(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3 ,icd_2_3)
        mae += (output.squeeze(-1) - hlgap).abs().mean().item()
mae /= len(test_dataloader)
print(f"Test MAE: {mae}")

results["test_mae"] = mae

with open(args.path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {args.path}")