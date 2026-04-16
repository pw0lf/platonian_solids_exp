import sys
import time
import json
import argparse
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from data_loader.mol3d import Mol3d_CycleLifting

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="Path to save results JSON")
args = parser.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

class CCConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        update_func: Literal["sigmoid", "relu", "leakyrelu", None] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_func = update_func
        self.W = nn.Parameter(torch.empty(out_channels, in_channels, dtype=torch.float32))
        nn.init.xavier_uniform_(self.W)

    def update(self, x):
        match self.update_func:
            case "sigmoid":
                return torch.sigmoid(x)
            case "relu":
                return F.relu(x)
            case "leakyrelu":
                return F.leaky_relu(x)
            case _:
                return x

    def forward(self, x, neighborhood):
        x_1 = x @ self.W.T
        x_2 = neighborhood @ x_1
        return self.update(x_2)

class CCAttLayer(nn.Module):
	def __init__(
		self,
		in_channels,
		out_channels,
		aggr_norm: bool = False,
		initialization: Literal["xavier_uniform"] = "xavier_uniform",
		initialzation_gain: float = 1.414,
		update_func: Literal["sigmoid","relu","leakyrelu",None] = None
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.aggr_norm = aggr_norm
		self.initialization = initialization
		self.initialization_gain = initialzation_gain
		self.update_func = update_func

		self.init_parameters()
	
	def init_parameters(self):
		self.Ws = nn.Parameter(torch.empty(self.out_channels,self.in_channels, dtype=torch.float32))
		self.Wt = nn.Parameter(torch.empty(self.in_channels,self.out_channels, dtype=torch.float32))
		self.a_s = nn.Parameter(torch.empty(self.out_channels, dtype=torch.float32))
		self.a_t = nn.Parameter(torch.empty(self.in_channels, dtype=torch.float32))
		nn.init.xavier_uniform_(self.Ws)
		nn.init.xavier_uniform_(self.Wt)
		nn.init.xavier_uniform_(self.a_s.unsqueeze(0))
		nn.init.xavier_uniform_(self.a_t.unsqueeze(0))

	def update(self,x):
		match self.update_func:
			case "sigmoid":
				return torch.sigmoid(x)
			case "relu":
				return torch.nn.functional.relu(x)
			case "leakyrelu":
				return torch.nn.functional.leaky_relu(x)
			case _:
				return x

	def forward(self, x, neighborhood, x_target):
		neighborhood = neighborhood.coalesce()
		Z_s = x @ self.Ws.T        # (N_src, out_ch)
		Z_t = x_target @ self.Wt.T # (N_tgt, out_ch) — fix Wt shape too
		
		rows, cols = neighborhood.indices()  # rows=tgt idx, cols=src idx
		e = (Z_s[cols] @ self.a_s) + (Z_t[rows] @ self.a_t)
		e = F.leaky_relu(e, negative_slope=0.2)
		
		# sparse softmax: subtract max per target for numerical stability
		e_max = torch.zeros(x_target.shape[0], device=x.device)
		e_max.scatter_reduce_(0, rows, e, reduce='amax', include_self=True)
		e_exp = torch.exp(e - e_max[rows])
		e_sum = torch.zeros(x_target.shape[0], device=x.device)
		e_sum.scatter_add_(0, rows, e_exp)
		att = e_exp / (e_sum[rows] + 1e-8)
		
		# weighted aggregation
		weighted = att.unsqueeze(1) * Z_s[cols]  # (nnz, out_ch)
		out = torch.zeros(x_target.shape[0], Z_s.shape[1], device=x.device)
		out.scatter_add_(0, rows.unsqueeze(1).expand_as(weighted), weighted)
		return self.update(out)

class TNN(nn.Module):
	def __init__(self,node_channels,channels_rk1,channels_rk2,channels_rk3, size_hidden_layer1, size_hidden_layer2,output_channels):
		super().__init__()

		self.conv_0_to_1 = CCConvLayer(in_channels = node_channels, out_channels =  channels_rk1, update_func="relu")
		self.conv_1_to_2 = CCConvLayer(in_channels = channels_rk1, out_channels =  channels_rk2, update_func="relu")
		self.conv_2_to_3 = CCConvLayer(in_channels = channels_rk2, out_channels = channels_rk3, update_func="relu")

		self.att_0_to_1 = CCAttLayer(in_channels = node_channels, out_channels =  channels_rk1, update_func="relu")
		self.att_1_to_2 = CCAttLayer(in_channels = channels_rk1, out_channels =  channels_rk2, update_func="relu")
		self.att_2_to_3 = CCAttLayer(in_channels = channels_rk2, out_channels = channels_rk3, update_func="relu")

		self.fc1 = nn.Linear(channels_rk3,size_hidden_layer1)
		self.fc2 = nn.Linear(size_hidden_layer1, size_hidden_layer2)
		self.fc3 = nn.Linear(size_hidden_layer2,output_channels)

	def forward(self,x_0,incidence_0_1,incidence_1_2,incidence_2_3):
		x_0 = x_0.to(torch.float32)
		#x_0 = x_0.view(28 * 28,1)
		x_1_out = self.conv_0_to_1(x_0,incidence_0_1.T)
		x_2_out = self.conv_1_to_2(x_1_out,incidence_1_2.T)
		x_3_out = self.conv_2_to_3(x_2_out,incidence_2_3.T)

		x_1_new = self.att_0_to_1(x_0,incidence_0_1.T, x_1_out)
		x_2_new = self.att_1_to_2(x_1_new,incidence_1_2.T, x_2_out)
		x_3_new = self.att_2_to_3(x_2_new,incidence_2_3.T, x_3_out)


		x = torch.flatten(x_3_new, start_dim=1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# ── Collation helpers ─────────────────────────────────────────────────────────

def sparse_block_diag(sparse_list):
    device = sparse_list[0].device
    dtype = sparse_list[0].dtype
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
    return torch.sparse_coo_tensor(indices, torch.cat(vals), size=(row_offset, col_offset), device=device, dtype=dtype)


def batch_vector(V_list):
    return torch.cat([torch.full((v.shape[0],), i, dtype=torch.int64) for i, v in enumerate(V_list)])


def collate(batch):
    x, icd01, icd02, icd12, icd23, homolumogap = zip(*batch)
    return torch.cat(x, dim=0), sparse_block_diag(icd01), sparse_block_diag(icd12), sparse_block_diag(icd23), torch.cat(homolumogap, dim=0), batch_vector(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

dataset = Mol3d_CycleLifting(root="data/data/raw", size=100000)
print(len(dataset))

batch_size = 64
train_dataloader = DataLoader(Subset(dataset, range(60000)), batch_size=batch_size, shuffle=True, collate_fn=collate)
val_dataloader = DataLoader(Subset(dataset, range(60000, 80000)), batch_size=batch_size, collate_fn=collate)
test_dataloader = DataLoader(Subset(dataset, range(80000, len(dataset))), batch_size=batch_size, collate_fn=collate)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

# ── Model init ────────────────────────────────────────────────────────────────

model = TNN(4, 64, 128, 256, 128, 64, 1)
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.MSELoss()
model.to(device)


def eval_val():
    val_mae = 0.0
    model.eval()
    with torch.no_grad():
        for x, icd_0_1, icd_1_2, icd_2_3, hlgap, batch in val_dataloader:
            x, icd_0_1, icd_1_2, icd_2_3, hlgap = x.to(device), icd_0_1.to(device), icd_1_2.to(device), icd_2_3.to(device), hlgap.to(device)
            output = model(x, icd_0_1, icd_1_2, icd_2_3).squeeze(-1)
            val_mae += (output - hlgap).abs().mean().item()
    val_mae /= len(val_dataloader)
    model.train()
    return {"val_mae": val_mae}


results = {"epochs": []}

# ── Training ──────────────────────────────────────────────────────────────────

for epoch in range(20):
    total_loss = 0
    collect_stats = (epoch + 1) % 10 == 0
    all_preds, all_targets = [], []
    model.train()
    t0 = time.time()
    for x, icd_0_1, icd_1_2, icd_2_3, hlgap, batch in train_dataloader:
        x, icd_0_1, icd_1_2, icd_2_3, hlgap = x.to(device), icd_0_1.to(device), icd_1_2.to(device), icd_2_3.to(device), hlgap.to(device)
        optimizer.zero_grad()
        output = model(x, icd_0_1, icd_1_2, icd_2_3).squeeze(-1)
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
    scheduler.step(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

    entry = {"epoch": epoch + 1, "train_loss": avg_loss, "epoch_time_s": epoch_time,
             "val_mae": None, "pred_mean": None, "pred_std": None, "target_mean": None, "target_std": None}

    if collect_stats:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        entry["pred_mean"] = all_preds.mean().item()
        entry["pred_std"] = all_preds.std().item()
        entry["target_mean"] = all_targets.mean().item()
        entry["target_std"] = all_targets.std().item()
        print(f"  Train pred mean: {entry['pred_mean']:.4f} std: {entry['pred_std']:.4f} | target mean: {entry['target_mean']:.4f} std: {entry['target_std']:.4f}")

    if epoch == 0 or collect_stats:
        stats = eval_val()
        entry["val_mae"] = stats["val_mae"]
        print(f"  Val MAE: {stats['val_mae']:.6f}")

    results["epochs"].append(entry)

# ── Test evaluation ───────────────────────────────────────────────────────────

mae = 0.0
model.eval()
with torch.no_grad():
    for x, icd_0_1, icd_1_2, icd_2_3, hlgap, batch in test_dataloader:
        x, icd_0_1, icd_1_2, icd_2_3, hlgap = x.to(device), icd_0_1.to(device), icd_1_2.to(device), icd_2_3.to(device), hlgap.to(device)
        output = model(x, icd_0_1, icd_1_2, icd_2_3)
        mae += (output.squeeze(-1) - hlgap).abs().mean().item()
mae /= len(test_dataloader)
print(f"Test MAE: {mae}")

results["test_mae"] = mae

with open(args.path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {args.path}")
