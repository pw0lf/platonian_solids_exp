import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet
from torch.optim.lr_scheduler import ExponentialLR

from data_loader.mol3d import Mol3d_PyG

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="Path to save results JSON")
args = parser.parse_args()

# Dataset
dataset = Mol3d_PyG(root="data/data/raw", size=100000)
print(len(dataset))

batch_size = 32
train_dataloader = DataLoader(Subset(dataset, range(60000)), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(Subset(dataset, range(60000, 80000)), batch_size=batch_size)
test_dataloader = DataLoader(Subset(dataset, range(80000, len(dataset))), batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Model
model = SchNet(num_interactions=6, hidden_channels=256, num_filters=256, cutoff=10, readout="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
gamma_per_step = 0.96 ** (1 / 100_000)
scheduler = ExponentialLR(optimizer, gamma=gamma_per_step)
criterion = nn.MSELoss()

model.to(device)

def eval_val():
    val_mae = 0.0
    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            data.to(device)
            output = model(data.node_features[:, 0].long(), data.node_features[:, 1:], data.batch)
            val_mae += (output.squeeze(-1) - data.homolumogap).abs().mean().item()
    val_mae /= len(val_dataloader)
    model.train()
    return val_mae

results = {"epochs": []}

# Training
for epoch in range(10):
    total_loss = 0
    model.train()
    t0 = time.time()
    for data in train_dataloader:
        data.to(device)
        optimizer.zero_grad()
        output = model(data.node_features[:, 0].long(), data.node_features[:, 1:], data.batch)
        loss = criterion(output.squeeze(-1), data.homolumogap)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler.step()
    epoch_time = time.time() - t0
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

    entry = {"epoch": epoch + 1, "train_loss": avg_loss, "epoch_time_s": epoch_time, "val_mae": None}

    if epoch == 0 or (epoch + 1) % 10 == 0:
        val_mae = eval_val()
        entry["val_mae"] = val_mae
        print(f"  Val MAE: {val_mae:.6f}")

    results["epochs"].append(entry)

# Test evaluation
mae = 0.0
model.eval()
with torch.no_grad():
    for data in test_dataloader:
        data.to(device)
        output = model(data.node_features[:, 0].long(), data.node_features[:, 1:], data.batch)
        mae += (output.squeeze(-1) - data.homolumogap).abs().mean().item()
mae /= len(test_dataloader)
print(f"Test MAE: {mae}")

results["test_mae"] = mae

with open(args.path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {args.path}")
