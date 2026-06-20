import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
from lrgb_gnn import LRGBGNNDataset
from gnn_models import GCN, GAT, GIN

DATASET_NAME = "Peptides-struct"
OUT_DIM = 11
NODE_DIM = 9
EDGE_DIM = 3
DATA_ROOT = str(Path(__file__).parent.parent / "data")


def make_model(args, edge_dim):
    common = dict(
        in_channels=NODE_DIM, hidden_channels=args.hidden_channels,
        num_conv_layers=args.num_conv_layers, readout_hidden_dim=args.readout_hidden_dim,
        num_readout_layers=args.num_readout_layers, out_dim=OUT_DIM, dropout=args.dropout,
    )
    if args.model == "GCN":
        return GCN(**common)
    if args.model == "GAT":
        return GAT(**common, num_heads=args.num_heads, edge_dim=edge_dim)
    if args.model == "GIN":
        return GIN(**common, mlp_hidden_dim=args.mlp_hidden_dim)


def evaluate(model, loader, device, use_edge_attr, y_mean, y_std):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_attr = batch.edge_attr if use_edge_attr else None
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
            out_denorm = out * y_std + y_mean
            all_preds.append(out_denorm.cpu())
            all_targets.append(batch.y.cpu())
    preds = torch.cat(all_preds)    # (N, 11)
    targets = torch.cat(all_targets)  # (N, 11)
    mae = float((preds - targets).abs().mean())
    return mae, preds, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",             type=str,   required=True, choices=["GCN", "GAT", "GIN"])
    parser.add_argument("--datapath",          type=str,   default=DATA_ROOT)
    parser.add_argument("--epochs",            type=int,   default=300)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--hidden_channels",   type=int,   default=128)
    parser.add_argument("--num_conv_layers",   type=int,   default=4)
    parser.add_argument("--readout_hidden_dim", type=int,  default=128)
    parser.add_argument("--num_readout_layers", type=int,  default=2)
    parser.add_argument("--dropout",           type=float, default=0.0)
    parser.add_argument("--num_heads",         type=int,   default=4)
    parser.add_argument("--mlp_hidden_dim",    type=int,   default=None)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--output",            type=str,   default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_struct_{args.model.lower()}.json"

    use_edge_attr = args.model == "GAT"
    edge_dim = EDGE_DIM if use_edge_attr else None

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | model: {args.model} | dataset: {DATASET_NAME}")

    print("Loading datasets...")
    train_ds = LRGBGNNDataset(args.datapath, DATASET_NAME, "train")
    val_ds   = LRGBGNNDataset(args.datapath, DATASET_NAME, "val")
    test_ds  = LRGBGNNDataset(args.datapath, DATASET_NAME, "test")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Per-target z-score normalization fitted on train
    train_labels = torch.cat([train_ds[i].y for i in range(len(train_ds))])  # (N, 11)
    y_mean = train_labels.mean(dim=0, keepdim=True).to(device)
    y_std  = train_labels.std(dim=0,  keepdim=True).clamp(min=1e-6).to(device)

    criterion = nn.L1Loss()
    results = {
        "model": args.model, "dataset": DATASET_NAME, "epochs": args.epochs, "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)
        model = make_model(args, edge_dim).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        run_result = {"run": run + 1, "train_losses": [], "epoch_times": []}
        run_start = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                edge_attr = batch.edge_attr if use_edge_attr else None
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
                loss = criterion(out, (batch.y - y_mean) / y_std)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            run_result["epoch_times"].append(round(time.time() - epoch_start, 2))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}  loss={train_loss:.4f}")

        val_mae,  _,          _            = evaluate(model, val_loader,  device, use_edge_attr, y_mean, y_std)
        test_mae, test_preds, test_targets = evaluate(model, test_loader, device, use_edge_attr, y_mean, y_std)
        run_result["val_mae"]   = round(val_mae,  4)
        run_result["test_mae"]  = round(test_mae, 4)
        run_result["runtime"]   = round(time.time() - run_start, 2)
        run_result["predictions"] = [
            {"index": int(idx),
             "pred": [round(float(v), 6) for v in pred],
             "true": [round(float(v), 6) for v in tgt]}
            for idx, pred, tgt in zip(test_ds.indices, test_preds.tolist(), test_targets.tolist())
        ]
        print(f"Val MAE: {val_mae:.4f}  Test MAE: {test_mae:.4f}")
        results["runs"].append(run_result)

    results["mean_val_mae"]  = round(sum(r["val_mae"]  for r in results["runs"]) / 3, 4)
    results["mean_test_mae"] = round(sum(r["test_mae"] for r in results["runs"]) / 3, 4)
    print(f"\nMean Val MAE: {results['mean_val_mae']:.4f}  Mean Test MAE: {results['mean_test_mae']:.4f}")

    out_path = Path(__file__).parent / "results" / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
