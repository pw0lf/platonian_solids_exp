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
from mol3d_gnn import load_gnn_data, NODE_DIM, EDGE_DIM
from gnn_models import GCN, GAT, GIN

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")
MODELS = {"GCN": GCN, "GAT": GAT, "GIN": GIN}


def make_model(args, edge_dim):
    common = dict(
        in_channels=NODE_DIM,
        hidden_channels=args.hidden_channels,
        num_conv_layers=args.num_conv_layers,
        readout_hidden_dim=args.readout_hidden_dim,
        num_readout_layers=args.num_readout_layers,
        dropout=args.dropout,
    )
    if args.model == "GCN":
        return GCN(**common)
    if args.model == "GAT":
        return GAT(**common, num_heads=args.num_heads, edge_dim=edge_dim)
    if args.model == "GIN":
        return GIN(**common, mlp_hidden_dim=args.mlp_hidden_dim)


def evaluate(model, loader, device, y_mean, y_std, criterion, use_edge_attr):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_attr = batch.edge_attr if use_edge_attr else None
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
            preds.append((out.squeeze(-1) * y_std + y_mean).cpu())
            targets.append(batch.y.cpu())
    p, t = torch.cat(preds).squeeze(-1), torch.cat(targets).squeeze(-1)
    rmse = criterion(p, t).sqrt().item()
    mae = (p - t).abs().mean().item()
    r2 = (1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()
    return rmse, mae, r2, p, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",            type=str,   required=True, choices=["GCN", "GAT", "GIN"])
    parser.add_argument("--split_file",       type=str,   default=str(Path(__file__).parent / "data_split.json"))
    parser.add_argument("--datapath",         type=str,   default=DATA_ROOT)
    parser.add_argument("--epochs",           type=int,   default=300)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--hidden_channels",  type=int,   default=64)
    parser.add_argument("--num_conv_layers",  type=int,   default=3)
    parser.add_argument("--readout_hidden_dim", type=int, default=64)
    parser.add_argument("--num_readout_layers", type=int, default=2)
    parser.add_argument("--dropout",          type=float, default=0.0)
    parser.add_argument("--num_heads",        type=int,   default=4,  help="GAT only")
    parser.add_argument("--mlp_hidden_dim",   type=int,   default=None, help="GIN only")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output",           type=str,   default=None,
                        help="filename (saved inside results/); defaults to results_<model>.json")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_mol3d_{args.model.lower()}.json"

    use_edge_attr = args.model == "GAT"
    edge_dim = EDGE_DIM if use_edge_attr else None

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | model: {args.model}")

    split = json.load(open(args.split_file))
    train_indices = split["train"]
    test_indices  = split["test"]
    print(f"Split: {len(train_indices)} train + {len(test_indices)} test (seed={split['seed']})")

    print("Loading train data...")
    train_data, _                  = load_gnn_data(train_indices, root=args.datapath, load_edge_features=use_edge_attr)
    print("Loading test data...")
    test_data, test_loaded_indices = load_gnn_data(test_indices,  root=args.datapath, load_edge_features=use_edge_attr)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    train_labels = torch.stack([d.y for d in train_data])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    criterion = nn.MSELoss()
    results = {
        "model": args.model, "split_file": args.split_file,
        "epochs": args.epochs, "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        model = make_model(args, edge_dim).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                loss = criterion(out.squeeze(-1), (batch.y - y_mean) / y_std)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            run_result["epoch_times"].append(round(time.time() - epoch_start, 2))
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}")

        test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(model, test_loader, device, y_mean, y_std, criterion, use_edge_attr)
        run_result["test_rmse"] = round(test_rmse, 4)
        run_result["test_mae"]  = round(test_mae, 4)
        run_result["test_r2"]   = round(test_r2, 4)
        run_result["runtime"]   = round(time.time() - run_start, 2)
        run_result["predictions"] = [
            {"index": int(idx), "pred": round(float(p), 6), "true": round(float(t), 6)}
            for idx, p, t in zip(test_loaded_indices, test_preds.tolist(), test_targets.tolist())
        ]
        print(f"Test  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  R2: {test_r2:.4f}")
        results["runs"].append(run_result)

    results["mean_test_rmse"] = round(sum(r["test_rmse"] for r in results["runs"]) / 3, 4)
    results["mean_test_mae"]  = round(sum(r["test_mae"]  for r in results["runs"]) / 3, 4)
    results["mean_test_r2"]   = round(sum(r["test_r2"]   for r in results["runs"]) / 3, 4)
    print(f"\nMean test  RMSE: {results['mean_test_rmse']:.4f}  "
          f"MAE: {results['mean_test_mae']:.4f}  R2: {results['mean_test_r2']:.4f}")

    out_path = Path(__file__).parent / "results" / args.output
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
