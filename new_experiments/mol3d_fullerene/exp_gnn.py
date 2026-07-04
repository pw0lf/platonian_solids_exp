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

MOL3D_DIR      = Path(__file__).parent.parent / "mol3d"
FULLERENE_ROOT = Path(__file__).parent.parent.parent / "FullereneNet"

sys.path.insert(0, str(MOL3D_DIR / "data_loader"))
sys.path.insert(0, str(MOL3D_DIR / "models"))
sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
from mol3d_gnn import load_gnn_data as load_mol3d_gnn, NODE_DIM, EDGE_DIM
from gnn_models import GCN, GAT, GIN
from fullerene_loader import load_fullerene_gnn

MOL3D_DATA_ROOT      = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")
MOL3D_SPLIT_FILE     = MOL3D_DIR / "data_split.json"
FULLERENE_SPLIT_FILE = Path(__file__).parent.parent / "fullerene_randomsplit" / "split.json"

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


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
    return rmse, mae, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",              type=str,   required=True, choices=["GCN", "GAT", "GIN"])
    parser.add_argument("--mol3d_datapath",     type=str,   default=MOL3D_DATA_ROOT)
    parser.add_argument("--mol3d_split",        type=str,   default=str(MOL3D_SPLIT_FILE))
    parser.add_argument("--fullerene_split",    type=str,   default=str(FULLERENE_SPLIT_FILE))
    parser.add_argument("--epochs",             type=int,   default=300)
    parser.add_argument("--batch_size",         type=int,   default=32)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--hidden_channels",    type=int,   default=64)
    parser.add_argument("--num_conv_layers",    type=int,   default=3)
    parser.add_argument("--readout_hidden_dim", type=int,   default=64)
    parser.add_argument("--num_readout_layers", type=int,   default=2)
    parser.add_argument("--dropout",            type=float, default=0.0)
    parser.add_argument("--num_heads",          type=int,   default=4,    help="GAT only")
    parser.add_argument("--mlp_hidden_dim",     type=int,   default=None, help="GIN only")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--output",             type=str,   default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_{args.model.lower()}.json"

    use_edge_attr = args.model == "GAT"
    edge_dim = EDGE_DIM if use_edge_attr else None

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | model: {args.model}")

    mol3d_split = json.load(open(args.mol3d_split))
    full_split  = json.load(open(args.fullerene_split))

    print("Loading mol3d data...")
    mol3d_train, _ = load_mol3d_gnn(mol3d_split["train"], root=args.mol3d_datapath, load_edge_features=use_edge_attr)
    mol3d_test,  _ = load_mol3d_gnn(mol3d_split["test"],  root=args.mol3d_datapath, load_edge_features=use_edge_attr)

    print("Loading fullerene data...")
    fullerene_all = load_fullerene_gnn(root=str(FULLERENE_ROOT), target="Gap", load_edge_features=use_edge_attr)
    assert len(fullerene_all) == full_split["n_total"], \
        f"Fullerene size mismatch: {len(fullerene_all)} vs {full_split['n_total']}"
    full_train = [fullerene_all[i] for i in full_split["train_idx"]]
    full_test  = [fullerene_all[i] for i in full_split["test_idx"]]

    train_set = mol3d_train + full_train
    test_set  = mol3d_test  + full_test
    print(f"Train: {len(train_set)} ({len(mol3d_train)} mol3d + {len(full_train)} fullerene)")
    print(f"Test:  {len(test_set)} ({len(mol3d_test)} mol3d + {len(full_test)} fullerene)")

    train_labels = torch.stack([d.y for d in train_set]).squeeze(-1)
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    criterion   = nn.MSELoss()
    results = {
        "model": args.model, "epochs": args.epochs,
        "mol3d_split": args.mol3d_split, "fullerene_split": args.fullerene_split,
        "n_train_mol3d": len(mol3d_train), "n_train_fullerene": len(full_train),
        "n_test_mol3d":  len(mol3d_test),  "n_test_fullerene":  len(full_test),
        "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
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
                loss = criterion(out.squeeze(-1), (batch.y.squeeze(-1) - y_mean) / y_std)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            run_result["epoch_times"].append(round(time.time() - epoch_start, 2))
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}")

        test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device, y_mean, y_std, criterion, use_edge_attr)
        run_result["test_rmse"] = round(test_rmse, 4)
        run_result["test_mae"]  = round(test_mae, 4)
        run_result["test_r2"]   = round(test_r2, 4)
        run_result["runtime"]   = round(time.time() - run_start, 2)
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
