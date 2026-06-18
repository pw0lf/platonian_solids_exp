import argparse
import json
import sys
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

FULLERENE_ROOT = Path(__file__).parent.parent / "FullereneNet"
sys.path.insert(0, str(FULLERENE_ROOT))
from model.FullereneNet import FullereneNet

LABEL_COLUMNS = {
    "HOMO": "HOMO(eV)", "LUMO": "LUMO(eV)", "Gap": "HOMO-LUMO(eV)", "Eb": "E_binding(eV)",
}


def load_pyg_data(node_features, edge_indices, edge_features, labels):
    return [
        Data(x=nf, edge_index=ei, edge_attr=ef, y=y)
        for nf, ei, ef, y in zip(node_features, edge_indices, edge_features, labels)
    ]


def evaluate(model, loader, device, y_mean, y_std, criterion):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch) * y_std + y_mean
            preds.append(out.squeeze(-1).cpu())
            targets.append(batch.y.cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    rmse = criterion(p, t).sqrt().item()
    mae = (p - t).abs().mean().item()
    return rmse, mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",         type=str, default="Eb")
    parser.add_argument("--epochs",         type=int, default=300)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--node_embedding", type=int, default=64)
    parser.add_argument("--num_conv_layer", type=int, default=3)
    parser.add_argument("--hidden_channels",type=int, default=128)
    parser.add_argument("--num_heads",      type=int, default=4)
    parser.add_argument("--dropout",        type=float, default=0.0)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--output",         type=str, default="results_fullerene_net.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    feat = FULLERENE_ROOT / "feature"
    data = FULLERENE_ROOT / "data"

    print("Loading pre-computed features...")
    nf_c60  = torch.load(feat / "node_feature_c60.pt",       weights_only=True)
    ei_c60  = torch.load(feat / "edge_index_c60.pt",         weights_only=True)
    ef_c60  = torch.load(feat / "edge_feature_c60.pt",       weights_only=True)

    nf_c70  = torch.load(feat / "node_feature_c70_non_IPR.pt",  weights_only=True)
    ei_c70  = torch.load(feat / "edge_index_c70_non_IPR.pt",    weights_only=True)
    ef_c70  = torch.load(feat / "edge_feature_c70_non_IPR.pt",  weights_only=True)

    nf_c72  = torch.load(feat / "node_feature_c72_100_IPR.pt",  weights_only=True)
    ei_c72  = torch.load(feat / "edge_index_c72_100_IPR.pt",    weights_only=True)
    ef_c72  = torch.load(feat / "edge_feature_c72_100_IPR.pt",  weights_only=True)

    target_col = LABEL_COLUMNS[args.target]
    labels_c60  = torch.tensor(pd.read_csv(data / "c20-c60-dft-all.csv")[target_col].values,     dtype=torch.float32)
    labels_c70  = torch.tensor(pd.read_csv(data / "c70-100-isomers-Eb-Eg-logP.csv")[target_col].values, dtype=torch.float32)
    labels_c72  = torch.tensor(pd.read_csv(data / "c62-c720-dft-all.csv")[target_col].values,    dtype=torch.float32)

    # split c20-c60 by atom count: train/val on <=58, test on ==60
    all_c60 = load_pyg_data(nf_c60, ei_c60, ef_c60, labels_c60)
    train_val = [d for d in all_c60 if d.num_nodes <= 58]
    c60_test  = [d for d in all_c60 if d.num_nodes == 60]

    c70_test  = load_pyg_data(nf_c70,  ei_c70,  ef_c70,  labels_c70)
    c72_test  = load_pyg_data(nf_c72,  ei_c72,  ef_c72,  labels_c72)

    n_train = int(0.9 * len(train_val))
    n_val   = len(train_val) - n_train
    print(f"Train: {n_train} | Val: {n_val} | Test C60: {len(c60_test)} | "
          f"Test C70 non-IPR: {len(c70_test)} | Test C72-100 IPR: {len(c72_test)}")

    train_labels = torch.stack([d.y for d in train_val[:n_train]])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    criterion = nn.MSELoss()
    results = {"target": args.target, "epochs": args.epochs, "runs": []}

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)
        train_set, val_set = random_split(
            train_val, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed + run),
        )

        train_loader = DataLoader(list(train_set), batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(list(val_set),   batch_size=args.batch_size, shuffle=False)
        c60_loader   = DataLoader(c60_test,         batch_size=args.batch_size, shuffle=False)
        c70_loader   = DataLoader(c70_test,         batch_size=args.batch_size, shuffle=False)
        c72_loader   = DataLoader(c72_test,         batch_size=args.batch_size, shuffle=False)

        model = FullereneNet(
            atom_input_features=4, node_fea=args.node_embedding, edge_fea=9,
            conv_layers=args.num_conv_layer, hidden_layer=args.hidden_channels,
            heads=args.num_heads, dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        run_result = {"run": run + 1, "train_losses": [], "val_rmses": []}
        best_val_rmse = float("inf")

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out.squeeze(-1), (batch.y - y_mean) / y_std)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))

            if (epoch + 1) % 5 == 0:
                val_rmse, val_mae = evaluate(model, val_loader, device, y_mean, y_std, criterion)
                run_result["val_rmses"].append(round(val_rmse, 4))
                run_result.setdefault("val_maes", []).append(round(val_mae, 4))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
                      f"val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}")

        for name, loader in [("c60", c60_loader), ("c70_non_IPR", c70_loader), ("c72_100_IPR", c72_loader)]:
            test_rmse, test_mae = evaluate(model, loader, device, y_mean, y_std, criterion)
            run_result[f"test_rmse_{name}"] = round(test_rmse, 4)
            run_result[f"test_mae_{name}"]  = round(test_mae, 4)
            print(f"Test [{name}]  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}")
        results["runs"].append(run_result)

    for name in ["c60", "c70_non_IPR", "c72_100_IPR"]:
        results[f"mean_test_rmse_{name}"] = round(sum(r[f"test_rmse_{name}"] for r in results["runs"]) / 3, 4)
        results[f"mean_test_mae_{name}"]  = round(sum(r[f"test_mae_{name}"]  for r in results["runs"]) / 3, 4)
        print(f"\nMean test [{name}]  RMSE: {results[f'mean_test_rmse_{name}']:.4f}  "
              f"MAE: {results[f'mean_test_mae_{name}']:.4f}")

    out_path = Path(args.output)
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
