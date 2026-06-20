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
from torch_geometric.nn.models import SchNet
from torch.optim.lr_scheduler import ExponentialLR

sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
from lrgb_schnet import load_schnet_data

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATASET_NAME = "Peptides-struct"
OUT_DIM = 11
DATA_ROOT = str(Path(__file__).parent.parent / "data")
SMILES_DIR = Path(__file__).parent.parent


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch)
            out_denorm = out * y_std + y_mean
            all_preds.append(out_denorm.cpu())
            all_targets.append(batch.y.cpu())
    preds = torch.cat(all_preds)    # (N, 11)
    targets = torch.cat(all_targets)  # (N, 11)
    mae = float((preds - targets).abs().mean())
    return mae, preds, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",         type=str,   default=DATA_ROOT)
    parser.add_argument("--epochs",           type=int,   default=300)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--num_interactions", type=int,   default=6)
    parser.add_argument("--hidden_channels",  type=int,   default=256)
    parser.add_argument("--num_filters",      type=int,   default=256)
    parser.add_argument("--cutoff",           type=float, default=10.0)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output",           type=str,   default="results_struct_schnet.json")
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        p = SMILES_DIR / f"smiles_{split}.csv"
        assert p.exists(), f"Missing {p}. Run: python new_experiments/lrgb/download_smiles.py"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device} | dataset: {DATASET_NAME}")

    print("Loading train data...")
    train_data, _ = load_schnet_data(
        args.datapath, DATASET_NAME, "train", str(SMILES_DIR / "smiles_train.csv"))
    print("Loading val data...")
    val_data, val_idx = load_schnet_data(
        args.datapath, DATASET_NAME, "val", str(SMILES_DIR / "smiles_val.csv"))
    print("Loading test data...")
    test_data, test_idx = load_schnet_data(
        args.datapath, DATASET_NAME, "test", str(SMILES_DIR / "smiles_test.csv"))
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    val_loader  = DataLoader(val_data,  batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Per-target normalization from train labels
    train_labels = torch.cat([d.y for d in train_data])  # (N, 11)
    y_mean = train_labels.mean(dim=0, keepdim=True).to(device)
    y_std  = train_labels.std(dim=0,  keepdim=True).clamp(min=1e-6).to(device)

    criterion = nn.L1Loss()
    gamma_per_step = 0.96 ** (1 / 100_000)
    results = {
        "dataset": DATASET_NAME, "epochs": args.epochs, "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        model = SchNet(
            num_interactions=args.num_interactions,
            hidden_channels=args.hidden_channels,
            num_filters=args.num_filters,
            cutoff=args.cutoff,
            readout="mean",
            out_channels=OUT_DIM,
        ).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = ExponentialLR(optimizer, gamma=gamma_per_step)

        run_result = {"run": run + 1, "train_losses": [], "epoch_times": []}
        run_start = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.z, batch.pos, batch.batch)
                loss = criterion(out, (batch.y - y_mean) / y_std)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            run_result["epoch_times"].append(round(time.time() - epoch_start, 2))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}  loss={train_loss:.4f}")

        val_mae,  _,          _            = evaluate(model, val_loader,  device, y_mean, y_std)
        test_mae, test_preds, test_targets = evaluate(model, test_loader, device, y_mean, y_std)
        run_result["val_mae"]   = round(val_mae,  4)
        run_result["test_mae"]  = round(test_mae, 4)
        run_result["runtime"]   = round(time.time() - run_start, 2)
        run_result["predictions"] = [
            {"index": int(idx),
             "pred": [round(float(v), 6) for v in pred.tolist()],
             "true": [round(float(v), 6) for v in tgt.tolist()]}
            for idx, pred, tgt in zip(test_idx, test_preds.tolist(), test_targets.tolist())
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
