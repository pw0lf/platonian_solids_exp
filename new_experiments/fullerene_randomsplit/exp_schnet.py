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

FULLERENE_DIR = Path(__file__).parent.parent / "fullerene"
sys.path.insert(0, str(FULLERENE_DIR / "data_loader"))
from schnet_dataset import load_schnet_data

DATA_ROOT = str(Path(__file__).parent.parent.parent / "FullereneNet")
SPLIT_FILE = Path(__file__).parent / "split.json"


def evaluate(model, loader, device, y_mean, y_std, criterion):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch) * y_std + y_mean
            preds.append(out.squeeze(-1).cpu())
            targets.append(batch.y.cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    rmse = criterion(p, t).sqrt().item()
    mae = (p - t).abs().mean().item()
    r2 = (1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()
    return rmse, mae, r2, p, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",           type=str,   default="Eb")
    parser.add_argument("--epochs",           type=int,   default=300)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--num_interactions", type=int,   default=6)
    parser.add_argument("--hidden_channels",  type=int,   default=256)
    parser.add_argument("--num_filters",      type=int,   default=256)
    parser.add_argument("--cutoff",           type=float, default=10.0)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output",           type=str,   default="results_schnet.json",
                        help="filename (saved inside results/)")
    parser.add_argument("--hp_file",          type=str,   default=None,
                        help="JSON from fullerene/hp_tuning_schnet.py (shared tuning). Values for "
                             "keys present in the file unconditionally override this script's CLI "
                             "defaults for lr/hidden_channels/num_interactions/num_filters/cutoff -- "
                             "even if you also pass those flags explicitly.")
    args = parser.parse_args()

    if args.hp_file:
        with open(args.hp_file) as f:
            hp = json.load(f)
        for key in ("lr", "hidden_channels", "num_interactions", "num_filters", "cutoff"):
            if key in hp:
                setattr(args, key, hp[key])
        print(f"Loaded hyperparameters from {args.hp_file}: "
              f"{ {k: getattr(args, k) for k in ('lr', 'hidden_channels', 'num_interactions', 'num_filters', 'cutoff')} }")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = "cpu"
    print("Device hardcoded")
    print(f"Device: {device}")

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    print(f"Split: {split['n_train']} train / {split['n_test']} test (seed={split['seed']})")

    print("Loading datasets...")
    all_data = (
        load_schnet_data("c60",         root=DATA_ROOT, target=args.target) +
        load_schnet_data("c70_non_IPR", root=DATA_ROOT, target=args.target) +
        load_schnet_data("c72_100_IPR", root=DATA_ROOT, target=args.target)
    )
    assert len(all_data) == split["n_total"], \
        f"Data size mismatch: {len(all_data)} vs split n_total={split['n_total']}"

    train_set = [all_data[i] for i in split["train_idx"]]
    test_set  = [all_data[i] for i in split["test_idx"]]

    train_labels = torch.stack([d.y for d in train_set])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    criterion   = nn.MSELoss()
    gamma_per_step = 0.96 ** (1 / 100_000)
    results = {"target": args.target, "epochs": args.epochs, "split_seed": split["seed"], "runs": []}

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        model = SchNet(
            num_interactions=args.num_interactions,
            hidden_channels=args.hidden_channels,
            num_filters=args.num_filters,
            cutoff=args.cutoff,
            readout="mean",
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
                loss = criterion(out.squeeze(-1), (batch.y - y_mean) / y_std)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            run_result["epoch_times"].append(round(time.time() - epoch_start, 2))
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}")

        test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
            model, test_loader, device, y_mean, y_std, criterion)
        run_result["test_rmse"] = round(test_rmse, 4)
        run_result["test_mae"]  = round(test_mae, 4)
        run_result["test_r2"]   = round(test_r2, 4)
        run_result["predictions"] = [
            {"index": idx, "pred": round(float(p), 6), "true": round(float(t), 6)}
            for idx, p, t in zip(split["test_idx"], test_preds.tolist(), test_targets.tolist())
        ]
        run_result["runtime"]   = round(time.time() - run_start, 2)
        print(f"Test  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  R2: {test_r2:.4f}")
        results["runs"].append(run_result)

    results["mean_test_rmse"] = round(sum(r["test_rmse"] for r in results["runs"]) / 3, 4)
    results["mean_test_mae"]  = round(sum(r["test_mae"]  for r in results["runs"]) / 3, 4)
    results["mean_test_r2"]   = round(sum(r["test_r2"]   for r in results["runs"]) / 3, 4)
    print(f"\nMean test  RMSE: {results['mean_test_rmse']:.4f}  "
          f"MAE: {results['mean_test_mae']:.4f}  R2: {results['mean_test_r2']:.4f}")

    out_path = Path(__file__).parent / "results" / args.output
    if args.hp_file:
        out_path = out_path.with_name(f"{out_path.stem}_hptuned{out_path.suffix}")
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
