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
from schnet_dataset import load_schnet_data

DATA_ROOT = str(Path(__file__).parent.parent.parent / "FullereneNet")


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
    return rmse, mae, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",           type=str,   default="Eb")
    parser.add_argument("--epochs",           type=int,   default=10)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--num_interactions", type=int,   default=6)
    parser.add_argument("--hidden_channels",  type=int,   default=256)
    parser.add_argument("--num_filters",      type=int,   default=256)
    parser.add_argument("--cutoff",           type=float, default=10.0)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output",           type=str,   default="results_schnet.json",
                        help="filename (saved inside results/)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print("Loading datasets...")
    all_c60  = load_schnet_data("c60",         root=DATA_ROOT, target=args.target)
    c70_test = load_schnet_data("c70_non_IPR", root=DATA_ROOT, target=args.target)
    c72_test = load_schnet_data("c72_100_IPR", root=DATA_ROOT, target=args.target)

    train_set = [d for d in all_c60 if d.z.shape[0] <= 58]
    c60_test  = [d for d in all_c60 if d.z.shape[0] == 60]

    print(f"Train: {len(train_set)} | Test C60: {len(c60_test)} | "
          f"Test C70 non-IPR: {len(c70_test)} | Test C72-100 IPR: {len(c72_test)}")

    train_labels = torch.stack([d.y for d in train_set])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    c60_loader = DataLoader(c60_test, batch_size=args.batch_size, shuffle=False)
    c70_loader = DataLoader(c70_test, batch_size=args.batch_size, shuffle=False)
    c72_loader = DataLoader(c72_test, batch_size=args.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    gamma_per_step = 0.96 ** (1 / 100_000)
    results = {"target": args.target, "epochs": args.epochs, "runs": []}

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

        for name, loader in [("c60", c60_loader), ("c70_non_IPR", c70_loader), ("c72_100_IPR", c72_loader)]:
            test_rmse, test_mae, test_r2 = evaluate(model, loader, device, y_mean, y_std, criterion)
            run_result[f"test_rmse_{name}"] = round(test_rmse, 4)
            run_result[f"test_mae_{name}"]  = round(test_mae, 4)
            run_result[f"test_r2_{name}"]   = round(test_r2, 4)
            print(f"Test [{name}]  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  R2: {test_r2:.4f}")

        run_result["runtime"] = round(time.time() - run_start, 2)
        results["runs"].append(run_result)

    for name in ["c60", "c70_non_IPR", "c72_100_IPR"]:
        results[f"mean_test_rmse_{name}"] = round(sum(r[f"test_rmse_{name}"] for r in results["runs"]) / 3, 4)
        results[f"mean_test_mae_{name}"]  = round(sum(r[f"test_mae_{name}"]  for r in results["runs"]) / 3, 4)
        results[f"mean_test_r2_{name}"]   = round(sum(r[f"test_r2_{name}"]   for r in results["runs"]) / 3, 4)
        print(f"\nMean test [{name}]  RMSE: {results[f'mean_test_rmse_{name}']:.4f}  "
              f"MAE: {results[f'mean_test_mae_{name}']:.4f}  "
              f"R2: {results[f'mean_test_r2_{name}']:.4f}")

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
