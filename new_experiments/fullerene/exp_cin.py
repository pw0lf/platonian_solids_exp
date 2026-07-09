import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
from fullerene_cin import FullereneComplexCINDataset, collate_cin
from cin import CIN

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT = str(Path(__file__).parent.parent.parent / "FullereneNet")
OUT_DIM = 1


def to_device(g, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in g.items()}


def make_model(args, dims):
    return CIN(
        x0_dim=dims[0], x1_dim=dims[1], x2_dim=dims[2], out_dim=OUT_DIM,
        num_layers=args.num_layers, hidden=args.hidden, variant=args.model,
        use_coboundaries=args.use_coboundaries, dropout=args.dropout,
        in_dropout=args.in_dropout, readout=args.readout,
        final_readout=args.final_readout, train_eps=args.train_eps,
    )


def evaluate(model, loader, device, y_mean, y_std, criterion):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            g = to_device(batch, device)
            out = model(g)
            preds.append((out * y_std + y_mean).cpu()); targets.append(g["y"].cpu())
    p, t = torch.cat(preds).squeeze(-1), torch.cat(targets).squeeze(-1)
    rmse = criterion(p, t).sqrt().item()
    mae = (p - t).abs().mean().item()
    r2 = (1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()
    return rmse, mae, r2, p, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",             type=str,   required=True, choices=["CIN", "CINpp"])
    parser.add_argument("--target",            type=str,   default="Eb")
    parser.add_argument("--epochs",            type=int,   default=300)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=5e-4)
    parser.add_argument("--warmup_epochs",     type=int,   default=5)
    parser.add_argument("--hidden",            type=int,   default=128)
    parser.add_argument("--num_layers",        type=int,   default=4)
    parser.add_argument("--dropout",           type=float, default=0.0)
    parser.add_argument("--in_dropout",        type=float, default=0.0)
    parser.add_argument("--use_coboundaries",  action="store_true", default=True)
    parser.add_argument("--no_use_coboundaries", dest="use_coboundaries", action="store_false")
    parser.add_argument("--train_eps",         action="store_true", default=False)
    parser.add_argument("--readout",           type=str,   default="mean", choices=["mean", "sum"])
    parser.add_argument("--final_readout",     type=str,   default="sum", choices=["mean", "sum"])
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--output",            type=str,   default=None,
                        help="filename (saved inside results/); defaults to results_<model>.json")
    parser.add_argument("--hp_file",           type=str,   default=None,
                        help="JSON from hp_tuning_cin.py / hp_tuning_cinpp.py. Values for keys "
                             "present in the file unconditionally override this script's CLI "
                             "defaults for lr/hidden/num_layers/dropout/in_dropout/readout/"
                             "final_readout -- even if you also pass those flags explicitly.")
    args = parser.parse_args()

    if args.hp_file:
        with open(args.hp_file) as f:
            hp = json.load(f)
        hp_keys = ("lr", "hidden", "num_layers", "dropout", "in_dropout", "readout", "final_readout")
        for key in hp_keys:
            if key in hp:
                setattr(args, key, hp[key])
        print(f"Loaded hyperparameters from {args.hp_file}: "
              f"{ {k: getattr(args, k) for k in hp_keys} }")

    if args.output is None:
        args.output = f"results_{args.model.lower()}.json"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | model: {args.model} | target: {args.target}")

    print("Loading datasets...")
    # c60 dataset spans C20-C60: train/val on C20-C58, internal test on C60 (held out by size)
    c20_c58_c60 = FullereneComplexCINDataset("c60", root=DATA_ROOT, target=args.target)
    train_val_idx = [i for i in range(len(c20_c58_c60)) if c20_c58_c60[i]["n_atoms"] <= 58]
    c60_test_idx  = [i for i in range(len(c20_c58_c60)) if c20_c58_c60[i]["n_atoms"] == 60]

    train_set    = Subset(c20_c58_c60, train_val_idx)
    c60_test_set = Subset(c20_c58_c60, c60_test_idx)

    # external test sets: C70 non-IPR and C72-C100 IPR
    c70_test_set     = FullereneComplexCINDataset("c70_non_IPR", root=DATA_ROOT, target=args.target)
    c72_100_test_set = FullereneComplexCINDataset("c72_100_IPR", root=DATA_ROOT, target=args.target)

    dims = (c20_c58_c60.x0_dim, c20_c58_c60.x1_dim, c20_c58_c60.x2_dim)
    print(f"Loaded: x0={dims[0]} x1={dims[1]} x2={dims[2]}")
    print(f"Train: {len(train_set)} | "
          f"Test C60: {len(c60_test_set)} | Test C70 non-IPR: {len(c70_test_set)} | "
          f"Test C72-100 IPR: {len(c72_100_test_set)}")

    c60_test_loader = DataLoader(c60_test_set,      batch_size=args.batch_size, shuffle=False, collate_fn=collate_cin)
    c70_test_loader = DataLoader(c70_test_set,      batch_size=args.batch_size, shuffle=False, collate_fn=collate_cin)
    c72_test_loader = DataLoader(c72_100_test_set,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_cin)

    # z-score normalization fitted on train labels only
    train_labels = torch.stack([train_set[i]["y"] for i in range(len(train_set))])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    criterion = nn.MSELoss()
    results = {"model": args.model, "target": args.target, "epochs": args.epochs, "runs": []}

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)
        model = make_model(args, dims).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-4)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_cin)
        run_result = {"run": run + 1, "train_losses": [], "epoch_times": []}
        run_start = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            for batch in train_loader:
                g = to_device(batch, device)
                optimizer.zero_grad()
                out = model(g)
                loss = criterion(out, (g["y"] - y_mean) / y_std)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            run_result["epoch_times"].append(round(time.time() - epoch_start, 2))
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  lr={lr:.2e}")

        predictions = []
        for name, loader in [("c60", c60_test_loader),
                             ("c70_non_IPR", c70_test_loader),
                             ("c72_100_IPR", c72_test_loader)]:
            test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
                model, loader, device, y_mean, y_std, criterion)
            run_result[f"test_rmse_{name}"] = round(test_rmse, 4)
            run_result[f"test_mae_{name}"]  = round(test_mae, 4)
            run_result[f"test_r2_{name}"]   = round(test_r2, 4)
            predictions.extend(
                {"test_set": name, "index": idx, "pred": round(float(p), 6), "true": round(float(t), 6)}
                for idx, (p, t) in enumerate(zip(test_preds.tolist(), test_targets.tolist()))
            )
            print(f"Test [{name}]  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  R2: {test_r2:.4f}")
        run_result["predictions"] = predictions
        run_result["runtime_s"] = round(time.time() - run_start, 2)
        results["runs"].append(run_result)

    for name in ["c60", "c70_non_IPR", "c72_100_IPR"]:
        results[f"mean_test_rmse_{name}"] = round(sum(r[f"test_rmse_{name}"] for r in results["runs"]) / 3, 4)
        results[f"mean_test_mae_{name}"]  = round(sum(r[f"test_mae_{name}"]  for r in results["runs"]) / 3, 4)
        results[f"mean_test_r2_{name}"]   = round(sum(r[f"test_r2_{name}"]   for r in results["runs"]) / 3, 4)
        print(f"\nMean test [{name}]  RMSE: {results[f'mean_test_rmse_{name}']:.4f}  "
              f"MAE: {results[f'mean_test_mae_{name}']:.4f}  "
              f"R2: {results[f'mean_test_r2_{name}']:.4f}")

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
