import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
from lrgb_cin import LRGBCINDataset, collate_cin
from cin import CIN

DATASET_NAME = "Peptides-struct"
OUT_DIM = 11
DATA_ROOT = str(Path(__file__).parent.parent / "data")


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


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            g = to_device(batch, device)
            out = model(g)
            out_denorm = out * y_std + y_mean
            all_preds.append(out_denorm.cpu())
            all_targets.append(g["y"].cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mae = float((preds - targets).abs().mean())
    return mae, preds, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",             type=str,   required=True, choices=["CIN", "CINpp"])
    parser.add_argument("--datapath",          type=str,   default=DATA_ROOT)
    parser.add_argument("--epochs",            type=int,   default=300)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=1e-3)
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
    parser.add_argument("--output",            type=str,   default=None)
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
        args.output = f"results_struct_{args.model.lower()}.json"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | model: {args.model} | dataset: {DATASET_NAME}")

    print("Loading train dataset...")
    train_ds = LRGBCINDataset(args.datapath, DATASET_NAME, "train")
    print("Loading val dataset...")
    val_ds   = LRGBCINDataset(args.datapath, DATASET_NAME, "val")
    print("Loading test dataset...")
    test_ds  = LRGBCINDataset(args.datapath, DATASET_NAME, "test")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    dims = (train_ds.x0_dim, train_ds.x1_dim, train_ds.x2_dim)

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_cin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cin)

    # Per-target z-score normalization fitted on train labels
    train_labels = torch.stack([train_ds[i]["y"] for i in range(len(train_ds))])  # (N, 11)
    y_mean = train_labels.mean(dim=0, keepdim=True).to(device)
    y_std  = train_labels.std(dim=0,  keepdim=True).clamp(min=1e-6).to(device)

    criterion = nn.L1Loss()
    results = {
        "model": args.model, "dataset": DATASET_NAME, "epochs": args.epochs, "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)
        model = make_model(args, dims).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_cin)
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
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}  loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        val_mae,  _,          _            = evaluate(model, val_loader,  device, y_mean, y_std)
        test_mae, test_preds, test_targets = evaluate(model, test_loader, device, y_mean, y_std)
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
    if args.hp_file:
        out_path = out_path.with_name(f"{out_path.stem}_hptuned{out_path.suffix}")
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
