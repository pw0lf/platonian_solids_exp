import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

MOL3D_DIR      = Path(__file__).parent.parent / "mol3d"
FULLERENE_ROOT = Path(__file__).parent.parent.parent / "FullereneNet"

sys.path.insert(0, str(MOL3D_DIR / "data_loader"))
sys.path.insert(0, str(MOL3D_DIR / "models"))
sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
from mol3d_cin import Mol3dCIN, collate_cin
from cin import CIN
from fullerene_loader import load_fullerene_cin

MOL3D_DATA_ROOT      = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")
MOL3D_SPLIT_FILE     = MOL3D_DIR / "data_split.json"
FULLERENE_SPLIT_FILE = Path(__file__).parent.parent / "fullerene_randomsplit" / "split.json"
OUT_DIM = 1

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


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
    parser.add_argument("--mol3d_datapath",    type=str,   default=MOL3D_DATA_ROOT)
    parser.add_argument("--mol3d_split",       type=str,   default=str(MOL3D_SPLIT_FILE))
    parser.add_argument("--fullerene_split",   type=str,   default=str(FULLERENE_SPLIT_FILE))
    parser.add_argument("--epochs",            type=int,   default=300)
    parser.add_argument("--batch_size",        type=int,   default=16)
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
        args.output = f"results_{args.model.lower()}.json"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | model: {args.model}")

    mol3d_split = json.load(open(args.mol3d_split))
    full_split  = json.load(open(args.fullerene_split))

    print("Loading mol3d train data...")
    mol3d_train_ds = Mol3dCIN(mol3d_split["train"], root=args.mol3d_datapath)
    print("Loading mol3d test data...")
    mol3d_test_ds  = Mol3dCIN(mol3d_split["test"],  root=args.mol3d_datapath)

    dims = (mol3d_train_ds.x0_dim, mol3d_train_ds.x1_dim, mol3d_train_ds.x2_dim)
    print(f"CIN dims: x0={dims[0]} x1={dims[1]} x2={dims[2]}")

    print("Loading fullerene data...")
    fullerene_all = load_fullerene_cin(root=str(FULLERENE_ROOT), target="Gap")
    assert len(fullerene_all) == full_split["n_total"], \
        f"Fullerene size mismatch: {len(fullerene_all)} vs {full_split['n_total']}"
    full_train = [fullerene_all[i] for i in full_split["train_idx"]]
    full_test  = [fullerene_all[i] for i in full_split["test_idx"]]

    train_dataset = ListDataset(list(mol3d_train_ds) + full_train)
    test_dataset  = ListDataset(list(mol3d_test_ds)  + full_test)
    print(f"Train: {len(train_dataset)} ({len(mol3d_train_ds)} mol3d + {len(full_train)} fullerene)")
    print(f"Test:  {len(test_dataset)} ({len(mol3d_test_ds)} mol3d + {len(full_test)} fullerene)")

    train_labels = torch.stack([train_dataset[i]["y"] for i in range(len(train_dataset))])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cin)
    criterion   = nn.MSELoss()
    results = {
        "model": args.model, "epochs": args.epochs,
        "mol3d_split": args.mol3d_split, "fullerene_split": args.fullerene_split,
        "n_train_mol3d": len(mol3d_train_ds), "n_train_fullerene": len(full_train),
        "n_test_mol3d":  len(mol3d_test_ds),  "n_test_fullerene":  len(full_test),
        "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_cin)
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

        test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
            model, test_loader, device, y_mean, y_std, criterion)
        run_result["test_rmse"] = round(test_rmse, 4)
        run_result["test_mae"]  = round(test_mae, 4)
        run_result["test_r2"]   = round(test_r2, 4)
        run_result["runtime"]   = round(time.time() - run_start, 2)
        n_mol3d_test = len(mol3d_test_ds)
        run_result["predictions"] = (
            [
                {"source": "mol3d", "index": int(idx), "pred": round(float(p), 6), "true": round(float(t), 6)}
                for idx, p, t in zip(mol3d_test_ds.indices,
                                     test_preds[:n_mol3d_test].tolist(),
                                     test_targets[:n_mol3d_test].tolist())
            ]
            + [
                {"source": "fullerene", "index": int(idx), "pred": round(float(p), 6), "true": round(float(t), 6)}
                for idx, p, t in zip(full_split["test_idx"],
                                     test_preds[n_mol3d_test:].tolist(),
                                     test_targets[n_mol3d_test:].tolist())
            ]
        )
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
