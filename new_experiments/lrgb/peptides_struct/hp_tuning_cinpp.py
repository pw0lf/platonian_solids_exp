"""Small hyperparameter search for CIN++ (variant="CINpp") on lrgb/peptides_struct.

CIN's architecture is already fully CLI-exposed in exp_cin.py
(--lr --hidden --num_layers --dropout), so this script's SEARCH_SPACE keys
map straight onto CIN's own constructor kwargs.

Subsamples the REAL train/val splits directly (LRGB ships them separately,
no split.json to carve from) -- this script never loads the test split.
Subsampling happens on the raw PyG *indices* before any per-molecule
processing (CIN's cell-complex construction runs networkx ring detection
per molecule, ~115ms/mol -- building the full 10873-molecule train set
first and subsampling after, like the real LRGBCINDataset class does, would
cost ~21 minutes regardless of how small MAX_TRAIN is).
"""
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
sys.path.insert(0, str(Path(__file__).parent))
from lrgb_cin import _LRGBNoDownload, _process_molecule, collate_cin
from cin import CIN
from hp_search_utils import subsample, sample_configs, save_best

DATASET_NAME = "Peptides-struct"
OUT_DIM = 11
DATA_ROOT = str(Path(__file__).parent.parent / "data")

SEARCH_SPACE = {
    "lr":            [1e-4, 3e-4, 1e-3, 3e-3],
    "hidden":        [32, 64, 128],
    "num_layers":    [2, 3, 4],
    "dropout":       [0.0, 0.1, 0.2, 0.3],
    "in_dropout":    [0.0, 0.1, 0.2],
    "readout":       ["mean", "sum"],
    "final_readout": ["mean", "sum"],
}
MAX_TRAIN = 1200
MAX_VAL = 300


def load_subset(datapath, split, max_n, seed):
    """Loads at most max_n molecules from `split`, subsampling the raw PyG
    indices BEFORE running the per-molecule cell-complex construction --
    never constructs the full split."""
    pyg_ds = _LRGBNoDownload(root=datapath, name=DATASET_NAME, split=split)
    raw_idx = subsample(range(len(pyg_ds)), max_n, seed=seed)
    data = []
    for i in raw_idx:
        try:
            data.append(_process_molecule(pyg_ds[i]))
        except Exception:
            pass
    return data, len(pyg_ds)


def to_device(g, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in g.items()}


def make_model(cfg, dims, variant="CIN"):
    return CIN(
        x0_dim=dims[0], x1_dim=dims[1], x2_dim=dims[2], out_dim=OUT_DIM,
        num_layers=cfg["num_layers"], hidden=cfg["hidden"], variant=variant,
        use_coboundaries=True, dropout=cfg["dropout"], in_dropout=cfg["in_dropout"],
        readout=cfg["readout"], final_readout=cfg["final_readout"], train_eps=False,
    )


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            g = to_device(batch, device)
            out = model(g)
            preds.append((out * y_std + y_mean).cpu()); targets.append(g["y"].cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",   type=str, default=DATA_ROOT)
    parser.add_argument("--n_trials",   type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--output",     type=str, default="best_hp_cinpp.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | tuning: CINpp")

    print("Loading train/val subsets (test never loaded by this script)...")
    train_data, n_train_total = load_subset(args.datapath, "train", MAX_TRAIN, args.seed)
    val_data, n_val_total     = load_subset(args.datapath, "val",   MAX_VAL,   args.seed)
    print(f"HP-train: {len(train_data)} (of {n_train_total} real train) | "
          f"HP-val: {len(val_data)} (of {n_val_total} real val)")

    dims = (train_data[0]["x_0"].shape[1], train_data[0]["x_1"].shape[1], train_data[0]["x_2"].shape[1])
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cin)

    train_labels = torch.stack([d["y"] for d in train_data])
    y_mean = train_labels.mean(dim=0, keepdim=True).to(device)
    y_std  = train_labels.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)

    criterion = nn.L1Loss()
    configs = sample_configs(SEARCH_SPACE, n_trials=args.n_trials, seed=args.seed)

    trials = []
    best_cfg, best_val = None, float("inf")
    for i, cfg in enumerate(configs):
        print(f"\n--- Trial {i + 1}/{len(configs)}: {cfg} ---")
        torch.manual_seed(args.seed)
        model = make_model(cfg, dims, variant="CINpp").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_cin)

        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                g = to_device(batch, device)
                optimizer.zero_grad()
                out = model(g)
                loss = criterion(out, (g["y"] - y_mean) / y_std)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        val_mae = evaluate(model, val_loader, device, y_mean, y_std)
        print(f"val_mae={val_mae:.4f}  ({time.time() - t0:.1f}s)")
        trials.append({"config": cfg, "val_mae": round(val_mae, 6)})
        if val_mae < best_val:
            best_val, best_cfg = val_mae, cfg

    print(f"\nBest config: {best_cfg}  val_mae={best_val:.4f}")
    out_path = Path(__file__).parent / "results" / args.output
    save_best(out_path, best_cfg, best_val, trials)
    print(f"Saved to {out_path}")
