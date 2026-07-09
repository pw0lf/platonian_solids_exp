"""Small hyperparameter search for CIN (variant="CIN") on fullerene.

CIN's architecture is already fully CLI-exposed in exp_cin.py
(--lr --hidden --num_layers --dropout), so this script's SEARCH_SPACE keys
map straight onto CIN's own constructor kwargs.

Carves a small hp-train/hp-val pair out of the <=58-atom TRAIN portion of the
"c60" group only -- this script never constructs the C60==60 / C70_non_IPR /
C72_100_IPR held-out test sets. Saves the winning config to
results/best_hp_cin.json for exp_cin.py --hp_file to pick up (shared by
fullerene and fullerene_randomsplit).
"""
import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
from fullerene_cin import FullereneComplexCINDataset, collate_cin
from cin import CIN
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT = str(Path(__file__).parent.parent.parent / "FullereneNet")
OUT_DIM = 1

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
    p, t = torch.cat(preds).squeeze(-1), torch.cat(targets).squeeze(-1)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     type=str, default="Eb")
    parser.add_argument("--n_trials",   type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--output",     type=str, default="best_hp_cin.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | tuning: CIN")

    print("Loading c60 group (train range only)...")
    c20_c58_c60 = FullereneComplexCINDataset("c60", root=DATA_ROOT, target=args.target)
    train_indices = [i for i in range(len(c20_c58_c60)) if c20_c58_c60[i]["n_atoms"] <= 58]

    hp_train_idx, hp_val_idx = carve_val_from_train(train_indices, val_frac=0.15, seed=args.seed)
    hp_train_idx = subsample(hp_train_idx, MAX_TRAIN, seed=args.seed)
    hp_val_idx   = subsample(hp_val_idx,   MAX_VAL,   seed=args.seed)
    print(f"HP-train: {len(hp_train_idx)} | HP-val: {len(hp_val_idx)} (carved from train only)")

    train_data = [c20_c58_c60[i] for i in hp_train_idx]
    val_data   = [c20_c58_c60[i] for i in hp_val_idx]
    dims = (c20_c58_c60.x0_dim, c20_c58_c60.x1_dim, c20_c58_c60.x2_dim)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_cin)

    train_labels = torch.stack([d["y"] for d in train_data])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)

    criterion = nn.MSELoss()
    configs = sample_configs(SEARCH_SPACE, n_trials=args.n_trials, seed=args.seed)

    trials = []
    best_cfg, best_val = None, float("inf")
    for i, cfg in enumerate(configs):
        print(f"\n--- Trial {i + 1}/{len(configs)}: {cfg} ---")
        torch.manual_seed(args.seed)
        model = make_model(cfg, dims, variant="CIN").to(device)
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
