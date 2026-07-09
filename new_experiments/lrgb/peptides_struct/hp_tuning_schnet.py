"""Small hyperparameter search for SchNet on lrgb/peptides_struct.

SchNet has no dropout parameter -- num_filters is tuned in its place, same
as mol3d/hp_tuning_schnet.py. Requires the SMILES CSVs (smiles_train.csv
etc.) same as exp_schnet.py -- the SMILES<->PyG alignment step runs once
(~30s) regardless of how much data is subsampled afterward.

Subsamples the REAL train/val splits directly -- this script never loads
the test split.
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
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet

sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent))
from lrgb_schnet import load_schnet_data, align_smiles_to_splits
from hp_search_utils import subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATASET_NAME = "Peptides-struct"
OUT_DIM = 11
DATA_ROOT = str(Path(__file__).parent.parent / "data")
SMILES_DIR = Path(__file__).parent.parent

SEARCH_SPACE = {
    "lr":               [1e-5, 3e-5, 1e-4, 3e-4],
    "hidden_channels":  [64, 128, 256],
    "num_interactions": [2, 4, 6],
    "num_filters":      [64, 128, 256],
    "cutoff":           [5.0, 7.5, 10.0],
}
MAX_TRAIN = 800
MAX_VAL = 200


def make_model(cfg):
    return SchNet(
        num_interactions=cfg["num_interactions"], hidden_channels=cfg["hidden_channels"],
        num_filters=cfg["num_filters"], cutoff=cfg["cutoff"], readout="mean", out_channels=OUT_DIM,
    )


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch) * y_std + y_mean
            preds.append(out.cpu())
            targets.append(batch.y.cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",   type=str, default=DATA_ROOT)
    parser.add_argument("--n_trials",   type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--output",     type=str, default="best_hp_schnet.json")
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        p = SMILES_DIR / f"smiles_{split}.csv"
        assert p.exists(), f"Missing {p}. Run: python new_experiments/lrgb/download_smiles.py"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | tuning: SchNet")

    print("Aligning SMILES to PyG graphs by content (may take under a minute)...")
    smiles_csv_paths = {s: str(SMILES_DIR / f"smiles_{s}.csv") for s in ("train", "val", "test")}
    smiles_pool, perms = align_smiles_to_splits(args.datapath, DATASET_NAME, smiles_csv_paths)

    print("Loading train/val splits (test never loaded by this script)...")
    train_data, _ = load_schnet_data(args.datapath, DATASET_NAME, "train", smiles_pool, perms["train"])
    val_data, _   = load_schnet_data(args.datapath, DATASET_NAME, "val",   smiles_pool, perms["val"])

    hp_train_idx = subsample(range(len(train_data)), MAX_TRAIN, seed=args.seed)
    hp_val_idx   = subsample(range(len(val_data)),   MAX_VAL,   seed=args.seed)
    print(f"HP-train: {len(hp_train_idx)} (of {len(train_data)} real train) | "
          f"HP-val: {len(hp_val_idx)} (of {len(val_data)} real val)")

    train_data = [train_data[i] for i in hp_train_idx]
    val_data   = [val_data[i]   for i in hp_val_idx]
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    train_labels = torch.cat([d.y for d in train_data])
    y_mean = train_labels.mean(dim=0, keepdim=True).to(device)
    y_std  = train_labels.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)

    criterion = nn.L1Loss()
    configs = sample_configs(SEARCH_SPACE, n_trials=args.n_trials, seed=args.seed)

    trials = []
    best_cfg, best_val = None, float("inf")
    for i, cfg in enumerate(configs):
        print(f"\n--- Trial {i + 1}/{len(configs)}: {cfg} ---")
        torch.manual_seed(args.seed)
        model = make_model(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.z, batch.pos, batch.batch)
                loss = criterion(out, (batch.y - y_mean) / y_std)
                loss.backward()
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
