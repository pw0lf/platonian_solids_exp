"""Small hyperparameter search for SchNet on mol3d.

SchNet (PyG's built-in model) has no dropout parameter -- num_filters is
tuned in its place as the 4th knob. Random search trained and validated on
a small subset carved out of the TRAIN split only (see
hp_search_utils.carve_val_from_train) -- this script never loads the test
split.

Note: mol3d/exp_schnet.py hardcodes device="cpu" unconditionally (a
pre-existing quirk, not touched here) -- this tuning script does NOT
replicate that, it uses the normal cuda/mps/cpu autodetection, so it may run
faster than the real exp_schnet.py job would.
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
from mol3d_schnet import load_schnet_data
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "Molecule3D" / "raw")

SEARCH_SPACE = {
    "lr":               [1e-5, 3e-5, 1e-4, 3e-4],
    "hidden_channels":  [64, 128, 256],
    "num_interactions": [2, 4, 6],
    "num_filters":      [64, 128, 256],
    "cutoff":           [5.0, 7.5, 10.0],
}
MAX_TRAIN = 1200
MAX_VAL = 300


def make_model(cfg):
    return SchNet(
        num_interactions=cfg["num_interactions"], hidden_channels=cfg["hidden_channels"],
        num_filters=cfg["num_filters"], cutoff=cfg["cutoff"], readout="mean",
    )


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch) * y_std + y_mean
            preds.append(out.squeeze(-1).cpu())
            targets.append(batch.y.cpu())
    p, t = torch.cat(preds).squeeze(-1), torch.cat(targets).squeeze(-1)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", type=str, default=str(Path(__file__).parent / "data_split.json"))
    parser.add_argument("--datapath",   type=str, default=DATA_ROOT)
    parser.add_argument("--n_trials",   type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--output",     type=str, default="best_hp_schnet.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | tuning: SchNet")

    split = json.load(open(args.split_file))
    train_indices = split["train"]  # test indices are never loaded by this script

    hp_train_idx, hp_val_idx = carve_val_from_train(train_indices, val_frac=0.15, seed=args.seed)
    hp_train_idx = subsample(hp_train_idx, MAX_TRAIN, seed=args.seed)
    hp_val_idx   = subsample(hp_val_idx,   MAX_VAL,   seed=args.seed)
    print(f"HP-train: {len(hp_train_idx)} | HP-val: {len(hp_val_idx)} (carved from train only)")

    train_data, _ = load_schnet_data(hp_train_idx, root=args.datapath)
    val_data, _   = load_schnet_data(hp_val_idx,   root=args.datapath)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    train_labels = torch.stack([d.y for d in train_data])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)

    criterion = nn.MSELoss()
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
                loss = criterion(out.squeeze(-1), (batch.y - y_mean) / y_std)
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
