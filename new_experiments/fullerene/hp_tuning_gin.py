"""Small hyperparameter search for GIN on fullerene (chem_features="full",
this search's fixed richest/default feature variant).

Carves a small hp-train/hp-val pair out of the <=58-atom TRAIN portion of the
"c60" group only (see hp_search_utils.carve_val_from_train) -- this script
never constructs the C60==60 / C70_non_IPR / C72_100_IPR held-out test sets
at all. Saves the winning config to results/best_hp_gin.json for
exp_gnn.py --hp_file to pick up (shared by fullerene and
fullerene_randomsplit).
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

sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
from gnn_dataset import load_gnn_data, NODE_DIM
from gnn_models import GIN
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT = str(Path(__file__).parent.parent.parent / "FullereneNet")

SEARCH_SPACE = {
    "lr":                 [1e-4, 3e-4, 1e-3, 3e-3],
    "hidden_channels":    [32, 64, 128],
    "num_conv_layers":    [2, 3, 4],
    "dropout":            [0.0, 0.1, 0.2, 0.3],
    "readout_hidden_dim": [32, 64, 128],
    "num_readout_layers": [1, 2, 3],
    "mlp_hidden_dim":     [32, 64, 128],
}
MAX_TRAIN = 1200
MAX_VAL = 300


def make_model(cfg):
    return GIN(
        in_channels=NODE_DIM["full"], hidden_channels=cfg["hidden_channels"],
        num_conv_layers=cfg["num_conv_layers"], readout_hidden_dim=cfg["readout_hidden_dim"],
        num_readout_layers=cfg["num_readout_layers"], dropout=cfg["dropout"],
        mlp_hidden_dim=cfg["mlp_hidden_dim"],
    )


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=None)
            preds.append((out.squeeze(-1) * y_std + y_mean).cpu())
            targets.append(batch.y.cpu())
    p, t = torch.cat(preds).squeeze(-1), torch.cat(targets).squeeze(-1)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     type=str, default="Eb")
    parser.add_argument("--n_trials",   type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--output",     type=str, default="best_hp_gin.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | tuning: GIN")

    print("Loading c60 group (train range only)...")
    all_c60 = load_gnn_data("c60", root=DATA_ROOT, target=args.target, chem_features="full")
    train_indices = [i for i, d in enumerate(all_c60) if d.num_nodes <= 58]
    # Note: this script never constructs the c60==60 / c70_non_IPR / c72_100_IPR
    # test sets at all -- there is nothing to leak from.

    hp_train_idx, hp_val_idx = carve_val_from_train(train_indices, val_frac=0.15, seed=args.seed)
    hp_train_idx = subsample(hp_train_idx, MAX_TRAIN, seed=args.seed)
    hp_val_idx   = subsample(hp_val_idx,   MAX_VAL,   seed=args.seed)
    print(f"HP-train: {len(hp_train_idx)} | HP-val: {len(hp_val_idx)} (carved from train only)")

    train_data = [all_c60[i] for i in hp_train_idx]
    val_data   = [all_c60[i] for i in hp_val_idx]
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
                out = model(batch.x, batch.edge_index, batch.batch, edge_attr=None)
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
