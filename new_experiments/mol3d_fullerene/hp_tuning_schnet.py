"""Small hyperparameter search for SchNet on mol3d_fullerene (combined mol3d
+ fullerene dataset). See hp_tuning_gcn.py in this directory for the shared
combined-dataset rationale. SchNet has no dropout parameter -- num_filters
is tuned in its place, same as mol3d/hp_tuning_schnet.py.
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

MOL3D_DIR      = Path(__file__).parent.parent / "mol3d"
FULLERENE_ROOT = Path(__file__).parent.parent.parent / "FullereneNet"

sys.path.insert(0, str(MOL3D_DIR / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent))
from mol3d_schnet import load_schnet_data as load_mol3d_schnet
from fullerene_loader import load_fullerene_schnet
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MOL3D_DATA_ROOT       = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "Molecule3D" / "raw")
MOL3D_SPLIT_FILE      = MOL3D_DIR / "data_split.json"
FULLERENE_SPLIT_FILE  = Path(__file__).parent.parent / "fullerene_randomsplit" / "split.json"

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
    parser.add_argument("--mol3d_datapath",  type=str, default=MOL3D_DATA_ROOT)
    parser.add_argument("--mol3d_split",     type=str, default=str(MOL3D_SPLIT_FILE))
    parser.add_argument("--fullerene_split", type=str, default=str(FULLERENE_SPLIT_FILE))
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

    mol3d_split = json.load(open(args.mol3d_split))
    full_split  = json.load(open(args.fullerene_split))

    mol3d_hp_train, mol3d_hp_val = carve_val_from_train(mol3d_split["train"], val_frac=0.15, seed=args.seed)
    mol3d_hp_train = subsample(mol3d_hp_train, MAX_TRAIN // 2, seed=args.seed)
    mol3d_hp_val   = subsample(mol3d_hp_val,   MAX_VAL // 2,   seed=args.seed)

    full_hp_train, full_hp_val = carve_val_from_train(full_split["train_idx"], val_frac=0.15, seed=args.seed)
    full_hp_train = subsample(full_hp_train, MAX_TRAIN // 2, seed=args.seed)
    full_hp_val   = subsample(full_hp_val,   MAX_VAL // 2,   seed=args.seed)

    print(f"mol3d HP-train: {len(mol3d_hp_train)} | HP-val: {len(mol3d_hp_val)} (carved from mol3d train only)")
    print(f"fullerene HP-train: {len(full_hp_train)} | HP-val: {len(full_hp_val)} (carved from fullerene train only)")

    mol3d_train_data, _ = load_mol3d_schnet(mol3d_hp_train, root=args.mol3d_datapath)
    mol3d_val_data, _   = load_mol3d_schnet(mol3d_hp_val,   root=args.mol3d_datapath)

    fullerene_all = load_fullerene_schnet(root=str(FULLERENE_ROOT), target="Gap")
    assert len(fullerene_all) == full_split["n_total"], \
        f"Fullerene size mismatch: {len(fullerene_all)} vs {full_split['n_total']}"
    full_train_data = [fullerene_all[i] for i in full_hp_train]
    full_val_data   = [fullerene_all[i] for i in full_hp_val]

    train_data = mol3d_train_data + full_train_data
    val_data   = mol3d_val_data + full_val_data
    print(f"Combined HP-train: {len(train_data)} | HP-val: {len(val_data)}")

    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    train_labels = torch.stack([d.y for d in train_data]).squeeze(-1)
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
                loss = criterion(out.squeeze(-1), (batch.y.squeeze(-1) - y_mean) / y_std)
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
