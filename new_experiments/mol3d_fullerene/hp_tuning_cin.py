"""Small hyperparameter search for CIN (variant="CIN") on mol3d_fullerene
(combined mol3d + fullerene dataset). See hp_tuning_gcn.py in this directory
for the shared combined-dataset rationale (carves hp-train/hp-val from BOTH
sources' TRAIN portions only, never touches either test split).
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

MOL3D_DIR      = Path(__file__).parent.parent / "mol3d"
FULLERENE_ROOT = Path(__file__).parent.parent.parent / "FullereneNet"

sys.path.insert(0, str(MOL3D_DIR / "data_loader"))
sys.path.insert(0, str(MOL3D_DIR / "models"))
sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent))
from mol3d_cin import Mol3dCIN, collate_cin
from cin import CIN
from fullerene_loader import load_fullerene_cin
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MOL3D_DATA_ROOT      = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")
MOL3D_SPLIT_FILE     = MOL3D_DIR / "data_split.json"
FULLERENE_SPLIT_FILE = Path(__file__).parent.parent / "fullerene_randomsplit" / "split.json"
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
    parser.add_argument("--mol3d_datapath",  type=str, default=MOL3D_DATA_ROOT)
    parser.add_argument("--mol3d_split",     type=str, default=str(MOL3D_SPLIT_FILE))
    parser.add_argument("--fullerene_split", type=str, default=str(FULLERENE_SPLIT_FILE))
    parser.add_argument("--n_trials",   type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
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

    mol3d_train_ds = Mol3dCIN(mol3d_hp_train, root=args.mol3d_datapath)
    mol3d_val_ds   = Mol3dCIN(mol3d_hp_val,   root=args.mol3d_datapath)
    dims = (mol3d_train_ds.x0_dim, mol3d_train_ds.x1_dim, mol3d_train_ds.x2_dim)

    fullerene_all = load_fullerene_cin(root=str(FULLERENE_ROOT), target="Gap")
    assert len(fullerene_all) == full_split["n_total"], \
        f"Fullerene size mismatch: {len(fullerene_all)} vs {full_split['n_total']}"
    full_train_data = [fullerene_all[i] for i in full_hp_train]
    full_val_data   = [fullerene_all[i] for i in full_hp_val]

    train_data = list(mol3d_train_ds) + full_train_data
    val_data   = list(mol3d_val_ds) + full_val_data
    print(f"Combined HP-train: {len(train_data)} | HP-val: {len(val_data)}")

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
