"""Small hyperparameter search for GIN on mol3d_fullerene (combined mol3d +
fullerene dataset). See mol3d_fullerene/hp_tuning_gcn.py for the shared
combined-dataset rationale (carves hp-train/hp-val from BOTH sources' TRAIN
portions only, never touches either test split)."""
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

MOL3D_DIR      = Path(__file__).parent.parent / "mol3d"
FULLERENE_ROOT = Path(__file__).parent.parent.parent / "FullereneNet"

sys.path.insert(0, str(MOL3D_DIR / "data_loader"))
sys.path.insert(0, str(MOL3D_DIR / "models"))
sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent))
from mol3d_gnn import load_gnn_data as load_mol3d_gnn, NODE_DIM
from gnn_models import GIN
from fullerene_loader import load_fullerene_gnn
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MOL3D_DATA_ROOT      = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")
MOL3D_SPLIT_FILE     = MOL3D_DIR / "data_split.json"
FULLERENE_SPLIT_FILE = Path(__file__).parent.parent / "fullerene_randomsplit" / "split.json"

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
        in_channels=NODE_DIM, hidden_channels=cfg["hidden_channels"],
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
    parser.add_argument("--mol3d_datapath",  type=str, default=MOL3D_DATA_ROOT)
    parser.add_argument("--mol3d_split",     type=str, default=str(MOL3D_SPLIT_FILE))
    parser.add_argument("--fullerene_split", type=str, default=str(FULLERENE_SPLIT_FILE))
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

    mol3d_train_data, _ = load_mol3d_gnn(mol3d_hp_train, root=args.mol3d_datapath, load_edge_features=False)
    mol3d_val_data, _   = load_mol3d_gnn(mol3d_hp_val,   root=args.mol3d_datapath, load_edge_features=False)

    fullerene_all = load_fullerene_gnn(root=str(FULLERENE_ROOT), target="Gap", load_edge_features=False)
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
                out = model(batch.x, batch.edge_index, batch.batch, edge_attr=None)
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
