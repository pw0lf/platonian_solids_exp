"""Small hyperparameter search for CT (CellularTransformer) on mol3d_fullerene
(combined mol3d + fullerene dataset).

Same special-casing as mol3d/hp_tuning_ct.py: CT's architecture is hardcoded
in exp_ct.py's make_model(), so this script builds CellularTransformer
directly with all of CT's own constructor kwargs sampled per trial:
num_layers, hidden_dim, num_heads, hidden_dim_per_head, att_dropout,
emb_dropout, readout_dropout, and num_readout_hidden_layers (the 3 dropout
knobs are tuned independently rather than sharing one value). num_heads is
restricted to values that evenly divide every hidden_dim candidate
(32/64/128 -> heads in {4, 8}) so every sampled config is structurally
valid. Uses feat_mode="full" (CT's richest/default variant).

Carves a small hp-train/hp-val pair out of BOTH data sources' TRAIN portions
only (never touches either test split) and combines them, mirroring the
real exp_ct.py's ListDataset combination. Data/epoch budget matches the
other mol3d_fullerene tuning scripts even though CT is slower per-epoch
than the other architectures.
"""
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
sys.path.insert(0, str(Path(__file__).parent))
from mol3d_ct_rand import Mol3dCT
from ct import CellularTransformer
from fullerene_loader import load_fullerene_ct
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MOL3D_DATA_ROOT      = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")
MOL3D_SPLIT_FILE     = MOL3D_DIR / "data_split.json"
FULLERENE_SPLIT_FILE = Path(__file__).parent.parent / "fullerene_randomsplit" / "split.json"

SEARCH_SPACE = {
    "lr":                        [1e-4, 3e-4, 1e-3, 3e-3],
    "hidden_dim":                [32, 64, 128],
    "num_layers":                [2, 4, 6],
    "num_heads":                 [4, 8],
    "hidden_dim_per_head":       [8, 16, 32],
    "att_dropout":               [0.0, 0.1, 0.2, 0.3],
    "emb_dropout":               [0.0, 0.1, 0.2, 0.3],
    "readout_dropout":           [0.0, 0.1, 0.2, 0.3],
    "num_readout_hidden_layers": [1, 2, 3],
}
MAX_TRAIN = 1200
MAX_VAL = 300


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def sparse_block_diag(sparse_list):
    rows, cols, vals = [], [], []
    row_offset = col_offset = 0
    for S in sparse_list:
        S = S.coalesce()
        i, v = S.indices(), S.values()
        rows.append(i[0] + row_offset); cols.append(i[1] + col_offset); vals.append(v)
        row_offset += S.shape[0]; col_offset += S.shape[1]
    return torch.sparse_coo_tensor(
        torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals),
        size=(row_offset, col_offset))


def collate(batch):
    x_0, x_1, x_2, icd01, icd02, icd12, adj00, adj11, adj22, y = zip(*batch)
    return (
        torch.cat(x_0), torch.cat(x_1), torch.cat(x_2),
        sparse_block_diag(adj00), sparse_block_diag(icd01), sparse_block_diag(adj11),
        sparse_block_diag(icd02), sparse_block_diag(icd12), sparse_block_diag(adj22),
        torch.tensor([x.shape[0] for x in x_0], dtype=torch.long),
        torch.stack(y),
    )


def make_model(cfg, rk0_dim, rk1_dim, rk2_dim):
    return CellularTransformer(
        rk0_dim=rk0_dim, rk1_dim=rk1_dim, rk2_dim=rk2_dim,
        output_dim=1, num_layers=cfg["num_layers"], hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"], hidden_dim_per_head=cfg["hidden_dim_per_head"],
        att_dropout=cfg["att_dropout"], emb_dropout=cfg["emb_dropout"],
        readout_dropout=cfg["readout_dropout"],
        num_readout_hidden_layers=cfg["num_readout_hidden_layers"],
    )


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
            out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
            preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
    p, t = torch.cat(preds).squeeze(-1), torch.cat(targets).squeeze(-1)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol3d_datapath",  type=str, default=MOL3D_DATA_ROOT)
    parser.add_argument("--mol3d_split",     type=str, default=str(MOL3D_SPLIT_FILE))
    parser.add_argument("--fullerene_split", type=str, default=str(FULLERENE_SPLIT_FILE))
    parser.add_argument("--pe_k",       type=int, default=5)
    parser.add_argument("--n_trials",   type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--output",     type=str, default="best_hp_ct.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | tuning: CT")

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

    mol3d_train_ds = Mol3dCT(mol3d_hp_train, root=args.mol3d_datapath, use_pe=True, pe_k=args.pe_k, feat_mode="full")
    mol3d_val_ds   = Mol3dCT(mol3d_hp_val,   root=args.mol3d_datapath, use_pe=True, pe_k=args.pe_k, feat_mode="full")

    fullerene_all = load_fullerene_ct(root=str(FULLERENE_ROOT), target="Gap", use_pe=True, pe_k=args.pe_k, feat_mode="full")
    assert len(fullerene_all) == full_split["n_total"], \
        f"Fullerene size mismatch: {len(fullerene_all)} vs {full_split['n_total']}"
    full_train_data = [fullerene_all[i] for i in full_hp_train]
    full_val_data   = [fullerene_all[i] for i in full_hp_val]

    train_dataset = ListDataset(list(mol3d_train_ds) + full_train_data)
    val_dataset   = ListDataset(list(mol3d_val_ds) + full_val_data)
    print(f"Combined HP-train: {len(train_dataset)} | HP-val: {len(val_dataset)}")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    train_labels = torch.stack([train_dataset[i][-1] for i in range(len(train_dataset))])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)

    criterion = nn.MSELoss()
    configs = sample_configs(SEARCH_SPACE, n_trials=args.n_trials, seed=args.seed)

    trials = []
    best_cfg, best_val = None, float("inf")
    for i, cfg in enumerate(configs):
        print(f"\n--- Trial {i + 1}/{len(configs)}: {cfg} ---")
        torch.manual_seed(args.seed)
        model = make_model(cfg, mol3d_train_ds.rk0_dim, mol3d_train_ds.rk1_dim, mol3d_train_ds.rk2_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                optimizer.zero_grad()
                out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                loss = criterion(out, (y - y_mean) / y_std)
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
