"""Small hyperparameter search for CT (CellularTransformer) on
lrgb/peptides_struct.

CT's architecture is normally hardcoded inside exp_ct.py's make_model() --
this script builds the model directly instead, with all of CT's own
constructor kwargs sampled per trial: num_layers, hidden_dim, num_heads,
hidden_dim_per_head, att_dropout, emb_dropout, readout_dropout, and
num_readout_hidden_layers (the 3 dropout knobs are tuned independently
rather than sharing one value). num_heads is restricted to values that
evenly divide every hidden_dim candidate (32/64/128 -> heads in {4, 8}) so
every sampled config is structurally valid.

Uses feat_mode="original" (the 9-dim OGB atom / 3-dim OGB bond features,
same as GCN/GAT/GIN/CIN use here) rather than "full" -- unlike mol3d/
fullerene, LRGB's CT has 3 feature variants and "original" is the one
directly comparable to every other model tuned in this experiment, and it
avoids the slow/previously-flagged-fragile SMILES<->PyG alignment step
"full"/"simple" require (see smiles_align.py).

Subsamples the REAL train/val splits directly (LRGB ships them separately,
no split.json to carve from) -- this script never loads the test split.
Subsampling happens on the raw PyG *indices* before any per-molecule
processing (feat_mode="original" runs networkx ring detection per molecule,
~115ms/mol -- building the full 10873-molecule train set first and
subsampling after, like the real LRGBCTDataset class does, would cost ~21
minutes regardless of how small MAX_TRAIN is).
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
from lrgb_ct import _LRGBNoDownload, _process_from_graph
from ct import CellularTransformer
from hp_search_utils import subsample, sample_configs, save_best

DATASET_NAME = "Peptides-struct"
OUT_DIM = 11
DATA_ROOT = str(Path(__file__).parent.parent / "data")

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
        output_dim=OUT_DIM, num_layers=cfg["num_layers"], hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"], hidden_dim_per_head=cfg["hidden_dim_per_head"],
        att_dropout=cfg["att_dropout"], emb_dropout=cfg["emb_dropout"],
        readout_dropout=cfg["readout_dropout"],
        num_readout_hidden_layers=cfg["num_readout_hidden_layers"],
    )


def load_subset(datapath, split, max_n, seed):
    """Loads at most max_n molecules from `split`, subsampling the raw PyG
    indices BEFORE running feat_mode="original"'s per-molecule networkx ring
    detection -- never constructs the full split."""
    pyg_ds = _LRGBNoDownload(root=datapath, name=DATASET_NAME, split=split)
    raw_idx = subsample(range(len(pyg_ds)), max_n, seed=seed)
    data = []
    for i in raw_idx:
        try:
            data.append(_process_from_graph(pyg_ds[i]))
        except Exception:
            pass
    return data, len(pyg_ds)


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
            out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
            preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",   type=str, default=DATA_ROOT)
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

    print("Loading train/val subsets (test never loaded by this script)...")
    train_data, n_train_total = load_subset(args.datapath, "train", MAX_TRAIN, args.seed)
    val_data, n_val_total     = load_subset(args.datapath, "val",   MAX_VAL,   args.seed)
    print(f"HP-train: {len(train_data)} (of {n_train_total} real train) | "
          f"HP-val: {len(val_data)} (of {n_val_total} real val)")

    rk0_dim, rk1_dim, rk2_dim = train_data[0][0].shape[1], train_data[0][1].shape[1], train_data[0][2].shape[1]
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    train_labels = torch.stack([d[-1] for d in train_data])
    y_mean = train_labels.mean(dim=0, keepdim=True).to(device)
    y_std  = train_labels.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)

    criterion = nn.L1Loss()
    configs = sample_configs(SEARCH_SPACE, n_trials=args.n_trials, seed=args.seed)

    trials = []
    best_cfg, best_val = None, float("inf")
    for i, cfg in enumerate(configs):
        print(f"\n--- Trial {i + 1}/{len(configs)}: {cfg} ---")
        torch.manual_seed(args.seed)
        model = make_model(cfg, rk0_dim, rk1_dim, rk2_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

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
