"""Small hyperparameter search for CT (CellularTransformer) on fullerene.

CT's architecture is normally hardcoded inside exp_ct.py's make_model() --
this script builds the model directly instead, with all of CT's own
constructor kwargs sampled per trial: num_layers, hidden_dim, num_heads,
hidden_dim_per_head, att_dropout, emb_dropout, readout_dropout, and
num_readout_hidden_layers (the 3 dropout knobs are tuned independently
rather than sharing one value). num_heads is restricted to values that
evenly divide every hidden_dim candidate (32/64/128 -> heads in {4, 8}) so
every sampled config is structurally valid.

Random search trained and validated on a small subset carved out of the
<=58-atom TRAIN portion of the "c60" group only (see
hp_search_utils.carve_val_from_train) -- this script never loads the C60/C70/
C72-100 held-out test sets at all. Saves the winning config to
results/best_hp_ct.json for exp_ct.py --hp_file to pick up (shared by
fullerene and fullerene_randomsplit -- see this repo's convention of
fullerene_randomsplit importing fullerene's data_loader/models directly).
"""
import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
sys.path.insert(0, str(Path(__file__).parent / "models"))
from fullerene_complex_dataset import FullereneComplexDataset
from ct import CellularTransformer
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT = str(Path(__file__).parent.parent.parent / "FullereneNet")

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
    p, t = torch.cat(preds), torch.cat(targets)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     type=str, default="Eb")
    parser.add_argument("--n_trials",   type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
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

    print("Loading c60 group (train range only)...")
    c20_c58_c60 = FullereneComplexDataset("c60", root=DATA_ROOT, target=args.target)
    train_indices = [i for i in range(len(c20_c58_c60)) if c20_c58_c60[i][0].shape[0] <= 58]
    # Note: this script never constructs the c60==60 / c70_non_IPR / c72_100_IPR
    # test sets at all -- there is nothing to leak from.

    hp_train_idx, hp_val_idx = carve_val_from_train(train_indices, val_frac=0.15, seed=args.seed)
    hp_train_idx = subsample(hp_train_idx, MAX_TRAIN, seed=args.seed)
    hp_val_idx   = subsample(hp_val_idx,   MAX_VAL,   seed=args.seed)
    print(f"HP-train: {len(hp_train_idx)} | HP-val: {len(hp_val_idx)} (carved from train only)")

    train_set = Subset(c20_c58_c60, hp_train_idx)
    val_set   = Subset(c20_c58_c60, hp_val_idx)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    train_labels = torch.stack([train_set[i][-1] for i in range(len(train_set))])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)

    criterion = nn.MSELoss()
    configs = sample_configs(SEARCH_SPACE, n_trials=args.n_trials, seed=args.seed)

    trials = []
    best_cfg, best_val = None, float("inf")
    for i, cfg in enumerate(configs):
        print(f"\n--- Trial {i + 1}/{len(configs)}: {cfg} ---")
        torch.manual_seed(args.seed)
        model = make_model(cfg, c20_c58_c60.rk0_dim, c20_c58_c60.rk1_dim, c20_c58_c60.rk2_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

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
