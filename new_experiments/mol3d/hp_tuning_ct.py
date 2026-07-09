"""Small hyperparameter search for CT (CellularTransformer) on mol3d.

CT's architecture is normally hardcoded inside exp_ct.py's make_model() --
this script builds the model directly instead, with all of CT's own
constructor kwargs sampled per trial: num_layers, hidden_dim, num_heads,
hidden_dim_per_head, att_dropout, emb_dropout, readout_dropout, and
num_readout_hidden_layers (the 3 dropout knobs are tuned independently
rather than sharing one value). num_heads is restricted to values that
evenly divide every hidden_dim candidate (32/64/128 -> heads in {4, 8}) so
every sampled config is structurally valid.

Uses feat_mode="full" (CT's richest/default feature variant -- this search
tunes architecture, not feature choice, per the "one search per model
architecture" scope decision).

Random search trained and validated on a small subset carved out of the
TRAIN split only (see hp_search_utils.carve_val_from_train) -- this script
never loads the test split. Data/epoch budget matches the other mol3d
tuning scripts (MAX_TRAIN/MAX_VAL/epochs) even though CT is slower
per-epoch than the other architectures.
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
from mol3d_ct_rand import Mol3dCT
from ct import CellularTransformer
from hp_search_utils import carve_val_from_train, subsample, sample_configs, save_best

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")

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
    p, t = torch.cat(preds).squeeze(-1), torch.cat(targets).squeeze(-1)
    return (p - t).abs().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", type=str, default=str(Path(__file__).parent / "data_split.json"))
    parser.add_argument("--datapath",   type=str, default=DATA_ROOT)
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

    split = json.load(open(args.split_file))
    train_indices = split["train"]  # test indices are never loaded by this script

    hp_train_idx, hp_val_idx = carve_val_from_train(train_indices, val_frac=0.15, seed=args.seed)
    hp_train_idx = subsample(hp_train_idx, MAX_TRAIN, seed=args.seed)
    hp_val_idx   = subsample(hp_val_idx,   MAX_VAL,   seed=args.seed)
    print(f"HP-train: {len(hp_train_idx)} | HP-val: {len(hp_val_idx)} (carved from train only)")

    train_dataset = Mol3dCT(hp_train_idx, root=args.datapath, use_pe=True, pe_k=args.pe_k, feat_mode="full")
    val_dataset   = Mol3dCT(hp_val_idx,   root=args.datapath, use_pe=True, pe_k=args.pe_k, feat_mode="full")
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
        model = make_model(cfg, train_dataset.rk0_dim, train_dataset.rk1_dim, train_dataset.rk2_dim).to(device)
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
