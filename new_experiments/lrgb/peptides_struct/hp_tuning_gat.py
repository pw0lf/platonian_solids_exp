"""Small hyperparameter search for GAT on lrgb/peptides_struct. See
hp_tuning_gcn.py for the shared rationale (subsamples the REAL train/val
splits directly, test never loaded)."""
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
sys.path.insert(0, str(Path(__file__).parent))
from lrgb_gnn import LRGBGNNDataset
from gnn_models import GAT
from hp_search_utils import subsample, sample_configs, save_best

DATASET_NAME = "Peptides-struct"
OUT_DIM = 11
NODE_DIM = 9
EDGE_DIM = 3
DATA_ROOT = str(Path(__file__).parent.parent / "data")

SEARCH_SPACE = {
    "lr":                 [1e-4, 3e-4, 1e-3, 3e-3],
    "hidden_channels":    [32, 64, 128],
    "num_conv_layers":    [2, 3, 4],
    "dropout":            [0.0, 0.1, 0.2, 0.3],
    "readout_hidden_dim": [32, 64, 128],
    "num_readout_layers": [1, 2, 3],
    "num_heads":          [2, 4, 8],
}
MAX_TRAIN = 1200
MAX_VAL = 300


def make_model(cfg):
    return GAT(
        in_channels=NODE_DIM, hidden_channels=cfg["hidden_channels"],
        num_conv_layers=cfg["num_conv_layers"], num_heads=cfg["num_heads"], edge_dim=EDGE_DIM,
        readout_hidden_dim=cfg["readout_hidden_dim"], num_readout_layers=cfg["num_readout_layers"],
        out_dim=OUT_DIM, dropout=cfg["dropout"],
    )


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            preds.append((out * y_std + y_mean).cpu())
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
    parser.add_argument("--output",     type=str, default="best_hp_gat.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | tuning: GAT")

    print("Loading train/val splits (test never loaded by this script)...")
    train_ds = LRGBGNNDataset(args.datapath, DATASET_NAME, "train")
    val_ds   = LRGBGNNDataset(args.datapath, DATASET_NAME, "val")

    hp_train_idx = subsample(range(len(train_ds)), MAX_TRAIN, seed=args.seed)
    hp_val_idx   = subsample(range(len(val_ds)),   MAX_VAL,   seed=args.seed)
    print(f"HP-train: {len(hp_train_idx)} (of {len(train_ds)} real train) | "
          f"HP-val: {len(hp_val_idx)} (of {len(val_ds)} real val)")

    train_data = [train_ds[i] for i in hp_train_idx]
    val_data   = [val_ds[i]   for i in hp_val_idx]
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
                out = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
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
