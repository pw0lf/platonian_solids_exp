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
from lrgb_ct import LRGBCTDataset
from ct import CellularTransformer

DATASET_NAME = "Peptides-struct"
OUT_DIM = 11
DATA_ROOT = str(Path(__file__).parent.parent / "data")
SMILES_DIR = Path(__file__).parent.parent


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


def make_model(rk0_dim, rk1_dim, rk2_dim):
    return CellularTransformer(
        rk0_dim=rk0_dim, rk1_dim=rk1_dim, rk2_dim=rk2_dim,
        output_dim=OUT_DIM, num_layers=4, hidden_dim=128, num_heads=4,
        hidden_dim_per_head=8, att_dropout=0.1, emb_dropout=0.2,
        readout_dropout=0.1, num_readout_hidden_layers=2,
    )


def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, nc, y = [b.to(device) for b in batch]
            out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, nc)
            out_denorm = out * y_std + y_mean
            all_preds.append(out_denorm.cpu())
            all_targets.append(y.cpu())
    preds = torch.cat(all_preds)    # (N, 11)
    targets = torch.cat(all_targets)  # (N, 11)
    mae = float((preds - targets).abs().mean())
    return mae, preds, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",      type=str,   default=DATA_ROOT)
    parser.add_argument("--feat_mode",     type=str,   default="original",
                        choices=["original", "full", "simple"])
    parser.add_argument("--pe_k",          type=int,   default=5)
    parser.add_argument("--epochs",        type=int,   default=300)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--output",        type=str,   default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results_struct_ct_{args.feat_mode}.json"

    if args.feat_mode in ("full", "simple"):
        for split in ("train", "val", "test"):
            p = SMILES_DIR / f"smiles_{split}.csv"
            assert p.exists(), f"Missing {p}. Run: python new_experiments/lrgb/download_smiles.py"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | dataset: {DATASET_NAME} | feat_mode: {args.feat_mode}")

    def get_smiles_csv(split):
        if args.feat_mode in ("full", "simple"):
            return str(SMILES_DIR / f"smiles_{split}.csv")
        return None

    print("Loading train dataset...")
    train_ds = LRGBCTDataset(args.datapath, DATASET_NAME, "train",
                             feat_mode=args.feat_mode, pe_k=args.pe_k,
                             smiles_csv=get_smiles_csv("train"))
    print("Loading val dataset...")
    val_ds   = LRGBCTDataset(args.datapath, DATASET_NAME, "val",
                             feat_mode=args.feat_mode, pe_k=args.pe_k,
                             smiles_csv=get_smiles_csv("val"))
    print("Loading test dataset...")
    test_ds  = LRGBCTDataset(args.datapath, DATASET_NAME, "test",
                             feat_mode=args.feat_mode, pe_k=args.pe_k,
                             smiles_csv=get_smiles_csv("test"))
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Per-target z-score normalization fitted on train labels
    train_labels = torch.stack([train_ds[i][-1] for i in range(len(train_ds))])  # (N, 11)
    y_mean = train_labels.mean(dim=0, keepdim=True).to(device)  # (1, 11)
    y_std  = train_labels.std(dim=0,  keepdim=True).clamp(min=1e-6).to(device)
    print(f"Label mean (first 3): {y_mean[0, :3].tolist()}")

    criterion = nn.L1Loss()
    results = {
        "dataset": DATASET_NAME, "feat_mode": args.feat_mode, "pe_k": args.pe_k,
        "epochs": args.epochs, "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)
        model = make_model(train_ds.rk0_dim, train_ds.rk1_dim, train_ds.rk2_dim).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        run_result = {"run": run + 1, "train_losses": [], "epoch_times": []}
        run_start = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            for batch in train_loader:
                x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, nc, y = [b.to(device) for b in batch]
                optimizer.zero_grad()
                out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, nc)
                loss = criterion(out, (y - y_mean) / y_std)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            run_result["epoch_times"].append(round(time.time() - epoch_start, 2))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}  loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        val_mae,  _,          _            = evaluate(model, val_loader,  device, y_mean, y_std)
        test_mae, test_preds, test_targets = evaluate(model, test_loader, device, y_mean, y_std)
        run_result["val_mae"]   = round(val_mae,  4)
        run_result["test_mae"]  = round(test_mae, 4)
        run_result["runtime"]   = round(time.time() - run_start, 2)
        run_result["predictions"] = [
            {"index": int(idx),
             "pred": [round(float(v), 6) for v in pred.tolist()],
             "true": [round(float(v), 6) for v in tgt.tolist()]}
            for idx, pred, tgt in zip(test_ds.indices, test_preds.tolist(), test_targets.tolist())
        ]
        print(f"Val MAE: {val_mae:.4f}  Test MAE: {test_mae:.4f}")
        results["runs"].append(run_result)

    results["mean_val_mae"]  = round(sum(r["val_mae"]  for r in results["runs"]) / 3, 4)
    results["mean_test_mae"] = round(sum(r["test_mae"] for r in results["runs"]) / 3, 4)
    print(f"\nMean Val MAE: {results['mean_val_mae']:.4f}  Mean Test MAE: {results['mean_test_mae']:.4f}")

    out_path = Path(__file__).parent / "results" / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
