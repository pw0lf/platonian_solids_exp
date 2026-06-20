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

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")


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
        torch.cat(x_0),
        torch.cat(x_1),
        torch.cat(x_2),
        sparse_block_diag(adj00), sparse_block_diag(icd01), sparse_block_diag(adj11),
        sparse_block_diag(icd02), sparse_block_diag(icd12), sparse_block_diag(adj22),
        torch.tensor([x.shape[0] for x in x_0], dtype=torch.long),
        torch.stack(y),
    )


def make_model(rk0_dim, rk1_dim, rk2_dim):
    return CellularTransformer(
        rk0_dim=rk0_dim, rk1_dim=rk1_dim, rk2_dim=rk2_dim,
        output_dim=1, num_layers=6, hidden_dim=128, num_heads=4,
        hidden_dim_per_head=8, att_dropout=0.007223573743289108,
        emb_dropout=0.2889364748304471, readout_dropout=0.1262504279429404,
        num_readout_hidden_layers=2,
    )


def evaluate(model, loader, device, y_mean, y_std, criterion):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
            out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
            preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
    p, t = torch.cat(preds).squeeze(-1), torch.cat(targets).squeeze(-1)
    rmse = criterion(p, t).sqrt().item()
    mae = (p - t).abs().mean().item()
    r2 = (1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()
    return rmse, mae, r2, p, t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file",     type=str,   default=str(Path(__file__).parent / "data_split.json"))
    parser.add_argument("--datapath",       type=str,   default=DATA_ROOT)
    parser.add_argument("--feat_mode",      type=str,   default="full", choices=["full", "simple", "coords"])
    parser.add_argument("--pe_k",           type=int,   default=5)
    parser.add_argument("--epochs",         type=int,   default=300)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=0.000965256564762386)
    parser.add_argument("--warmup_epochs",  type=int,   default=5)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output",         type=str,   default=None,
                        help="filename (saved inside results/); defaults to results_mol3d_ct_<feat_mode>.json")
    args = parser.parse_args()

    use_pe = args.feat_mode == "full"

    if args.output is None:
        args.output = f"results_mol3d_ct_{args.feat_mode}.json"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | feat_mode: {args.feat_mode} | use_pe: {use_pe} | pe_k: {args.pe_k}")

    split = json.load(open(args.split_file))
    train_indices = split["train"]
    test_indices  = split["test"]
    print(f"Split: {len(train_indices)} train + {len(test_indices)} test (seed={split['seed']})")

    print("Loading train dataset...")
    train_dataset = Mol3dCT(train_indices, root=args.datapath, use_pe=use_pe, pe_k=args.pe_k, feat_mode=args.feat_mode)
    print("Loading test dataset...")
    test_dataset  = Mol3dCT(test_indices,  root=args.datapath, use_pe=use_pe, pe_k=args.pe_k, feat_mode=args.feat_mode)
    print(f"Loaded: rk0={train_dataset.rk0_dim} rk1={train_dataset.rk1_dim} rk2={train_dataset.rk2_dim}")
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    train_labels = torch.stack([train_dataset[i][-1] for i in range(len(train_dataset))])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    criterion = nn.MSELoss()
    results = {
        "feat_mode": args.feat_mode, "pe_k": args.pe_k,
        "split_file": args.split_file, "epochs": args.epochs, "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)
        model = make_model(train_dataset.rk0_dim, train_dataset.rk1_dim, train_dataset.rk2_dim).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-4)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        run_result = {"run": run + 1, "train_losses": [], "epoch_times": []}
        run_start = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0
            for batch in train_loader:
                x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                optimizer.zero_grad()
                out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                loss = criterion(out, (y - y_mean) / y_std)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            run_result["epoch_times"].append(round(time.time() - epoch_start, 2))
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  lr={lr:.2e}")

        test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(model, test_loader, device, y_mean, y_std, criterion)
        run_result["test_rmse"] = round(test_rmse, 4)
        run_result["test_mae"]  = round(test_mae, 4)
        run_result["test_r2"]   = round(test_r2, 4)
        run_result["runtime"]   = round(time.time() - run_start, 2)
        run_result["predictions"] = [
            {"index": int(idx), "pred": round(float(p), 6), "true": round(float(t), 6)}
            for idx, p, t in zip(test_dataset.indices, test_preds.tolist(), test_targets.tolist())
        ]
        print(f"Test  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  R2: {test_r2:.4f}")
        results["runs"].append(run_result)

    results["mean_test_rmse"] = round(sum(r["test_rmse"] for r in results["runs"]) / 3, 4)
    results["mean_test_mae"]  = round(sum(r["test_mae"]  for r in results["runs"]) / 3, 4)
    results["mean_test_r2"]   = round(sum(r["test_r2"]   for r in results["runs"]) / 3, 4)
    print(f"\nMean test  RMSE: {results['mean_test_rmse']:.4f}  "
          f"MAE: {results['mean_test_mae']:.4f}  R2: {results['mean_test_r2']:.4f}")

    out_path = Path(__file__).parent / "results" / args.output
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
