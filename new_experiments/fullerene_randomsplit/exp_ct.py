import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from pathlib import Path

FULLERENE_DIR = Path(__file__).parent.parent / "fullerene"
sys.path.insert(0, str(FULLERENE_DIR / "data_loader"))
sys.path.insert(0, str(FULLERENE_DIR / "models"))
from fullerene_complex_dataset import FullereneComplexDataset
from ct import CellularTransformer

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

DATA_ROOT  = str(Path(__file__).parent.parent.parent / "FullereneNet")
SPLIT_FILE = Path(__file__).parent / "split.json"


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


def make_model(args, rk0_dim, rk1_dim, rk2_dim):
    return CellularTransformer(
        rk0_dim=rk0_dim, rk1_dim=rk1_dim, rk2_dim=rk2_dim,
        output_dim=1, num_layers=args.num_layers, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, hidden_dim_per_head=args.hidden_dim_per_head,
        att_dropout=args.att_dropout, emb_dropout=args.emb_dropout,
        readout_dropout=args.readout_dropout,
        num_readout_hidden_layers=args.num_readout_hidden_layers,
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
    parser.add_argument("--chem_features", type=str,   default="full", choices=["full", "simple", "none"])
    parser.add_argument("--topo_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pe_k",         type=int,   default=5)
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--num_layers",   type=int,   default=8)
    parser.add_argument("--hidden_dim",   type=int,   default=64)
    parser.add_argument("--num_heads",    type=int,   default=8)
    parser.add_argument("--hidden_dim_per_head", type=int, default=16)
    parser.add_argument("--att_dropout",     type=float, default=0.1)
    parser.add_argument("--emb_dropout",     type=float, default=0.1)
    parser.add_argument("--readout_dropout", type=float, default=0.1)
    parser.add_argument("--num_readout_hidden_layers", type=int, default=3)
    parser.add_argument("--warmup_epochs",type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--output",       type=str,   default="results_ct.json",
                        help="filename (saved inside results/)")
    parser.add_argument("--hp_file",      type=str,   default=None,
                        help="JSON from fullerene/hp_tuning_ct.py (shared tuning). Values for keys "
                             "present in the file unconditionally override this script's CLI "
                             "defaults for lr/num_layers/hidden_dim/num_heads/hidden_dim_per_head/"
                             "att_dropout/emb_dropout/readout_dropout/num_readout_hidden_layers -- "
                             "even if you also pass those flags explicitly.")
    args = parser.parse_args()

    if args.hp_file:
        with open(args.hp_file) as f:
            hp = json.load(f)
        hp_keys = ("lr", "num_layers", "hidden_dim", "num_heads", "hidden_dim_per_head",
                   "att_dropout", "emb_dropout", "readout_dropout", "num_readout_hidden_layers")
        for key in hp_keys:
            if key in hp:
                setattr(args, key, hp[key])
        print(f"Loaded hyperparameters from {args.hp_file}: "
              f"{ {k: getattr(args, k) for k in hp_keys} }")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | chem_features: {args.chem_features} | topo_features: {args.topo_features} | pe_k: {args.pe_k}")

    with open(SPLIT_FILE) as f:
        split = json.load(f)
    print(f"Split: {split['n_train']} train / {split['n_test']} test (seed={split['seed']})")

    print("Loading datasets...")
    ds_c60 = FullereneComplexDataset("c60",         root=DATA_ROOT, target="Eb",
                                     pe_k=args.pe_k, chem_features=args.chem_features,
                                     topo_features=args.topo_features)
    ds_c70 = FullereneComplexDataset("c70_non_IPR", root=DATA_ROOT, target="Eb",
                                     pe_k=args.pe_k, chem_features=args.chem_features,
                                     topo_features=args.topo_features)
    ds_c72 = FullereneComplexDataset("c72_100_IPR", root=DATA_ROOT, target="Eb",
                                     pe_k=args.pe_k, chem_features=args.chem_features,
                                     topo_features=args.topo_features)

    all_data = ConcatDataset([ds_c60, ds_c70, ds_c72])
    assert len(all_data) == split["n_total"], \
        f"Data size mismatch: {len(all_data)} vs split n_total={split['n_total']}"

    train_set = Subset(all_data, split["train_idx"])
    test_set  = Subset(all_data, split["test_idx"])

    print(f"Loaded: rk0={ds_c60.rk0_dim} rk1={ds_c60.rk1_dim} rk2={ds_c60.rk2_dim}")
    print(f"Train: {len(train_set)} | Test: {len(test_set)}")

    # z-score normalization fitted on train labels only
    train_labels = torch.stack([train_set[i][-1] for i in range(len(train_set))])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    criterion   = nn.MSELoss()
    results = {
        "chem_features": args.chem_features, "topo_features": args.topo_features,
        "pe_k": args.pe_k, "epochs": args.epochs, "split_seed": split["seed"], "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        model = make_model(args, ds_c60.rk0_dim, ds_c60.rk1_dim, ds_c60.rk2_dim).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.999), eps=1e-8)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-4)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

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

        test_rmse, test_mae, test_r2, test_preds, test_targets = evaluate(
            model, test_loader, device, y_mean, y_std, criterion)
        run_result["test_rmse"]  = round(test_rmse, 4)
        run_result["test_mae"]   = round(test_mae, 4)
        run_result["test_r2"]    = round(test_r2, 4)
        run_result["predictions"] = [
            {"index": idx, "pred": round(float(p), 6), "true": round(float(t), 6)}
            for idx, p, t in zip(split["test_idx"], test_preds.tolist(), test_targets.tolist())
        ]
        run_result["runtime_s"]  = round(time.time() - run_start, 2)
        print(f"Test  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}  R2: {test_r2:.4f}")
        results["runs"].append(run_result)

    results["mean_test_rmse"] = round(sum(r["test_rmse"] for r in results["runs"]) / 3, 4)
    results["mean_test_mae"]  = round(sum(r["test_mae"]  for r in results["runs"]) / 3, 4)
    results["mean_test_r2"]   = round(sum(r["test_r2"]   for r in results["runs"]) / 3, 4)
    print(f"\nMean test  RMSE: {results['mean_test_rmse']:.4f}  "
          f"MAE: {results['mean_test_mae']:.4f}  R2: {results['mean_test_r2']:.4f}")

    out_path = Path(__file__).parent / "results" / args.output
    if args.hp_file:
        out_path = out_path.with_name(f"{out_path.stem}_hptuned{out_path.suffix}")
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
