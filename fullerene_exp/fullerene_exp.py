import argparse
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from pathlib import Path

FULLERENE_ROOT = Path(__file__).parent.parent / "FullereneNet"
sys.path.insert(0, str(Path(__file__).parent.parent / "gcb"))
sys.path.insert(0, str(FULLERENE_ROOT))
from fullerene_complex_dataset import FullereneComplexDataset
from ct import CellularTransformer

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


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
        output_dim=1, num_layers=8, hidden_dim=64, num_heads=8,
        hidden_dim_per_head=16, att_dropout=0.1, emb_dropout=0.1,
        readout_dropout=0.1, num_readout_hidden_layers=3,
    )


def evaluate(model, loader, device, y_mean, y_std, criterion):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
            out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
            preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    rmse = criterion(p, t).sqrt().item()
    mae = (p - t).abs().mean().item()
    return rmse, mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pe",       action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fullerenet_features_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pe_k",         type=int,   default=5)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--warmup_epochs",type=int,   default=5)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--output",       type=str,   default="results_fullerene.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | use_pe: {args.use_pe} | pe_k: {args.pe_k} | fullerenet_features_only: {args.fullerenet_features_only}")

    print("Loading datasets...")
    # c60 dataset spans C20-C60: train/val on C20-C58, internal test on C60 (held out by size)
    c20_c58_c60 = FullereneComplexDataset("c60", root=str(FULLERENE_ROOT), target="Eb",
                                          use_pe=args.use_pe, pe_k=args.pe_k,
                                          fullerenet_features_only=args.fullerenet_features_only)
    train_val_idx = [i for i in range(len(c20_c58_c60)) if c20_c58_c60[i][0].shape[0] <= 58]
    c60_test_idx  = [i for i in range(len(c20_c58_c60)) if c20_c58_c60[i][0].shape[0] == 60]

    train_val_set = Subset(c20_c58_c60, train_val_idx)
    c60_test_set  = Subset(c20_c58_c60, c60_test_idx)

    n_train = int(0.9 * len(train_val_set))
    n_val   = len(train_val_set) - n_train
    train_set, val_set = random_split(train_val_set, [n_train, n_val],
                                       generator=torch.Generator().manual_seed(args.seed))

    # external test sets: C70 non-IPR and C72-C100 IPR
    c70_test_set     = FullereneComplexDataset("c70_non_IPR", root=str(FULLERENE_ROOT), target="Eb",
                                                use_pe=args.use_pe, pe_k=args.pe_k,
                                                fullerenet_features_only=args.fullerenet_features_only)
    c72_100_test_set = FullereneComplexDataset("c72_100_IPR", root=str(FULLERENE_ROOT), target="Eb",
                                                use_pe=args.use_pe, pe_k=args.pe_k,
                                                fullerenet_features_only=args.fullerenet_features_only)

    print(f"Loaded: rk0={c20_c58_c60.rk0_dim} rk1={c20_c58_c60.rk1_dim} rk2={c20_c58_c60.rk2_dim}")
    print(f"Train: {n_train} | Val: {n_val} | "
          f"Test C60: {len(c60_test_set)} | Test C70 non-IPR: {len(c70_test_set)} | "
          f"Test C72-100 IPR: {len(c72_100_test_set)}")

    train_loader     = DataLoader(train_set,         batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    val_loader       = DataLoader(val_set,           batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    c60_test_loader  = DataLoader(c60_test_set,      batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    c70_test_loader  = DataLoader(c70_test_set,      batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    c72_test_loader  = DataLoader(c72_100_test_set,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # z-score normalization fitted on train labels only
    train_labels = torch.stack([train_val_set[i][-1] for i in train_set.indices])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    criterion = nn.MSELoss()
    results = {"use_pe": args.use_pe, "pe_k": args.pe_k, "epochs": args.epochs, "runs": []}

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        model = make_model(c20_c58_c60.rk0_dim, c20_c58_c60.rk1_dim, c20_c58_c60.rk2_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4,
                                      betas=(0.9, 0.999), eps=1e-8)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-4)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

        run_result = {"run": run + 1, "train_losses": [], "val_rmses": []}
        best_val_rmse = float("inf")
        patience_count = 0
        stop = False

        for epoch in range(args.epochs):
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
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  lr={lr:.2e}", end="")

            if (epoch + 1) % 5 == 0:
                val_rmse, val_mae = evaluate(model, val_loader, device, y_mean, y_std, criterion)
                run_result["val_rmses"].append(round(val_rmse, 4))
                run_result.setdefault("val_maes", []).append(round(val_mae, 4))
                print(f"  |  val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}", end="")
                if val_rmse < best_val_rmse and val_rmse <= 1.9:
                    best_val_rmse = val_rmse
                    patience_count = 0
                elif args.early_stopping:
                    patience_count += 1
                    if patience_count >= args.patience:
                        print(f"  |  early stop", end="")
                        stop = True
            print()
            if stop:
                break

        for name, loader in [("c60", c60_test_loader),
                             ("c70_non_IPR", c70_test_loader),
                             ("c72_100_IPR", c72_test_loader)]:
            test_rmse, test_mae = evaluate(model, loader, device, y_mean, y_std, criterion)
            run_result[f"test_rmse_{name}"] = round(test_rmse, 4)
            run_result[f"test_mae_{name}"]  = round(test_mae, 4)
            print(f"Test [{name}]  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}")
        results["runs"].append(run_result)

    for name in ["c60", "c70_non_IPR", "c72_100_IPR"]:
        results[f"mean_test_rmse_{name}"] = round(sum(r[f"test_rmse_{name}"] for r in results["runs"]) / 3, 4)
        results[f"mean_test_mae_{name}"]  = round(sum(r[f"test_mae_{name}"]  for r in results["runs"]) / 3, 4)
        print(f"\nMean test [{name}]  RMSE: {results[f'mean_test_rmse_{name}']:.4f}  "
              f"MAE: {results[f'mean_test_mae_{name}']:.4f}")

    out_path = Path(args.output)
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")
