import argparse
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "gcb"))
sys.path.insert(0, str(Path(__file__).parent))
from data_loader.mol3d_ct import Mol3dCT
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pe",       action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pe_k",         type=int,   default=5)
    parser.add_argument("--size",         type=int,   default=10000)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--warmup_epochs",type=int,   default=5)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--output",       type=str,   default="results_mol3d.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | use_pe: {args.use_pe} | pe_k: {args.pe_k} | size: {args.size}")

    print("Loading dataset...")
    dataset = Mol3dCT(root="./data/data/raw", size=args.size, use_pe=args.use_pe, pe_k=args.pe_k)
    print(f"Loaded: {len(dataset)} | rk0={dataset.rk0_dim} rk1={dataset.rk1_dim} rk2={dataset.rk2_dim}")

    n_train = int(0.8 * len(dataset))
    n_val   = int(0.1 * len(dataset))
    n_test  = len(dataset) - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                                 generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # z-score normalization fitted on train labels only
    train_labels = torch.stack([dataset[i][-1] for i in train_set.indices])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    criterion = nn.MSELoss()
    results = {"use_pe": args.use_pe, "pe_k": args.pe_k, "size": args.size,
               "epochs": args.epochs, "runs": []}

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        model = make_model(dataset.rk0_dim, dataset.rk1_dim, dataset.rk2_dim).to(device)
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
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                        out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                        preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
                val_rmse = criterion(torch.cat(preds), torch.cat(targets)).sqrt().item()
                run_result["val_rmses"].append(round(val_rmse, 4))
                print(f"  |  val_rmse={val_rmse:.4f}", end="")
                if val_rmse < best_val_rmse and val_rmse <= 1.9:
                    best_val_rmse = val_rmse
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= args.patience:
                        print(f"  |  early stop", end="")
                        stop = True
            print()
            if stop:
                break

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
        test_rmse = criterion(torch.cat(preds), torch.cat(targets)).sqrt().item()
        run_result["test_rmse"] = round(test_rmse, 4)
        print(f"Test RMSE: {test_rmse:.4f}")
        results["runs"].append(run_result)

    results["mean_test_rmse"] = round(sum(r["test_rmse"] for r in results["runs"]) / 3, 4)
    print(f"\nMean test RMSE: {results['mean_test_rmse']:.4f}")

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
