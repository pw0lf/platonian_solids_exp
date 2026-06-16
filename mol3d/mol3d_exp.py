import argparse
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "gcb"))
sys.path.insert(0, str(Path(__file__).parent))
from data_loader.mol3d_ct_rand import Mol3dCTRand
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
        output_dim=1, num_layers=6, hidden_dim=128, num_heads=4,
        hidden_dim_per_head=8, att_dropout=0.007223573743289108,
        emb_dropout=0.2889364748304471, readout_dropout=0.1262504279429404,
        num_readout_hidden_layers=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pe",          action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pe_k",            type=int,   default=5)
    parser.add_argument("--per_file_size",   type=int,   default=250000)
    parser.add_argument("--epochs",          type=int,   default=400)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=0.000965256564762386)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--checkpoint_every",type=int,   default=50)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--datapath",        type=str,   default="./data/data/raw")
    parser.add_argument("--output",          type=str,   default="results_mol3d.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | use_pe: {args.use_pe} | pe_k: {args.pe_k} | per_file_size: {args.per_file_size}")

    print("Loading dataset...")
    dataset = Mol3dCTRand(
        root=args.datapath, per_file_size=args.per_file_size,
        use_pe=args.use_pe, pe_k=args.pe_k, seed=args.seed,
    )
    print(f"Loaded: {len(dataset)} | rk0={dataset.rk0_dim} rk1={dataset.rk1_dim} rk2={dataset.rk2_dim}")

    n_train = int(0.8 * len(dataset))
    n_test  = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test],
                                       generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"Train: {n_train} | Test: {n_test}")

    # z-score normalization fitted on train labels only
    train_labels = torch.stack([dataset[i][-1] for i in train_set.indices])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    criterion = nn.MSELoss()
    results = {
        "use_pe": args.use_pe, "pe_k": args.pe_k, "per_file_size": args.per_file_size,
        "epochs": args.epochs, "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        model = make_model(dataset.rk0_dim, dataset.rk1_dim, dataset.rk2_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.999), eps=1e-8)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-4)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

        run_result = {"run": run + 1, "train_losses": [], "checkpoints": []}

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

            if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                        out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                        preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
                p, t = torch.cat(preds), torch.cat(targets)
                test_rmse = criterion(p, t).sqrt().item()
                test_mae  = (p - t).abs().mean().item()
                print(f"  |  test_rmse={test_rmse:.4f}  test_mae={test_mae:.4f}", end="")

                ckpt_path = f"checkpoint_run{run+1}_epoch{epoch+1}.pt"
                torch.save(model.state_dict(), ckpt_path)

                run_result["checkpoints"].append({
                    "epoch": epoch + 1,
                    "test_rmse": round(test_rmse, 4),
                    "test_mae":  round(test_mae, 4),
                })
            print()

        run_result["test_rmse"] = run_result["checkpoints"][-1]["test_rmse"]
        run_result["test_mae"]  = run_result["checkpoints"][-1]["test_mae"]
        print(f"Final test RMSE: {run_result['test_rmse']:.4f}  MAE: {run_result['test_mae']:.4f}")
        results["runs"].append(run_result)

    results["mean_test_rmse"] = round(sum(r["test_rmse"] for r in results["runs"]) / 3, 4)
    results["mean_test_mae"]  = round(sum(r["test_mae"]  for r in results["runs"]) / 3, 4)
    print(f"\nMean test RMSE: {results['mean_test_rmse']:.4f}  MAE: {results['mean_test_mae']:.4f}")

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
