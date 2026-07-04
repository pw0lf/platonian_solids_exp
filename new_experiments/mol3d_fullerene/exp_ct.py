import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

MOL3D_DIR      = Path(__file__).parent.parent / "mol3d"
FULLERENE_ROOT = Path(__file__).parent.parent.parent / "FullereneNet"

sys.path.insert(0, str(MOL3D_DIR / "data_loader"))
sys.path.insert(0, str(MOL3D_DIR / "models"))
sys.path.insert(0, str(Path(__file__).parent / "data_loader"))
from mol3d_ct_rand import Mol3dCT
from ct import CellularTransformer
from fullerene_loader import load_fullerene_ct

MOL3D_DATA_ROOT      = str(Path(__file__).parent.parent.parent / "mol3d" / "data" / "data" / "raw")
MOL3D_SPLIT_FILE     = MOL3D_DIR / "data_split.json"
FULLERENE_SPLIT_FILE = Path(__file__).parent.parent / "fullerene_randomsplit" / "split.json"

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


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
    return rmse, mae, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol3d_datapath",  type=str,   default=MOL3D_DATA_ROOT)
    parser.add_argument("--mol3d_split",     type=str,   default=str(MOL3D_SPLIT_FILE))
    parser.add_argument("--fullerene_split", type=str,   default=str(FULLERENE_SPLIT_FILE))
    parser.add_argument("--feat_mode",       type=str,   default="full", choices=["full", "simple", "coords"])
    parser.add_argument("--pe_k",            type=int,   default=5)
    parser.add_argument("--epochs",          type=int,   default=300)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=0.000965256564762386)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--output",          type=str,   default=None)
    args = parser.parse_args()

    use_pe = args.feat_mode == "full"

    if args.output is None:
        args.output = f"results_ct_{args.feat_mode}.json"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | feat_mode: {args.feat_mode} | use_pe: {use_pe} | pe_k: {args.pe_k}")

    mol3d_split = json.load(open(args.mol3d_split))
    full_split  = json.load(open(args.fullerene_split))

    print("Loading mol3d train data...")
    mol3d_train_ds = Mol3dCT(mol3d_split["train"], root=args.mol3d_datapath,
                              use_pe=use_pe, pe_k=args.pe_k, feat_mode=args.feat_mode)
    print("Loading mol3d test data...")
    mol3d_test_ds  = Mol3dCT(mol3d_split["test"],  root=args.mol3d_datapath,
                              use_pe=use_pe, pe_k=args.pe_k, feat_mode=args.feat_mode)

    rk0_dim = mol3d_train_ds.rk0_dim
    rk1_dim = mol3d_train_ds.rk1_dim
    rk2_dim = mol3d_train_ds.rk2_dim
    print(f"CT dims: rk0={rk0_dim} rk1={rk1_dim} rk2={rk2_dim}")

    print("Loading fullerene data...")
    fullerene_all = load_fullerene_ct(root=str(FULLERENE_ROOT), target="Gap",
                                      use_pe=use_pe, pe_k=args.pe_k, feat_mode=args.feat_mode)
    assert len(fullerene_all) == full_split["n_total"], \
        f"Fullerene size mismatch: {len(fullerene_all)} vs {full_split['n_total']}"
    full_train = [fullerene_all[i] for i in full_split["train_idx"]]
    full_test  = [fullerene_all[i] for i in full_split["test_idx"]]

    train_dataset = ListDataset(list(mol3d_train_ds) + full_train)
    test_dataset  = ListDataset(list(mol3d_test_ds)  + full_test)
    print(f"Train: {len(train_dataset)} ({len(mol3d_train_ds)} mol3d + {len(full_train)} fullerene)")
    print(f"Test:  {len(test_dataset)} ({len(mol3d_test_ds)} mol3d + {len(full_test)} fullerene)")

    train_labels = torch.stack([train_dataset[i][-1] for i in range(len(train_dataset))]).squeeze(-1)
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    criterion   = nn.MSELoss()
    results = {
        "feat_mode": args.feat_mode, "pe_k": args.pe_k, "epochs": args.epochs,
        "mol3d_split": args.mol3d_split, "fullerene_split": args.fullerene_split,
        "n_train_mol3d": len(mol3d_train_ds), "n_train_fullerene": len(full_train),
        "n_test_mol3d":  len(mol3d_test_ds),  "n_test_fullerene":  len(full_test),
        "runs": [],
    }

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        torch.manual_seed(args.seed + run)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        model = make_model(rk0_dim, rk1_dim, rk2_dim).to(device)
        if run == 0:
            results["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-4)
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
                loss = criterion(out, (y.squeeze(-1) - y_mean) / y_std)
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

        test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device, y_mean, y_std, criterion)
        run_result["test_rmse"]  = round(test_rmse, 4)
        run_result["test_mae"]   = round(test_mae, 4)
        run_result["test_r2"]    = round(test_r2, 4)
        run_result["runtime"]    = round(time.time() - run_start, 2)
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
