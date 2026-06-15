import argparse
import json
import sys
import torch
import torch.nn as nn
import optuna
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


def train_and_eval(model, train_loader, val_loader, y_mean, y_std, lr, args, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=1e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    best_val_rmse = float("inf")
    patience_count = 0

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
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                    out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                    preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
            p, t = torch.cat(preds), torch.cat(targets)
            val_rmse = criterion(p, t).sqrt().item()

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_count = 0
            elif args.early_stopping:
                patience_count += 1
                if patience_count >= args.patience:
                    break

    return best_val_rmse


def make_objective(dataset, train_set, val_set, y_mean, y_std, args, device):
    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        if hidden_dim % num_heads != 0:
            raise optuna.TrialPruned()

        hp = {
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "hidden_dim_per_head": trial.suggest_int("hidden_dim_per_head", 8, 32, step=8),
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "att_dropout": trial.suggest_float("att_dropout", 0.0, 0.3),
            "emb_dropout": trial.suggest_float("emb_dropout", 0.0, 0.3),
            "readout_dropout": trial.suggest_float("readout_dropout", 0.0, 0.3),
            "num_readout_hidden_layers": trial.suggest_int("num_readout_hidden_layers", 1, 3),
        }
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        torch.manual_seed(args.seed + trial.number)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate)

        model = CellularTransformer(
            rk0_dim=dataset.rk0_dim, rk1_dim=dataset.rk1_dim, rk2_dim=dataset.rk2_dim,
            output_dim=1, **hp,
        ).to(device)

        try:
            return train_and_eval(model, train_loader, val_loader, y_mean, y_std, lr, args, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise optuna.TrialPruned()
            raise

    return objective


def make_callback(output_path, meta):
    def callback(study, trial):
        out = dict(meta)
        out["trials"] = [
            {"number": t.number, "params": t.params, "value": t.value, "state": str(t.state)}
            for t in study.trials
        ]
        try:
            best = study.best_trial
            out["best_trial"] = {"number": best.number, "params": best.params, "value": best.value}
        except ValueError:
            pass
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
    return callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pe",       action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pe_k",         type=int,   default=5)
    parser.add_argument("--size",         type=int,   default=10000)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--warmup_epochs",type=int,   default=5)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ntrials",      type=int,   default=20)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--datapath",     type=str,   default="./data/data/raw")
    parser.add_argument("--output",       type=str,   default="results_hp_mol3d_ct.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | use_pe: {args.use_pe} | pe_k: {args.pe_k} | size: {args.size} | seed: {args.seed}")

    print("Loading dataset...")
    dataset = Mol3dCTRand(root=args.datapath, size=args.size, use_pe=args.use_pe, pe_k=args.pe_k, seed=args.seed)
    print(f"Loaded: {len(dataset)} | rk0={dataset.rk0_dim} rk1={dataset.rk1_dim} rk2={dataset.rk2_dim}")

    n_train = int(0.8 * len(dataset))
    n_val   = int(0.1 * len(dataset))
    n_test  = len(dataset) - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test],
                                                 generator=torch.Generator().manual_seed(args.seed))
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # z-score normalization fitted on train labels only
    train_labels = torch.stack([dataset[i][-1] for i in train_set.indices])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    meta = {
        "use_pe": args.use_pe, "pe_k": args.pe_k, "size": args.size,
        "epochs": args.epochs, "seed": args.seed,
    }

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    objective = make_objective(dataset, train_set, val_set, y_mean, y_std, args, device)
    callback = make_callback(args.output, meta)
    study.optimize(objective, n_trials=args.ntrials, callbacks=[callback])

    print(f"\nBest val RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Results saved to {args.output}")
