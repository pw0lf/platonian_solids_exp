import argparse
import json
import sys
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path

FULLERENE_ROOT = Path(__file__).parent.parent / "FullereneNet"
sys.path.insert(0, str(Path(__file__).parent.parent / "gcb"))
sys.path.insert(0, str(FULLERENE_ROOT))
from fullerene_complex_dataset import FullereneComplexDataset
from ct import CellularTransformer

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# Leave-size(s)-out validation splits over the C20-C58 train/val pool.
# C60 stays a held-out internal test set (as in fullerene_exp.py), independent of the split.
SPLIT_VAL_SIZES = [
    set(range(20, 51, 2)),  # split 0: val = C20-C50
    {52},                   # split 1: val = C52
    {54},                   # split 2: val = C54
    {56},                   # split 3: val = C56
    {58},                   # split 4: val = C58
]
TRAIN_VAL_SIZES = set(range(20, 59, 2))  # C20-C58


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


def build_model(rk0_dim, rk1_dim, rk2_dim, hp):
    return CellularTransformer(
        rk0_dim=rk0_dim, rk1_dim=rk1_dim, rk2_dim=rk2_dim, output_dim=1,
        num_layers=hp["num_layers"], hidden_dim=hp["hidden_dim"],
        num_heads=hp["num_heads"], hidden_dim_per_head=hp["hidden_dim_per_head"],
        att_dropout=hp["att_dropout"], emb_dropout=hp["emb_dropout"],
        readout_dropout=hp["readout_dropout"],
        num_readout_hidden_layers=hp["num_readout_hidden_layers"],
    )


def sample_hp(trial):
    return {
        "epochs":                    trial.suggest_int("epochs", 40, 200, step=10),
        "num_layers":               trial.suggest_int("num_layers", 2, 8),
        "hidden_dim":                trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "num_heads":                 trial.suggest_categorical("num_heads", [2, 4, 8]),
        "hidden_dim_per_head":       trial.suggest_categorical("hidden_dim_per_head", [8, 16, 32]),
        "att_dropout":               trial.suggest_float("att_dropout", 0.0, 0.3),
        "emb_dropout":               trial.suggest_float("emb_dropout", 0.0, 0.3),
        "readout_dropout":           trial.suggest_float("readout_dropout", 0.0, 0.3),
        "num_readout_hidden_layers": trial.suggest_int("num_readout_hidden_layers", 0, 3),
        "lr":                        trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "batch_size":                trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def train_and_eval(hp, args, dataset, train_idx, val_idx, y_mean, y_std, device, trial=None):
    epochs = hp["epochs"]
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=hp["batch_size"], shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=hp["batch_size"], shuffle=False, collate_fn=collate)

    model = build_model(dataset.rk0_dim, dataset.rk1_dim, dataset.rk2_dim, hp).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], betas=(0.9, 0.999), eps=1e-8)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - args.warmup_epochs, 1), eta_min=1e-4)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])

    best_val_rmse = float("inf")
    patience_count = 0

    for epoch in range(epochs):
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

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_rmse, _, _ = evaluate_full(model, val_loader, device, y_mean, y_std)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_count = 0
            else:
                patience_count += 1

            if trial is not None:
                trial.report(val_rmse, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if args.early_stopping and patience_count >= args.patience:
                break

    return model, best_val_rmse


def evaluate_full(model, loader, device, y_mean, y_std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
            out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
            preds.append((out * y_std + y_mean).cpu()); targets.append(y.cpu())
    p, t = torch.cat(preds), torch.cat(targets)
    rmse = ((p - t) ** 2).mean().sqrt().item()
    mae = (p - t).abs().mean().item()
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    return rmse, mae, r2


def make_objective(args, dataset, train_idx, val_idx, y_mean, y_std, device):
    def objective(trial):
        torch.manual_seed(args.seed)
        hp = sample_hp(trial)
        _, best_val_rmse = train_and_eval(hp, args, dataset, train_idx, val_idx, y_mean, y_std, device, trial=trial)
        return best_val_rmse
    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=[0, 1, 2, 3, 4], required=True,
                         help="0: val=C20-C50 | 1: val=C52 | 2: val=C54 | 3: val=C56 | 4: val=C58")
    parser.add_argument("--use_pe",       action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pe_k",         type=int,   default=5)
    parser.add_argument("--warmup_epochs",type=int,   default=5)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_trials",     type=int,   default=30)
    parser.add_argument("--seed",         type=int,   default=42, help="random seed for optuna sampler and torch")
    parser.add_argument("--study_name",   type=str,   default=None)
    parser.add_argument("--storage",      type=str,   default=None,
                         help="optuna storage URL, e.g. sqlite:///fullerene_hp.db (enables resuming)")
    parser.add_argument("--output",       type=str,   default="hp_results_fullerene.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | use_pe: {args.use_pe} | pe_k: {args.pe_k} | seed: {args.seed}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading dataset...")
    dataset = FullereneComplexDataset("c60", root=str(FULLERENE_ROOT), target="Eb",
                                       use_pe=args.use_pe, pe_k=args.pe_k)
    sizes = [dataset[i][0].shape[0] for i in range(len(dataset))]

    val_sizes = SPLIT_VAL_SIZES[args.split]
    train_idx = [i for i, s in enumerate(sizes) if s in TRAIN_VAL_SIZES and s not in val_sizes]
    val_idx   = [i for i, s in enumerate(sizes) if s in val_sizes]

    print(f"Loaded: rk0={dataset.rk0_dim} rk1={dataset.rk1_dim} rk2={dataset.rk2_dim}")
    print(f"Split {args.split}  val_sizes={sorted(val_sizes)}  train={len(train_idx)}  val={len(val_idx)}")

    train_labels = torch.stack([dataset[i][-1] for i in train_idx])
    y_mean = train_labels.mean().to(device)
    y_std  = train_labels.std().to(device)
    print(f"Label mean={y_mean.item():.4f}  std={y_std.item():.4f}")

    out_path = Path(args.output)
    if out_path.exists():
        stem, suffix = out_path.stem, out_path.suffix
        i = 1
        while out_path.exists():
            out_path = out_path.with_name(f"{stem}_{i}{suffix}")
            i += 1

    def build_results(final_eval=None):
        results = {
            "split": args.split,
            "val_sizes": sorted(val_sizes),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "use_pe": args.use_pe,
            "pe_k": args.pe_k,
            "n_trials": args.n_trials,
            "seed": args.seed,
            "trials": [
                {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
                for t in study.trials
            ],
        }
        completed = [t for t in study.trials if t.value is not None]
        if completed:
            results["best_value"] = round(study.best_value, 4)
            results["best_params"] = study.best_params
        if final_eval is not None:
            results["final_eval"] = final_eval
        return results

    def save_results(final_eval=None):
        with open(out_path, "w") as f:
            json.dump(build_results(final_eval), f, indent=2)
        print(f"Saved to {out_path}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                 study_name=args.study_name, storage=args.storage, load_if_exists=True)
    study.optimize(make_objective(args, dataset, train_idx, val_idx, y_mean, y_std, device),
                    n_trials=args.n_trials, callbacks=[lambda study, trial: save_results()])

    print(f"\nBest val RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    print("\nRetraining best config for final evaluation...")
    best_model, best_val_rmse = train_and_eval(study.best_params, args, dataset,
                                                 train_idx, val_idx, y_mean, y_std, device)

    c60_test_idx = [i for i, s in enumerate(sizes) if s == 60]
    c70_test_set     = FullereneComplexDataset("c70_non_IPR", root=str(FULLERENE_ROOT), target="Eb",
                                                 use_pe=args.use_pe, pe_k=args.pe_k)
    c72_100_test_set = FullereneComplexDataset("c72_100_IPR", root=str(FULLERENE_ROOT), target="Eb",
                                                 use_pe=args.use_pe, pe_k=args.pe_k)

    test_results = {}
    for name, loader in [
        ("val",         DataLoader(Subset(dataset, val_idx), batch_size=study.best_params["batch_size"],
                                    shuffle=False, collate_fn=collate)),
        ("c60",         DataLoader(Subset(dataset, c60_test_idx), batch_size=study.best_params["batch_size"],
                                    shuffle=False, collate_fn=collate)),
        ("c70_non_IPR", DataLoader(c70_test_set, batch_size=study.best_params["batch_size"],
                                    shuffle=False, collate_fn=collate)),
        ("c72_100_IPR", DataLoader(c72_100_test_set, batch_size=study.best_params["batch_size"],
                                    shuffle=False, collate_fn=collate)),
    ]:
        rmse, mae, r2 = evaluate_full(best_model, loader, device, y_mean, y_std)
        test_results[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
        print(f"Test [{name}]  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R2: {r2:.4f}")

    save_results(final_eval=test_results)
