"""Shared helpers for this experiment's hp_tuning_<model>.py scripts.

These NEVER touch the real test split -- only carve/subsample from the
train split (or, where a real val split already exists elsewhere in the
repo, e.g. lrgb, subsample that instead of carving a new one out of train).
Kept small and duplicated per-experiment rather than shared across
experiment families, matching this repo's existing convention (e.g.
lrgb/smiles_align.py living only under lrgb/).
"""
import itertools
import json
import random
from pathlib import Path


def carve_val_from_train(train_indices, val_frac=0.15, seed=0):
    """Deterministic split of TRAIN indices only into (hp_train, hp_val).
    Never touches test -- test indices are never passed into this function
    at all, by construction of every hp_tuning_<model>.py script."""
    rng = random.Random(seed)
    idx = list(train_indices)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    hp_val = idx[:n_val]
    hp_train = idx[n_val:]
    assert set(hp_val).isdisjoint(hp_train)
    assert set(hp_val) <= set(train_indices) and set(hp_train) <= set(train_indices)
    return hp_train, hp_val


def subsample(indices, max_n, seed=0):
    """Caps a list of indices to at most max_n, for fast tuning runs."""
    if len(indices) <= max_n:
        return list(indices)
    rng = random.Random(seed)
    return rng.sample(list(indices), max_n)


def sample_configs(space, n_trials=10, seed=0):
    """space: dict[str, list-of-candidate-values]. Returns up to n_trials
    hyperparameter dicts, sampled without replacement from the cartesian
    product of the candidate grid (random search, not full grid search)."""
    rng = random.Random(seed)
    all_combos = list(itertools.product(*space.values()))
    rng.shuffle(all_combos)
    n = min(n_trials, len(all_combos))
    keys = list(space.keys())
    return [dict(zip(keys, combo)) for combo in all_combos[:n]]


def save_best(path, best_config, best_val_metric, all_trials, metric_name="val_mae"):
    """Writes {**best_config, metric_name: best_val_metric, "trials": [...]}."""
    result = {**best_config, metric_name: round(float(best_val_metric), 6), "trials": all_trials}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path
