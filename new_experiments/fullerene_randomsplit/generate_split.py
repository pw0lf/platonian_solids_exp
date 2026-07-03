"""
Run once to generate split.json.
Counts molecules from the 3 label CSVs (same order all exp scripts load data):
  1. c20-c60-dft-all.csv       (c60 subset)
  2. c70-100-isomers-Eb-Eg-logP.csv  (c70_non_IPR subset)
  3. c62-c720-dft-all.csv      (c72_100_IPR subset)
Then shuffles indices with --seed and saves train/test split.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch

DATA_ROOT = Path(__file__).parent.parent.parent / "FullereneNet" / "data"

LABEL_FILES = [
    "c20-c60-dft-all.csv",
    "c70-100-isomers-Eb-Eg-logP.csv",
    "c62-c720-dft-all.csv",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--test_ratio",   type=float, default=0.2)
    args = parser.parse_args()

    counts = [len(pd.read_csv(DATA_ROOT / f)) for f in LABEL_FILES]
    n_total = sum(counts)
    print(f"Dataset sizes: c60={counts[0]}, c70_non_IPR={counts[1]}, c72_100_IPR={counts[2]}, total={n_total}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g).tolist()

    n_test = int(n_total * args.test_ratio)
    test_idx  = sorted(perm[:n_test])
    train_idx = sorted(perm[n_test:])

    split = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "n_total": n_total,
        "n_train": len(train_idx),
        "n_test":  len(test_idx),
        "counts": {"c60": counts[0], "c70_non_IPR": counts[1], "c72_100_IPR": counts[2]},
        "train_idx": train_idx,
        "test_idx":  test_idx,
    }

    out = Path(__file__).parent / "split.json"
    with open(out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Saved split to {out}: {len(train_idx)} train / {len(test_idx)} test")
