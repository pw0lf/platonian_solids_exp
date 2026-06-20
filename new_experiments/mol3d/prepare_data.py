import json
import random
from pathlib import Path
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA_ROOT = Path(__file__).parent.parent.parent / "mol3d" / "data" / "Molecule3D" / "raw"
OUT_PATH  = Path(__file__).parent / "data_split.json"

SIZE       = 10000
OVERSAMPLE = 15000  # extra buffer for molecules that fail validation
SEED       = 42

_FILES = [
    ("combined_mols_0_to_1000000.sdf",         0,         1_000_000),
    ("combined_mols_1000000_to_2000000.sdf",    1_000_000, 2_000_000),
    ("combined_mols_2000000_to_3000000.sdf",    2_000_000, 3_000_000),
    ("combined_mols_3000000_to_3899647.sdf",    3_000_000, 3_899_647),
]


def validate_indices(candidates, root):
    """Stream SDF files for candidate indices; return those passing all CT checks."""
    root = Path(root)

    file_targets = {fname: {} for fname, _, _ in _FILES}
    for g_idx in sorted(candidates):
        for fname, g_start, g_end in _FILES:
            if g_start <= g_idx < g_end:
                file_targets[fname][g_idx - g_start] = g_idx
                break

    valid = []
    skipped = 0
    for fname, g_start, _ in _FILES:
        local_targets = file_targets[fname]
        if not local_targets:
            continue
        max_local = max(local_targets.keys())
        suppl = Chem.SDMolSupplier(str(root / fname), sanitize=False)
        for local_idx, mol in enumerate(suppl):
            if local_idx > max_local:
                break
            if local_idx not in local_targets:
                continue
            if mol is None:
                skipped += 1
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                skipped += 1
                continue
            if len(list(Chem.GetSymmSSSR(mol))) == 0:
                skipped += 1
                continue
            valid.append(local_targets[local_idx])

    return valid, skipped


def main():
    split = json.load(open(DATA_ROOT / "random_split_inds.json"))
    all_valid = split["train"] + split["valid"] + split["test"]
    print(f"Total molecules in split file: {len(all_valid)}")

    rng = random.Random(SEED)
    candidates = rng.sample(all_valid, min(OVERSAMPLE, len(all_valid)))
    print(f"Validating {len(candidates)} candidate molecules...")

    valid_indices, skipped = validate_indices(candidates, DATA_ROOT)
    print(f"Passed: {len(valid_indices)}  |  Skipped: {skipped}")

    if len(valid_indices) < SIZE:
        raise RuntimeError(
            f"Only {len(valid_indices)} valid molecules found after oversampling {OVERSAMPLE}. "
            f"Increase OVERSAMPLE and retry."
        )

    rng.shuffle(valid_indices)
    selected = valid_indices[:SIZE]

    rng.shuffle(selected)
    n_train = int(0.8 * len(selected))
    train_indices = sorted(selected[:n_train])
    test_indices  = sorted(selected[n_train:])

    result = {
        "train": train_indices,
        "test":  test_indices,
        "size":  len(selected),
        "seed":  SEED,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {len(train_indices)} train + {len(test_indices)} test → {OUT_PATH}")


if __name__ == "__main__":
    main()
