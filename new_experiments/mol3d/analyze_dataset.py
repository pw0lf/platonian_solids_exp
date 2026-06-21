import json
import argparse
import numpy as np
from collections import deque
from pathlib import Path
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATA_ROOT = Path(__file__).parent.parent.parent / "mol3d" / "data" / "Molecule3D" / "raw"

_FILES = [
    ("combined_mols_0_to_1000000.sdf",         0,         1_000_000),
    ("combined_mols_1000000_to_2000000.sdf",    1_000_000, 2_000_000),
    ("combined_mols_2000000_to_3000000.sdf",    2_000_000, 3_000_000),
    ("combined_mols_3000000_to_3899647.sdf",    3_000_000, 3_899_647),
]


def graph_diameter(mol):
    """Longest shortest path between any two atoms (BFS from each atom)."""
    n = mol.GetNumAtoms()
    if n <= 1:
        return 0
    adj = [[] for _ in range(n)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        reachable = [d for d in dist if d >= 0]
        if reachable:
            max_dist = max(max_dist, max(reachable))
    return max_dist


def cyclomatic_number(mol):
    """Number of independent cycles: |E| - |V| + 1."""
    return mol.GetNumBonds() - mol.GetNumAtoms() + 1


def count_rings(mol):
    """Number of rings in the SSSR (smallest set of smallest rings)."""
    return mol.GetRingInfo().NumRings()


def spatial_diameter(mol):
    """Max Euclidean distance between any two atoms (Angstrom)."""
    n = mol.GetNumAtoms()
    if n <= 1:
        return 0.0
    conf = mol.GetConformer()
    coords = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(n)
    ])
    diff = coords[:, None, :] - coords[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=-1)).max())


def percentile_stats(arr):
    return {
        "min":    float(arr.min()),
        "p01":    float(np.percentile(arr, 1)),
        "p25":    float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean":   float(arr.mean()),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "max":    float(arr.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath",  type=str, default=str(DATA_ROOT))
    parser.add_argument("--max_mols",  type=int, default=None,
                        help="stop early after N molecules (for testing)")
    parser.add_argument("--output",    type=str,
                        default=str(Path(__file__).parent / "dataset_analysis.json"))
    args = parser.parse_args()

    root = Path(args.datapath)
    global_indices    = []
    graph_diameters   = []
    spatial_diameters = []
    cyclomatic_numbers = []
    num_rings_list    = []
    n_components_list = []
    n_total   = 0
    n_skipped = 0

    for fname, g_start, g_end in _FILES:
        print(f"\nProcessing {fname}  ({g_end - g_start:,} molecules)...")
        suppl = Chem.SDMolSupplier(str(root / fname), sanitize=False)
        for local_idx, mol in enumerate(suppl):
            if args.max_mols and n_total >= args.max_mols:
                break
            n_total += 1
            if n_total % 100_000 == 0:
                print(f"  {n_total:,} processed | {n_skipped:,} skipped | "
                      f"{len(graph_diameters):,} valid")
            if mol is None:
                n_skipped += 1
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                n_skipped += 1
                continue
            try:
                gd = graph_diameter(mol)
                sd = spatial_diameter(mol)
                cn = cyclomatic_number(mol)
                nr = count_rings(mol)
                nc = len(Chem.GetMolFrags(mol))
            except Exception:
                n_skipped += 1
                continue
            global_indices.append(g_start + local_idx)
            graph_diameters.append(gd)
            spatial_diameters.append(sd)
            cyclomatic_numbers.append(cn)
            num_rings_list.append(nr)
            n_components_list.append(nc)
        else:
            continue
        break  # early stop triggered by max_mols

    print(f"\nDone: {n_total:,} total | {len(graph_diameters):,} valid | {n_skipped:,} skipped")

    n_valid        = len(graph_diameters)
    n_disconnected = sum(1 for c in n_components_list if c > 1)
    n_neg_cyclo    = sum(1 for c in cyclomatic_numbers if c < 0)
    print(f"\nConnectivity check:")
    print(f"  num_components > 1 (disconnected): {n_disconnected:,} ({100*n_disconnected/n_valid:.3f}%)")
    print(f"  cyclomatic_number < 0:             {n_neg_cyclo:,} ({100*n_neg_cyclo/n_valid:.3f}%)")

    idx = np.array(global_indices, dtype=np.int32)
    gd  = np.array(graph_diameters, dtype=np.int16)
    sd  = np.array(spatial_diameters, dtype=np.float32)
    cn  = np.array(cyclomatic_numbers, dtype=np.int16)
    nr  = np.array(num_rings_list, dtype=np.int16)
    nc  = np.array(n_components_list, dtype=np.int8)

    per_mol_path = Path(args.output).with_name("dataset_per_molecule.npz")
    np.savez_compressed(per_mol_path, index=idx, graph_diameter=gd, spatial_diameter=sd,
                        cyclomatic_number=cn, num_rings=nr, num_components=nc)
    print(f"Per-molecule data saved to {per_mol_path}")

    results = {
        "n_total":        n_total,
        "n_valid":        n_valid,
        "n_skipped":      n_skipped,
        "n_disconnected": n_disconnected,
        "graph_diameter_hops":       percentile_stats(gd),
        "spatial_diameter_angstrom": percentile_stats(sd),
        "cyclomatic_number":         percentile_stats(cn),
        "num_rings":                 percentile_stats(nr),
    }

    print("\n--- Graph Diameter (hops) ---")
    for k, v in results["graph_diameter_hops"].items():
        print(f"  {k:8s}: {v:.2f}")

    print("\n--- Spatial Diameter (Å) ---")
    for k, v in results["spatial_diameter_angstrom"].items():
        print(f"  {k:8s}: {v:.2f}")

    print("\n--- Cyclomatic Number ---")
    for k, v in results["cyclomatic_number"].items():
        print(f"  {k:8s}: {v:.2f}")

    print("\n--- Num Rings (SSSR) ---")
    for k, v in results["num_rings"].items():
        print(f"  {k:8s}: {v:.2f}")

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
