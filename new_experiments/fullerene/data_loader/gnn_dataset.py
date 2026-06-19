import glob
import sys
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent))
from fullerene_complex_dataset import (
    SIZE_DIRS, LABEL_FILES, TARGET_COLUMNS, custom_sort, load_mol,
    make_atom_features, make_simple_atom_features,
    make_bond_features, make_simple_bond_features,
)

NODE_DIM = {"full": 11, "simple": 3}
EDGE_DIM = {"full": 8,  "simple": 1}


def load_gnn_data(name, root=".", target="Eb", chem_features="full", load_edge_features=False):
    if name not in SIZE_DIRS:
        raise ValueError(f"Unknown dataset name '{name}', expected one of {list(SIZE_DIRS)}")
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target '{target}', expected one of {list(TARGET_COLUMNS)}")
    if chem_features not in ("full", "simple"):
        raise ValueError(f"chem_features must be 'full' or 'simple', got '{chem_features}'")

    root = Path(root)
    xyz_dir = root / "data" / "optimized_xyz"

    files = []
    for d in SIZE_DIRS[name]:
        files.extend(glob.glob(str(xyz_dir / d / "*.xyz")))
    files = sorted(files, key=custom_sort)

    labels = pd.read_csv(root / "data" / LABEL_FILES[name])
    assert len(files) == len(labels), f"Mismatch: {len(files)} xyz files vs {len(labels)} labels"

    target_col = TARGET_COLUMNS[target]
    targets = torch.tensor(labels[target_col].values, dtype=torch.float32)

    data_list = []
    for f, y in zip(files, targets):
        mol = load_mol(f)

        x = make_atom_features(mol) if chem_features == "full" else make_simple_atom_features(mol)

        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        if bonds:
            src = [e[0] for e in bonds] + [e[1] for e in bonds]
            dst = [e[1] for e in bonds] + [e[0] for e in bonds]
        else:
            src, dst = [], []
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        if load_edge_features:
            sssr = list(Chem.GetSymmSSSR(mol)) if chem_features == "full" else []
            ef = make_bond_features(mol, sssr) if chem_features == "full" else make_simple_bond_features(mol)
            edge_attr = torch.cat([ef, ef], dim=0)
        else:
            edge_attr = None

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

    return data_list
