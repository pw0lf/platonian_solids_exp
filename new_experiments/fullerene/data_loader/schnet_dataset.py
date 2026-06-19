import glob
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from torch_geometric.data import Data

SIZE_DIRS = {
    "c60": [f"c{n}" for n in [20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]],
    "c70_non_IPR": ["c70"],
    "c72_100_IPR": [f"c{n}" for n in [72, 74, 76, 78, 80, 82, 84, 86, 90, 92, 94, 96, 98, 100]],
}

LABEL_FILES = {
    "c60": "c20-c60-dft-all.csv",
    "c70_non_IPR": "c70-100-isomers-Eb-Eg-logP.csv",
    "c72_100_IPR": "c62-c720-dft-all.csv",
}

TARGET_COLUMNS = {
    "HOMO": "HOMO(eV)",
    "LUMO": "LUMO(eV)",
    "Gap": "HOMO-LUMO(eV)",
    "Eb": "E_binding(eV)",
}


def custom_sort(file):
    n1 = int(file.split("/")[-1].split("-")[0].split("c")[-1])
    n2 = int(file.split("/")[-1].split("-")[-1].split("_")[0])
    return (n1, n2)


def load_schnet_data(name, root=".", target="Eb"):
    if name not in SIZE_DIRS:
        raise ValueError(f"Unknown dataset name '{name}', expected one of {list(SIZE_DIRS)}")
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target '{target}', expected one of {list(TARGET_COLUMNS)}")

    root = Path(root)
    xyz_dir = root / "data" / "optimized_xyz"

    files = []
    for d in SIZE_DIRS[name]:
        files.extend(glob.glob(str(xyz_dir / d / "*.xyz")))
    files = sorted(files, key=custom_sort)

    labels = pd.read_csv(root / "data" / LABEL_FILES[name])
    assert len(files) == len(labels), f"Mismatch between xyz files ({len(files)}) and labels ({len(labels)})"

    target_col = TARGET_COLUMNS[target]
    targets = torch.tensor(labels[target_col].values, dtype=torch.float32)

    data_list = []
    for f, y in zip(files, targets):
        mol = Chem.MolFromXYZFile(f)
        conn_mol = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(conn_mol, charge=0)
        conf = conn_mol.GetConformer()
        z = torch.tensor([a.GetAtomicNum() for a in conn_mol.GetAtoms()], dtype=torch.long)
        pos = torch.tensor(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
             for i in range(conn_mol.GetNumAtoms())],
            dtype=torch.float32,
        )
        data_list.append(Data(z=z, pos=pos, y=y))

    return data_list
