"""SchNet dataloader for LRGB Peptides datasets.

Generates 3D conformers with RDKit ETKDG from SMILES.
Requires SMILES CSV — run new_experiments/lrgb/download_smiles.py first.
"""

import torch
import pandas as pd
from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def load_schnet_data(root, name, split, smiles_csv):
    """
    Returns (data_list, loaded_indices) where loaded_indices are 0-based
    positions in the PyG split that were successfully embedded.
    """
    pyg_ds = LRGBDataset(root=root, name=name, split=split)

    df = pd.read_csv(smiles_csv)
    assert len(df) == len(pyg_ds), (
        f"SMILES CSV has {len(df)} rows but PyG split '{split}' has {len(pyg_ds)} items."
    )
    smiles_list = df["smiles"].tolist()

    data_list, loaded_indices = [], []
    failed = 0
    for i, (smi, pyg_item) in enumerate(zip(smiles_list, pyg_ds)):
        y = pyg_item.y.squeeze(0).float()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42) == -1:
                failed += 1
                continue
            mol = Chem.RemoveHs(mol)
            conf = mol.GetConformer()
            z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
            pos = torch.tensor(
                [[conf.GetAtomPosition(j).x,
                  conf.GetAtomPosition(j).y,
                  conf.GetAtomPosition(j).z]
                 for j in range(mol.GetNumAtoms())],
                dtype=torch.float32,
            )
            data_list.append(Data(z=z, pos=pos, y=y))
            loaded_indices.append(i)
        except Exception:
            failed += 1
    print(f"  SchNet {split}: loaded {len(data_list)} / {len(pyg_ds)}  (failed: {failed})")
    return data_list, loaded_indices
