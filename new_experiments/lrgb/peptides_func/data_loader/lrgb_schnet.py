"""SchNet dataloader for LRGB Peptides datasets.

Generates 3D conformers with RDKit ETKDG from SMILES.
Requires SMILES CSV — run new_experiments/lrgb/download_smiles.py first.
"""

import torch
import pandas as pd
from multiprocessing import Pool, cpu_count
from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


class _LRGBNoDownload(LRGBDataset):
    def download(self):
        pass


def _embed_one(args):
    i, smi, y_list = args
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            return None
        mol = Chem.RemoveHs(mol)
        conf = mol.GetConformer()
        z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        pos = [[conf.GetAtomPosition(j).x,
                conf.GetAtomPosition(j).y,
                conf.GetAtomPosition(j).z]
               for j in range(mol.GetNumAtoms())]
        return (i, z, pos)
    except Exception:
        return None


def load_schnet_data(root, name, split, smiles_csv, num_workers=None):
    """
    Returns (data_list, loaded_indices) where loaded_indices are 0-based
    positions in the PyG split that were successfully embedded.
    Uses multiprocessing for conformer generation.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    pyg_ds = _LRGBNoDownload(root=root, name=name, split=split)

    df = pd.read_csv(smiles_csv)
    assert len(df) == len(pyg_ds), (
        f"SMILES CSV has {len(df)} rows but PyG split '{split}' has {len(pyg_ds)} items."
    )
    smiles_list = df["smiles"].tolist()
    ys = [item.y.float().tolist() for item in pyg_ds]

    tasks = [(i, smi, y) for i, (smi, y) in enumerate(zip(smiles_list, ys))]
    print(f"  SchNet {split}: embedding {len(tasks)} molecules with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = pool.map(_embed_one, tasks)

    data_list, loaded_indices = [], []
    for res, y_list in zip(results, ys):
        if res is None:
            continue
        i, z, pos = res
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(0)
        data_list.append(Data(
            z=torch.tensor(z, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float32),
            y=y,
        ))
        loaded_indices.append(i)

    failed = len(tasks) - len(data_list)
    print(f"  SchNet {split}: loaded {len(data_list)} / {len(pyg_ds)}  (failed: {failed})")
    return data_list, loaded_indices
