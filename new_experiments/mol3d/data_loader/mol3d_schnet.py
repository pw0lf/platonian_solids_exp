import torch
import pandas as pd
from rdkit import Chem, RDLogger
from pathlib import Path
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

_FILES = [
    ("combined_mols_0_to_1000000.sdf",         0,         1_000_000),
    ("combined_mols_1000000_to_2000000.sdf",    1_000_000, 2_000_000),
    ("combined_mols_2000000_to_3000000.sdf",    2_000_000, 3_000_000),
    ("combined_mols_3000000_to_3899647.sdf",    3_000_000, 3_899_647),
]


def load_schnet_data(indices, root="."):
    root = Path(root)
    properties_df = pd.read_csv(root / "properties.csv")

    file_targets = {fname: {} for fname, _, _ in _FILES}
    for g_idx in sorted(indices):
        for fname, g_start, g_end in _FILES:
            if g_start <= g_idx < g_end:
                file_targets[fname][g_idx - g_start] = g_idx
                break

    data_list, loaded_indices = [], []
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
                continue
            try:
                Chem.SanitizeMol(mol)
                conf = mol.GetConformer()
            except Exception:
                continue
            z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
            pos = torch.tensor(
                [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                 for i in range(mol.GetNumAtoms())],
                dtype=torch.float32,
            )
            g_idx = local_targets[local_idx]
            y = torch.tensor([properties_df.iloc[g_idx].homolumogap], dtype=torch.float32)
            data_list.append(Data(z=z, pos=pos, y=y))
            loaded_indices.append(g_idx)

    return data_list, loaded_indices
