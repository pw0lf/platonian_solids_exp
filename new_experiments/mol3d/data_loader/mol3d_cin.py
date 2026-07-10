"""CIN / CIN++ dataloader for mol3d.

Loads molecules via the same SDF pipeline as mol3d_ct_rand.py's
_load_from_indices, reusing its atom/bond/ring feature builders
(make_atom_features/make_bond_features/make_ring_features -- same features
CT's feat_mode="full" uses, no positional encoding) and its CIN cell-complex
index builder (make_cin_indices). See lrgb/peptides_struct/data_loader/
lrgb_cin.py for the up/down/boundary semantics.
"""
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from mol3d_ct_rand import (
    _FILES, make_atom_features, make_bond_features, make_ring_features, make_cin_indices,
)


def _process_mol_cin(mol, y):
    sssr = list(Chem.GetSymmSSSR(mol))
    x_0 = make_atom_features(mol)
    x_1 = make_bond_features(mol, sssr)
    x_2 = make_ring_features(mol, sssr) if len(sssr) > 0 else torch.zeros(0, 6, dtype=torch.float32)
    cin = make_cin_indices(mol, sssr)
    return dict(x_0=x_0, x_1=x_1, x_2=x_2, y=y, **cin)


def _load_from_indices_cin(indices, root):
    root = Path(root)
    properties_df = pd.read_csv(root / "properties.csv")

    file_targets = {fname: {} for fname, _, _ in _FILES}
    for g_idx in sorted(indices):
        for fname, g_start, g_end in _FILES:
            if g_start <= g_idx < g_end:
                file_targets[fname][g_idx - g_start] = g_idx
                break

    data, loaded_indices = [], []
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
            except Exception:
                continue
            g_idx = local_targets[local_idx]
            y = torch.tensor([properties_df.iloc[g_idx].homolumogap], dtype=torch.float32)
            data.append(_process_mol_cin(mol, y))
            loaded_indices.append(g_idx)

    return data, loaded_indices


class Mol3dCIN(Dataset):
    """Load CIN data for an explicit list of global molecule indices."""
    def __init__(self, indices, root="."):
        self.data, self.indices = _load_from_indices_cin(indices, root)
        assert len(self.data) > 0, "No molecules successfully loaded."
        self.x0_dim = self.data[0]["x_0"].shape[1]
        self.x1_dim = self.data[0]["x_1"].shape[1]
        self.x2_dim = self.data[0]["x_2"].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_cin(batch):
    """Batches a list of per-molecule dicts (see _process_mol_cin) into one
    big disjoint-union complex, offsetting all indices and building per-rank
    batch-assignment vectors for pooling."""
    off = [0, 0, 0]  # cumulative atom/bond/ring counts
    x0_list, x1_list, x2_list, y_list = [], [], [], []
    batch0, batch1, batch2 = [], [], []
    up0_i, up0_a, up1_i, up1_a = [], [], [], []
    down1_i, down1_a, down2_i, down2_a = [], [], [], []
    b1_i, b2_i = [], []

    for g, item in enumerate(batch):
        n0, n1, n2 = item["n_atoms"], item["n_bonds"], item["n_rings"]
        x0_list.append(item["x_0"]); x1_list.append(item["x_1"]); x2_list.append(item["x_2"])
        y_list.append(item["y"])
        batch0.append(torch.full((n0,), g, dtype=torch.long))
        batch1.append(torch.full((n1,), g, dtype=torch.long))
        batch2.append(torch.full((n2,), g, dtype=torch.long))

        up0_i.append(item["up0_index"] + off[0])
        up0_a.append(item["up0_attr_idx"] + off[1])
        up1_i.append(item["up1_index"] + off[1])
        up1_a.append(item["up1_attr_idx"] + off[2])
        down1_i.append(item["down1_index"] + off[1])
        down1_a.append(item["down1_attr_idx"] + off[0])
        down2_i.append(item["down2_index"] + off[2])
        down2_a.append(item["down2_attr_idx"] + off[1])
        b1_i.append(item["boundary1_index"] + torch.tensor([[off[0]], [off[1]]]))
        b2_i.append(item["boundary2_index"] + torch.tensor([[off[1]], [off[2]]]))

        off[0] += n0; off[1] += n1; off[2] += n2

    def cat_idx(lst):
        return torch.cat(lst, dim=1) if lst else torch.zeros(2, 0, dtype=torch.long)

    def cat_attr(lst):
        return torch.cat(lst, dim=0) if lst else torch.zeros(0, dtype=torch.long)

    return dict(
        x_0=torch.cat(x0_list), x_1=torch.cat(x1_list), x_2=torch.cat(x2_list),
        batch0=torch.cat(batch0), batch1=torch.cat(batch1), batch2=torch.cat(batch2),
        up0_index=cat_idx(up0_i), up0_attr_idx=cat_attr(up0_a),
        up1_index=cat_idx(up1_i), up1_attr_idx=cat_attr(up1_a),
        down1_index=cat_idx(down1_i), down1_attr_idx=cat_attr(down1_a),
        down2_index=cat_idx(down2_i), down2_attr_idx=cat_attr(down2_a),
        boundary1_index=cat_idx(b1_i), boundary2_index=cat_idx(b2_i),
        y=torch.stack(y_list),
        num_graphs=len(batch),
    )
