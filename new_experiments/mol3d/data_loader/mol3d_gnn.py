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

EN = {1:2.20,5:2.04,6:2.55,7:3.04,8:3.44,9:3.98,14:1.90,15:2.19,16:2.58,17:3.16,35:2.96,53:2.66}

HYBRIDIZATION = {
    Chem.rdchem.HybridizationType.S:     0,
    Chem.rdchem.HybridizationType.SP:    1,
    Chem.rdchem.HybridizationType.SP2:   2,
    Chem.rdchem.HybridizationType.SP3:   3,
    Chem.rdchem.HybridizationType.SP3D:  4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
    Chem.rdchem.HybridizationType.OTHER: 0,
}

BOND_TYPE = {
    Chem.rdchem.BondType.SINGLE:   1,
    Chem.rdchem.BondType.DOUBLE:   2,
    Chem.rdchem.BondType.TRIPLE:   3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

STEREO = {
    Chem.rdchem.BondStereo.STEREONONE:  0,
    Chem.rdchem.BondStereo.STEREOANY:   1,
    Chem.rdchem.BondStereo.STEREOZ:     2,
    Chem.rdchem.BondStereo.STEREOE:     3,
    Chem.rdchem.BondStereo.STEREOCIS:   4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}

NODE_DIM = 11
EDGE_DIM = 8


def _make_atom_features(mol):
    conf = mol.GetConformer()
    feats = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        feats.append([
            atom.GetAtomicNum(), atom.GetTotalValence(), atom.GetDegree(),
            atom.GetImplicitValence(), int(atom.GetIsAromatic()),
            int(atom.GetChiralTag()), atom.GetFormalCharge(),
            HYBRIDIZATION.get(atom.GetHybridization(), 0),
            pos.x, pos.y, pos.z,
        ])
    return torch.tensor(feats, dtype=torch.float32)


def _make_bond_features(mol):
    sssr = list(Chem.GetSymmSSSR(mol))
    bond_min_ring = {}
    for ring in sssr:
        ring = list(ring)
        for i in range(len(ring)):
            a1, a2 = ring[i], ring[(i + 1) % len(ring)]
            b = mol.GetBondBetweenAtoms(a1, a2)
            if b:
                bid = b.GetIdx()
                bond_min_ring[bid] = min(bond_min_ring.get(bid, 999), len(ring))
    feats = []
    for bond in mol.GetBonds():
        bid = bond.GetIdx()
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        en1, en2 = EN.get(a1.GetAtomicNum(), 2.5), EN.get(a2.GetAtomicNum(), 2.5)
        feats.append([
            BOND_TYPE.get(bond.GetBondType(), 0), int(bond.GetIsConjugated()),
            int(bond.IsInRing()), STEREO.get(bond.GetStereo(), 0),
            int(not bond.IsInRing() and not bond.GetIsConjugated() and bond.GetBondTypeAsDouble() == 1.0),
            bond_min_ring.get(bid, 0),
            int(a1.GetAtomicNum() in (7, 8) or a2.GetAtomicNum() in (7, 8)),
            abs(en1 - en2),
        ])
    if not feats:
        return torch.zeros(0, EDGE_DIM, dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32)


def load_gnn_data(indices, root=".", load_edge_features=False):
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
            except Exception:
                continue

            x = _make_atom_features(mol)

            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
            if bonds:
                src = [e[0] for e in bonds] + [e[1] for e in bonds]
                dst = [e[1] for e in bonds] + [e[0] for e in bonds]
            else:
                src, dst = [], []
            edge_index = torch.tensor([src, dst], dtype=torch.long)

            if load_edge_features:
                ef = _make_bond_features(mol)
                edge_attr = torch.cat([ef, ef], dim=0)
            else:
                edge_attr = None

            g_idx = local_targets[local_idx]
            y = torch.tensor([properties_df.iloc[g_idx].homolumogap], dtype=torch.float32)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            loaded_indices.append(g_idx)

    return data_list, loaded_indices
