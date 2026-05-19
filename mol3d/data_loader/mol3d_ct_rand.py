import sys
import json
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gcb"))
from pe import CC_RWBSPe

# file ranges: (filename, global_start, global_end_exclusive)
_FILES = [
    ("combined_mols_0_to_1000000.sdf",         0,       1_000_000),
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


def make_atom_features(mol):
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


def make_bond_features(mol, sssr):
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
        return torch.zeros(0, 8, dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32)


def make_ring_features(mol, sssr):
    if len(sssr) == 0:
        return torch.zeros(1, 6, dtype=torch.float32)
    feats = []
    for ring_idx, ring in enumerate(sssr):
        ring = list(ring)
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        bonds = [mol.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)]) for i in range(len(ring))]
        bonds = [b for b in bonds if b]
        feats.append([
            len(ring), int(all(a.GetIsAromatic() for a in atoms)),
            sum(1 for a in atoms if a.GetAtomicNum() != 6),
            int(all(b.GetBondTypeAsDouble() == 1.0 for b in bonds)),
            int(any(len(set(ring) & set(list(other))) >= 2 for j, other in enumerate(sssr) if j != ring_idx)),
            sum(EN.get(a.GetAtomicNum(), 2.5) for a in atoms) / len(atoms),
        ])
    return torch.tensor(feats, dtype=torch.float32)


def make_matrices(mol, sssr):
    n_atoms, n_bonds, n_rings = mol.GetNumAtoms(), mol.GetNumBonds(), len(sssr)

    icd01_rows, icd01_cols = [], []
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        icd01_rows.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        icd01_cols.extend([bond_idx, bond_idx])
    icd01 = torch.sparse_coo_tensor(
        torch.tensor([icd01_rows, icd01_cols], dtype=torch.long),
        torch.ones(len(icd01_rows), dtype=torch.float32),
        size=(n_atoms, n_bonds)).coalesce()

    if n_rings == 0:
        icd02 = torch.zeros(n_atoms, 1).to_sparse_coo().coalesce()
        icd12 = torch.zeros(n_bonds, 1).to_sparse_coo().coalesce()
        n_rings = 1
    else:
        icd02_rows, icd02_cols = [], []
        for r_idx, ring in enumerate(sssr):
            for atom_idx in ring:
                icd02_rows.append(atom_idx); icd02_cols.append(r_idx)
        icd02 = torch.sparse_coo_tensor(
            torch.tensor([icd02_rows, icd02_cols], dtype=torch.long),
            torch.ones(len(icd02_rows), dtype=torch.float32),
            size=(n_atoms, n_rings)).coalesce()
        icd12_rows, icd12_cols = [], []
        for r_idx, ring in enumerate(sssr):
            ring = list(ring)
            for i in range(len(ring)):
                b = mol.GetBondBetweenAtoms(ring[i], ring[(i+1) % len(ring)])
                if b:
                    icd12_rows.append(b.GetIdx()); icd12_cols.append(r_idx)
        icd12 = torch.sparse_coo_tensor(
            torch.tensor([icd12_rows, icd12_cols], dtype=torch.long),
            torch.ones(len(icd12_rows), dtype=torch.float32),
            size=(n_bonds, n_rings)).coalesce()

    def make_adj(M):
        d = M.to_dense()
        return ((d @ d.T) > 0).float().to_sparse_coo().coalesce()

    return icd01, icd02, icd12, make_adj(icd01), make_adj(icd01.T), make_adj(icd12.T)


def _process_mol(mol, properties_df, global_idx, use_pe, pe_k):
    sssr = list(Chem.GetSymmSSSR(mol))
    if len(sssr) == 0:
        return None
    icd01, icd02, icd12, adj00, adj11, adj22 = make_matrices(mol, sssr)
    x_0 = make_atom_features(mol)
    x_1 = make_bond_features(mol, sssr)
    x_2 = make_ring_features(mol, sssr)
    if use_pe:
        n_atoms, n_bonds, n_rings = x_0.shape[0], x_1.shape[0], x_2.shape[0]
        pe = CC_RWBSPe(pe_k, n_atoms, n_bonds, n_rings, icd01, icd02, icd12, 'cpu')
        x_0 = torch.cat([x_0, pe[:n_atoms]], dim=1)
        x_1 = torch.cat([x_1, pe[n_atoms:(n_atoms + n_bonds)]], dim=1)
        x_2 = torch.cat([x_2, pe[(n_atoms + n_bonds):]], dim=1)
    y = torch.tensor([properties_df.iloc[global_idx].homolumogap], dtype=torch.float32)
    return (x_0, x_1, x_2, icd01, icd02, icd12, adj00, adj11, adj22, y)


class Mol3dCTRand(Dataset):
    def __init__(self, root=".", size=10000, use_pe=True, pe_k=5, seed=42):
        root = Path(root)
        rng = random.Random(seed)

        # load all known-valid indices from the precomputed split file
        split_path = root / "random_split_inds.json"
        split = json.load(open(split_path))
        all_valid = split["train"] + split["valid"] + split["test"]

        # sample without replacement
        sampled = sorted(rng.sample(all_valid, min(size, len(all_valid))))

        properties_df = pd.read_csv(root / "properties.csv")

        # group sampled global indices by file
        file_targets = {fname: {} for fname, _, _ in _FILES}
        for g_idx in sampled:
            for fname, g_start, g_end in _FILES:
                if g_start <= g_idx < g_end:
                    file_targets[fname][g_idx - g_start] = g_idx
                    break

        self.data = []
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
                item = _process_mol(mol, properties_df, local_targets[local_idx], use_pe, pe_k)
                if item is not None:
                    self.data.append(item)

        self.rk0_dim = self.data[0][0].shape[1]
        self.rk1_dim = self.data[0][1].shape[1]
        self.rk2_dim = self.data[0][2].shape[1]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
