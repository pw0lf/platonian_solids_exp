import glob
import sys
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "gcb"))
from pe import CC_RWBSPe

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

EN = {1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}

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


def custom_sort(file):
    n1 = int(file.split("/")[-1].split("-")[0].split("c")[-1])
    n2 = int(file.split("/")[-1].split("-")[-1].split("_")[0])
    return (n1, n2)


def load_mol(xyz_path):
    mol = Chem.MolFromXYZFile(xyz_path)
    conn_mol = Chem.Mol(mol)
    rdDetermineBonds.DetermineBonds(conn_mol, charge=0)
    return conn_mol


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


def _process_mol(mol, target, use_pe, pe_k):
    sssr = list(Chem.GetSymmSSSR(mol))
    icd01, icd02, icd12, adj00, adj11, adj22 = make_matrices(mol, sssr)
    x_0 = make_atom_features(mol)
    x_1 = make_bond_features(mol, sssr)
    x_2 = make_ring_features(mol, sssr)
    if use_pe:
        n_atoms, n_bonds, n_rings = x_0.shape[0], x_1.shape[0], x_2.shape[0]
        pe = CC_RWBSPe(pe_k, n_atoms, n_bonds, n_rings, icd01, icd02, icd12, "cpu")
        x_0 = torch.cat([x_0, pe[:n_atoms]], dim=1)
        x_1 = torch.cat([x_1, pe[n_atoms:(n_atoms + n_bonds)]], dim=1)
        x_2 = torch.cat([x_2, pe[(n_atoms + n_bonds):]], dim=1)
    return (x_0, x_1, x_2, icd01, icd02, icd12, adj00, adj11, adj22, target)


class FullereneComplexDataset(Dataset):
    """Fullerene cage as a 2-dimensional cell complex (atoms / bonds / rings),
    using the same node/edge/ring featurization and incidence/adjacency matrix
    construction as `mol3d_ct_rand.Mol3dCTRand` (atom features, bond features,
    SSSR-ring features, icd01/icd02/icd12 incidence and adj00/adj11/adj22
    up-adjacency matrices).
    """

    def __init__(self, name, root=".", target="Eb", use_pe=True, pe_k=5):
        super().__init__()
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
        targets = torch.tensor(labels[target_col].values, dtype=torch.float32).unsqueeze(-1)

        self.data = []
        for f, y in zip(files, targets):
            mol = load_mol(f)
            self.data.append(_process_mol(mol, y, use_pe, pe_k))

        self.rk0_dim = self.data[0][0].shape[1]
        self.rk1_dim = self.data[0][1].shape[1]
        self.rk2_dim = self.data[0][2].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
