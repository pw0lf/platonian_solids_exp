from torch.utils.data import Dataset
import torch
from rdkit import Chem
from pathlib import Path
import pandas as pd

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
            atom.GetAtomicNum(),
            atom.GetTotalValence(),
            atom.GetDegree(),
            atom.GetImplicitValence(),
            int(atom.GetIsAromatic()),
            int(atom.GetChiralTag()),
            atom.GetFormalCharge(),
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
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        en1 = EN.get(a1.GetAtomicNum(), 2.5)
        en2 = EN.get(a2.GetAtomicNum(), 2.5)
        feats.append([
            BOND_TYPE.get(bond.GetBondType(), 0),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            STEREO.get(bond.GetStereo(), 0),
            int(not bond.IsInRing() and not bond.GetIsConjugated() and bond.GetBondTypeAsDouble() == 1.0),
            bond_min_ring.get(bid, 0),
            int(a1.GetAtomicNum() in (7, 8) or a2.GetAtomicNum() in (7, 8)),
            abs(en1 - en2),
        ])
    return torch.tensor(feats, dtype=torch.float32)


def make_ring_features(mol, sssr):
    feats = []
    for ring_idx, ring in enumerate(sssr):
        ring = list(ring)
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        bonds = [mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
        bonds = [b for b in bonds if b]
        feats.append([
            len(ring),
            int(all(a.GetIsAromatic() for a in atoms)),
            sum(1 for a in atoms if a.GetAtomicNum() != 6),
            int(all(b.GetBondTypeAsDouble() == 1.0 for b in bonds)),
            int(any(len(set(ring) & set(list(other))) >= 2 for j, other in enumerate(sssr) if j != ring_idx)),
            sum(EN.get(a.GetAtomicNum(), 2.5) for a in atoms) / len(atoms),
        ])
    return torch.tensor(feats, dtype=torch.float32)


def make_icd_to_3(n_cells):
    row = torch.arange(n_cells)
    col = torch.zeros(n_cells, dtype=torch.long)
    idx = torch.stack([row, col], dim=0)
    vals = torch.ones(n_cells, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, size=(n_cells, 1)).coalesce()


def make_matrices(mol, sssr):
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    n_rings = len(sssr)

    # atoms × bonds
    icd01_rows, icd01_cols = [], []
    for bond in mol.GetBonds():
        k = bond.GetIdx()
        icd01_rows.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        icd01_cols.extend([k, k])
    icd01 = torch.sparse_coo_tensor(
        torch.tensor([icd01_rows, icd01_cols], dtype=torch.long),
        torch.ones(len(icd01_rows), dtype=torch.float32),
        size=(n_atoms, n_bonds),
    ).coalesce()

    # atoms × rings
    icd02_rows, icd02_cols = [], []
    for r_idx, ring in enumerate(sssr):
        for atom_idx in ring:
            icd02_rows.append(atom_idx)
            icd02_cols.append(r_idx)
    icd02 = torch.sparse_coo_tensor(
        torch.tensor([icd02_rows, icd02_cols], dtype=torch.long),
        torch.ones(len(icd02_rows), dtype=torch.float32),
        size=(n_atoms, n_rings),
    ).coalesce()

    # bonds × rings
    icd12_rows, icd12_cols = [], []
    for r_idx, ring in enumerate(sssr):
        ring = list(ring)
        for i in range(len(ring)):
            b = mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])
            if b:
                icd12_rows.append(b.GetIdx())
                icd12_cols.append(r_idx)
    icd12 = torch.sparse_coo_tensor(
        torch.tensor([icd12_rows, icd12_cols], dtype=torch.long),
        torch.ones(len(icd12_rows), dtype=torch.float32),
        size=(n_bonds, n_rings),
    ).coalesce()

    icd03 = make_icd_to_3(n_atoms)
    icd13 = make_icd_to_3(n_bonds)
    icd23 = make_icd_to_3(n_rings)

    return icd01, icd02, icd12, icd03, icd13, icd23


class Mol3d_Aug(Dataset):
    def __init__(self, root=".", size=1000000):
        super().__init__()
        root = Path(root)
        sdf_path = root / "combined_mols_1000000_to_2000000.sdf"
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        properties_path = root / "properties.csv"
        properties_df = pd.read_csv(properties_path)

        self.homolumogap = []
        self.atom_feature = []
        self.bond_feature = []
        self.ring_feature = []
        self.icd01 = []
        self.icd02 = []
        self.icd12 = []
        self.icd03 = []
        self.icd13 = []
        self.icd23 = []

        for i, mol in enumerate(suppl):
            if size and i >= size:
                break
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue

            sssr = list(Chem.GetSymmSSSR(mol))
            if len(sssr) == 0:
                continue

            icd01, icd02, icd12, icd03, icd13, icd23 = make_matrices(mol, sssr)

            self.atom_feature.append(make_atom_features(mol))
            self.bond_feature.append(make_bond_features(mol, sssr))
            self.ring_feature.append(make_ring_features(mol, sssr))
            self.icd01.append(icd01)
            self.icd02.append(icd02)
            self.icd12.append(icd12)
            self.icd03.append(icd03)
            self.icd13.append(icd13)
            self.icd23.append(icd23)

            row = properties_df.iloc[i]
            self.homolumogap.append(torch.tensor(row.homolumogap, dtype=torch.float32).unsqueeze(-1))

    def __len__(self):
        return len(self.homolumogap)

    def __getitem__(self, index):
        return (
            self.atom_feature[index],
            self.bond_feature[index],
            self.ring_feature[index],
            self.icd01[index], self.icd02[index], self.icd12[index],
            self.icd03[index], self.icd13[index], self.icd23[index],
            self.homolumogap[index],
        )
