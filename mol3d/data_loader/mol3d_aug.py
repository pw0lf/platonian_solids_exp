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


def _to_global(n_cells):
    row = torch.arange(n_cells)
    col = torch.zeros(n_cells, dtype=torch.long)
    idx = torch.stack([row, col], dim=0)
    vals = torch.ones(n_cells, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, size=(n_cells, 1)).coalesce()


def make_matrices(mol, sssr, k=2):
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    n_rings = len(sssr)

    # atoms × bonds
    icd01_rows, icd01_cols = [], []
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        icd01_rows.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        icd01_cols.extend([bond_idx, bond_idx])
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

    # --- k-hop lifting: rank-3 cells ---
    # atom-atom adjacency
    A = torch.zeros(n_atoms, n_atoms)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[a, b] = A[b, a] = 1.0

    # k-hop reachability: P[i,j] > 0 iff j reachable from i in ≤k steps
    P = A.clone()
    for _ in range(2, k + 1):
        P = P @ A
    khop = torch.unique((P > 0).long(), dim=1).float()  # (n_atoms, n_khop_cells)
    n_khop = khop.shape[1]

    # atoms × k-hop cells
    icd03 = khop.to_sparse_coo().coalesce()

    # bonds × k-hop cells: bond in cell j if both atoms are in cell j
    icd13_dense = torch.zeros(n_bonds, n_khop)
    for i in range(n_bonds):
        icd13_dense[i] = (khop[icd01_rows[2 * i]].bool() & khop[icd01_rows[2 * i + 1]].bool()).float()
    icd13 = icd13_dense.to_sparse_coo().coalesce()

    # rings × k-hop cells: ring in cell j if all ring atoms are in cell j
    icd23_dense = torch.zeros(n_rings, n_khop)
    for r_idx, ring in enumerate(sssr):
        ring = list(ring)
        membership = khop[ring[0]].bool()
        for atom_idx in ring[1:]:
            membership = membership & khop[atom_idx].bool()
        icd23_dense[r_idx] = membership.float()
    icd23 = icd23_dense.to_sparse_coo().coalesce()

    # k-hop cells × global node (rank 4)
    icd34 = _to_global(n_khop)

    # same-rank adjacency
    def make_adj(M):
        d = M.to_dense()
        adj = (d @ d.T > 0).float()
        return adj.to_sparse_coo().coalesce()

    adj00 = make_adj(icd01)    # atoms × atoms  (share a bond)
    adj11 = make_adj(icd01.T)  # bonds × bonds  (share an atom)
    adj22 = make_adj(icd12.T)  # rings × rings  (share a bond)
    adj33 = make_adj(icd23.T)  # k-hop × k-hop  (share a ring)

    return icd01, icd02, icd12, icd03, icd13, icd23, icd34, adj00, adj11, adj22, adj33


class Mol3d_Aug(Dataset):
    def __init__(self, root=".", size=1000000, k=3):
        super().__init__()
        root = Path(root)
        sdf_path = root / "combined_mols_2000000_to_3000000.sdf"
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
        self.icd34 = []
        self.adj00 = []
        self.adj11 = []
        self.adj22 = []
        self.adj33 = []

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

            icd01, icd02, icd12, icd03, icd13, icd23, icd34, adj00, adj11, adj22, adj33 = make_matrices(mol, sssr, k)

            self.atom_feature.append(make_atom_features(mol))
            self.bond_feature.append(make_bond_features(mol, sssr))
            self.ring_feature.append(make_ring_features(mol, sssr))
            self.icd01.append(icd01)
            self.icd02.append(icd02)
            self.icd12.append(icd12)
            self.icd03.append(icd03)
            self.icd13.append(icd13)
            self.icd23.append(icd23)
            self.icd34.append(icd34)
            self.adj00.append(adj00)
            self.adj11.append(adj11)
            self.adj22.append(adj22)
            self.adj33.append(adj33)

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
            self.icd34[index],
            self.adj00[index], self.adj11[index], self.adj22[index], self.adj33[index],
            self.homolumogap[index],
        )
