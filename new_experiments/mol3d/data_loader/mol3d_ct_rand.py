import sys
import bisect
import json
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem, RDLogger
from pathlib import Path

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from pe import CC_RWBSPe

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

_PT = Chem.GetPeriodicTable()


# ── full features ─────────────────────────────────────────────────────────────

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


# ── simple features ───────────────────────────────────────────────────────────

def make_simple_atom_features(mol):
    feats = []
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        feats.append([
            atom.GetMass(),
            EN.get(an, 2.5),
            _PT.GetRvdw(an),
        ])
    return torch.tensor(feats, dtype=torch.float32)


def make_simple_bond_features(mol):
    feats = [[float(BOND_TYPE.get(b.GetBondType(), 0))] for b in mol.GetBonds()]
    if not feats:
        return torch.zeros(0, 1, dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32)


def make_simple_ring_features(sssr):
    if len(sssr) == 0:
        return torch.zeros(1, 1, dtype=torch.float32)
    return torch.tensor([[float(len(list(ring)))] for ring in sssr], dtype=torch.float32)


# ── coord-only features ───────────────────────────────────────────────────────

def make_coord_atom_features(mol):
    conf = mol.GetConformer()
    feats = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        feats.append([p.x, p.y, p.z])
    return torch.tensor(feats, dtype=torch.float32)


def make_coord_bond_features(mol):
    conf = mol.GetConformer()
    feats = []
    for bond in mol.GetBonds():
        p1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        p2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
        feats.append([(p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2])
    if not feats:
        return torch.zeros(0, 3, dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32)


def make_coord_ring_features(mol, sssr):
    conf = mol.GetConformer()
    if len(sssr) == 0:
        return torch.zeros(1, 3, dtype=torch.float32)
    feats = []
    for ring in sssr:
        ring = list(ring)
        positions = [conf.GetAtomPosition(i) for i in ring]
        feats.append([
            sum(p.x for p in positions) / len(positions),
            sum(p.y for p in positions) / len(positions),
            sum(p.z for p in positions) / len(positions),
        ])
    return torch.tensor(feats, dtype=torch.float32)


# ── incidence / adjacency matrices ────────────────────────────────────────────

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


# ── CIN / CIN++ cell-complex index builder ─────────────────────────────────────
# Same up/down/boundary semantics as lrgb/peptides_struct/data_loader/lrgb_cin.py,
# adapted from a PyG edge_index to an RDKit mol + its SSSR ring list (ring = a
# sequence of atom indices, e.g. from Chem.GetSymmSSSR(mol)).

def _idx_or_empty(src, dst):
    return torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)


def _pairs_within_groups(groups):
    """groups: dict[key -> list[cell_idx]]. Returns (src, dst, shared_key) for
    every ordered pair of distinct cells within the same group."""
    src, dst, shared = [], [], []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        for i in members:
            for j in members:
                if i != j:
                    src.append(i); dst.append(j); shared.append(key)
    return src, dst, shared


def make_cin_indices(mol, sssr):
    """Returns dict with up0/up1/down1/down2/boundary1/boundary2 index (+attr)
    tensors and n_atoms/n_bonds/n_rings, for CIN-style message passing over the
    atom/bond/ring cell complex. All index tensors are always present (possibly
    empty, shape (2, 0))."""
    n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()

    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    edge_to_idx = {}
    for b, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = b
        edge_to_idx[(v, u)] = b

    ring_bonds = []
    for ring in sssr:
        ring = list(ring)
        n = len(ring)
        bonds = []
        for i in range(n):
            b = edge_to_idx.get((ring[i], ring[(i + 1) % n]))
            if b is not None:
                bonds.append(b)
        ring_bonds.append(bonds)
    n_rings = len(ring_bonds)

    # boundary1: atom -> bond (each bond's 2 atom faces)
    b1_face, b1_cell = [], []
    for b, (u, v) in enumerate(edges):
        b1_face.extend([u, v]); b1_cell.extend([b, b])
    boundary1_index = _idx_or_empty(b1_face, b1_cell)

    # boundary2: bond -> ring (each ring's boundary bonds)
    b2_face, b2_cell = [], []
    for r, bonds in enumerate(ring_bonds):
        for b in bonds:
            b2_face.append(b); b2_cell.append(r)
    boundary2_index = _idx_or_empty(b2_face, b2_cell)

    # up0: atoms sharing a bond (coboundary = the bond itself)
    up0_src, up0_dst, up0_attr = [], [], []
    for b, (u, v) in enumerate(edges):
        up0_src.extend([u, v]); up0_dst.extend([v, u]); up0_attr.extend([b, b])
    up0_index = _idx_or_empty(up0_src, up0_dst)
    up0_attr_idx = torch.tensor(up0_attr, dtype=torch.long)

    # up1: bonds sharing a ring (coboundary = the ring)
    ring_groups = {r: bonds for r, bonds in enumerate(ring_bonds)}
    up1_src, up1_dst, up1_attr = _pairs_within_groups(ring_groups)
    up1_index = _idx_or_empty(up1_src, up1_dst)
    up1_attr_idx = torch.tensor(up1_attr, dtype=torch.long)

    # down1: bonds sharing an atom (boundary = the atom), for CIN++
    atom_groups = {}
    for b, (u, v) in enumerate(edges):
        atom_groups.setdefault(u, []).append(b)
        atom_groups.setdefault(v, []).append(b)
    down1_src, down1_dst, down1_attr = _pairs_within_groups(atom_groups)
    down1_index = _idx_or_empty(down1_src, down1_dst)
    down1_attr_idx = torch.tensor(down1_attr, dtype=torch.long)

    # down2: rings sharing a bond (boundary = the bond), for CIN++
    bond_groups = {}
    for r, bonds in enumerate(ring_bonds):
        for b in bonds:
            bond_groups.setdefault(b, []).append(r)
    down2_src, down2_dst, down2_attr = _pairs_within_groups(bond_groups)
    down2_index = _idx_or_empty(down2_src, down2_dst)
    down2_attr_idx = torch.tensor(down2_attr, dtype=torch.long)

    return dict(
        n_atoms=n_atoms, n_bonds=n_bonds, n_rings=n_rings,
        boundary1_index=boundary1_index, boundary2_index=boundary2_index,
        up0_index=up0_index, up0_attr_idx=up0_attr_idx,
        up1_index=up1_index, up1_attr_idx=up1_attr_idx,
        down1_index=down1_index, down1_attr_idx=down1_attr_idx,
        down2_index=down2_index, down2_attr_idx=down2_attr_idx,
    )


# ── per-molecule processing ───────────────────────────────────────────────────

def _process_mol(mol, properties_df, global_idx, use_pe, pe_k, feat_mode):
    sssr = list(Chem.GetSymmSSSR(mol))
    if len(sssr) == 0:
        return None
    icd01, icd02, icd12, adj00, adj11, adj22 = make_matrices(mol, sssr)

    if feat_mode == "full":
        x_0 = make_atom_features(mol)
        x_1 = make_bond_features(mol, sssr)
        x_2 = make_ring_features(mol, sssr)
    elif feat_mode == "simple":
        x_0 = make_simple_atom_features(mol)
        x_1 = make_simple_bond_features(mol)
        x_2 = make_simple_ring_features(sssr)
    else:  # coords
        x_0 = make_coord_atom_features(mol)
        x_1 = make_coord_bond_features(mol)
        x_2 = make_coord_ring_features(mol, sssr)

    if use_pe:
        n_atoms, n_bonds, n_rings = x_0.shape[0], x_1.shape[0], x_2.shape[0]
        pe = CC_RWBSPe(pe_k, n_atoms, n_bonds, n_rings, icd01, icd02, icd12, 'cpu')
        x_0 = torch.cat([x_0, pe[:n_atoms]], dim=1)
        x_1 = torch.cat([x_1, pe[n_atoms:(n_atoms + n_bonds)]], dim=1)
        x_2 = torch.cat([x_2, pe[(n_atoms + n_bonds):]], dim=1)

    y = torch.tensor([properties_df.iloc[global_idx].homolumogap], dtype=torch.float32)
    return (x_0, x_1, x_2, icd01, icd02, icd12, adj00, adj11, adj22, y)


def _load_from_indices(indices, root, use_pe, pe_k, feat_mode):
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
            item = _process_mol(mol, properties_df, g_idx, use_pe, pe_k, feat_mode)
            if item is not None:
                data.append(item)
                loaded_indices.append(g_idx)
    return data, loaded_indices


class Mol3dCT(Dataset):
    """Load CT data for an explicit list of global molecule indices."""
    def __init__(self, indices, root=".", use_pe=True, pe_k=5, feat_mode="full"):
        self.data, self.indices = _load_from_indices(indices, root, use_pe, pe_k, feat_mode)
        self.rk0_dim = self.data[0][0].shape[1]
        self.rk1_dim = self.data[0][1].shape[1]
        self.rk2_dim = self.data[0][2].shape[1]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


class Mol3dCTRand(Dataset):
    """Sample random molecules and load CT data (kept for backward compatibility)."""
    def __init__(self, root=".", size=10000, use_pe=True, pe_k=5, seed=42,
                 per_file_size=None, feat_mode="full"):
        root = Path(root)
        rng = random.Random(seed)

        split_path = root / "random_split_inds.json"
        split = json.load(open(split_path))
        all_valid = split["train"] + split["valid"] + split["test"]

        if per_file_size is not None:
            all_valid_sorted = sorted(all_valid)
            sampled = []
            boundaries = [g_start for _, g_start, _ in _FILES] + [_FILES[-1][2]]
            for i, (fname, _, _) in enumerate(_FILES):
                lo = bisect.bisect_left(all_valid_sorted, boundaries[i])
                hi = bisect.bisect_left(all_valid_sorted, boundaries[i + 1])
                file_valid = all_valid_sorted[lo:hi]
                sampled.extend(rng.sample(file_valid, min(per_file_size, len(file_valid))))
            sampled = sorted(sampled)
        else:
            sampled = sorted(rng.sample(all_valid, min(size, len(all_valid))))

        self.data, self.indices = _load_from_indices(sampled, str(root), use_pe, pe_k, feat_mode)
        self.rk0_dim = self.data[0][0].shape[1]
        self.rk1_dim = self.data[0][1].shape[1]
        self.rk2_dim = self.data[0][2].shape[1]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
