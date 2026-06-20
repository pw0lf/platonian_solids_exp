"""CT dataloader for LRGB Peptides datasets.

feat_mode options:
  'original' — 9-dim OGB node features, 3-dim OGB edge features, 1-dim ring size.
               No SMILES needed; rings detected via networkx on the graph structure.
  'full'     — 8-dim RDKit chem features (no 3D coords), 8-dim bond, 6-dim ring + PE.
               Requires SMILES CSV (run new_experiments/lrgb/download_smiles.py first).
  'simple'   — 3-dim (mass/EN/vdW), 1-dim bond type, 1-dim ring size. Requires SMILES.
"""

import sys
import torch
import networkx as nx
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.datasets import LRGBDataset
from pathlib import Path


class _LRGBNoDownload(LRGBDataset):
    def download(self):
        pass

sys.path.insert(0, str(Path(__file__).parent))
from pe import CC_RWBSPe

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

_PT = Chem.GetPeriodicTable()

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


# ── RDKit feature functions (full/simple modes) ───────────────────────────────

def _full_atom_features(mol):
    """8-dim: mol3d chem features without xyz coords."""
    feats = []
    for atom in mol.GetAtoms():
        feats.append([
            atom.GetAtomicNum(), atom.GetTotalValence(), atom.GetDegree(),
            atom.GetImplicitValence(), int(atom.GetIsAromatic()),
            int(atom.GetChiralTag()), atom.GetFormalCharge(),
            HYBRIDIZATION.get(atom.GetHybridization(), 0),
        ])
    return torch.tensor(feats, dtype=torch.float32)


def _full_bond_features(mol, sssr):
    bond_min_ring = {}
    for ring in sssr:
        ring = list(ring)
        for i in range(len(ring)):
            b = mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])
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


def _full_ring_features(mol, sssr):
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


def _simple_atom_features(mol):
    feats = []
    for atom in mol.GetAtoms():
        an = atom.GetAtomicNum()
        feats.append([atom.GetMass(), EN.get(an, 2.5), _PT.GetRvdw(an)])
    return torch.tensor(feats, dtype=torch.float32)


def _simple_bond_features(mol):
    feats = [[float(BOND_TYPE.get(b.GetBondType(), 0))] for b in mol.GetBonds()]
    if not feats:
        return torch.zeros(0, 1, dtype=torch.float32)
    return torch.tensor(feats, dtype=torch.float32)


def _simple_ring_features(sssr):
    if len(sssr) == 0:
        return torch.zeros(1, 1, dtype=torch.float32)
    return torch.tensor([[float(len(list(r)))] for r in sssr], dtype=torch.float32)


# ── Matrix building (shared) ───────────────────────────────────────────────────

def _make_matrices_rdkit(mol, sssr):
    """Build CT matrices from RDKit mol object (full/simple modes)."""
    n_atoms, n_bonds, n_rings = mol.GetNumAtoms(), mol.GetNumBonds(), len(sssr)

    icd01_rows, icd01_cols = [], []
    for bond in mol.GetBonds():
        bi = bond.GetIdx()
        icd01_rows.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        icd01_cols.extend([bi, bi])
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
        for ri, ring in enumerate(sssr):
            for atom_idx in ring:
                icd02_rows.append(atom_idx); icd02_cols.append(ri)
        icd02 = torch.sparse_coo_tensor(
            torch.tensor([icd02_rows, icd02_cols], dtype=torch.long),
            torch.ones(len(icd02_rows), dtype=torch.float32),
            size=(n_atoms, n_rings)).coalesce()
        icd12_rows, icd12_cols = [], []
        for ri, ring in enumerate(sssr):
            ring = list(ring)
            for i in range(len(ring)):
                b = mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])
                if b:
                    icd12_rows.append(b.GetIdx()); icd12_cols.append(ri)
        icd12 = torch.sparse_coo_tensor(
            torch.tensor([icd12_rows, icd12_cols], dtype=torch.long),
            torch.ones(len(icd12_rows), dtype=torch.float32),
            size=(n_bonds, n_rings)).coalesce()

    def make_adj(M):
        d = M.to_dense()
        return ((d @ d.T) > 0).float().to_sparse_coo().coalesce()

    return icd01, icd02, icd12, make_adj(icd01), make_adj(icd01.T), make_adj(icd12.T)


def _make_matrices_graph(n_atoms, edges, cycles):
    """Build CT matrices from graph edges + networkx cycles (original mode)."""
    n_bonds = len(edges)
    n_rings = max(len(cycles), 1)

    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

    # icd01: atom-bond
    icd01_rows, icd01_cols = [], []
    for j, (u, v) in enumerate(edges):
        icd01_rows.extend([u, v])
        icd01_cols.extend([j, j])
    if icd01_rows:
        icd01 = torch.sparse_coo_tensor(
            torch.tensor([icd01_rows, icd01_cols], dtype=torch.long),
            torch.ones(len(icd01_rows), dtype=torch.float32),
            size=(n_atoms, n_bonds)).coalesce()
    else:
        icd01 = torch.zeros(n_atoms, 0).to_sparse_coo().coalesce()

    if not cycles:
        icd02 = torch.zeros(n_atoms, 1).to_sparse_coo().coalesce()
        icd12 = torch.zeros(n_bonds, 1).to_sparse_coo().coalesce()
    else:
        icd02_rows, icd02_cols = [], []
        for k, cycle in enumerate(cycles):
            for node in cycle:
                icd02_rows.append(node); icd02_cols.append(k)
        icd02 = torch.sparse_coo_tensor(
            torch.tensor([icd02_rows, icd02_cols], dtype=torch.long),
            torch.ones(len(icd02_rows), dtype=torch.float32),
            size=(n_atoms, n_rings)).coalesce()

        icd12_rows, icd12_cols = [], []
        for k, cycle in enumerate(cycles):
            n = len(cycle)
            for i in range(n):
                u, v = cycle[i], cycle[(i + 1) % n]
                j = edge_to_idx.get((u, v))
                if j is not None:
                    icd12_rows.append(j); icd12_cols.append(k)
        if icd12_rows:
            icd12 = torch.sparse_coo_tensor(
                torch.tensor([icd12_rows, icd12_cols], dtype=torch.long),
                torch.ones(len(icd12_rows), dtype=torch.float32),
                size=(n_bonds, n_rings)).coalesce()
        else:
            icd12 = torch.zeros(n_bonds, n_rings).to_sparse_coo().coalesce()

    def make_adj(M):
        d = M.to_dense()
        return ((d @ d.T) > 0).float().to_sparse_coo().coalesce()

    return icd01, icd02, icd12, make_adj(icd01), make_adj(icd01.T), make_adj(icd12.T)


# ── Per-item processors ────────────────────────────────────────────────────────

def _process_from_graph(pyg_item):
    """Build CT tuple for 'original' mode using graph structure + networkx rings."""
    x_node = pyg_item.x          # (N, 9) int64
    edge_index = pyg_item.edge_index  # (2, 2E) int64
    edge_attr = pyg_item.edge_attr    # (2E, 3) int64
    y = pyg_item.y.squeeze(0).float()

    n_atoms = x_node.shape[0]
    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    src_u, dst_u = src[mask].tolist(), dst[mask].tolist()
    edge_attr_u = edge_attr[mask]

    x_0 = x_node.float()
    x_1 = edge_attr_u.float()

    edges = list(zip(src_u, dst_u))
    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))
    G.add_edges_from(edges)

    try:
        cycles = list(nx.minimum_cycle_basis(G))
    except Exception:
        cycles = []

    x_2 = (torch.tensor([[float(len(c))] for c in cycles], dtype=torch.float32)
            if cycles else torch.zeros(1, 1, dtype=torch.float32))

    icd01, icd02, icd12, adj00, adj11, adj22 = _make_matrices_graph(n_atoms, edges, cycles)
    return (x_0, x_1, x_2, icd01, icd02, icd12, adj00, adj11, adj22, y)


def _process_from_smiles(smiles, y, feat_mode, pe_k):
    """Build CT tuple for 'full'/'simple' modes using RDKit from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    sssr = list(Chem.GetSymmSSSR(mol))
    icd01, icd02, icd12, adj00, adj11, adj22 = _make_matrices_rdkit(mol, sssr)

    if feat_mode == "full":
        x_0 = _full_atom_features(mol)
        x_1 = _full_bond_features(mol, sssr)
        x_2 = _full_ring_features(mol, sssr)
    else:  # simple
        x_0 = _simple_atom_features(mol)
        x_1 = _simple_bond_features(mol)
        x_2 = _simple_ring_features(sssr)

    if feat_mode == "full" and pe_k > 0:
        n_a, n_b, n_r = x_0.shape[0], x_1.shape[0], x_2.shape[0]
        pe = CC_RWBSPe(pe_k, n_a, n_b, n_r, icd01, icd02, icd12, 'cpu')
        x_0 = torch.cat([x_0, pe[:n_a]], dim=1)
        x_1 = torch.cat([x_1, pe[n_a:(n_a + n_b)]], dim=1)
        x_2 = torch.cat([x_2, pe[(n_a + n_b):]], dim=1)

    return (x_0, x_1, x_2, icd01, icd02, icd12, adj00, adj11, adj22, y)


# ── Dataset class ─────────────────────────────────────────────────────────────

class LRGBCTDataset(Dataset):
    def __init__(self, root, name, split, feat_mode="original", pe_k=5, smiles_csv=None):
        """
        root: PyG data root directory
        name: 'Peptides-func' or 'Peptides-struct'
        split: 'train', 'val', 'test'
        feat_mode: 'original' | 'full' | 'simple'
        smiles_csv: path to CSV with 'smiles' column (needed for full/simple modes)
        """
        pyg_ds = _LRGBNoDownload(root=root, name=name, split=split)

        self.data = []
        self.indices = []

        if feat_mode == "original":
            for i, item in enumerate(pyg_ds):
                try:
                    ct = _process_from_graph(item)
                    self.data.append(ct)
                    self.indices.append(i)
                except Exception:
                    pass
        else:
            assert smiles_csv is not None, (
                f"feat_mode='{feat_mode}' requires SMILES. "
                "Run new_experiments/lrgb/download_smiles.py first and pass smiles_csv=..."
            )
            df = pd.read_csv(smiles_csv)
            assert len(df) == len(pyg_ds), (
                f"SMILES CSV has {len(df)} rows but PyG split '{split}' has {len(pyg_ds)} items. "
                "Ensure the CSV was generated for this split."
            )
            smiles_list = df["smiles"].tolist()
            for i, (item, smi) in enumerate(zip(pyg_ds, smiles_list)):
                y = item.y.squeeze(0).float()
                try:
                    ct = _process_from_smiles(smi, y, feat_mode, pe_k)
                    if ct is not None:
                        self.data.append(ct)
                        self.indices.append(i)
                except Exception:
                    pass

        assert len(self.data) > 0, "No molecules successfully loaded."
        self.rk0_dim = self.data[0][0].shape[1]
        self.rk1_dim = self.data[0][1].shape[1]
        self.rk2_dim = self.data[0][2].shape[1]
        print(f"  Loaded {len(self.data)}/{len(pyg_ds)} molecules "
              f"| rk0={self.rk0_dim} rk1={self.rk1_dim} rk2={self.rk2_dim}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
