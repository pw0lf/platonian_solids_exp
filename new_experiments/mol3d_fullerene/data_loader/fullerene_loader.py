"""
Loads fullerene XYZ files using mol3d-compatible feature functions so the
two datasets can be combined with identical feature dimensions.

Molecule loading order matches fullerene_randomsplit/split.json:
  1. c60 subset  (c20-c60-dft-all.csv)
  2. c70_non_IPR (c70-100-isomers-Eb-Eg-logP.csv)
  3. c72_100_IPR (c62-c720-dft-all.csv)
"""
import glob
import sys
from pathlib import Path

import torch
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

MOL3D_LOADER = Path(__file__).parent.parent.parent / "mol3d" / "data_loader"
sys.path.insert(0, str(MOL3D_LOADER))
from mol3d_ct_rand import (
    make_atom_features, make_bond_features, make_ring_features,
    make_simple_atom_features, make_simple_bond_features, make_simple_ring_features,
    make_coord_atom_features, make_coord_bond_features, make_coord_ring_features,
    make_matrices, make_cin_indices,
)
from pe import CC_RWBSPe

SIZE_DIRS = {
    "c60":         [f"c{n}" for n in [20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]],
    "c70_non_IPR": ["c70"],
    "c72_100_IPR": [f"c{n}" for n in [72, 74, 76, 78, 80, 82, 84, 86, 90, 92, 94, 96, 98, 100]],
}
LABEL_FILES = {
    "c60":         "c20-c60-dft-all.csv",
    "c70_non_IPR": "c70-100-isomers-Eb-Eg-logP.csv",
    "c72_100_IPR": "c62-c720-dft-all.csv",
}
TARGET_COLUMNS = {
    "HOMO": "HOMO(eV)",
    "LUMO": "LUMO(eV)",
    "Gap":  "HOMO-LUMO(eV)",
    "Eb":   "E_binding(eV)",
}
SUBSETS = ["c60", "c70_non_IPR", "c72_100_IPR"]


def _custom_sort(file):
    n1 = int(file.split("/")[-1].split("-")[0].split("c")[-1])
    n2 = int(file.split("/")[-1].split("-")[-1].split("_")[0])
    return (n1, n2)


def _load_mol(xyz_path):
    mol = Chem.MolFromXYZFile(xyz_path)
    conn = Chem.Mol(mol)
    rdDetermineBonds.DetermineBonds(conn, charge=0)
    return conn


def _iter_fullerene_mols(root, target):
    root = Path(root)
    xyz_dir = root / "data" / "optimized_xyz"
    target_col = TARGET_COLUMNS[target]

    for name in SUBSETS:
        files = []
        for d in SIZE_DIRS[name]:
            files.extend(glob.glob(str(xyz_dir / d / "*.xyz")))
        files = sorted(files, key=_custom_sort)
        labels = pd.read_csv(root / "data" / LABEL_FILES[name])
        assert len(files) == len(labels)
        targets = torch.tensor(labels[target_col].values, dtype=torch.float32)
        for f, y in zip(files, targets):
            yield _load_mol(f), y.unsqueeze(0)


# ── SchNet ────────────────────────────────────────────────────────────────────

def load_fullerene_schnet(root, target="Gap"):
    data = []
    for mol, y in _iter_fullerene_mols(root, target):
        conf = mol.GetConformer()
        z   = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
        pos = torch.tensor(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
             for i in range(mol.GetNumAtoms())],
            dtype=torch.float32,
        )
        data.append(Data(z=z, pos=pos, y=y))
    return data


# ── GNN ───────────────────────────────────────────────────────────────────────

def load_fullerene_gnn(root, target="Gap", load_edge_features=False):
    data = []
    for mol, y in _iter_fullerene_mols(root, target):
        x = make_atom_features(mol)

        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        if bonds:
            src = [e[0] for e in bonds] + [e[1] for e in bonds]
            dst = [e[1] for e in bonds] + [e[0] for e in bonds]
        else:
            src, dst = [], []
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        if load_edge_features:
            sssr = list(Chem.GetSymmSSSR(mol))
            ef = make_bond_features(mol, sssr)
            edge_attr = torch.cat([ef, ef], dim=0)
        else:
            edge_attr = None

        data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data


# ── CT ────────────────────────────────────────────────────────────────────────

def _process_fullerene_mol(mol, y, use_pe, pe_k, feat_mode):
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

    return (x_0, x_1, x_2, icd01, icd02, icd12, adj00, adj11, adj22, y)


def load_fullerene_ct(root, target="Gap", use_pe=True, pe_k=5, feat_mode="full"):
    data = []
    for mol, y in _iter_fullerene_mols(root, target):
        item = _process_fullerene_mol(mol, y, use_pe, pe_k, feat_mode)
        if item is not None:
            data.append(item)
    return data


# ── CIN / CIN++ ──────────────────────────────────────────────────────────────

def _process_fullerene_mol_cin(mol, y):
    sssr = list(Chem.GetSymmSSSR(mol))
    x_0 = make_atom_features(mol)
    x_1 = make_bond_features(mol, sssr)
    x_2 = make_ring_features(mol, sssr) if len(sssr) > 0 else torch.zeros(0, 6, dtype=torch.float32)
    cin = make_cin_indices(mol, sssr)
    return dict(x_0=x_0, x_1=x_1, x_2=x_2, y=y, **cin)


def load_fullerene_cin(root, target="Gap"):
    return [_process_fullerene_mol_cin(mol, y) for mol, y in _iter_fullerene_mols(root, target)]
