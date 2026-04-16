from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
from rdkit import Chem
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix

def make_node_features(mol):
    atomic_numbers = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.float32).unsqueeze(-1)
    conf = mol.GetConformer()
    positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    positions = torch.tensor(list(map(lambda a: [a.x,a.y,a.z], positions)), dtype=torch.float32)
    return torch.cat([atomic_numbers, positions], dim=1)

def make_node_features_new(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetHybridization().real,   # sp=2, sp2=3, sp3=4
            atom.GetTotalNumHs(),
        ])
    conf = mol.GetConformer()
    positions = torch.tensor(
        [[conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=torch.float32)
    chem = torch.tensor(feats, dtype=torch.float32)
    return torch.cat([chem, positions], dim=1)  # 10 features total

def make_incidence_0_2(adj, n_nodes):
    G = nx.from_scipy_sparse_array(adj)
    rows = []
    cols = []
    cycles = nx.simple_cycles(G)
    n_faces = 0
    for k,cycle in enumerate(cycles):
        if len(cycle) > 3:
            rows.extend(cycle)
            cols.extend([k for _ in cycle])
            n_faces += 1
    if n_faces != 0:
        indices = torch.tensor([rows,cols], dtype=torch.long)
    else:
        n_faces = 1
        indices = torch.tensor([[i for i in range(n_nodes)], [0 for _ in range(n_nodes)]], dtype=torch.long)
    values = torch.ones(indices.shape[1],dtype=torch.float32)
    icd_matrix = torch.sparse_coo_tensor(indices, values, size=(n_nodes,n_faces)).coalesce()
    return icd_matrix

def make_incidence_1_2(icd01, icd02):
    idx = icd01.indices()
    nodes = idx[0]
    edges = idx[1]

    perm= edges.argsort()
    nodes_sorted = nodes[perm]
    n_nodes, n_edges = icd01.shape
    n_faces = icd02.shape[1]

    edge_nodes = nodes_sorted.view(n_edges,2)
    u = edge_nodes[:,0]
    v = edge_nodes[:,1]

    icd02_bool = icd02.bool().to_dense()
    U = icd02_bool[u,:]
    V = icd02_bool[v,:]

    mask = (U & V)
    idx = mask.nonzero().T
    vals = torch.ones(idx.size(1),dtype=icd02.dtype)
    icd12 = torch.sparse_coo_tensor(idx, vals, (n_edges,n_faces)).coalesce()
    return icd12

def make_incidence_2_3(icd12):
    n_faces = icd12.shape[1]
    row = torch.arange(n_faces)
    col = torch.zeros(n_faces, dtype=torch.long)
    idx = torch.stack([row, col], dim=0)  # [2, n_faces]
    vals = torch.ones(n_faces, dtype=icd12.dtype,)
    return torch.sparse_coo_tensor(idx, vals, size=(n_faces, 1)).coalesce()

def normalize_incidence_col(B):
    """Column normalization: each rank-2 cell's incoming messages sum to 1"""
    B = B.coalesce()
    indices, values = B.indices(), B.values()
    col_deg = torch.zeros(B.shape[1], device=B.device)\
                   .scatter_add(0, indices[1], values.abs()).clamp(min=1)
    return torch.sparse_coo_tensor(
        indices, values / col_deg[indices[1]], B.shape
    ).coalesce()

def make_matrices(mol):
    n_vertices = mol.GetNumAtoms()
    n_edges = mol.GetNumBonds()
    icd_rows = []
    icd_cols = []
    adj_rows = []
    adj_cols = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        k = bond.GetIdx()
        icd_rows.extend([a,b])
        icd_cols.extend([k,k])
        adj_rows.extend([a,b])
        adj_cols.extend([b,a])

    icd_indices = torch.tensor([icd_rows, icd_cols], dtype=torch.long)
    icd_values = torch.ones(icd_indices.shape[1],dtype=torch.float32)
    icd01_matrix = torch.sparse_coo_tensor(icd_indices, icd_values, size=(n_vertices,n_edges)).coalesce()
    adj_matrix = coo_matrix(
        (np.ones(len(adj_cols)),(np.array(adj_rows,dtype=np.int64), np.array(adj_cols,dtype=np.int64))),
        shape=(n_vertices,n_vertices)
    )
    icd02_matrix = make_incidence_0_2(adj_matrix, n_vertices)
    if icd02_matrix._nnz() == 0:
        raise ValueError("No rank 2 cells")
    else:
        icd02_matrix = normalize_incidence_col(icd02_matrix)
    icd12_matrix = make_incidence_1_2(icd01_matrix, icd02_matrix)
    if icd12_matrix._nnz() == 0:
        raise ValueError("No rank 2 cells")
    else:
        icd12_matrix = normalize_incidence_col(icd12_matrix)
    icd23_matrix = normalize_incidence_col(make_incidence_2_3(icd12_matrix))
    return icd01_matrix, icd02_matrix, icd12_matrix, icd23_matrix

class Mol3d_CycleLifting(Dataset):
    def __init__(self, root=".", size=1000000):
        super().__init__()
        root = Path(root)
        sdf_path = root / "combined_mols_1000000_to_2000000.sdf"
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        properties_path = root / "properties.csv"
        properties_df = pd.read_csv(properties_path)
        self.homolumogap = []
        self.node_feature = []
        self.icd01 = []
        self.icd02 = []
        self.icd12 = []
        self.icd23 = []
        for i, mol in enumerate(suppl):
            if size:
                if i >= size:
                    break
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            try:
                icd01, icd02, icd12, icd23 = make_matrices(mol)
            except (ValueError, RuntimeError):  # catch both just in case
                #print("no rank 2 cells")
                continue
            self.icd01.append(icd01)
            self.icd02.append(icd02)
            self.icd12.append(icd12)
            self.icd23.append(icd23)
            self.node_feature.append(make_node_features(mol))
            row = properties_df.iloc[i]
            self.homolumogap.append(torch.tensor(row.homolumogap,dtype=torch.float32).unsqueeze(-1))
    
    def __len__(self):
        return len(self.homolumogap)
    
    def __getitem__(self, index):
        return self.node_feature[index], self.icd01[index],self.icd02[index],self.icd12[index],self.icd23[index], self.homolumogap[index]
    
class Mol3d_CycleLifting_morefeatures(Dataset):
    def __init__(self, root=".", size=1000000):
        super().__init__()
        root = Path(root)
        sdf_path = root / "combined_mols_1000000_to_2000000.sdf"
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        properties_path = root / "properties.csv"
        properties_df = pd.read_csv(properties_path)
        self.homolumogap = []
        self.node_feature = []
        self.icd01 = []
        self.icd02 = []
        self.icd12 = []
        self.icd23 = []
        for i, mol in enumerate(suppl):
            if size:
                if i >= size:
                    break
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            try:
                icd01, icd02, icd12, icd23 = make_matrices(mol)
            except (ValueError, RuntimeError):  # catch both just in case
                #print("no rank 2 cells")
                continue
            self.icd01.append(icd01)
            self.icd02.append(icd02)
            self.icd12.append(icd12)
            self.icd23.append(icd23)
            self.node_feature.append(make_node_features_new(mol))
            row = properties_df.iloc[i]
            self.homolumogap.append(torch.tensor(row.homolumogap,dtype=torch.float32).unsqueeze(-1))
    
    def __len__(self):
        return len(self.homolumogap)
    
    def __getitem__(self, index):
        return self.node_feature[index], self.icd01[index],self.icd02[index],self.icd12[index],self.icd23[index], self.homolumogap[index]

def make_node_features_atomicnum(mol):
    """Atomic number only — shape (N_atoms, 1)."""
    return torch.tensor([[a.GetAtomicNum()] for a in mol.GetAtoms()], dtype=torch.float32)


def make_bond_features_dist(mol):
    """Euclidean distance between bonded atoms — shape (N_bonds, 1).
    Bond ordering matches icd01 (bond.GetIdx())."""
    conf = mol.GetConformer()
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=torch.float32)
    dists = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dists.append([(pos[i] - pos[j]).norm().item()])
    return torch.tensor(dists, dtype=torch.float32)


class Mol3d_CycleLifting_distfeatures(Dataset):
    """Node features: atomic number (1-dim).
    Edge features: bond length / pairwise distance (1-dim).
    __getitem__ returns (node_feat, edge_feat, icd01, icd02, icd12, icd23, homolumogap).
    """
    def __init__(self, root=".", size=1000000):
        super().__init__()
        root = Path(root)
        sdf_path = root / "combined_mols_1000000_to_2000000.sdf"
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        properties_path = root / "properties.csv"
        properties_df = pd.read_csv(properties_path)
        self.homolumogap = []
        self.node_feature = []
        self.edge_feature = []
        self.icd01 = []
        self.icd02 = []
        self.icd12 = []
        self.icd23 = []
        for i, mol in enumerate(suppl):
            if size:
                if i >= size:
                    break
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            try:
                icd01, icd02, icd12, icd23 = make_matrices(mol)
            except (ValueError, RuntimeError):
                continue
            self.icd01.append(icd01)
            self.icd02.append(icd02)
            self.icd12.append(icd12)
            self.icd23.append(icd23)
            self.node_feature.append(make_node_features_atomicnum(mol))
            self.edge_feature.append(make_bond_features_dist(mol))
            row = properties_df.iloc[i]
            self.homolumogap.append(torch.tensor(row.homolumogap, dtype=torch.float32).unsqueeze(-1))

    def __len__(self):
        return len(self.homolumogap)

    def __getitem__(self, index):
        return (self.node_feature[index], self.edge_feature[index],
                self.icd01[index], self.icd02[index],
                self.icd12[index], self.icd23[index],
                self.homolumogap[index])


class Mol3d_KHopLifting(Dataset):
    def __init__(self,root=".", size=1000000, k=3):
        super().__init__()
        root = Path(root)
        sdf_path = root / "combined_mols_1000000_to_2000000.sdf"
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        properties_path = root / "properties.csv"
        properties_df = pd.read_csv(properties_path)
        self.homolumogap = []
        self.node_feature = []
        self.icd01 = []
        self.icd02 = []
        self.icd12 = []
        self.icd23 = []
        for i, mol in enumerate(suppl):
            if size:
                if i >= size:
                    break
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue

            #TODO: Make incidence matrices with k-Hop distance

            self.node_feature.append(make_node_features(mol))
            row = properties_df.iloc[i]
            self.homolumogap.append(torch.tensor(row.homolumogap,dtype=torch.float32).unsqueeze(-1))

def make_edge_index(mol):
    u = []
    v = []
    for bond in mol.GetBonds():
        u.append(bond.GetBeginAtomIdx())
        v.append(bond.GetEndAtomIdx())
    
    edge_index = torch.tensor([u,v])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index

class Mol3d_PyG(Dataset):
    def __init__(self, root=".", size=1000000, lifting=None):
        super().__init__()
        root = Path(root)
        sdf_path = root / "combined_mols_0_to_1000000.sdf"
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        properties_path = root / "properties.csv"
        properties_df = pd.read_csv(properties_path)
        self.homolumogap = []
        self.node_feature = []
        self.edge_index = []
        for i, mol in enumerate(suppl):
            if size:
                if i >= size:
                    break
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                # hier kannst du invalid molecules droppen
                continue
            self.node_feature.append(make_node_features(mol))
            self.edge_index.append(make_edge_index(mol))
            row = properties_df.iloc[i]
            self.homolumogap.append(torch.tensor(row.homolumogap,dtype=torch.float32))
    
    def __len__(self):
        return len(self.homolumogap)
    
    def __getitem__(self, index):
        #return self.node_feature[index], self.edge_index[index], self.homolumogap[index]
        return Data(node_features= self.node_feature[index], edge_index = self.edge_index[index],homolumogap = self.homolumogap[index])
            