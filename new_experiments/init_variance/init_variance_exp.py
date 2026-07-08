"""
Initialization-variance experiment.

Trains each model 100 times (100 different seeds), all runs sharing the exact
same train/test data (noisy platonic solids at the highest noise level used
in the original sweep: m=150 extra vertices, eps=0.3) and the exact same
frozen best hyperparameters (hard-coded in BEST_HPS below, taken from the
top row of "results 2/<MODEL>_*/top5_params.csv").

Only the model's weight initialization varies across runs (DataLoader
shuffling is disabled so batch order is identical across seeds too -- the
only source of run-to-run variance is the random weight init).

No restart-on-stuck: every seed is run to completion once, even if the loss
gets stuck. The "stuck" flag is recorded, not resampled.

Model/dataset code below is a standalone copy of experiment.py (not an
import), so this script has no dependency on it.
"""

import argparse
import json
import os
import time

import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, global_add_pool, global_mean_pool
from typing import Literal


################# Making of dataset #################

def scale_to_volume(mesh, target_volume):
    scale = (target_volume / mesh.volume) ** (1 / 3)
    mesh.apply_scale(scale)
    return mesh


def subdivide_face(vertices, faces, face_index, eps_factor=0.1):
    face = faces[face_index]        # [i, j, k]
    A, B, C = vertices[face]

    r1, r2 = np.random.random(2)
    if r1 + r2 > 1:                 # mirror back into the lower triangle
        r1, r2 = 1 - r1, 1 - r2
    P = A + r1 * (B - A) + r2 * (C - A)

    # moving the point a little bit
    normal_offset = np.random.randn(1) * eps_factor
    normal = np.cross(B - A, C - A)
    normal /= np.linalg.norm(normal)
    P += normal * normal_offset

    new_idx = len(vertices)
    vertices = np.vstack([vertices, P[np.newaxis, :]])

    i, j, k = face
    new_faces = np.array([
        [i, j, new_idx],
        [j, k, new_idx],
        [k, i, new_idx],
    ], dtype=faces.dtype)

    faces = np.vstack([
        faces[:face_index],
        new_faces,
        faces[face_index + 1:]
    ])

    return vertices, faces


def noisy_tetrahedron(target_volume, num_of_extra_vertices, eps_factor):
    V_tet = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1]
    ], dtype=float)

    F_tet = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ], dtype=np.int64)

    for _ in range(num_of_extra_vertices):
        face_idx = np.random.choice(range(len(F_tet)))
        V_tet, F_tet = subdivide_face(V_tet, F_tet, face_idx, eps_factor)

    tetra = scale_to_volume(trimesh.Trimesh(V_tet, F_tet), target_volume)
    return tetra


def noisy_cube(target_volume, num_of_extra_vertices, eps_factor):
    V_cube = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], dtype=float)

    F_cube = np.array([
        [0, 3, 2], [0, 2, 1],  # -Z
        [4, 5, 6], [4, 6, 7],  # +Z
        [0, 1, 5], [0, 5, 4],  # -Y
        [2, 3, 7], [2, 7, 6],  # +Y
        [0, 4, 7], [0, 7, 3],  # -X
        [1, 2, 6], [1, 6, 5],  # +X
    ], dtype=np.int64)

    for _ in range(num_of_extra_vertices):
        face_idx = np.random.choice(range(len(F_cube)))
        V_cube, F_cube = subdivide_face(V_cube, F_cube, face_idx, eps_factor)

    cube = scale_to_volume(trimesh.Trimesh(V_cube, F_cube), target_volume)
    return cube


def noisy_octahedron(target_volume, num_of_extra_vertices, eps_factor):
    V_oct = np.array([
        [ 1,  0,  0],  # 0  +X
        [-1,  0,  0],  # 1  -X
        [ 0,  1,  0],  # 2  +Y
        [ 0, -1,  0],  # 3  -Y
        [ 0,  0,  1],  # 4  +Z
        [ 0,  0, -1],  # 5  -Z
    ], dtype=np.float64)

    F_oct = np.array([
        [0, 2, 4],  # +X +Y +Z
        [0, 4, 3],  # +X -Y +Z
        [0, 3, 5],  # +X -Y -Z
        [0, 5, 2],  # +X +Y -Z
        [1, 4, 2],  # -X +Y +Z
        [1, 3, 4],  # -X -Y +Z
        [1, 5, 3],  # -X -Y -Z
        [1, 2, 5],  # -X +Y -Z
    ], dtype=np.int64)

    for _ in range(num_of_extra_vertices):
        face_idx = np.random.choice(range(len(F_oct)))
        V_oct, F_oct = subdivide_face(V_oct, F_oct, face_idx, eps_factor)

    octa = scale_to_volume(trimesh.Trimesh(V_oct, F_oct), target_volume)
    return octa


def noisy_icosahedron(target_volume, num_of_extra_vertices, eps_factor):
    phi = (1 + np.sqrt(5)) / 2

    V_ico = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ], dtype=float)

    F_ico = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ])

    for _ in range(num_of_extra_vertices):
        face_idx = np.random.choice(range(len(F_ico)))
        V_ico, F_ico = subdivide_face(V_ico, F_ico, face_idx, eps_factor)

    icosa = scale_to_volume(trimesh.Trimesh(V_ico, F_ico), target_volume)
    return icosa


def noisy_dodecahedron(target_volume, num_of_extra_vertices, eps_factor):
    phi = (1 + np.sqrt(5)) / 2  # golden ratio ~ 1.618

    # 20 vertices of a regular dodecahedron
    V_dod = np.array([
        # Cube corners (+-1, +-1, +-1)
        [ 1,  1,  1],   # 0
        [ 1,  1, -1],   # 1
        [ 1, -1,  1],   # 2
        [ 1, -1, -1],   # 3
        [-1,  1,  1],   # 4
        [-1,  1, -1],   # 5
        [-1, -1,  1],   # 6
        [-1, -1, -1],   # 7
        # (0, +-1/phi, +-phi)
        [ 0,  1/phi,  phi],  # 8
        [ 0,  1/phi, -phi],  # 9
        [ 0, -1/phi,  phi],  # 10
        [ 0, -1/phi, -phi],  # 11
        # (+-1/phi, +-phi, 0)
        [ 1/phi,  phi, 0],   # 12
        [ 1/phi, -phi, 0],   # 13
        [-1/phi,  phi, 0],   # 14
        [-1/phi, -phi, 0],   # 15
        # (+-phi, 0, +-1/phi)
        [ phi, 0,  1/phi],   # 16
        [ phi, 0, -1/phi],   # 17
        [-phi, 0,  1/phi],   # 18
        [-phi, 0, -1/phi],   # 19
    ], dtype=np.float64)

    pent_faces = np.array([
        [13, 15,  7, 11,  3],
        [13,  3, 17, 16,  2],
        [ 6, 18, 19,  7, 15],
        [ 1,  9,  5, 14, 12],
        [12, 14,  4,  8,  0],
        [ 2, 10,  6, 15, 13],
        [ 0, 16, 17,  1, 12],
        [18,  6, 10,  8,  4],
        [ 9, 11,  7, 19,  5],
        [14,  5, 19, 18,  4],
        [ 0,  8, 10,  2, 16],
        [17,  3, 11,  9,  1],
    ], dtype=np.int64)

    faces = []
    for pent in pent_faces:
        # fan triangulation
        faces.append([pent[0], pent[1], pent[2]])
        faces.append([pent[0], pent[2], pent[3]])
        faces.append([pent[0], pent[3], pent[4]])

    F_dod = np.array(faces)

    for _ in range(num_of_extra_vertices):
        face_idx = np.random.choice(range(len(F_dod)))
        V_dod, F_dod = subdivide_face(V_dod, F_dod, face_idx, eps_factor)

    dodeca = scale_to_volume(trimesh.Trimesh(V_dod, F_dod, process=True), target_volume)
    return dodeca


def make_matrices(mesh):
    V = torch.from_numpy(mesh.vertices.copy()).float()
    F = torch.from_numpy(mesh.faces.copy()).long()
    E = torch.from_numpy(mesh.edges_unique.copy()).long()

    n_vertices = V.shape[0]
    n_faces    = F.shape[0]
    n_edges    = E.shape[0]

    # V x F
    rows = F.reshape(-1)
    cols = torch.arange(n_faces).repeat_interleave(3)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    VF = torch.sparse_coo_tensor(indices, values, size=(n_vertices, n_faces))

    # V x E
    rows = E.reshape(-1)
    cols = torch.arange(n_edges).repeat_interleave(2)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    VE = torch.sparse_coo_tensor(indices, values, size=(n_vertices, n_edges))

    # E x F
    E_np = mesh.edges_unique
    F_np = mesh.faces

    face_edges_np = np.stack([
        np.sort(F_np[:, [0, 1]], axis=1),
        np.sort(F_np[:, [1, 2]], axis=1),
        np.sort(F_np[:, [2, 0]], axis=1)
    ], axis=1).reshape(-1, 2)

    edges_sorted_np = np.sort(E_np, axis=1)
    edge_index_map = {tuple(e): i for i, e in enumerate(edges_sorted_np)}
    edge_indices_np = np.array(
        [edge_index_map[tuple(e)] for e in face_edges_np], dtype=np.int64
    )
    face_indices_np = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    edge_indices = torch.from_numpy(edge_indices_np)
    face_indices = torch.from_numpy(face_indices_np)

    indices = torch.stack([edge_indices, face_indices], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    EF = torch.sparse_coo_tensor(indices, values, size=(n_edges, n_faces))

    indices = torch.stack([
        torch.arange(n_faces, dtype=torch.long),
        torch.zeros(n_faces, dtype=torch.long)
    ], dim=0)

    values = torch.ones(n_faces, dtype=torch.float32)

    FC = torch.sparse_coo_tensor(indices, values, size=(n_faces, 1))

    return V, VF.coalesce(), VE.coalesce(), EF.coalesce(), FC.coalesce()


SOLID_TYPES = ["tetra", "cube", "octa", "dodeca", "icosa"]  # order fixed
name_to_label = {name: i for i, name in enumerate(SOLID_TYPES)}

create_platonic_functions = {
    "tetra": noisy_tetrahedron, "cube": noisy_cube, "octa": noisy_octahedron,
    "dodeca": noisy_dodecahedron, "icosa": noisy_icosahedron,
}


def make_noisy_platonic(name, num_of_extra_vertices, eps_factor):
    return create_platonic_functions[name](1.0, num_of_extra_vertices, eps_factor)


class NoisyPlatonicSolids(Dataset):
    def __init__(self, counts_per_type, max_num_of_extra_vertices, eps_factor, save_path=None):
        self.samples = []
        for name, count in counts_per_type.items():
            label = name_to_label[name]
            for i in range(count):
                mesh = make_noisy_platonic(name, np.random.randint(10, max_num_of_extra_vertices), eps_factor)
                V, VF, VE, EF, FC = make_matrices(mesh)
                self.samples.append({
                    "V": V, "VF": VF, "VE": VE, "EF": EF, "FC": FC,
                    "label": torch.tensor(label, dtype=torch.long).view(1)
                })

        if save_path is not None:
            torch.save(self.samples, save_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["V"], s["VF"], s["VE"], s["EF"], s["FC"], s["label"]


def sparse_block_diag(sparse_list):
    device = sparse_list[0].device
    dtype = sparse_list[0].dtype

    rows, cols, vals = [], [], []
    row_offset = 0
    col_offset = 0

    for S in sparse_list:
        S = S.coalesce()
        i = S.indices()
        v = S.values()
        n_rows, n_cols = S.shape

        rows.append(i[0] + row_offset)
        cols.append(i[1] + col_offset)
        vals.append(v)

        row_offset += n_rows
        col_offset += n_cols

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    vals = torch.cat(vals)

    indices = torch.stack([rows, cols])
    shape = (row_offset, col_offset)

    return torch.sparse_coo_tensor(indices, vals, size=shape, device=device, dtype=dtype)


def batch_vector(V_list):
    batch_vector_list = []
    for i, v in enumerate(V_list):
        batch_vector_list.append(torch.full((v.shape[0],), i, dtype=torch.int64))
    return torch.cat(batch_vector_list, dim=0)


def platonic_collate(batch):
    V_list, VF_list, VE_list, EF_list, FC_list, label_list = zip(*batch)
    V_stacked = torch.cat(V_list, dim=0)
    b_vec = batch_vector(V_list)

    return (V_stacked, sparse_block_diag(VF_list), sparse_block_diag(VE_list),
            sparse_block_diag(EF_list), sparse_block_diag(FC_list),
            torch.cat(label_list, dim=0), b_vec)


def make_matrices_for_graph(mesh):
    V = torch.from_numpy(mesh.vertices)
    E = torch.from_numpy(mesh.edges_unique).long()

    edge_index = torch.cat([E.t(), E.flip(1).t()], dim=1)
    return V, edge_index


def mesh_to_data(name, num_of_extra_vertices, eps_factor):
    mesh = make_noisy_platonic(name, num_of_extra_vertices, eps_factor)
    V, edge_index = make_matrices_for_graph(mesh)

    x = V.float()
    y = torch.tensor([name_to_label[name]], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=torch.tensor([y]))


def build_dataset(num_per_type, num_of_extra_vertices, eps_factor):
    data_list = []
    for name, count in num_per_type.items():
        for _ in range(count):
            data = mesh_to_data(name, np.random.randint(10, num_of_extra_vertices), eps_factor)
            data_list.append(data)
    return data_list


################# TNN #################

class CCConvLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, aggr_norm: bool = False, att: bool = False,
        initialization: Literal["xavier_uniform"] = "xavier_uniform",
        initialzation_gain: float = 1.414,
        update_func: Literal["sigmoid", "relu", None] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggr_norm = aggr_norm
        self.att = att
        self.initialization = initialization
        self.initialization_gain = initialzation_gain
        self.update_func = update_func

        self.init_parameters()

    def init_parameters(self):
        self.W = nn.Parameter(torch.empty(self.out_channels, self.in_channels, dtype=torch.float32))
        nn.init.xavier_uniform_(self.W)

    def update(self, x):
        match self.update_func:
            case "sigmoid":
                return torch.sigmoid(x)
            case "relu":
                return torch.nn.functional.relu(x)
            case _:
                return x

    def forward(self, x, neighborhood):
        x_1 = x @ self.W.T
        x_2 = neighborhood @ x_1
        x_3 = self.update(x_2)
        return x_3


class CCAttLayer(nn.Module):
    def __init__(self, in_channels, out_channels, update_func: Literal["sigmoid", "relu", "leakyrelu", None] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_func = update_func

        self.Ws = nn.Parameter(torch.empty(out_channels, in_channels, dtype=torch.float32))
        self.Wt = nn.Parameter(torch.empty(in_channels, out_channels, dtype=torch.float32))
        self.a_s = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
        self.a_t = nn.Parameter(torch.empty(in_channels, dtype=torch.float32))

        nn.init.xavier_uniform_(self.Ws)
        nn.init.xavier_uniform_(self.Wt)
        nn.init.normal_(self.a_s, std=0.1)
        nn.init.normal_(self.a_t, std=0.1)

    def _activate(self, x):
        match self.update_func:
            case "sigmoid":   return torch.sigmoid(x)
            case "relu":      return F.relu(x)
            case "leakyrelu": return F.leaky_relu(x)
            case _:           return x

    def forward(self, x, neighborhood, x_target):
        neighborhood = neighborhood.coalesce()
        Z_s = x        @ self.Ws.T
        Z_t = x_target @ self.Wt.T

        rows, cols = neighborhood.indices()
        e = (Z_s[cols] @ self.a_s) + (Z_t[rows] @ self.a_t)
        e = F.leaky_relu(e, negative_slope=0.2)

        e_max = torch.zeros(x_target.shape[0], device=x.device)
        e_max.scatter_reduce_(0, rows, e, reduce='amax', include_self=True)
        e_exp = torch.exp(e - e_max[rows])
        e_sum = torch.zeros(x_target.shape[0], device=x.device)
        e_sum.scatter_add_(0, rows, e_exp)
        att = e_exp / (e_sum[rows] + 1e-8)

        weighted = att.unsqueeze(1) * Z_s[cols]
        out = torch.zeros(x_target.shape[0], Z_s.shape[1], device=x.device)
        out.scatter_add_(0, rows.unsqueeze(1).expand_as(weighted), weighted)
        return self._activate(out)


class TNN_Att(nn.Module):
    def __init__(self, node_channels, channels_rk1, channels_rk2, channels_rk3, size_hidden_layer, lr, epochs):
        super().__init__()

        self.lr = lr
        self.epochs = epochs

        self.conv_0_to_1 = CCConvLayer(in_channels=node_channels, out_channels=channels_rk1, update_func="relu")
        self.conv_1_to_2 = CCConvLayer(in_channels=channels_rk1, out_channels=channels_rk2, update_func="relu")
        self.conv_2_to_3 = CCConvLayer(in_channels=channels_rk2, out_channels=channels_rk3, update_func="relu")

        self.att_0_to_1 = CCAttLayer(node_channels, channels_rk1, update_func="relu")
        self.att_1_to_2 = CCAttLayer(channels_rk1, channels_rk2, update_func="relu")
        self.att_2_to_3 = CCAttLayer(channels_rk2, channels_rk3, update_func="relu")

        self.fc1 = nn.Linear(channels_rk3, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x_0, incidence_0_1, incidence_1_2, incidence_2_3):
        x_0 = x_0.to(torch.float32)
        x_1_out = self.conv_0_to_1(x_0, incidence_0_1.T)
        x_2_out = self.conv_1_to_2(x_1_out, incidence_1_2.T)
        x_3_out = self.conv_2_to_3(x_2_out, incidence_2_3.T)

        x_1_new = self.att_0_to_1(x_0, incidence_0_1.T, x_1_out)
        x_2_new = self.att_1_to_2(x_1_new, incidence_1_2.T, x_2_out)
        x_3_new = self.att_2_to_3(x_2_new, incidence_2_3.T, x_3_out)
        x = torch.flatten(x_3_new, start_dim=1)
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self, train_loader, device="cpu"):
        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.train()
        start = time.perf_counter()
        stuck_count = 0
        stuck = False
        for epoch in range(self.epochs):
            total_loss = 0
            for V, VF, VE, EF, FC, label, _ in train_loader:
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = self(V, VE, EF, FC)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch: {epoch+1}, Loss: {avg_loss}")
            if np.isnan(avg_loss):
                print(f"Early stopping: NaN loss at epoch {epoch+1}")
                stuck = True
                break
            if abs(avg_loss - STUCK_LOSS) < STUCK_EPS:
                stuck_count += 1
                if stuck_count >= STUCK_PATIENCE:
                    print(f"Loss stuck at {avg_loss:.4f} (not restarting)")
                    stuck = True
            else:
                stuck_count = 0
        end = time.perf_counter()
        print(f"Training time: {end-start} seconds")

        return losses, end - start, (end - start) / len(losses), stuck

    def test(self, test_loader, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for V, VF, VE, EF, FC, label, _ in test_loader:
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                output = self(V, VE, EF, FC)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                total += label.size(0)
            accuracy = 100. * correct / total
            print(f"Test accuracy: {accuracy:.2f}%")

        return accuracy


class TNN(nn.Module):
    def __init__(self, node_channels, channels_rk1, channels_rk2, channels_rk3, size_hidden_layer, lr, epochs):
        super().__init__()

        self.lr = lr
        self.epochs = epochs

        self.conv_0_to_1 = CCConvLayer(in_channels=node_channels, out_channels=channels_rk1, update_func="relu")
        self.conv_1_to_2 = CCConvLayer(in_channels=channels_rk1, out_channels=channels_rk2, update_func="relu")
        self.conv_2_to_3 = CCConvLayer(in_channels=channels_rk2, out_channels=channels_rk3, update_func="relu")
        self.fc1 = nn.Linear(channels_rk3, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x_0, incidence_0_1, incidence_1_2, incidence_2_3):
        x_0 = x_0.to(torch.float32)
        x_1_out = self.conv_0_to_1(x_0, incidence_0_1.T)
        x_2_out = self.conv_1_to_2(x_1_out, incidence_1_2.T)
        x_3_out = self.conv_2_to_3(x_2_out, incidence_2_3.T)
        x = torch.flatten(x_3_out, start_dim=1)
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def fit(self, train_loader, device="cpu"):
        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.train()
        start = time.perf_counter()
        stuck_count = 0
        stuck = False
        for epoch in range(self.epochs):
            total_loss = 0
            for V, VF, VE, EF, FC, label, _ in train_loader:
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = self(V, VE, EF, FC)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch: {epoch+1}, Loss: {avg_loss}")
            if np.isnan(avg_loss):
                print(f"Early stopping: NaN loss at epoch {epoch+1}")
                stuck = True
                break
            if abs(avg_loss - STUCK_LOSS) < STUCK_EPS:
                stuck_count += 1
                if stuck_count >= STUCK_PATIENCE:
                    print(f"Loss stuck at {avg_loss:.4f} (not restarting)")
                    stuck = True
            else:
                stuck_count = 0
        end = time.perf_counter()
        print(f"Training time: {end-start} seconds")

        return losses, end - start, (end - start) / len(losses), stuck

    def test(self, test_loader, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for V, VF, VE, EF, FC, label, _ in test_loader:
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                output = self(V, VE, EF, FC)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                total += label.size(0)
            accuracy = 100. * correct / total
            print(f"Test accuracy: {accuracy:.2f}%")

        return accuracy


################# GCN #################

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, out_channels, lr, epochs):
        super().__init__()

        self.lr = lr
        self.epochs = epochs

        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.fc1 = torch.nn.Linear(hidden_channels2, hidden_channels3)
        self.fc2 = torch.nn.Linear(hidden_channels3, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

    def fit(self, train_loader, device="cpu"):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.train()
        start = time.perf_counter()
        stuck_count = 0
        stuck = False
        for epoch in range(self.epochs):
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = self(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch: {epoch+1}, Loss: {avg_loss}")
            if np.isnan(avg_loss):
                print(f"Early stopping: NaN loss at epoch {epoch+1}")
                stuck = True
                break
            if abs(avg_loss - STUCK_LOSS) < STUCK_EPS:
                stuck_count += 1
                if stuck_count >= STUCK_PATIENCE:
                    print(f"Loss stuck at {avg_loss:.4f} (not restarting)")
                    stuck = True
            else:
                stuck_count = 0
        end = time.perf_counter()
        print(f"Training time: {end-start} seconds")

        return losses, end - start, (end - start) / len(losses), stuck

    def test(self, test_loader, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                data = data.to(device)
                output = self(data.x, data.edge_index, data.batch)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(data.y.view_as(pred)).sum().item()
                total += data.y.size(0)
            accuracy = 100. * correct / total
            print(f"Test accuracy: {accuracy:.2f}%")

        return accuracy


################# GAN #################

class GAN(nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, out_channels, lr, epochs, heads1=4, heads2=4):
        super().__init__()

        self.lr = lr
        self.epochs = epochs

        self.conv1 = GATv2Conv(in_channels, hidden_channels1, heads=heads1, concat=True)
        self.conv2 = GATv2Conv(hidden_channels1 * heads1, hidden_channels2, heads=heads2, concat=True)

        self.fc1 = torch.nn.Linear(hidden_channels2 * heads2, hidden_channels3)
        self.fc2 = torch.nn.Linear(hidden_channels3, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

    def fit(self, train_loader, device="cpu"):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.train()
        start = time.perf_counter()
        stuck_count = 0
        stuck = False
        for epoch in range(self.epochs):
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = self(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch: {epoch+1}, Loss: {avg_loss}")
            if np.isnan(avg_loss):
                print(f"Early stopping: NaN loss at epoch {epoch+1}")
                stuck = True
                break
            if abs(avg_loss - STUCK_LOSS) < STUCK_EPS:
                stuck_count += 1
                if stuck_count >= STUCK_PATIENCE:
                    print(f"Loss stuck at {avg_loss:.4f} (not restarting)")
                    stuck = True
            else:
                stuck_count = 0
        end = time.perf_counter()
        print(f"Training time: {end-start} seconds")

        return losses, end - start, (end - start) / len(losses), stuck

    def test(self, test_loader, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                data = data.to(device)
                output = self(data.x, data.edge_index, data.batch)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(data.y.view_as(pred)).sum().item()
                total += data.y.size(0)
            accuracy = 100. * correct / total
            print(f"Test accuracy: {accuracy:.2f}%")

        return accuracy


################# GIN #################

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, hidden_channels3, out_channels, lr, epochs):
        super().__init__()

        self.lr = lr
        self.epochs = epochs

        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels1),
            nn.ReLU(),
            nn.Linear(hidden_channels1, hidden_channels1),
        )
        self.conv1 = GINConv(nn1, train_eps=True)

        nn2 = nn.Sequential(
            nn.Linear(hidden_channels1, hidden_channels2),
            nn.ReLU(),
            nn.Linear(hidden_channels2, hidden_channels2),
        )
        self.conv2 = GINConv(nn2, train_eps=True)

        self.fc1 = nn.Linear(hidden_channels2, hidden_channels3)
        self.fc2 = nn.Linear(hidden_channels3, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_add_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def fit(self, train_loader, device="cpu"):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.train()
        start = time.perf_counter()
        stuck_count = 0
        stuck = False
        for epoch in range(self.epochs):
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = self(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch: {epoch+1}, Loss: {avg_loss}")
            if np.isnan(avg_loss):
                print(f"Early stopping: NaN loss at epoch {epoch+1}")
                stuck = True
                break
            if abs(avg_loss - STUCK_LOSS) < STUCK_EPS:
                stuck_count += 1
                if stuck_count >= STUCK_PATIENCE:
                    print(f"Loss stuck at {avg_loss:.4f} (not restarting)")
                    stuck = True
            else:
                stuck_count = 0
        end = time.perf_counter()
        print(f"Training time: {end-start} seconds")

        return losses, end - start, (end - start) / len(losses), stuck

    def test(self, test_loader, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                data = data.to(device)
                output = self(data.x, data.edge_index, data.batch)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(data.y.view_as(pred)).sum().item()
                total += data.y.size(0)
            accuracy = 100. * correct / total
            print(f"Test accuracy: {accuracy:.2f}%")

        return accuracy


STUCK_LOSS = np.log(5)  # uniform random loss for 5 classes
STUCK_EPS = 0.005
STUCK_PATIENCE = 3


################# Init-variance experiment #################

MODELS = {"TNN": TNN, "GCN": GCN, "GAN": GAN, "GIN": GIN, "TNN_Att": TNN_Att}
CC_MODELS = {"TNN", "TNN_Att"}

# Highest noise level from the original sweep (mnoev max=150, epsf max=0.3)
DATA_M = 150
DATA_EPS = 0.3
TRAIN_COUNT_PER_CLASS = 500
TEST_COUNT_PER_CLASS = 100

DATA_SEED_TRAIN = 0
DATA_SEED_TEST = 1

# Best hyperparameters per model, hard-coded from the top row of
# "results 2/<MODEL>_*/top5_params.csv" (highest test accuracy from the
# original Optuna sweep). Frozen here, no retuning.
BEST_HPS = {
    "TNN": {  # results 2/TNN_12-05-2026_12-36-32, test acc=98.0
        "node_channels": 3, "channels_rk1": 25, "channels_rk2": 54, "channels_rk3": 111,
        "epochs": 47, "lr": 0.0005405455511401, "size_hidden_layer": 115,
    },
    "TNN_Att": {  # results 2/TNN_Att_12-05-2026_12-40-25, test acc=100.0
        "node_channels": 3, "channels_rk1": 26, "channels_rk2": 43, "channels_rk3": 215,
        "epochs": 28, "lr": 0.0008883738107271, "size_hidden_layer": 89,
    },
    "GCN": {  # results 2/GCN_12-05-2026_12-40-15, test acc=63.8
        "in_channels": 3, "out_channels": 5, "epochs": 42,
        "hidden_channels1": 74, "hidden_channels2": 123, "hidden_channels3": 94,
        "lr": 0.0005991916206625,
    },
    "GAN": {  # results 2/GAN_12-05-2026_12-40-23, test acc=100.0
        "in_channels": 3, "out_channels": 5, "epochs": 36, "heads1": 4, "heads2": 3,
        "hidden_channels1": 40, "hidden_channels2": 41, "hidden_channels3": 101,
        "lr": 0.0006245491335884,
    },
    "GIN": {  # results 2/GIN_12-05-2026_12-40-22, test acc=98.6
        "in_channels": 3, "out_channels": 5, "epochs": 41,
        "hidden_channels1": 96, "hidden_channels2": 115, "hidden_channels3": 66,
        "lr": 0.0007796324399785,
    },
}


def load_best_hps(model_name):
    hps = BEST_HPS[model_name]
    print(f"[{model_name}] using hard-coded best HPs: {hps}")
    return hps


def build_fixed_datasets(model_name):
    if model_name in CC_MODELS:
        np.random.seed(DATA_SEED_TRAIN)
        train_dataset = NoisyPlatonicSolids(
            {name: TRAIN_COUNT_PER_CLASS for name in SOLID_TYPES}, DATA_M, DATA_EPS)
        np.random.seed(DATA_SEED_TEST)
        test_dataset = NoisyPlatonicSolids(
            {name: TEST_COUNT_PER_CLASS for name in SOLID_TYPES}, DATA_M, DATA_EPS)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=platonic_collate)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=platonic_collate)
    else:
        np.random.seed(DATA_SEED_TRAIN)
        train_list = build_dataset({name: TRAIN_COUNT_PER_CLASS for name in SOLID_TYPES}, DATA_M, DATA_EPS)
        np.random.seed(DATA_SEED_TEST)
        test_list = build_dataset({name: TEST_COUNT_PER_CLASS for name in SOLID_TYPES}, DATA_M, DATA_EPS)

        train_loader = PyGDataLoader(train_list, batch_size=32, shuffle=False)
        test_loader = PyGDataLoader(test_list, batch_size=32, shuffle=False)

    return train_loader, test_loader


def run_init_variance(model_name, num_seeds, device, out_path):
    hps = load_best_hps(model_name)
    train_loader, test_loader = build_fixed_datasets(model_name)
    model_cls = MODELS[model_name]

    runs = []
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_cls(**hps)
        losses, total_time, time_per_epoch, stuck = model.fit(train_loader, device=device)
        acc = model.test(test_loader, device=device)

        runs.append({
            "seed": seed,
            "test_accuracy": acc,
            "losses": losses,
            "stuck": stuck,
            "total_time": total_time,
            "time_per_epoch": time_per_epoch,
        })

        with open(out_path, "w") as f:
            json.dump({"model": model_name, "hps": hps, "data_params": {
                "m": DATA_M, "eps": DATA_EPS,
                "train_per_class": TRAIN_COUNT_PER_CLASS, "test_per_class": TEST_COUNT_PER_CLASS,
            }, "runs": runs}, f, indent=2)

        print(f"[{model_name}] seed {seed}: acc={acc:.2f}% stuck={stuck}")

    return runs


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()))
    argparser.add_argument("--num_seeds", type=int, default=100)
    argparser.add_argument("--device", type=str, default="cpu")
    argparser.add_argument("--out_dir", type=str, default="init_variance_results")
    args = argparser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model}_init_variance.json")

    run_init_variance(args.model, args.num_seeds, args.device, out_path)
