import trimesh
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu, max_pool2d
from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv
from torch.nn.functional import relu, max_pool2d
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
import json
from itertools import product
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.nn import GINConv

import argparse
import optuna
import os
from datetime import datetime
import time
from typing import Literal

################# Making of dataset #################
def scale_to_volume(mesh, target_volume):
    scale = (target_volume / mesh.volume) ** (1/3)
    mesh.apply_scale(scale)
    return mesh

def subdivide_face(vertices,faces,face_index, eps_factor=0.1):
    face = faces[face_index]        # [i, j, k]
    A, B, C = vertices[face]

    r1, r2 = np.random.random(2)
    if r1 + r2 > 1:                 # mirror back into the lower triangle
        r1, r2 = 1 - r1, 1 - r2
    P = A + r1 * (B - A) + r2 * (C - A)

    # moving the point a little bit
    normal_offset = np.random.randn(1)*eps_factor
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

def noisy_tetrahedron(target_volume,num_of_extra_vertices, eps_factor):
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

def noisy_cube(target_volume,num_of_extra_vertices, eps_factor):
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

def noisy_octahedron(target_volume,num_of_extra_vertices, eps_factor):
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

def noisy_icosahedron(target_volume,num_of_extra_vertices, eps_factor):
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

def noisy_dodecahedron(target_volume,num_of_extra_vertices, eps_factor):
    phi = (1 + np.sqrt(5)) / 2  # golden ratio ≈ 1.618

    # 20 vertices of a regular dodecahedron
    V_dod = np.array([
        # Cube corners (±1, ±1, ±1)
        [ 1,  1,  1],   # 0
        [ 1,  1, -1],   # 1
        [ 1, -1,  1],   # 2
        [ 1, -1, -1],   # 3
        [-1,  1,  1],   # 4
        [-1,  1, -1],   # 5
        [-1, -1,  1],   # 6
        [-1, -1, -1],   # 7
        # (0, ±1/φ, ±φ)
        [ 0,  1/phi,  phi],  # 8
        [ 0,  1/phi, -phi],  # 9
        [ 0, -1/phi,  phi],  # 10
        [ 0, -1/phi, -phi],  # 11
        # (±1/φ, ±φ, 0)
        [ 1/phi,  phi, 0],   # 12
        [ 1/phi, -phi, 0],   # 13
        [-1/phi,  phi, 0],   # 14
        [-1/phi, -phi, 0],   # 15
        # (±φ, 0, ±1/φ)
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
    V = torch.from_numpy(mesh.vertices)        # (n_vertices, 3), float
    F = torch.from_numpy(mesh.faces).long()    # (n_faces, 3), long
    E = torch.from_numpy(mesh.edges_unique).long()  # (n_edges, 2), long

    n_vertices = V.shape[0]
    n_faces    = F.shape[0]
    n_edges    = E.shape[0]

    #V x F
    rows = F.reshape(-1)                                      # (3 * n_faces,)
    cols = torch.arange(n_faces).repeat_interleave(3)         # (3 * n_faces,)
    indices = torch.stack([rows, cols], dim=0)                # (2, nnz)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    VF = torch.sparse_coo_tensor(indices, values,
                                 size=(n_vertices, n_faces))
    # V x E
    rows = E.reshape(-1)                                      # (2 * n_edges,)
    cols = torch.arange(n_edges).repeat_interleave(2)         # (2 * n_edges,)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    VE = torch.sparse_coo_tensor(indices, values,
                             size=(n_vertices, n_edges))

    # E x F
    E_np = mesh.edges_unique
    F_np = mesh.faces

    # all face edges, sorted by vertex index
    face_edges_np = np.stack([
        np.sort(F_np[:, [0, 1]], axis=1),
        np.sort(F_np[:, [1, 2]], axis=1),
        np.sort(F_np[:, [2, 0]], axis=1)
    ], axis=1).reshape(-1, 2)          # (3*n_faces, 2)

    edges_sorted_np = np.sort(E_np, axis=1)
    edge_index_map = {tuple(e): i for i, e in enumerate(edges_sorted_np)}
    edge_indices_np = np.array(
        [edge_index_map[tuple(e)] for e in face_edges_np], dtype=np.int64
    )
    face_indices_np = np.repeat(np.arange(n_faces, dtype=np.int64), 3)

    # to torch
    edge_indices = torch.from_numpy(edge_indices_np)
    face_indices = torch.from_numpy(face_indices_np)

    indices = torch.stack([edge_indices, face_indices], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32)

    EF = torch.sparse_coo_tensor(indices, values,
                                 size=(n_edges, n_faces))


    indices = torch.stack([
        torch.arange(n_faces, dtype=torch.long),  # row indices: 0..n_faces-1 (faces)
        torch.zeros(n_faces, dtype=torch.long)    # col indices: all 0 (cell 0)
    ], dim=0)

    values = torch.ones(n_faces, dtype=torch.float32)

    FC = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n_faces, 1)
    )

    return V, VF.coalesce(), VE.coalesce(), EF.coalesce(), FC.coalesce()

SOLID_TYPES = ["tetra", "cube", "octa", "dodeca", "icosa"]  # order fixed
name_to_label = {name: i for i, name in enumerate(SOLID_TYPES)}

create_platonic_functions = {"tetra":noisy_tetrahedron, "cube":noisy_cube,"octa":noisy_octahedron, "dodeca":noisy_dodecahedron ,"icosa":noisy_icosahedron}

def make_noisy_platonic(name, num_of_extra_vertices, eps_factor):
    return create_platonic_functions[name](1.0,num_of_extra_vertices,eps_factor)

class NoisyPlatonicSolids(Dataset):
    def __init__(self, counts_per_type, max_num_of_extra_vertices,eps_factor ,save_path=None):
        self.samples = []
        for name, count in counts_per_type.items():
            label = name_to_label[name]
            for i in range(count):
                mesh = make_noisy_platonic(name, np.random.randint(10,max_num_of_extra_vertices),eps_factor)
                V, VF, VE, EF, FC = make_matrices(mesh)
                self.samples.append({
                    "V": V,
                    "VF": VF,
                    "VE": VE,
                    "EF": EF,
                    "FC": FC,
                    "label": torch.tensor(label, dtype=torch.long).view(1)
                })

        if save_path is not None:
            torch.save(self.samples, save_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["V"], s["VF"], s["VE"], s["EF"], s["FC"] ,s["label"] 

def sparse_block_diag(sparse_list):
    # sparse_list: list of 2D sparse COO tensors
    device = sparse_list[0].device
    dtype = sparse_list[0].dtype

    rows = []
    cols = []
    vals = []

    row_offset = 0
    col_offset = 0

    for S in sparse_list:
        S = S.coalesce()
        i = S.indices()      # (2, nnz)
        v = S.values()       # (nnz,)
        n_rows, n_cols = S.shape

        # shift indices for this block
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
    for i,v in enumerate(V_list):
        batch_vector_list.append(torch.full((v.shape[0],), i, dtype=torch.int64))
    return torch.cat(batch_vector_list,dim=0)

def platonic_collate(batch):
    V_list, VF_list, VE_list, EF_list, FC_list, label_list = zip(*batch)
    V_stacked = torch.cat(V_list,dim=0)
    b_vec = batch_vector(V_list)

    return V_stacked, sparse_block_diag(VF_list), sparse_block_diag(VE_list), sparse_block_diag(EF_list), sparse_block_diag(FC_list), torch.cat(label_list,dim=0), b_vec

def ve_convert(VE):
    # VE: torch.sparse_coo_tensor of shape [num_vertices, num_edges]
    VE = VE.coalesce()
    row, col = VE.indices()      # row: vertex indices, col: edge indices

    # For each edge index e, find the two vertices with nonzero incidence:
    num_edges = VE.size(1)
    edge_u = []
    edge_v = []

    for e in range(num_edges):
        # vertices incident to edge e
        verts = row[col == e]
        assert verts.numel() == 2  # if you expect simple edges only
        edge_u.append(verts[0].item())
        edge_v.append(verts[1].item())

    edge_index = torch.tensor([edge_u, edge_v])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    return edge_index

################# TNN #################
class CCConvLayer(nn.Module):
	def __init__(
		self,
		in_channels,
		out_channels,
		aggr_norm: bool = False,
		att: bool = False,
		initialization: Literal["xavier_uniform"] = "xavier_uniform",
		initialzation_gain: float = 1.414,
		update_func: Literal["sigmoid","relu",None] = None
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
		self.W = nn.Parameter(torch.empty(self.out_channels,self.in_channels, dtype=torch.float32))
		nn.init.xavier_uniform_(self.W)

	def update(self,x):
		match self.update_func:
			case "sigmoid":
				return torch.sigmoid(x)
			case "relu":
				return torch.nn.functional.relu(x)
			case _:
				return x

	def forward(self,x,neighborhood):
		x_1 = x @ self.W.T
		x_2 = neighborhood @ x_1 
		x_3 = self.update(x_2)
		return x_3
	
class TNN(nn.Module):
	def __init__(self,node_channels,channels_rk1,channels_rk2,channels_rk3, size_hidden_layer,lr, epochs):
		super().__init__()

		self.lr = lr
		self.epochs = epochs

		self.conv_0_to_1 = CCConvLayer(in_channels = node_channels, out_channels =  channels_rk1, update_func="relu")
		self.conv_1_to_2 = CCConvLayer(in_channels = channels_rk1, out_channels =  channels_rk2, update_func="relu")
		self.conv_2_to_3 = CCConvLayer(in_channels = channels_rk2, out_channels = channels_rk3, update_func="relu")
		# Add later 0 -> 2
		#self.conv_0_to_2 = Conv(in_channels = node_channels, out_channels =  channels_rk2, update_func="relu")
		self.fc1 = nn.Linear(channels_rk3,64)
		self.fc2 = nn.Linear(64,5)

	def forward(self,x_0,incidence_0_1,incidence_1_2,incidence_2_3):
		x_0 = x_0.to(torch.float32)
		#x_0 = x_0.view(28 * 28,1)
		x_1_out = self.conv_0_to_1(x_0,incidence_0_1.T)
		x_2_out = self.conv_1_to_2(x_1_out,incidence_1_2.T)
		x_3_out = self.conv_2_to_3(x_2_out,incidence_2_3.T)
		x = torch.flatten(x_3_out, start_dim=1)
		x = relu(self.fc1(x))
		x = self.fc2(x)
		return x
		
	def fit(self,train_loader,device="cpu"):
		self.to(device)
		optimizer = optim.AdamW(self.parameters(),lr=self.lr)
		criterion = nn.CrossEntropyLoss()
		
		losses = []
		self.train()
		start = time.perf_counter()
		for epoch in range(self.epochs):
			total_loss = 0
			for V, VF, VE, EF, FC,label,_ in train_loader:
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
			losses.append(total_loss/len(train_loader))
			print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}")
		end = time.perf_counter()
		print(f"Training time: {end-start} seconds")
				
		return losses, end-start, (end-start)/self.epochs
		
	def	test(self, test_loader,device="cpu"):
		self.to(device)
		self.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			for V, VF, VE, EF, FC,label,_ in test_loader:
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
			accuracy = 100. * correct/total
			print(f"Test accuracy: {accuracy:.2f}%")
				
		return accuracy

################# GCN #################

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2,hidden_channels3 ,out_channels, lr, epochs):
        super().__init__()
        
        self.lr = lr
        self.epochs = epochs
        
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.fc1   = torch.nn.Linear(hidden_channels2, hidden_channels3)
        self.fc2   = torch.nn.Linear(hidden_channels3, out_channels)	

    def forward(self, x, edge_index, batch):
    	# 1) Node-level GCN
    	x = self.conv1(x, edge_index).relu()
    	x = self.conv2(x, edge_index).relu()	
    	# 2) Graph-level pooling
    	x = global_mean_pool(x, batch)	
    	x = self.fc1(x)  # logits per graph
    	x = self.fc2(x)
    	return x
        
    def fit(self, train_loader, device="cpu"):
        self.to(device)
        optimizer = optim.Adam(self.parameters(),lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.train()
        start = time.perf_counter()
        for epoch in range(self.epochs):
            total_loss = 0
            for V, VF, VE, EF, FC,label, batch_vec in train_loader:
                V = V.to(torch.float32)
                VE = ve_convert(VE)
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = self(V, VE, batch_vec)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss/len(train_loader))
            print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}")
        end = time.perf_counter()
        print(f"Training time: {end-start} seconds")
        
        return losses, end-start, (end-start)/self.epochs

    def test(self, test_loader, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for V, VF, VE, EF, FC,label, batch_vec in test_loader:
                V = V.to(torch.float32)
                VE = ve_convert(VE)
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                output = self(V, VE, batch_vec)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                total += label.size(0)
            accuracy = 100. * correct/total
            print(f"Test accuracy: {accuracy:.2f}%")
        
        return accuracy

################# GAN #################
class GAN(nn.Module):
    def __init__(self, in_channels, hidden_channels1,hidden_channels2,hidden_channels3 ,out_channels,lr,epochs, heads1=4,heads2=4):
        super().__init__()

        self.lr = lr
        self.epochs = epochs

        self.conv1 = GATv2Conv(in_channels, hidden_channels1,heads=heads1,concat=True)
        self.conv2 = GATv2Conv(hidden_channels1 * heads1, hidden_channels2,heads=heads2,concat=True)

        self.fc1   = torch.nn.Linear(hidden_channels2 * heads2 , hidden_channels3)
        self.fc2   = torch.nn.Linear(hidden_channels3, out_channels)	

    def forward(self, x, edge_index, batch):
    	# 1) Node-level GCN
    	x = self.conv1(x, edge_index).relu()
    	x = self.conv2(x, edge_index).relu()	
    	# 2) Graph-level pooling
    	x = global_mean_pool(x, batch)	
    	x = self.fc1(x).relu()  # logits per graph
    	x = self.fc2(x)
    	return x
        
    def fit(self, train_loader, device="cpu"):
        self.to(device)
        optimizer = optim.Adam(self.parameters(),lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.train()
        start = time.perf_counter()
        for epoch in range(self.epochs):
            total_loss = 0
            for V, VF, VE, EF, FC,label, batch_vec in train_loader:
                V = V.to(torch.float32)
                VE = ve_convert(VE)
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = self(V, VE, batch_vec)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss/len(train_loader))
            print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}")
        end = time.perf_counter()
        print(f"Training time: {end-start} seconds")
        return losses, end-start, (end-start)/self.epochs

    def test(self, test_loader, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for V, VF, VE, EF, FC,label, batch_vec in test_loader:
                V = V.to(torch.float32)
                VE = ve_convert(VE)
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                output = self(V, VE, batch_vec)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                total += label.size(0)
            accuracy = 100. * correct/total
            print(f"Test accuracy: {accuracy:.2f}%")
        
        return accuracy

################# GIN #################

class GIN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels1,
        hidden_channels2,
        hidden_channels3,
        out_channels,
        lr, epochs
    ):
        super().__init__()
        
        self.lr = lr
        self.epochs= epochs
        
        # GIN layer 1 MLP
        nn1 = nn.Sequential(
        	nn.Linear(in_channels, hidden_channels1),
        	nn.ReLU(),
        	nn.Linear(hidden_channels1, hidden_channels1),
        )
        self.conv1 = GINConv(nn1, train_eps=True)
        
        # GIN layer 2 MLP
        nn2 = nn.Sequential(
        	nn.Linear(hidden_channels1, hidden_channels2),
        	nn.ReLU(),
        	nn.Linear(hidden_channels2, hidden_channels2),
        )
        self.conv2 = GINConv(nn2, train_eps=True)
        
        # Graph-level MLP
        self.fc1 = nn.Linear(hidden_channels2, hidden_channels3)
        self.fc2 = nn.Linear(hidden_channels3, out_channels)

    def forward(self, x, edge_index, batch):
    	# 1) Node-level GIN
    	x = self.conv1(x, edge_index)
    	x = F.relu(x)

    	x = self.conv2(x, edge_index)
    	x = F.relu(x)

    	# 2) Graph-level pooling (sum is standard for GIN)
    	x = global_add_pool(x, batch)  # [num_graphs, hidden_channels2]

    	# 3) Graph-level MLP
    	x = self.fc1(x)
    	x = F.relu(x)
    	x = self.fc2(x)
    	return x

    def fit(self, train_loader, device="cpu"):
        self.to(device)
        optimizer = optim.Adam(self.parameters(),lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        self.train()
        start = time.perf_counter()
        for epoch in range(self.epochs):
            total_loss = 0
            for V, VF, VE, EF, FC,label, batch_vec in train_loader:
                V = V.to(torch.float32)
                VE = ve_convert(VE)
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = self(V, VE, batch_vec)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss/len(train_loader))
            print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}")
        end = time.perf_counter()
        print(f"Training time: {end-start} seconds")
        
        return losses, end-start, (end-start)/self.epochs

    def test(self, test_loader, device="cpu"):
        self.to(device)
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for V, VF, VE, EF, FC,label, batch_vec in test_loader:
                V = V.to(torch.float32)
                VE = ve_convert(VE)
                V = V.to(device)
                VF = VF.to(device)
                VE = VE.to(device)
                EF = EF.to(device)
                FC = FC.to(device)
                label = label.to(device)
                output = self(V, VE, batch_vec)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                total += label.size(0)
            accuracy = 100. * correct/total
            print(f"Test accuracy: {accuracy:.2f}%")
        
        return accuracy

################# HP-Tuning #################
def hp_optimization(model, hps, data_params, n_trials,runs_per_hp_comb, random_seed, device="cpu"):
    fixed_params = {}
    for hp in hps:
        if hp[1] == "fixed":
            fixed_params[hp[0]] = hp[2]
    
    # Load data
    dataset = NoisyPlatonicSolids({name: 500 for name in SOLID_TYPES},data_params[0],data_params[1])
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=platonic_collate)

    dataset = NoisyPlatonicSolids({name: 100 for name in SOLID_TYPES},data_params[0],data_params[1])
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=platonic_collate)

    def objective(trial):
        hp_trial_dict = {}
        
        for hp in hps:
            if hp[1] == "categorical":
                hp_trial_dict[hp[0]] = trial.suggest_categorical(hp[0], hp[2])
            elif hp[1] == "int":
                hp_trial_dict[hp[0]] = trial.suggest_int(hp[0], hp[2][0], hp[2][1])
            elif hp[1] == "float":
                hp_trial_dict[hp[0]] = trial.suggest_float(hp[0], hp[2][0], hp[2][1])
        
        accuracies = []
        for i in range(runs_per_hp_comb):
            cur_model = model(**hp_trial_dict,**fixed_params)
            cur_model.fit(train_loader,device=device)
            acc = cur_model.test(test_loader,device=device)
            accuracies.append(acc)
        
        return np.max(accuracies)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.trials_dataframe(),{**study.best_params, **fixed_params}

HYPERPARAMS = {
    "TNN": [("epochs", "int",(5,15)),("channels_rk1","int",(4,32)),("channels_rk2","int",(4,64)),
    ("channels_rk3","int",(64,256)),("size_hidden_layer","int",(32,128)),("lr","float",(1e-3,1e-2)), ("node_channels","fixed",3)],

    "GCN": [("lr","float",(1e-3,1e-1)),("epochs", "int",(5,20)),("in_channels","fixed",3),("out_channels","fixed",5),
    ("hidden_channels1","int",(16,128)),("hidden_channels2","int",(16,128)),("hidden_channels3","int",(16,128)) ],
    
    "GAN": [("lr","float",(1e-3,1e-1)),("epochs", "int",(5,20)),("in_channels","fixed",3),("out_channels","fixed",5),
    ("hidden_channels1","int",(16,128)),("hidden_channels2","int",(16,128)),("hidden_channels3","int",(16,128)),
    ("heads1","int",(1,4)),("heads2","int",(1,4)) ],

    "GIN": [("lr","float",(1e-3,1e-1)),("epochs", "int",(5,20)),("in_channels","fixed",3),("out_channels","fixed",5),
    ("hidden_channels1","int",(16,128)),("hidden_channels2","int",(16,128)),("hidden_channels3","int",(16,128)) ]
    }

################# MAIN #################

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--path", type=str, default=os.getcwd())
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--device",type=str, default="cpu")

    args = argparser.parse_args()
    SEED = args.seed
    DEVICE = args.device


    MODELS = {"TNN": TNN, "GCN": GCN, "GAN": GAN, "GIN": GIN}

    model = MODELS[args.model]

    path = f"{args.path}/{args.model}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)

    data_params = (40, 0.01)
    hps = HYPERPARAMS[args.model]
    n_trials = 50
    runs_per_hp_comb = 3
    study_dataframe, best_hps = hp_optimization(model, hps, data_params, n_trials, runs_per_hp_comb, SEED,device=DEVICE)

    top5 = study_dataframe[study_dataframe["state"] == "COMPLETE"].sort_values("value", ascending=False).head(5)
    top5.to_csv(f"{path}/top5_params.csv")

    mnoev = np.arange(40, 160, 10)
    epsf = [0.01,0.1,0.2,0.3]

    results = {}
    runs_per_data_params = 5
    for m,e in product(mnoev, epsf):
        results[f"{m}_{e}"] = {}
        dataset = NoisyPlatonicSolids({name: 500 for name in SOLID_TYPES},m,e)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=platonic_collate)

        dataset = NoisyPlatonicSolids({name: 100 for name in SOLID_TYPES},m,e)
        test_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=platonic_collate)

        accuracies = []
        for i in range(runs_per_data_params):
            cur_model = model(**best_hps)
            losses, total_time, time_per_epoch = cur_model.fit(train_loader, device=DEVICE)
            accs = cur_model.test(test_loader, device=DEVICE)
            accuracies.append(accs)

            results[f"{m}_{e}"][f"run_{i+1}"] = {"losses":losses,"accuracy":accs,"total_time":total_time,"time_per_epoch":time_per_epoch}
        
        results[f"{m}_{e}"]["results"] = {"best_acc": float(np.max(accuracies)),"best_run": int(np.argmax(accuracies)) + 1, "best_hps":best_hps}
        with open(f"{path}/results.json","w") as f:
            json.dump(results,f, indent=2, sort_keys=True)




