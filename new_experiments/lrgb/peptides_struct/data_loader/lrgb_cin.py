"""CIN / CIN++ dataloader for LRGB Peptides datasets.

Builds a 2-dimensional cell complex per molecule (atoms=0-cells,
bonds=1-cells, rings=2-cells; rings via networkx minimum_cycle_basis,
same as the CT 'original' feat_mode) and precomputes the sparse index
lists CIN-style message passing needs:

  up{r}_index      : (2, E) pairs of rank-r cells sharing a rank-(r+1)
                      coboundary cell, up{r}_attr_idx points into rank-(r+1)
                      features for that shared cell.
  down{r}_index    : (2, E) pairs of rank-r cells sharing a rank-(r-1)
                      boundary cell, down{r}_attr_idx points into rank-(r-1)
                      features for that shared cell. Only used by CIN++.
  boundary{r}_index: (2, E) incidence edges (face, cell) from rank-(r-1)
                      to rank-r, used to sum boundary-face features per cell.

All index tensors are always present (possibly empty, shape (2, 0)) so
batching/scatter never has to special-case missing structure.
"""

import torch
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.datasets import LRGBDataset


class _LRGBNoDownload(LRGBDataset):
    def download(self):
        pass


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
                    src.append(i)
                    dst.append(j)
                    shared.append(key)
    return src, dst, shared


def _process_molecule(pyg_item):
    x_node = pyg_item.x            # (N, 9) int64
    edge_index = pyg_item.edge_index    # (2, 2E) int64
    edge_attr = pyg_item.edge_attr      # (2E, 3) int64
    y = pyg_item.y.squeeze(0).float()

    n_atoms = x_node.shape[0]
    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    src_u, dst_u = src[mask].tolist(), dst[mask].tolist()
    edge_attr_u = edge_attr[mask]

    x_0 = x_node.float()
    x_1 = edge_attr_u.float()
    n_bonds = len(src_u)

    edges = list(zip(src_u, dst_u))
    edge_to_idx = {}
    for b, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = b
        edge_to_idx[(v, u)] = b

    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))
    G.add_edges_from(edges)
    try:
        cycles = [list(c) for c in nx.minimum_cycle_basis(G)]
    except Exception:
        cycles = []

    ring_bonds = []
    for cycle in cycles:
        n = len(cycle)
        bonds = []
        for i in range(n):
            b = edge_to_idx.get((cycle[i], cycle[(i + 1) % n]))
            if b is not None:
                bonds.append(b)
        ring_bonds.append(bonds)
    n_rings = len(ring_bonds)

    x_2 = (torch.tensor([[float(len(c))] for c in cycles], dtype=torch.float32)
           if n_rings > 0 else torch.zeros(0, 1, dtype=torch.float32))

    # boundary1: atom -> bond (each bond's 2 atom faces)
    b1_face, b1_cell = [], []
    for b, (u, v) in enumerate(edges):
        b1_face.extend([u, v])
        b1_cell.extend([b, b])
    boundary1_index = torch.tensor([b1_face, b1_cell], dtype=torch.long) if b1_face \
        else torch.zeros(2, 0, dtype=torch.long)

    # boundary2: bond -> ring (each ring's boundary bonds)
    b2_face, b2_cell = [], []
    for r, bonds in enumerate(ring_bonds):
        for b in bonds:
            b2_face.append(b)
            b2_cell.append(r)
    boundary2_index = torch.tensor([b2_face, b2_cell], dtype=torch.long) if b2_face \
        else torch.zeros(2, 0, dtype=torch.long)

    # up0: atoms sharing a bond (coboundary = the bond itself)
    up0_src, up0_dst, up0_attr = [], [], []
    for b, (u, v) in enumerate(edges):
        up0_src.extend([u, v]); up0_dst.extend([v, u]); up0_attr.extend([b, b])
    up0_index = torch.tensor([up0_src, up0_dst], dtype=torch.long) if up0_src \
        else torch.zeros(2, 0, dtype=torch.long)
    up0_attr_idx = torch.tensor(up0_attr, dtype=torch.long)

    # up1: bonds sharing a ring (coboundary = the ring)
    ring_groups = {r: bonds for r, bonds in enumerate(ring_bonds)}
    up1_src, up1_dst, up1_attr = _pairs_within_groups(ring_groups)
    up1_index = torch.tensor([up1_src, up1_dst], dtype=torch.long) if up1_src \
        else torch.zeros(2, 0, dtype=torch.long)
    up1_attr_idx = torch.tensor(up1_attr, dtype=torch.long)

    # down1: bonds sharing an atom (boundary = the atom), for CIN++
    atom_groups = {}
    for b, (u, v) in enumerate(edges):
        atom_groups.setdefault(u, []).append(b)
        atom_groups.setdefault(v, []).append(b)
    down1_src, down1_dst, down1_attr = _pairs_within_groups(atom_groups)
    down1_index = torch.tensor([down1_src, down1_dst], dtype=torch.long) if down1_src \
        else torch.zeros(2, 0, dtype=torch.long)
    down1_attr_idx = torch.tensor(down1_attr, dtype=torch.long)

    # down2: rings sharing a bond (boundary = the bond), for CIN++
    bond_groups = {}
    for r, bonds in enumerate(ring_bonds):
        for b in bonds:
            bond_groups.setdefault(b, []).append(r)
    down2_src, down2_dst, down2_attr = _pairs_within_groups(bond_groups)
    down2_index = torch.tensor([down2_src, down2_dst], dtype=torch.long) if down2_src \
        else torch.zeros(2, 0, dtype=torch.long)
    down2_attr_idx = torch.tensor(down2_attr, dtype=torch.long)

    return dict(
        x_0=x_0, x_1=x_1, x_2=x_2,
        n_atoms=n_atoms, n_bonds=n_bonds, n_rings=n_rings,
        boundary1_index=boundary1_index, boundary2_index=boundary2_index,
        up0_index=up0_index, up0_attr_idx=up0_attr_idx,
        up1_index=up1_index, up1_attr_idx=up1_attr_idx,
        down1_index=down1_index, down1_attr_idx=down1_attr_idx,
        down2_index=down2_index, down2_attr_idx=down2_attr_idx,
        y=y,
    )


class LRGBCINDataset(Dataset):
    def __init__(self, root, name, split):
        pyg_ds = _LRGBNoDownload(root=root, name=name, split=split)
        self.data = []
        self.indices = []
        for i, item in enumerate(pyg_ds):
            try:
                self.data.append(_process_molecule(item))
                self.indices.append(i)
            except Exception:
                pass
        assert len(self.data) > 0, "No molecules successfully loaded."
        self.x0_dim = self.data[0]["x_0"].shape[1]
        self.x1_dim = self.data[0]["x_1"].shape[1]
        self.x2_dim = self.data[0]["x_2"].shape[1]
        print(f"  Loaded {len(self.data)}/{len(pyg_ds)} molecules "
              f"| x0={self.x0_dim} x1={self.x1_dim} x2={self.x2_dim}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_cin(batch):
    """Batches a list of per-molecule dicts (see _process_molecule) into one
    big disjoint-union complex, offsetting all indices and building per-rank
    batch-assignment vectors for pooling."""
    keys_idx_attr = [
        ("up0_index", "up0_attr_idx", 0, 1),      # (index key, attr key, cell rank, attr rank)
        ("up1_index", "up1_attr_idx", 1, 2),
        ("down1_index", "down1_attr_idx", 1, 0),
        ("down2_index", "down2_attr_idx", 2, 1),
    ]
    boundary_keys = [
        ("boundary1_index", 0, 1),  # (index key, face rank, cell rank)
        ("boundary2_index", 1, 2),
    ]

    off = [0, 0, 0]  # cumulative atom/bond/ring counts
    x0_list, x1_list, x2_list, y_list = [], [], [], []
    batch0, batch1, batch2 = [], [], []
    up0_i, up0_a, up1_i, up1_a = [], [], [], []
    down1_i, down1_a, down2_i, down2_a = [], [], [], []
    b1_i, b2_i = [], []

    for g, item in enumerate(batch):
        n0, n1, n2 = item["n_atoms"], item["n_bonds"], item["n_rings"]
        x0_list.append(item["x_0"]); x1_list.append(item["x_1"]); x2_list.append(item["x_2"])
        y_list.append(item["y"])
        batch0.append(torch.full((n0,), g, dtype=torch.long))
        batch1.append(torch.full((n1,), g, dtype=torch.long))
        batch2.append(torch.full((n2,), g, dtype=torch.long))

        up0_i.append(item["up0_index"] + off[0])
        up0_a.append(item["up0_attr_idx"] + off[1])
        up1_i.append(item["up1_index"] + off[1])
        up1_a.append(item["up1_attr_idx"] + off[2])
        down1_i.append(item["down1_index"] + off[1])
        down1_a.append(item["down1_attr_idx"] + off[0])
        down2_i.append(item["down2_index"] + off[2])
        down2_a.append(item["down2_attr_idx"] + off[1])
        b1_i.append(item["boundary1_index"] + torch.tensor([[off[0]], [off[1]]]))
        b2_i.append(item["boundary2_index"] + torch.tensor([[off[1]], [off[2]]]))

        off[0] += n0; off[1] += n1; off[2] += n2

    def cat_idx(lst):
        return torch.cat(lst, dim=1) if lst else torch.zeros(2, 0, dtype=torch.long)

    def cat_attr(lst):
        return torch.cat(lst, dim=0) if lst else torch.zeros(0, dtype=torch.long)

    return dict(
        x_0=torch.cat(x0_list), x_1=torch.cat(x1_list), x_2=torch.cat(x2_list),
        batch0=torch.cat(batch0), batch1=torch.cat(batch1), batch2=torch.cat(batch2),
        up0_index=cat_idx(up0_i), up0_attr_idx=cat_attr(up0_a),
        up1_index=cat_idx(up1_i), up1_attr_idx=cat_attr(up1_a),
        down1_index=cat_idx(down1_i), down1_attr_idx=cat_attr(down1_a),
        down2_index=cat_idx(down2_i), down2_attr_idx=cat_attr(down2_a),
        boundary1_index=cat_idx(b1_i), boundary2_index=cat_idx(b2_i),
        y=torch.stack(y_list),
        num_graphs=len(batch),
    )
