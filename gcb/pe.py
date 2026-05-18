import torch

def barycentric_subdivision(num_nodes, num_edges, num_rings, icd01, icd02, icd12):
    num_cells = num_nodes + num_edges + num_rings
    e = num_nodes
    r = num_nodes + num_edges

    def block(sparse, row_off, col_off, transpose=False):
        s = sparse.coalesce()
        ri, ci = s.indices()
        if transpose:
            ri, ci = ci, ri
        return ri + row_off, ci + col_off, s.values()

    parts = [
        block(icd01, 0, e),        # nodes -> edges
        block(icd01, e, 0, True),  # edges -> nodes
        block(icd02, 0, r),        # nodes -> rings
        block(icd02, r, 0, True),  # rings -> nodes
        block(icd12, e, r),        # edges -> rings
        block(icd12, r, e, True),  # rings -> edges
    ]

    rows = torch.cat([p[0] for p in parts])
    cols = torch.cat([p[1] for p in parts])
    vals = torch.cat([p[2] for p in parts])

    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(num_cells, num_cells)
    ).coalesce()

def random_walk(num_nodes, num_edges, num_rings, icd01, icd02, icd12, device):
    adj = barycentric_subdivision(num_nodes, num_edges, num_rings,icd01,icd02,icd12)
    sum = torch.sparse.sum(adj,dim=1).to_dense()
    mask = (sum == 0).float()
    n = sum.size(0)
    idx = torch.arange(n, device=device)
    diag = torch.sparse_coo_tensor(torch.stack([idx, idx]), mask, size=(n, n))
    adj = adj + diag
    sum = sum + mask  # isolated nodes now have sum=1, preventing 1/0=inf
    D_inv = torch.sparse_coo_tensor(torch.stack([idx, idx]), 1/sum, size=(n, n))
    rw = adj @ D_inv
    return rw

def get_diag_values(rw, device):
    n = rw.size(0)
    eye = torch.eye(n, device=device).to_sparse()
    diag = (rw * eye).coalesce().values()
    return diag

def CC_RWBSPe(k,num_nodes, num_edges, num_rings, icd01, icd02, icd12, device):
    num_cells = num_nodes + num_edges + num_rings
    rw = random_walk(num_nodes, num_edges, num_rings, icd01, icd02, icd12, device)
    powers = [torch.matrix_power(rw.to_dense(),i+1) for i in range(k)]
    out = torch.zeros(num_cells,k, device=device)
    for i,rw in enumerate(powers):
        out[:,i] = get_diag_values(rw, device)
    return out