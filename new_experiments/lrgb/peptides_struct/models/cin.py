"""CIN and CIN++ (cellular isomorphism networks) ported from
https://github.com/twitter-research/cwn (mp/layers.py, mp/molec_models.py)
to operate on the sparse index-list cell-complex batches produced by
data_loader/lrgb_cin.py, instead of the original repo's Cochain/ComplexBatch
classes.

Per rank r cell i, one CIN layer computes:
  up_out       = update_up_nn( sum_j msg_up(x_j, x_up_shared[shared cell]) + (1+eps1)*x_i )
  boundary_out = update_boundary_nn( sum_f x_f (faces of i)                + (1+eps2)*x_i )
  [CIN++ only]
  down_out     = update_down_nn( sum_j msg_down(x_j, x_down_shared[shared cell]) + (1+eps3)*x_i )
  x_i' = combine_nn( concat(up_out, [down_out], boundary_out) )

msg_up/msg_down optionally concatenate the shared (co)boundary cell's feature
("use_coboundaries", matches the official cin++-pep-s.sh config).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, LayerNorm as LN
from torch_geometric.nn import global_mean_pool, global_add_pool


def _scatter_add(src, index, dim_size):
    if src.shape[0] == 0:
        return torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


def _mlp2(in_dim, hidden, act_module):
    return Sequential(
        Linear(in_dim, hidden), LN(hidden), act_module(),
        Linear(hidden, hidden), LN(hidden), act_module(),
    )


class Catter(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=-1)


def _msg_nn(dim, use_coboundaries, act_module):
    if use_coboundaries:
        return Sequential(Catter(), Linear(2 * dim, dim), act_module())
    return None  # identity: message = neighbor feature


class CINCochainLayer(nn.Module):
    """One CIN(/CIN++) message-passing layer for a single rank."""

    def __init__(self, in_dim, hidden, has_down, use_coboundaries, act_module, train_eps=False, eps=0.0):
        super().__init__()
        self.has_down = has_down
        self.msg_up_nn = _msg_nn(in_dim, use_coboundaries, act_module)
        self.msg_down_nn = _msg_nn(in_dim, use_coboundaries, act_module) if has_down else None
        self.update_up_nn = _mlp2(in_dim, hidden, act_module)
        self.update_boundary_nn = _mlp2(in_dim, hidden, act_module)
        if has_down:
            self.update_down_nn = _mlp2(in_dim, hidden, act_module)
        n_components = 3 if has_down else 2
        self.combine_nn = Sequential(
            Linear(hidden * n_components, hidden), LN(hidden), act_module())

        def _eps_param():
            return nn.Parameter(torch.tensor([eps])) if train_eps else eps
        self.eps1 = _eps_param()
        self.eps2 = _eps_param()
        self.eps3 = _eps_param() if has_down else None

    def forward(self, x, n_cells,
                up_index, up_attr_idx, x_up_shared,
                boundary_index, x_boundary,
                down_index=None, down_attr_idx=None, x_down_shared=None):
        # up branch
        if up_index.shape[1] > 0:
            x_j = x.index_select(0, up_index[1])
            if self.msg_up_nn is not None:
                attr = x_up_shared.index_select(0, up_attr_idx)
                msg = self.msg_up_nn((x_j, attr))
            else:
                msg = x_j
            up_sum = _scatter_add(msg, up_index[0], n_cells)
        else:
            up_sum = torch.zeros(n_cells, x.shape[1], device=x.device, dtype=x.dtype)
        out_up = self.update_up_nn(up_sum + (1 + self.eps1) * x)

        # boundary branch (identity message, no shared attr)
        if boundary_index.shape[1] > 0:
            face_feats = x_boundary.index_select(0, boundary_index[0])
            boundary_sum = _scatter_add(face_feats, boundary_index[1], n_cells)
        else:
            boundary_sum = torch.zeros(n_cells, x.shape[1], device=x.device, dtype=x.dtype)
        out_boundary = self.update_boundary_nn(boundary_sum + (1 + self.eps2) * x)

        parts = [out_up]

        if self.has_down:
            if down_index.shape[1] > 0:
                x_j = x.index_select(0, down_index[1])
                if self.msg_down_nn is not None:
                    attr = x_down_shared.index_select(0, down_attr_idx)
                    msg = self.msg_down_nn((x_j, attr))
                else:
                    msg = x_j
                down_sum = _scatter_add(msg, down_index[0], n_cells)
            else:
                down_sum = torch.zeros(n_cells, x.shape[1], device=x.device, dtype=x.dtype)
            out_down = self.update_down_nn(down_sum + (1 + self.eps3) * x)
            parts.append(out_down)

        parts.append(out_boundary)
        return self.combine_nn(torch.cat(parts, dim=-1))


class CINLayer(nn.Module):
    """A full layer updating all 3 ranks (nodes, bonds, rings) at once."""

    def __init__(self, in_dim, hidden, variant, use_coboundaries, act_module, train_eps=False):
        super().__init__()
        assert variant in ("CIN", "CINpp")
        has_down = variant == "CINpp"
        self.rank0 = CINCochainLayer(in_dim, hidden, has_down, use_coboundaries, act_module, train_eps)
        self.rank1 = CINCochainLayer(in_dim, hidden, has_down, use_coboundaries, act_module, train_eps)
        self.rank2 = CINCochainLayer(in_dim, hidden, has_down, use_coboundaries, act_module, train_eps)

    def forward(self, x0, x1, x2, g):
        n0, n1, n2 = x0.shape[0], x1.shape[0], x2.shape[0]
        empty_idx = torch.zeros(2, 0, dtype=torch.long, device=x0.device)
        empty_attr = torch.zeros(0, dtype=torch.long, device=x0.device)

        x0_new = self.rank0(
            x0, n0,
            up_index=g["up0_index"], up_attr_idx=g["up0_attr_idx"], x_up_shared=x1,
            boundary_index=empty_idx, x_boundary=x0,
            down_index=empty_idx, down_attr_idx=empty_attr, x_down_shared=x0,
        )
        x1_new = self.rank1(
            x1, n1,
            up_index=g["up1_index"], up_attr_idx=g["up1_attr_idx"], x_up_shared=x2,
            boundary_index=g["boundary1_index"], x_boundary=x0,
            down_index=g["down1_index"], down_attr_idx=g["down1_attr_idx"], x_down_shared=x0,
        )
        x2_new = self.rank2(
            x2, n2,
            up_index=empty_idx, up_attr_idx=empty_attr, x_up_shared=x2,
            boundary_index=g["boundary2_index"], x_boundary=x1,
            down_index=g["down2_index"], down_attr_idx=g["down2_attr_idx"], x_down_shared=x1,
        )
        return x0_new, x1_new, x2_new


class CIN(nn.Module):
    """CIN / CIN++ for graph-level regression on molecular cell complexes.

    variant='CIN'   -> up + boundary messages only (SparseCIN in the paper).
    variant='CINpp' -> also adds down messages (CIN++).
    """

    def __init__(self, x0_dim, x1_dim, x2_dim, out_dim, num_layers=3, hidden=128,
                 variant="CINpp", use_coboundaries=True, dropout=0.0, in_dropout=0.0,
                 final_hidden_multiplier=2, readout="mean", final_readout="sum",
                 train_eps=False):
        super().__init__()
        assert variant in ("CIN", "CINpp")
        self.dropout = dropout
        self.in_dropout = in_dropout
        self.readout_fn = global_mean_pool if readout == "mean" else global_add_pool
        self.final_readout = final_readout
        act_module = nn.ReLU

        self.emb0 = Linear(x0_dim, hidden)
        self.emb1 = Linear(x1_dim, hidden)
        self.emb2 = Linear(x2_dim, hidden)

        self.layers = nn.ModuleList([
            CINLayer(hidden, hidden, variant, use_coboundaries, act_module, train_eps)
            for _ in range(num_layers)
        ])

        self.lin1s = nn.ModuleList([Linear(hidden, final_hidden_multiplier * hidden) for _ in range(3)])
        self.lin2 = Linear(final_hidden_multiplier * hidden, out_dim)

    def forward(self, g):
        x0 = F.dropout(self.emb0(g["x_0"]), self.in_dropout, self.training)
        x1 = F.dropout(self.emb1(g["x_1"]), self.in_dropout, self.training)
        x2 = F.dropout(self.emb2(g["x_2"]), self.in_dropout, self.training)

        for layer in self.layers:
            x0, x1, x2 = layer(x0, x1, x2, g)

        B = g["num_graphs"]
        p0 = self.readout_fn(x0, g["batch0"], size=B)
        p1 = self.readout_fn(x1, g["batch1"], size=B) if x1.shape[0] > 0 else torch.zeros(B, x1.shape[1], device=x0.device)
        p2 = self.readout_fn(x2, g["batch2"], size=B) if x2.shape[0] > 0 else torch.zeros(B, x2.shape[1], device=x0.device)

        outs = []
        for pooled, lin in zip((p0, p1, p2), self.lin1s):
            outs.append(torch.relu(lin(pooled)))
        stacked = torch.stack(outs, dim=0)
        combined = stacked.sum(0) if self.final_readout == "sum" else stacked.mean(0)

        combined = F.dropout(combined, self.dropout, self.training)
        return self.lin2(combined)
