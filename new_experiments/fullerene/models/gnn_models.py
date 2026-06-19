import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, global_mean_pool, global_add_pool


def _make_mlp(in_dim, hidden_dim, num_hidden_layers, out_dim, dropout):
    dims = [in_dim] + [hidden_dim] * num_hidden_layers
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim if num_hidden_layers > 0 else in_dim, out_dim))
    return nn.Sequential(*layers)


class GCN(nn.Module):
    """
    Hyperparameters:
      in_channels          : input node feature dim
      hidden_channels      : hidden dim in conv layers
      num_conv_layers      : number of GCN conv layers
      readout_hidden_dim   : hidden dim in readout MLP
      num_readout_layers   : number of hidden layers in readout MLP
      dropout              : dropout probability (applied after each conv and in readout)
    """
    def __init__(self, in_channels, hidden_channels, num_conv_layers,
                 readout_hidden_dim, num_readout_layers, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_conv_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.readout = _make_mlp(hidden_channels, readout_hidden_dim, num_readout_layers, 1, dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(global_mean_pool(x, batch))


class GAT(nn.Module):
    """
    Hyperparameters:
      in_channels          : input node feature dim
      hidden_channels      : hidden dim per head in conv layers
      num_conv_layers      : number of GAT conv layers
      num_heads            : attention heads (output dim = hidden_channels * num_heads)
      edge_dim             : edge feature dim (None = no edge features)
      readout_hidden_dim   : hidden dim in readout MLP
      num_readout_layers   : number of hidden layers in readout MLP
      dropout              : dropout probability (attention + post-conv + readout)
    """
    def __init__(self, in_channels, hidden_channels, num_conv_layers, num_heads,
                 edge_dim=None, readout_hidden_dim=64, num_readout_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads,
                                    edge_dim=edge_dim, concat=True, dropout=dropout))
        for _ in range(num_conv_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads,
                                        edge_dim=edge_dim, concat=True, dropout=dropout))
        self.readout = _make_mlp(hidden_channels * num_heads, readout_hidden_dim, num_readout_layers, 1, dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr=edge_attr))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(global_mean_pool(x, batch))


class GIN(nn.Module):
    """
    Hyperparameters:
      in_channels          : input node feature dim
      hidden_channels      : hidden dim in conv layers
      num_conv_layers      : number of GIN conv layers
      mlp_hidden_dim       : hidden dim inside each GIN MLP (defaults to hidden_channels)
      readout_hidden_dim   : hidden dim in readout MLP
      num_readout_layers   : number of hidden layers in readout MLP
      dropout              : dropout probability (post-conv + readout)
    """
    def __init__(self, in_channels, hidden_channels, num_conv_layers,
                 mlp_hidden_dim=None, readout_hidden_dim=64, num_readout_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        mlp_hidden_dim = mlp_hidden_dim or hidden_channels
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(
            nn.Sequential(nn.Linear(in_channels, mlp_hidden_dim), nn.ReLU(),
                          nn.Linear(mlp_hidden_dim, hidden_channels)),
            train_eps=True,
        ))
        for _ in range(num_conv_layers - 1):
            self.convs.append(GINConv(
                nn.Sequential(nn.Linear(hidden_channels, mlp_hidden_dim), nn.ReLU(),
                              nn.Linear(mlp_hidden_dim, hidden_channels)),
                train_eps=True,
            ))
        self.readout = _make_mlp(hidden_channels, readout_hidden_dim, num_readout_layers, 1, dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(global_add_pool(x, batch))
