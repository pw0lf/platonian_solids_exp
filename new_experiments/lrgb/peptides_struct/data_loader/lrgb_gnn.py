"""GNN dataloader for LRGB Peptides datasets.

Wraps PyG LRGBDataset directly. Node/edge features cast to float32.
"""

from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Data


class LRGBGNNDataset:
    """Thin wrapper around PyG LRGBDataset that casts integer features to float."""
    def __init__(self, root, name, split):
        ds = LRGBDataset(root=root, name=name, split=split)
        self._data = []
        for i, item in enumerate(ds):
            self._data.append(Data(
                x=item.x.float(),
                edge_index=item.edge_index,
                edge_attr=item.edge_attr.float(),
                y=item.y.squeeze(0).float(),
            ))
        self.indices = list(range(len(self._data)))
        self.num_node_features = 9
        self.num_edge_features = 3

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]
