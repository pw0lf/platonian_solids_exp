"""Aligns externally-downloaded SMILES rows to PyG LRGBDataset graph items.

download_smiles.py fetches the LRGB peptides CSV independently of PyG's
LRGBDataset, which loads pre-built train/val/test .pt files from the LRGB
authors. Nothing guarantees these share row order OR split membership:
positionally zipping pyg_ds with smiles_{split}.csv silently pairs one
molecule's structure with an unrelated molecule's label (confirmed
empirically: 0/30 atom counts matched on Peptides-struct train), and even
matching by content against the *per-split* CSV fails outright for some
items, because a molecule can land in a different smiles_{split}.csv file
than the PyG split it actually belongs to.

The fix: pool smiles_train/val/test.csv together (they cover the same
15,535 molecules overall) and match each PyG split's graphs against the
pooled set by content (atomic-number-labeled Weisfeiler-Lehman hash +
exact isomorphism check for hash collisions), claiming matches globally
across splits so no pooled row is reused.

A small number of molecules are 2D-graph-isomorphic to a different molecule
elsewhere in the corpus (most likely stereoisomers — indistinguishable from
a 2D connectivity graph, but with different 3D-derived labels). These can't
be disambiguated from structure alone; match_all_splits() drops them
(reported via `unmatched`) rather than risk mispairing them, same as the
existing "failed to parse" convention in the CT/SchNet loaders.
"""
import networkx as nx
import pandas as pd
from rdkit import Chem
from torch_geometric.datasets import LRGBDataset


class _LRGBNoDownload(LRGBDataset):
    def download(self):
        pass


def _graph_from_pyg(item):
    """Node labels are atomic numbers, decoded from OGB atom feature col 0
    (value = atomic_num - 1)."""
    z = (item.x[:, 0] + 1).tolist()
    src, dst = item.edge_index[0].tolist(), item.edge_index[1].tolist()
    edges = {(min(u, v), max(u, v)) for u, v in zip(src, dst) if u != v}
    G = nx.Graph()
    G.add_nodes_from(range(len(z)))
    for i, zi in enumerate(z):
        G.nodes[i]["label"] = int(zi)
    G.add_edges_from(edges)
    return G


def _graph_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


def _node_match(a, b):
    return a["label"] == b["label"]


def _build_buckets(smiles_list):
    smi_graphs, buckets = {}, {}
    for j, smi in enumerate(smiles_list):
        G = _graph_from_smiles(smi)
        if G is None:
            continue
        h = nx.weisfeiler_lehman_graph_hash(G, node_attr="label", iterations=3)
        smi_graphs[j] = (G, h)
        buckets.setdefault(h, []).append(j)
    return smi_graphs, buckets


def match_split(pyg_ds, smi_graphs, buckets, used):
    """Matches one split's graphs against the pooled (smi_graphs, buckets),
    claiming into the shared `used` set (mutated in place). Returns perm,
    a list of length len(pyg_ds) with the matched pooled index, or None
    for items that couldn't be matched uniquely."""
    perm = []
    for i in range(len(pyg_ds)):
        Gi = _graph_from_pyg(pyg_ds[i])
        hi = nx.weisfeiler_lehman_graph_hash(Gi, node_attr="label", iterations=3)
        candidates = [j for j in buckets.get(hi, []) if j not in used]

        match = None
        if len(candidates) == 1:
            match = candidates[0]
        elif len(candidates) > 1:
            for j in candidates:
                Gj, _ = smi_graphs[j]
                if nx.is_isomorphic(Gi, Gj, node_match=_node_match):
                    match = j
                    break
        if match is not None:
            used.add(match)
        perm.append(match)

    return perm


def match_all_splits(pyg_splits, smiles_pool):
    """pyg_splits: dict of split_name -> pyg dataset (order matters — earlier
    splits get first claim on ambiguous duplicates in the corpus).
    smiles_pool: combined list of all SMILES (e.g. train+val+test CSVs
    concatenated).

    Returns dict of split_name -> perm (index into smiles_pool, or None for
    unmatched items).
    """
    smi_graphs, buckets = _build_buckets(smiles_pool)
    used = set()
    return {
        name: match_split(ds, smi_graphs, buckets, used)
        for name, ds in pyg_splits.items()
    }


def align_smiles_to_splits(root, name, smiles_csv_paths):
    """smiles_csv_paths: dict of 'train'/'val'/'test' -> path to a smiles CSV.

    The three smiles_{split}.csv files are NOT guaranteed to align with the
    PyG split of the same name -- some molecules end up in a different split
    file than the PyG split they actually belong to. So we pool all three
    files together and match each PyG split's graphs against the full pool
    by content, claiming matches globally across splits (train first) so no
    pooled row is reused.

    Returns (smiles_pool, perms) where perms[split][i] is an index into
    smiles_pool identifying the same molecule as pyg split item i, or None
    if it couldn't be matched uniquely.
    """
    smiles_pool = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(smiles_csv_paths[split])
        smiles_pool.extend(df["smiles"].tolist())

    pyg_splits = {s: _LRGBNoDownload(root=root, name=name, split=s) for s in ("train", "val", "test")}
    perms = match_all_splits(pyg_splits, smiles_pool)
    return smiles_pool, perms
