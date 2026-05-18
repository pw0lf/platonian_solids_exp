import argparse
import json
import numpy as np
import torch
import networkx as nx
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, ".")
from ct import CellularTransformer
from pe import CC_RWBSPe

PE_K = 5


def _make_matrices(A_scipy, n_nodes):
    A_coo = A_scipy.tocoo()
    adj00 = torch.sparse_coo_tensor(
        torch.tensor([A_coo.row.astype(np.int64), A_coo.col.astype(np.int64)]),
        torch.ones(len(A_coo.row), dtype=torch.float32),
        size=(n_nodes, n_nodes),
    ).coalesce()
    mask  = A_coo.row < A_coo.col
    e_src = A_coo.row[mask].astype(np.int64)
    e_dst = A_coo.col[mask].astype(np.int64)
    n_edges = len(e_src)
    icd01 = torch.sparse_coo_tensor(
        torch.tensor([np.concatenate([e_src, e_dst]),
                      np.concatenate([np.arange(n_edges), np.arange(n_edges)])]),
        torch.ones(2 * n_edges, dtype=torch.float32),
        size=(n_nodes, n_edges),
    ).coalesce()
    node_to_edges = defaultdict(list)
    for k in range(n_edges):
        node_to_edges[int(e_src[k])].append(k)
        node_to_edges[int(e_dst[k])].append(k)
    adj11_rows, adj11_cols = [], []
    for nbrs in node_to_edges.values():
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                adj11_rows += [nbrs[i], nbrs[j]]
                adj11_cols += [nbrs[j], nbrs[i]]
    adj11 = (torch.sparse_coo_tensor(
        torch.tensor([adj11_rows, adj11_cols]),
        torch.ones(len(adj11_rows), dtype=torch.float32),
        size=(n_edges, n_edges)).coalesce()
        if adj11_rows else
        torch.zeros(n_edges, n_edges).to_sparse_coo().coalesce())
    G = nx.from_scipy_sparse_array(A_scipy)
    cycles = nx.cycle_basis(G)
    edge_lookup = {(min(int(e_src[k]), int(e_dst[k])), max(int(e_src[k]), int(e_dst[k]))): k
                   for k in range(n_edges)}
    n_cycles = len(cycles)
    if n_cycles == 0:
        icd02 = torch.zeros(n_nodes, 1).to_sparse_coo().coalesce()
        icd12 = torch.zeros(n_edges, 1).to_sparse_coo().coalesce()
        adj22 = torch.zeros(1, 1).to_sparse_coo().coalesce()
        n_cycles = 1
    else:
        icd02_rows, icd02_cols = [], []
        for c, cycle in enumerate(cycles):
            for node in cycle:
                icd02_rows.append(node); icd02_cols.append(c)
        icd02 = torch.sparse_coo_tensor(
            torch.tensor([icd02_rows, icd02_cols]),
            torch.ones(len(icd02_rows), dtype=torch.float32),
            size=(n_nodes, n_cycles)).coalesce()
        icd12_rows, icd12_cols = [], []
        for c, cycle in enumerate(cycles):
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i+1) % len(cycle)]
                key = (min(u, v), max(u, v))
                if key in edge_lookup:
                    icd12_rows.append(edge_lookup[key]); icd12_cols.append(c)
        icd12 = (torch.sparse_coo_tensor(
            torch.tensor([icd12_rows, icd12_cols]),
            torch.ones(len(icd12_rows), dtype=torch.float32),
            size=(n_edges, n_cycles)).coalesce()
            if icd12_rows else
            torch.zeros(n_edges, n_cycles).to_sparse_coo().coalesce())
        edge_to_cycles = defaultdict(list)
        for r, c in zip(icd12_rows, icd12_cols):
            edge_to_cycles[r].append(c)
        adj22_rows, adj22_cols = [], []
        for cyc_list in edge_to_cycles.values():
            for i in range(len(cyc_list)):
                for j in range(i+1, len(cyc_list)):
                    adj22_rows += [cyc_list[i], cyc_list[j]]
                    adj22_cols += [cyc_list[j], cyc_list[i]]
        adj22 = (torch.sparse_coo_tensor(
            torch.tensor([adj22_rows, adj22_cols]),
            torch.ones(len(adj22_rows), dtype=torch.float32),
            size=(n_cycles, n_cycles)).coalesce()
            if adj22_rows else
            torch.zeros(n_cycles, n_cycles).to_sparse_coo().coalesce())
    return adj00, icd01, adj11, icd02, icd12, adj22


def make_cell_features(x, icd01, icd02):
    icd01_c = icd01.coalesce()
    node_idx = icd01_c.indices()[0]
    edge_idx = icd01_c.indices()[1]
    order = torch.argsort(edge_idx, stable=True)
    nodes = node_idx[order]
    x_1 = torch.cat([x[nodes[0::2]], x[nodes[1::2]]], dim=1)
    x_2 = torch.sparse.mm(icd02.t(), x)
    return x_1, x_2


class GBChard(Dataset):
    def __init__(self, root=".", split="train", use_pe=True):
        loaded = np.load(f"{root}/hard.npz", allow_pickle=True)
        prefix = {"train": "tr", "val": "val", "test": "te"}[split]
        X = loaded[f"{prefix}_feat"]
        A = loaded[f"{prefix}_adj"]
        Y = loaded[f"{prefix}_class"]
        self.data = []
        for x, a, y in zip(X, A, Y):
            n_nodes = x.shape[0]
            x_t = torch.tensor(x, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            adj00, icd01, adj11, icd02, icd12, adj22 = _make_matrices(a, n_nodes)
            x_1, x_2 = make_cell_features(x_t, icd01, icd02)
            if use_pe:
                n_edges = icd01.shape[1]
                n_rings  = icd02.shape[1]
                pe = CC_RWBSPe(PE_K, n_nodes, n_edges, n_rings, icd01, icd02, icd12, 'cpu')
                x_0 = torch.cat([x_t, pe[:n_nodes]], dim=1)
                x_1 = torch.cat([x_1, pe[n_nodes:(n_nodes + n_edges)]], dim=1)
                x_2 = torch.cat([x_2, pe[(n_nodes + n_edges):]], dim=1)
            else:
                x_0 = x_t
            self.data.append((x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, y_t))
        self.rk0_dim = self.data[0][0].shape[1]
        self.rk1_dim = self.data[0][1].shape[1]
        self.rk2_dim = self.data[0][2].shape[1]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def sparse_block_diag(sparse_list):
    rows, cols, vals = [], [], []
    row_offset = col_offset = 0
    for S in sparse_list:
        S = S.coalesce()
        i, v = S.indices(), S.values()
        rows.append(i[0] + row_offset); cols.append(i[1] + col_offset); vals.append(v)
        row_offset += S.shape[0]; col_offset += S.shape[1]
    return torch.sparse_coo_tensor(
        torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals),
        size=(row_offset, col_offset))


def collate(batch):
    x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, y = zip(*batch)
    return (
        torch.cat(x_0, dim=0),
        torch.cat(x_1, dim=0),
        torch.cat(x_2, dim=0),
        sparse_block_diag(adj00), sparse_block_diag(icd01), sparse_block_diag(adj11),
        sparse_block_diag(icd02), sparse_block_diag(icd12), sparse_block_diag(adj22),
        torch.tensor([xi.shape[0] for xi in x_0], dtype=torch.long),
        torch.stack(y, dim=0),
    )


def make_model(rk0_dim, rk1_dim, rk2_dim):
    return CellularTransformer(
        rk0_dim=rk0_dim, rk1_dim=rk1_dim, rk2_dim=rk2_dim,
        output_dim=3, num_layers=4, hidden_dim=80, num_heads=8,
        hidden_dim_per_head=10, att_dropout=0, emb_dropout=0,
        readout_dropout=0, num_readout_hidden_layers=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device} | use_pe: {args.use_pe}")

    BATCH_SIZE = 32
    train_dataset = GBChard(root=".", split="train", use_pe=args.use_pe)
    val_dataset   = GBChard(root=".", split="val",   use_pe=args.use_pe)
    test_dataset  = GBChard(root=".", split="test",  use_pe=args.use_pe)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    criterion = torch.nn.CrossEntropyLoss()
    results = {"use_pe": args.use_pe, "epochs": args.epochs, "runs": []}

    for run in range(3):
        print(f"\n--- Run {run + 1}/3 ---")
        model = make_model(train_dataset.rk0_dim, train_dataset.rk1_dim, train_dataset.rk2_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        run_result = {"run": run + 1, "train_losses": [], "val_accs": []}

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                optimizer.zero_grad()
                out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            run_result["train_losses"].append(round(train_loss, 4))
            print(f"Epoch {epoch+1:3d}  loss={train_loss:.4f}", end="")

            if (epoch + 1) % 5 == 0:
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                        out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                        preds.append(out.argmax(dim=1).cpu())
                        targets.append(y.argmax(dim=1).cpu())
                val_acc = (torch.cat(preds) == torch.cat(targets)).float().mean().item()
                run_result["val_accs"].append(round(val_acc, 4))
                print(f"  |  val acc={val_acc*100:.2f}%", end="")
            print()

        model.eval()
        preds, targets = [], []
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts, y = [b.to(device) for b in batch]
                out = model(x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts)
                total_loss += criterion(out, y).item()
                preds.append(out.argmax(dim=1).cpu())
                targets.append(y.argmax(dim=1).cpu())
        test_acc  = (torch.cat(preds) == torch.cat(targets)).float().mean().item()
        test_loss = total_loss / len(test_loader)
        run_result["test_acc"]  = round(test_acc, 4)
        run_result["test_loss"] = round(test_loss, 4)
        print(f"Test acc={test_acc*100:.2f}%  loss={test_loss:.4f}")
        results["runs"].append(run_result)

    results["mean_test_acc"]  = round(sum(r["test_acc"]  for r in results["runs"]) / 3, 4)
    results["mean_test_loss"] = round(sum(r["test_loss"] for r in results["runs"]) / 3, 4)
    print(f"\nMean test acc: {results['mean_test_acc']*100:.2f}%")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")
