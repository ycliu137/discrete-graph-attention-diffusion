"""Adjacency matrix utilities."""

import torch


def edge_index_to_adj(edge_index, num_nodes):
    """Dense 0/1 adjacency from edge_index (2, E)."""
    adjacency = torch.zeros(
        (num_nodes, num_nodes), dtype=edge_index.dtype, device=edge_index.device
    )
    adjacency[edge_index[0], edge_index[1]] = 1
    return adjacency


def edge_index_labels(edge_index, dense_adj=None):
    """Per-edge labels for sparse adjacency loss."""
    if dense_adj is not None:
        return dense_adj[edge_index[0], edge_index[1]]
    return torch.ones(
        edge_index.size(1),
        dtype=torch.float32,
        device=edge_index.device,
    )
