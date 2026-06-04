"""Adjacency matrix utilities."""

import torch


def edge_index_to_adj(edge_index, num_nodes):
    """Dense 0/1 adjacency from edge_index (2, E)."""
    adjacency = torch.zeros(
        (num_nodes, num_nodes), dtype=edge_index.dtype, device=edge_index.device
    )
    adjacency[edge_index[0], edge_index[1]] = 1
    return adjacency
