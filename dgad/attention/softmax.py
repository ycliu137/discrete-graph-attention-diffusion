"""Neighborhood-aware softmax utilities for sparse graph attention."""

import torch


def neighborhood_aware_softmax(scores_per_edge, trg_index, num_of_nodes, nodes_dim=0):
    """
    Softmax over incoming edges for each target node (sparse neighborhood).

    scores_per_edge: (E, NH)
    trg_index: (E,) target node indices
    """
    node_max = _scatter_reduce(scores_per_edge, trg_index, num_of_nodes, nodes_dim, reduce="amax")
    max_per_edge = node_max.index_select(nodes_dim, trg_index)
    scores_per_edge = scores_per_edge - max_per_edge
    exp_scores_per_edge = scores_per_edge.exp()

    denominator = sum_edge_scores_neighborhood_aware(
        exp_scores_per_edge, trg_index, num_of_nodes, nodes_dim=nodes_dim
    )
    attentions_per_edge = exp_scores_per_edge / (denominator + 1e-16)
    return attentions_per_edge.unsqueeze(-1)


def sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes, nodes_dim=0):
    trg_index_broadcasted = trg_index.unsqueeze(-1).expand_as(exp_scores_per_edge)

    size = list(exp_scores_per_edge.shape)
    size[nodes_dim] = num_of_nodes
    neighborhood_sums = torch.zeros(
        size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device
    )
    neighborhood_sums.scatter_add_(nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
    return neighborhood_sums.index_select(nodes_dim, trg_index)


def _scatter_reduce(values, index, num_of_nodes, nodes_dim, reduce):
    size = list(values.shape)
    size[nodes_dim] = num_of_nodes
    init = float("-inf") if reduce == "amax" else 0.0
    out = torch.full(size, init, dtype=values.dtype, device=values.device)
    index_broadcasted = index.unsqueeze(-1).expand_as(values)
    out.scatter_reduce_(nodes_dim, index_broadcasted, values, reduce=reduce, include_self=True)
    return out


def explicit_broadcast(this, other):
    for _ in range(this.dim(), other.dim()):
        this = this.unsqueeze(-1)
    return this.expand_as(other)
