"""Neighborhood-aware softmax utilities for sparse graph attention."""

import torch


def neighborhood_aware_softmax(scores_per_edge, trg_index, num_of_nodes, nodes_dim=0):
    """
    Softmax over incoming edges for each target node (sparse neighborhood).

    scores_per_edge: (E, NH)
    trg_index: (E,) target node indices
    """
    scores_per_edge = scores_per_edge - scores_per_edge.max()
    exp_scores_per_edge = scores_per_edge.exp()

    denominator = sum_edge_scores_neighborhood_aware(
        exp_scores_per_edge, trg_index, num_of_nodes, nodes_dim=nodes_dim
    )
    attentions_per_edge = exp_scores_per_edge / (denominator + 1e-16)
    return attentions_per_edge.unsqueeze(-1)


def sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes, nodes_dim=0):
    trg_index_broadcasted = explicit_broadcast(trg_index, exp_scores_per_edge)

    size = list(exp_scores_per_edge.shape)
    size[nodes_dim] = num_of_nodes
    neighborhood_sums = torch.zeros(
        size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device
    )
    neighborhood_sums.scatter_add_(nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
    return neighborhood_sums.index_select(nodes_dim, trg_index)


def explicit_broadcast(this, other):
    for _ in range(this.dim(), other.dim()):
        this = this.unsqueeze(-1)
    return this.expand_as(other)
