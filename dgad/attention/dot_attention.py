"""Scaled dot-product edge attention (attention_type='dot')."""

import math

import torch
import torch.nn as nn

from dgad.attention.softmax import neighborhood_aware_softmax


class AttentionDotProduct(nn.Module):
    """
    Scaled dot-product attention: e_ij = (Q h_i)^T (K h_j) / sqrt(d).

    Separate learnable query and key projections per head. Unlike ``prod``,
    both sides are projected before the inner product.
    """

    src_nodes_dim = 1
    trg_nodes_dim = 0
    nodes_dim = 0

    def __init__(
        self,
        num_features,
        num_heads,
        recover=False,
        query_weights=None,
        key_weights=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.scale = math.sqrt(num_features)

        if recover:
            self.query_weights = query_weights
            self.key_weights = key_weights
        else:
            self.query_weights = nn.Parameter(
                torch.Tensor(1, num_heads, num_features, num_features)
            )
            self.key_weights = nn.Parameter(
                torch.Tensor(1, num_heads, num_features, num_features)
            )
            nn.init.xavier_uniform_(self.query_weights)
            nn.init.xavier_uniform_(self.key_weights)

    def forward(self, data):
        nodes_features, edge_index = data
        num_of_nodes = nodes_features.shape[self.nodes_dim]

        queries = self._project(nodes_features, self.query_weights)
        keys = self._project(nodes_features, self.key_weights)

        trg_index = edge_index[self.trg_nodes_dim]
        src_index = edge_index[self.src_nodes_dim]
        q_edge = queries.index_select(self.nodes_dim, trg_index)
        k_edge = keys.index_select(self.nodes_dim, src_index)
        sources = nodes_features.index_select(self.nodes_dim, src_index)

        edge_scores = (q_edge * k_edge).sum(dim=-1) / self.scale
        attentions_per_edge = neighborhood_aware_softmax(
            edge_scores, edge_index[self.trg_nodes_dim], num_of_nodes, nodes_dim=self.nodes_dim
        )
        return attentions_per_edge, sources

    def _project(self, nodes_features, weight):
        x = nodes_features.view(-1, 1, self.num_features, 1)
        return weight.matmul(x).view(-1, self.num_heads, self.num_features)
