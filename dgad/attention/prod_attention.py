"""Bilinear / inner-product attention (attention_type='prod')."""

import torch
import torch.nn as nn

from dgad.attention.softmax import neighborhood_aware_softmax


class AttentionInnerProduct(nn.Module):
    """
    Bilinear attention: e_ij = LeakyReLU(h_j^T W_h h_i).

    Sparse-graph variant inspired by Transformer dot-product attention (Vaswani et al., 2017).
    Only the target side is projected by W_h (asymmetric).
    """

    src_nodes_dim = 1
    trg_nodes_dim = 0
    nodes_dim = 0

    def __init__(self, num_features, num_heads, recover=False, metric_weights=None):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads

        if recover:
            self.metric_weights = metric_weights
        else:
            self.metric_weights = nn.Parameter(
                torch.Tensor(1, num_heads, num_features, num_features)
            )
            nn.init.xavier_uniform_(self.metric_weights)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, data):
        nodes_features, edge_index = data
        num_of_nodes = nodes_features.shape[self.nodes_dim]

        nodes_features_source, nodes_features_target = self._lift(nodes_features, edge_index)
        nodes_features_target = nodes_features_target.view(-1, 1, self.num_features, 1)
        nodes_features_target = (
            self.metric_weights.matmul(nodes_features_target)
            .view(-1, self.num_heads, self.num_features)
        )

        edge_scores = (nodes_features_source * nodes_features_target).sum(dim=-1)
        edge_scores = self.leaky_relu(edge_scores)
        attentions_per_edge = neighborhood_aware_softmax(
            edge_scores, edge_index[self.trg_nodes_dim], num_of_nodes, nodes_dim=self.nodes_dim
        )
        return attentions_per_edge, nodes_features_source

    def _lift(self, nodes_features, edge_index):
        src_index = edge_index[self.src_nodes_dim]
        trg_index = edge_index[self.trg_nodes_dim]
        sources = nodes_features.index_select(self.nodes_dim, src_index)
        targets = nodes_features.index_select(self.nodes_dim, trg_index)
        return sources, targets
