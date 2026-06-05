"""Distance-based attention (attention_type='dist').

DGAD-specific edge scoring from learned weighted feature-space distance.
See README "Attention types" for the formula and discussion.
"""

import torch
import torch.nn as nn

from dgad.attention.softmax import neighborhood_aware_softmax


class AttentionDistance(nn.Module):
    """
    Attention from weighted feature-space distance between neighbors.

    e_ij derived from weighted squared differences of (h_target - h_source).

    Reference: DGAD distance-based edge scoring (see README).
    """

    src_nodes_dim = 1
    trg_nodes_dim = 0
    nodes_dim = 0

    def __init__(self, num_features, num_heads, recover=False, edge_dims_weights=None, distance_dims_weights=None):
        super().__init__()
        if recover:
            self.edge_dims_weights = edge_dims_weights
            self.distance_dims_weights = distance_dims_weights
        else:
            self.edge_dims_weights = nn.Parameter(torch.Tensor(1, num_heads, num_features))
            self.distance_dims_weights = nn.Parameter(torch.Tensor(1, num_heads, num_features))
            nn.init.xavier_uniform_(self.edge_dims_weights)
            nn.init.xavier_uniform_(self.distance_dims_weights)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, data):
        nodes_features, edge_index = data
        num_of_nodes = nodes_features.shape[self.nodes_dim]

        edge_vectors, nodes_features_source = self._edge_vectors(nodes_features, edge_index)
        edge_vectors_weighted = edge_vectors * self.edge_dims_weights
        edge_distance_vectors = torch.square(edge_vectors_weighted)
        edge_distances = (edge_distance_vectors * self.distance_dims_weights).sum(dim=-1)

        edge_distance_mean = edge_distances.mean(dim=0, keepdim=True)
        edge_scores = -1.0 * self.leaky_relu(edge_distances + edge_distance_mean)
        attentions_per_edge = neighborhood_aware_softmax(
            edge_scores, edge_index[self.trg_nodes_dim], num_of_nodes, nodes_dim=self.nodes_dim
        )
        return attentions_per_edge, nodes_features_source

    def _edge_vectors(self, nodes_features, edge_index):
        src_index = edge_index[self.src_nodes_dim]
        trg_index = edge_index[self.trg_nodes_dim]
        sources = nodes_features.index_select(self.nodes_dim, src_index)
        targets = nodes_features.index_select(self.nodes_dim, trg_index)
        return targets - sources, sources
