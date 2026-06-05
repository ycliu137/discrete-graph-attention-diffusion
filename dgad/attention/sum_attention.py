"""Additive edge attention (attention_type='sum').

Reference: Veličković et al., Graph Attention Networks, ICLR 2018.
https://arxiv.org/abs/1710.10903
"""

import torch
import torch.nn as nn

from dgad.attention.softmax import neighborhood_aware_softmax


class AttentionWeightSum(nn.Module):
    """
    Additive attention: e_ij = LeakyReLU(a_s^T h_j + a_t^T h_i).

    Learnable source and target scoring vectors per attention head.

    Reference: Veličković et al., Graph Attention Networks, ICLR 2018.
    """

    src_nodes_dim = 1
    trg_nodes_dim = 0
    nodes_dim = 0

    def __init__(self, num_features, num_heads, recover=False, scoring_fn_target=None, scoring_fn_source=None):
        super().__init__()
        if recover:
            self.scoring_fn_target = scoring_fn_target
            self.scoring_fn_source = scoring_fn_source
        else:
            self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_heads, num_features))
            self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_heads, num_features))
            nn.init.xavier_uniform_(self.scoring_fn_target)
            nn.init.xavier_uniform_(self.scoring_fn_source)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, data):
        nodes_features, edge_index = data
        num_of_nodes = nodes_features.shape[self.nodes_dim]

        scores_source = (nodes_features * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features * self.scoring_fn_target).sum(dim=-1)
        scores_lifted, src_features_lifted = self._lift(scores_source, scores_target, nodes_features, edge_index)
        scores_per_edge = self.leaky_relu(scores_lifted)
        attentions_per_edge = neighborhood_aware_softmax(
            scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes, nodes_dim=self.nodes_dim
        )
        return attentions_per_edge, src_features_lifted

    def _lift(self, scores_source, scores_target, nodes_features, edge_index):
        src_index = edge_index[self.src_nodes_dim]
        trg_index = edge_index[self.trg_nodes_dim]
        scores_source_lifted = scores_source.index_select(self.nodes_dim, src_index)
        scores_target_lifted = scores_target.index_select(self.nodes_dim, trg_index)
        src_features_lifted = nodes_features.index_select(self.nodes_dim, src_index)
        return scores_source_lifted + scores_target_lifted, src_features_lifted
