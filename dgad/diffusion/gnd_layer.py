"""Single step of discrete graph attention diffusion (DGAD layer)."""

import torch
import torch.nn as nn

from dgad.attention import (
    AttentionWeightSum,
    AttentionInnerProduct,
    AttentionDotProduct,
    AttentionDistance,
)
from dgad.attention.softmax import explicit_broadcast
from dgad.graph.knn import features_to_edge_index_knn_no_self_edge


class GNDLayer(nn.Module):
    """
    One Euler step: x^{k+1} = tau * Agg(attn, x^k) + (1 - tau) * x^k.

    Attention types: sum (GAT), prod (bilinear), dot (Transformer), dist (feature distance).
    """

    src_nodes_dim = 1
    trg_nodes_dim = 0
    nodes_dim = 0

    def __init__(
        self,
        num_features,
        num_heads,
        time_increment,
        attention_type="sum",
        activation=nn.ELU(),
        dropout_prob=0.0,
        log_attention_weights=False,
        rebuild_graph=False,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.time_increment = time_increment
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.rebuild_graph = rebuild_graph
        self.log_attention_weights = log_attention_weights
        self.attention_weights = None
        self.attention_layer = self._build_attention(attention_type)

    def forward(self, data):
        if self.rebuild_graph:
            in_features, rebuild_args = data
            edge_index = features_to_edge_index_knn_no_self_edge(in_features, rebuild_args)
        else:
            in_features, edge_index = data

        num_nodes = in_features.shape[self.nodes_dim]
        in_features = self.dropout(in_features)
        in_features = in_features.view(-1, 1, self.num_features)

        data = (in_features, edge_index)
        attentions_per_edge, nodes_features_source = self.attention_layer(data)
        attentions_per_edge = self.dropout(attentions_per_edge)

        weighted = nodes_features_source * attentions_per_edge
        aggregated = self._aggregate_neighbors(weighted, edge_index, in_features, num_nodes)
        aggregated = aggregated.mean(dim=1, keepdim=True)

        out_features = self._propagate(in_features, aggregated)

        if self.log_attention_weights:
            self.attention_weights = attentions_per_edge

        self.last_edge_index = edge_index

        if self.rebuild_graph:
            return out_features, rebuild_args
        return out_features, edge_index

    def _aggregate_neighbors(self, weighted, edge_index, in_features, num_nodes):
        size = list(weighted.shape)
        size[self.nodes_dim] = num_nodes
        out = torch.zeros(size, dtype=in_features.dtype, device=in_features.device)
        trg_broadcast = explicit_broadcast(edge_index[self.trg_nodes_dim], weighted)
        out.scatter_add_(self.nodes_dim, trg_broadcast, weighted)
        return out

    def _propagate(self, in_features, aggregated):
        tau = self.time_increment
        if isinstance(tau, (int, float)):
            out = tau * aggregated + (1.0 - tau) * in_features
        else:
            tau_t = torch.tensor(tau, dtype=in_features.dtype, device=in_features.device)
            repeat = list(in_features.shape)
            repeat[0] = 1
            tau_t = tau_t.repeat(*repeat).reshape(in_features.shape)
            out = tau_t * aggregated + (1.0 - tau_t) * in_features
        return out.view(-1, self.num_features)

    def _build_attention(self, attention_type):
        if attention_type == "sum":
            return AttentionWeightSum(self.num_features, self.num_heads)
        if attention_type == "prod":
            return AttentionInnerProduct(self.num_features, self.num_heads)
        if attention_type == "dot":
            return AttentionDotProduct(self.num_features, self.num_heads)
        if attention_type == "dist":
            return AttentionDistance(self.num_features, self.num_heads)
        raise ValueError(
            f'attention_type must be one of ("sum", "prod", "dot", "dist"), got "{attention_type}".'
        )
