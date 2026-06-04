"""Tests for attention modules and neighborhood softmax."""

import pytest
import torch

from dgad.attention.dist_attention import AttentionDistance
from dgad.attention.prod_attention import AttentionInnerProduct
from dgad.attention.softmax import neighborhood_aware_softmax
from dgad.attention.sum_attention import AttentionWeightSum


def _attention_sums_to_one(attentions, edge_index, num_nodes):
    """Sum attention weights per target node; each should be ~1."""
    trg = edge_index[0]  # GND uses edge_index[0] as aggregation target
    num_heads = attentions.shape[1]
    for head in range(num_heads):
        sums = torch.zeros(num_nodes)
        for e in range(edge_index.shape[1]):
            sums[trg[e]] += attentions[e, head, 0].item()
        active = sums > 0
        assert torch.allclose(sums[active], torch.ones(active.sum()), atol=1e-5)


def test_neighborhood_aware_softmax_sums_to_one():
    torch.manual_seed(0)
    num_nodes = 4
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
    scores = torch.randn(edge_index.shape[1], 2)
    attentions = neighborhood_aware_softmax(scores, edge_index[0], num_nodes)
    _attention_sums_to_one(attentions, edge_index, num_nodes)


def test_attention_weight_sum_forward(small_graph):
    x, edge_index, n, f = small_graph
    layer = AttentionWeightSum(f, num_heads=2)
    x_3d = x.view(-1, 1, f)
    attentions, src_features = layer((x_3d, edge_index))
    assert attentions.shape[0] == edge_index.shape[1]
    assert attentions.shape[1] == 2
    assert src_features.shape[0] == edge_index.shape[1]
    _attention_sums_to_one(attentions, edge_index, n)


@pytest.mark.parametrize("attention_cls", [AttentionInnerProduct, AttentionDistance])
def test_attention_variants_forward(attention_cls, small_graph):
    x, edge_index, n, f = small_graph
    layer = attention_cls(f, num_heads=2)
    x_3d = x.view(-1, 1, f)
    attentions, src_features = layer((x_3d, edge_index))
    assert attentions.shape[0] == edge_index.shape[1]
    assert src_features.shape[0] == edge_index.shape[1]
    _attention_sums_to_one(attentions, edge_index, n)
