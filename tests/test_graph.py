"""Tests for graph construction and adjacency utilities."""

import torch

from dgad.graph.adjacency import edge_index_to_adj
from dgad.graph.knn import (
    features_to_edge_index_knn_no_self_edge,
    knn_graph,
    knn_indices_to_edge_index,
)


def test_knn_graph_shape():
    x = torch.randn(10, 4)
    edge_index = knn_graph(x, k_min=0, k_max=3, self_edge=False)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0


def test_knn_graph_excludes_self_edges():
    x = torch.randn(10, 4)
    edge_index = knn_graph(x, k_min=0, k_max=3, self_edge=False)
    assert (edge_index[0] == edge_index[1]).sum() == 0


def test_knn_graph_includes_self_edges():
    x = torch.randn(10, 4)
    edge_index = knn_graph(x, k_min=0, k_max=3, self_edge=True)
    assert (edge_index[0] == edge_index[1]).sum() > 0


def test_knn_indices_to_edge_index():
    knn_indices = torch.tensor([[1, 2], [0, 2], [0, 1]])
    edge_index = knn_indices_to_edge_index(knn_indices)
    assert edge_index.shape == (2, 6)
    assert edge_index[0].tolist() == [0, 0, 1, 1, 2, 2]
    assert edge_index[1].tolist() == [1, 2, 0, 2, 0, 1]


def test_features_to_edge_index_knn_no_self_edge_tuple_args():
    x = torch.randn(5, 8)
    edge_index = features_to_edge_index_knn_no_self_edge(x, (0, 3, None))
    assert edge_index.shape[0] == 2
    assert (edge_index[0] == edge_index[1]).sum() == 0


def test_features_to_edge_index_knn_no_self_edge_dict_args():
    x = torch.randn(5, 8)
    edge_index = features_to_edge_index_knn_no_self_edge(
        x, {"k_min": 0, "k_max": 3, "remov_edge_prob": None}
    )
    assert edge_index.shape[0] == 2


def test_knn_graph_chunked_matches_full():
    torch.manual_seed(0)
    x = torch.randn(20, 8)
    edge_full = knn_graph(x, k_min=0, k_max=5, self_edge=False)
    edge_chunked = knn_graph(x, k_min=0, k_max=5, self_edge=False, chunk_size=7)
    assert torch.equal(edge_full, edge_chunked)


def test_edge_index_to_adj():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    adj = edge_index_to_adj(edge_index, num_nodes=3)
    expected = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    assert torch.equal(adj, expected)
