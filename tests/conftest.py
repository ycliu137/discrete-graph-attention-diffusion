"""Shared fixtures for DGAD tests."""

import torch
import pytest

from dgad.graph.knn import knn_graph


@pytest.fixture
def small_graph():
    """Small synthetic graph: 8 nodes, 16 features, KNN edges."""
    torch.manual_seed(42)
    n, f = 8, 16
    x = torch.randn(n, f)
    edge_index = knn_graph(x, k_min=0, k_max=4, self_edge=False)
    return x, edge_index, n, f
