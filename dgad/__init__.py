"""
DGAD: Discrete Graph Attention Diffusion.

General GNN backbone (GND) for node embedding on fixed or feature-built graphs.
Use with supervised, semi-supervised, or self-supervised downstream tasks.
"""

from dgad.diffusion import GND, GNDLayer
from dgad.models import GraphFeatureEncoder, InnerProductDecoder
from dgad.models.dgad_model import DGADModel
from dgad.graph import knn_graph, edge_index_to_adj, features_to_edge_index_knn_no_self_edge
from dgad.attention import (
    AttentionWeightSum,
    AttentionInnerProduct,
    AttentionDotProduct,
    AttentionDistance,
)
from dgad.training import fit_dgad

__version__ = "0.1.0"

__all__ = [
    "GND",
    "GNDLayer",
    "DGADModel",
    "GraphFeatureEncoder",
    "InnerProductDecoder",
    "knn_graph",
    "edge_index_to_adj",
    "features_to_edge_index_knn_no_self_edge",
    "AttentionWeightSum",
    "AttentionInnerProduct",
    "AttentionDotProduct",
    "AttentionDistance",
    "fit_dgad",
]
