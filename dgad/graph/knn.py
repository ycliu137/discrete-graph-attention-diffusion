"""Feature-space KNN graph construction for dynamic rewiring."""

import torch


def knn_graph(
    feature_matrix,
    k_min=0,
    k_max=10,
    self_edge=False,
    remov_edge_prob=None,
):
    """
    Build directed KNN edges from node features.

    Args:
        feature_matrix: (N, F) node features
        k_min, k_max: neighbor rank range (exclusive upper bound style as in CellDiffusion)
        self_edge: include self-loops when True
        remov_edge_prob: if set, randomly drop edges with this probability

    Returns:
        edge_index: (2, E) with edge_index[0]=source, edge_index[1]=target
    """
    dist_matrix = torch.cdist(feature_matrix, feature_matrix, p=2)

    if self_edge:
        knn_indices = torch.argsort(dist_matrix, dim=1)[:, k_min:k_max]
        edge_index = knn_indices_to_edge_index(knn_indices)
    else:
        knn_indices = torch.argsort(dist_matrix, dim=1)[:, k_min + 1 : k_max + 1]
        edge_index = knn_indices_to_edge_index(knn_indices)
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

    if remov_edge_prob is not None:
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > remov_edge_prob
        edge_index = edge_index[:, mask]

    return edge_index


def knn_indices_to_edge_index(knn_indices):
    """Convert (N, k) neighbor indices to PyG-style edge_index (2, N*k)."""
    num_points, k = knn_indices.shape
    src_nodes = torch.arange(num_points, device=knn_indices.device).view(-1, 1).repeat(1, k).view(-1)
    trg_nodes = knn_indices.reshape(-1)
    return torch.stack([src_nodes, trg_nodes], dim=0)


def features_to_edge_index_knn_no_self_edge(feature_matrix, rebuild_args):
    """
    KNN graph for one diffusion step (no self edges).

    rebuild_args: (k_min, k_max, remov_edge_prob) or dict with those keys.
    """
    if isinstance(rebuild_args, dict):
        k_min = rebuild_args["k_min"]
        k_max = rebuild_args["k_max"]
        remov_edge_prob = rebuild_args.get("remov_edge_prob")
    else:
        k_min, k_max, remov_edge_prob = rebuild_args

    return knn_graph(
        feature_matrix,
        k_min=k_min,
        k_max=k_max,
        self_edge=False,
        remov_edge_prob=remov_edge_prob,
    )
