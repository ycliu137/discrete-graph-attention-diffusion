"""Feature-space KNN graph construction for dynamic rewiring."""

import torch


def knn_graph(
    feature_matrix,
    k_min=0,
    k_max=10,
    self_edge=False,
    remov_edge_prob=None,
    chunk_size=None,
):
    """
    Build directed KNN edges from node features.

    Args:
        feature_matrix: (N, F) node features
        k_min, k_max: neighbor rank range (exclusive upper bound style as in CellDiffusion)
        self_edge: include self-loops when True
        remov_edge_prob: if set, randomly drop edges with this probability
        chunk_size: if set, compute distances in row chunks to reduce peak memory

    Returns:
        edge_index: (2, E) with edge_index[0]=source, edge_index[1]=target
    """
    num_nodes = feature_matrix.size(0)
    if chunk_size is not None and chunk_size < num_nodes:
        knn_indices = _knn_indices_chunked(feature_matrix, k_min, k_max, self_edge, chunk_size)
    else:
        dist_matrix = torch.cdist(feature_matrix, feature_matrix, p=2)
        knn_indices = _knn_indices_from_dist(dist_matrix, k_min, k_max, self_edge)

    edge_index = knn_indices_to_edge_index(knn_indices)

    if not self_edge:
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]

    if remov_edge_prob is not None:
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > remov_edge_prob
        edge_index = edge_index[:, mask]

    return edge_index


def _knn_indices_from_dist(dist_matrix, k_min, k_max, self_edge):
    num_nodes = dist_matrix.size(0)
    k_pick = min(k_max if self_edge else k_max + 1, num_nodes)
    _, sorted_indices = torch.topk(dist_matrix, k=k_pick, dim=1, largest=False, sorted=True)
    if self_edge:
        return sorted_indices[:, k_min:k_max]
    return sorted_indices[:, k_min + 1 : k_max + 1]


def _knn_indices_chunked(feature_matrix, k_min, k_max, self_edge, chunk_size):
    num_nodes = feature_matrix.size(0)
    chunks = []
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        chunk_dist = torch.cdist(feature_matrix[start:end], feature_matrix, p=2)
        chunks.append(_knn_indices_from_dist(chunk_dist, k_min, k_max, self_edge))
    return torch.cat(chunks, dim=0)


def knn_indices_to_edge_index(knn_indices):
    """Convert (N, k) neighbor indices to PyG-style edge_index (2, N*k)."""
    num_points, k = knn_indices.shape
    src_nodes = torch.arange(num_points, device=knn_indices.device).repeat_interleave(k)
    trg_nodes = knn_indices.reshape(-1)
    return torch.stack([src_nodes, trg_nodes], dim=0)


def features_to_edge_index_knn_no_self_edge(feature_matrix, rebuild_args):
    """
    KNN graph for one diffusion step (no self edges).

    rebuild_args: (k_min, k_max, remov_edge_prob) or dict with those keys.
    Optional dict key ``chunk_size`` enables chunked distance computation.
    """
    if isinstance(rebuild_args, dict):
        k_min = rebuild_args["k_min"]
        k_max = rebuild_args["k_max"]
        remov_edge_prob = rebuild_args.get("remov_edge_prob")
        chunk_size = rebuild_args.get("chunk_size")
    else:
        k_min, k_max, remov_edge_prob = rebuild_args
        chunk_size = None

    return knn_graph(
        feature_matrix,
        k_min=k_min,
        k_max=k_max,
        self_edge=False,
        remov_edge_prob=remov_edge_prob,
        chunk_size=chunk_size,
    )
