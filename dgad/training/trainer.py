"""Task-agnostic training loop for DGAD."""

import torch
import torch.nn.functional as F

from dgad.graph.adjacency import edge_index_labels, edge_index_to_adj


def _adjacency_targets(model, edge_index, target_adj, num_nodes):
    """Build dense or sparse adjacency targets depending on decoder mode."""
    sparse = getattr(model, "sparse_adj_decoder", False)
    if sparse:
        return edge_index_labels(edge_index, dense_adj=target_adj)
    if target_adj is None:
        target_adj = edge_index_to_adj(edge_index, num_nodes)
    return target_adj


def fit_dgad(
    model,
    node_features,
    edge_index=None,
    target_features=None,
    target_adj=None,
    *,
    edge_rewire=False,
    edge_rewire_args=None,
    max_epochs=2000,
    lr=1e-3,
    loss_adj=0.0,
    loss_reduction="sum",
    device="cpu",
    verbose=True,
    log_every=50,
):
    """
    Train a DGADModel on node features (and optionally adjacency).

    Args:
        model: DGADModel instance
        node_features: (N, F_in) tensor
        edge_index: (2, E) required if edge_rewire=False
        target_features: reconstruction target for MSE (default: node_features)
        target_adj: dense adjacency for BCE loss when loss_adj > 0
        edge_rewire: rebuild KNN graph each diffusion step
        edge_rewire_args: dict with k_min, k_max, remov_edge_prob

    Returns:
        dict with keys: model, loss_history, embedding
    """
    model = model.to(device)
    node_features = node_features.to(device)

    if target_features is None:
        target_features = node_features
    else:
        target_features = target_features.to(device)

    if edge_rewire:
        if edge_rewire_args is None:
            edge_rewire_args = {"k_min": 0, "k_max": 50, "remov_edge_prob": None}
        data = (node_features, edge_rewire_args)
    else:
        if edge_index is None:
            raise ValueError("edge_index is required when edge_rewire=False")
        edge_index = edge_index.to(device)
        data = (node_features, edge_index)

    adj_targets = None
    if loss_adj > 0:
        adj_edge_index = edge_index
        if adj_edge_index is None:
            raise ValueError("target_adj or edge_index required for adjacency loss")
        if target_adj is not None:
            target_adj = target_adj.to(device)
        num_nodes = node_features.shape[0]
        adj_targets = _adjacency_targets(model, adj_edge_index, target_adj, num_nodes)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out_features, recon_adj, embedding = model(data)

        if loss_adj == 1.0:
            loss = F.binary_cross_entropy_with_logits(
                recon_adj, adj_targets, reduction=loss_reduction
            )
        elif loss_adj == 0.0:
            loss = F.mse_loss(out_features, target_features, reduction=loss_reduction)
        else:
            loss_1 = F.binary_cross_entropy_with_logits(
                recon_adj, adj_targets, reduction=loss_reduction
            )
            loss_2 = F.mse_loss(out_features, target_features, reduction=loss_reduction)
            fold = loss_1.detach() / loss_2.detach().clamp_min(1e-12)
            loss = loss_adj * loss_1 + (1.0 - loss_adj) * fold * loss_2

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if verbose and epoch % log_every == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, loss={loss.item():.4f}")

    return {
        "model": model,
        "loss_history": loss_history,
        "embedding": embedding.detach(),
    }
