"""Tests for DGADModel and training utilities."""

import torch
import pytest

from dgad.models.dgad_model import DGADModel
from dgad.training.trainer import fit_dgad


def test_dgad_model_forward_shapes(small_graph):
    x, edge_index, n, f = small_graph
    model = DGADModel(num_features=f, num_heads=2, num_steps=2)
    out_features, recon_adj, embedding = model((x, edge_index))
    assert out_features.shape == (n, f)
    assert recon_adj.shape == (n, n)
    assert embedding.shape == (n, f)


def test_dgad_model_with_encoder_decoder(small_graph):
    x, edge_index, n, f_in = small_graph
    f_dif = 12
    model = DGADModel(
        num_features=f_dif,
        num_heads=2,
        num_steps=2,
        encoder=[f_in, f_dif],
        decoder=[f_dif, f_in],
    )
    out_features, recon_adj, embedding = model((x, edge_index))
    assert out_features.shape == (n, f_in)
    assert embedding.shape == (n, f_dif)
    assert recon_adj.shape == (n, n)


def test_fit_dgad_runs(small_graph):
    x, edge_index, n, f_in = small_graph
    f_dif = 8
    model = DGADModel(
        num_features=f_dif,
        num_heads=2,
        num_steps=2,
        encoder=[f_in, f_dif],
        decoder=[f_dif, f_in],
    )
    result = fit_dgad(
        model,
        x,
        edge_index=edge_index,
        max_epochs=5,
        lr=1e-2,
        device="cpu",
        verbose=False,
    )
    assert "model" in result
    assert "loss_history" in result
    assert "embedding" in result
    assert len(result["loss_history"]) == 5
    assert result["embedding"].shape == (n, f_dif)


def test_dgad_model_sparse_adj_decoder(small_graph):
    x, edge_index, n, f = small_graph
    model = DGADModel(num_features=f, num_heads=2, num_steps=2, sparse_adj_decoder=True)
    out_features, recon_adj, embedding = model((x, edge_index))
    assert out_features.shape == (n, f)
    assert recon_adj.shape == (edge_index.shape[1],)
    assert embedding.shape == (n, f)


def test_fit_dgad_sparse_adj_loss(small_graph):
    x, edge_index, n, f_in = small_graph
    f_dif = 8
    model = DGADModel(
        num_features=f_dif,
        num_heads=2,
        num_steps=2,
        encoder=[f_in, f_dif],
        decoder=[f_dif, f_in],
        sparse_adj_decoder=True,
    )
    result = fit_dgad(
        model,
        x,
        edge_index=edge_index,
        max_epochs=3,
        lr=1e-2,
        loss_adj=1.0,
        device="cpu",
        verbose=False,
    )
    assert len(result["loss_history"]) == 3


def test_fit_dgad_requires_edge_index_when_not_rewiring():
    x = torch.randn(4, 8)
    model = DGADModel(num_features=8, num_heads=2, num_steps=1)
    with pytest.raises(ValueError, match="edge_index is required"):
        fit_dgad(model, x, edge_rewire=False, max_epochs=1, verbose=False)
