"""Tests for GND and GNDLayer diffusion modules."""

import torch
import pytest

from dgad.diffusion.gnd import GND
from dgad.diffusion.gnd_layer import GNDLayer


@pytest.mark.parametrize("attention_type", ["sum", "prod", "dist"])
def test_gnd_forward_fixed_graph(small_graph, attention_type):
    x, edge_index, n, f = small_graph
    gnd = GND(num_features=f, num_heads=2, num_steps=3, attention_type=attention_type)
    out, embedding = gnd((x, edge_index))
    assert out[0].shape == (n, f)
    assert embedding.shape == (n, f)


def test_gnd_preserves_feature_dim_with_encoder_decoder(small_graph):
    x, edge_index, n, f_in = small_graph
    f_dif = 12
    gnd = GND(
        num_features=f_dif,
        num_heads=2,
        num_steps=2,
        encoder=[f_in, f_dif],
        decoder=[f_dif, f_in],
    )
    out, embedding = gnd((x, edge_index))
    assert out[0].shape == (n, f_in)
    assert embedding.shape == (n, f_dif)


def test_gnd_edge_rewire():
    torch.manual_seed(0)
    n, f = 10, 8
    x = torch.randn(n, f)
    gnd = GND(num_features=f, num_heads=2, num_steps=2, rebuild_graph=True)
    out, embedding = gnd((x, {"k_min": 0, "k_max": 4}))
    assert out[0].shape == (n, f)
    assert embedding.shape == (n, f)


def test_gnd_log_diffusion(small_graph):
    x, edge_index, n, f = small_graph
    gnd = GND(num_features=f, num_heads=2, num_steps=3, log_diffusion=True)
    gnd((x, edge_index))
    assert len(gnd.diffusion_step_outputs) == 4  # initial + 3 steps


def test_gnd_layer_propagate_time_increment():
    layer = GNDLayer(num_features=4, num_heads=1, time_increment=0.3)
    in_features = torch.ones(2, 1, 4)
    aggregated = torch.zeros(2, 1, 4)
    out = layer._propagate(in_features, aggregated)
    expected = 0.7 * in_features.squeeze(1)
    assert torch.allclose(out, expected)


def test_gnd_backward(small_graph):
    x, edge_index, n, f = small_graph
    gnd = GND(num_features=f, num_heads=2, num_steps=2)
    out, _ = gnd((x, edge_index))
    loss = out[0].sum()
    loss.backward()
    assert any(p.grad is not None for p in gnd.parameters())
