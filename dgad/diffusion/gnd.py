"""
Discrete Graph Attention Diffusion (DGAD / GND).

Fixed K-step explicit diffusion with attention-weighted neighborhood aggregation
and discrete Euler-style propagation steps.
"""

import torch
import torch.nn as nn

from dgad.diffusion.gnd_layer import GNDLayer
from dgad.models.encoder import GraphFeatureEncoder


class GND(nn.Module):
    """
    K-step discrete graph attention diffusion with optional dynamic KNN rewiring.

    Forward input:
        - Fixed graph: (node_features, edge_index)
        - Dynamic graph: (node_features, rebuild_args) where rebuild_args is
          dict or tuple (k_min, k_max, remov_edge_prob)
    """

    def __init__(
        self,
        num_features,
        num_heads=8,
        num_steps=8,
        time_increment=0.2,
        attention_type="sum",
        activation=nn.ELU(),
        dropout=0.0,
        log_diffusion=False,
        encoder=None,
        decoder=None,
        rebuild_graph=False,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_steps = num_steps
        self.num_heads = num_heads
        self.rebuild_graph = rebuild_graph
        self.log_diffusion = log_diffusion
        self.diffusion_step_outputs = []
        self.activation = activation

        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

        if encoder is not None:
            assert encoder[-1] == num_features, "encoder last dim must equal num_features"
            self.encoder = GraphFeatureEncoder(encoder, activation=activation, last_activation=False)
        if decoder is not None:
            assert decoder[0] == num_features, "decoder first dim must equal num_features"
            self.decoder = GraphFeatureEncoder(decoder, activation=activation, last_activation=False)

        self.gnd_layer = GNDLayer(
            num_features=num_features,
            num_heads=num_heads,
            time_increment=time_increment,
            attention_type=attention_type,
            activation=activation,
            dropout_prob=dropout,
            rebuild_graph=rebuild_graph,
        )

    def forward(self, data):
        data = self._normalize_rebuild_args(data)
        data = self.encoder(data)

        if self.rebuild_graph:
            features, rebuild_args = data
            if isinstance(rebuild_args, dict):
                rebuild_args = (
                    rebuild_args["k_min"],
                    rebuild_args["k_max"],
                    rebuild_args.get("remov_edge_prob"),
                )
            data = (features, rebuild_args)

        if self.log_diffusion:
            self.diffusion_step_outputs = [data[0].detach().cpu()]

        for _ in range(self.num_steps):
            data = self.gnd_layer(data)
            if self.log_diffusion:
                self.diffusion_step_outputs.append(data[0].detach().cpu())

        embedding, edge_or_args = data
        last_embedding = embedding

        out = self.decoder((embedding, edge_or_args))
        return out, last_embedding

    @staticmethod
    def _normalize_rebuild_args(data):
        features, second = data
        if isinstance(second, dict):
            return (
                features,
                (second["k_min"], second["k_max"], second.get("remov_edge_prob")),
            )
        return data
