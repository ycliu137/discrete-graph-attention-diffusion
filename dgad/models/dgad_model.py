"""Optional reconstruction model: diffusion embeddings + adjacency decoder."""

import torch.nn as nn

from dgad.diffusion.gnd import GND
from dgad.models.decoder import InnerProductDecoder


class DGADModel(nn.Module):
    """
    GND with an inner-product adjacency decoder for reconstruction-based training.

    For supervised or other tasks, use GND directly and attach your own head/loss.
    Pipeline: encode (optional) -> K-step diffusion -> decode (optional) -> adjacency logits.
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
        edge_rewire=False,
    ):
        super().__init__()
        self.log_diffusion = log_diffusion
        self.diffusion_step_outputs = None

        self.diffusion = GND(
            num_features=num_features,
            num_heads=num_heads,
            num_steps=num_steps,
            time_increment=time_increment,
            attention_type=attention_type,
            activation=activation,
            dropout=dropout,
            log_diffusion=log_diffusion,
            encoder=encoder,
            decoder=decoder,
            rebuild_graph=edge_rewire,
        )
        self.adj_decoder = InnerProductDecoder(dropout=0.0, act=lambda x: x)

    def forward(self, data):
        (out_features, edge_index), last_embedding = self.diffusion(data)
        recon_adj = self.adj_decoder(out_features)

        if self.log_diffusion:
            self.diffusion_step_outputs = self.diffusion.diffusion_step_outputs

        return out_features, recon_adj, last_embedding
