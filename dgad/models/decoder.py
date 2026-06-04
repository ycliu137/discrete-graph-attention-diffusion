"""Adjacency reconstruction decoder (VGAE-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InnerProductDecoder(nn.Module):
    """Predict adjacency via inner product of node embeddings."""

    def __init__(self, dropout=0.0, act=None):
        super().__init__()
        self.dropout = dropout
        self.act = act if act is not None else (lambda x: x)

    def forward(self, z, edge_index=None):
        """
        Args:
            z: (N, F) node embeddings
            edge_index: optional (2, E); when set, return sparse (E,) edge logits

        Returns:
            Dense (N, N) logits if edge_index is None, else sparse (E,) logits.
        """
        z = F.dropout(z, self.dropout, training=self.training)
        if edge_index is None:
            return self.act(torch.mm(z, z.t()))
        src, trg = edge_index[0], edge_index[1]
        logits = (z[src] * z[trg]).sum(dim=-1)
        return self.act(logits)
