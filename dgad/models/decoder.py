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

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        return self.act(torch.mm(z, z.t()))
