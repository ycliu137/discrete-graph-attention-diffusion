"""Linear encoder/decoder wrapping node features around diffusion."""

import torch
import torch.nn as nn


class GraphFeatureEncoder(nn.Module):
    """
    MLP encoder/decoder that maps (N, F_in) <-> (N, F_diff) while preserving edge_index.

    num_features_list: layer dimensions, e.g. [F_in, 128, F_diff].
    """

    def __init__(self, num_features_list, activation, last_activation=False, pre_activation=False):
        super().__init__()
        self.activation = activation
        layers = []
        if pre_activation:
            layers.append(self.activation)

        for i in range(len(num_features_list) - 1):
            layer = nn.Linear(num_features_list[i], num_features_list[i + 1], bias=False)
            nn.init.xavier_uniform_(layer.weight)
            layers.append(layer)
            layers.append(self.activation)

        if not last_activation:
            layers = layers[:-1]

        self.net = nn.Sequential(*layers)

    def forward(self, data):
        node_features, edge_index = data
        return self.net(node_features), edge_index
