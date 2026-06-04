"""Minimal DGAD example on synthetic node features."""

import torch

from dgad import DGADModel, knn_graph, fit_dgad

torch.manual_seed(0)
n, f_in, f_dif = 200, 64, 32
x = torch.randn(n, f_in)
edge_index = knn_graph(x, k_min=0, k_max=10, self_edge=False)

model = DGADModel(
    num_features=f_dif,
    num_heads=4,
    num_steps=4,
    time_increment=0.2,
    attention_type="sum",
    encoder=[f_in, f_dif],
    decoder=[f_dif, f_in],
    edge_rewire=False,
)

result = fit_dgad(
    model,
    x,
    edge_index=edge_index,
    max_epochs=100,
    lr=1e-2,
    device="cpu",
    verbose=True,
    log_every=25,
)

print("embedding shape:", result["embedding"].shape)
