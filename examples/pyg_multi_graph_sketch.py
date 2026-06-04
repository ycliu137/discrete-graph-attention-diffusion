"""
Sketch: multiple graphs via PyG DataLoader without changing GND.

Each step processes one graph (batch_size=1): forward GND, mean-pool node
embeddings for graph-level labels, backprop. For true batched multi-graph
forward (PyG Batch in one call), use a separate integration layer—GND core
expects a single (N, F) + edge_index pair.

Install:
    pip install -e ".[pyg]"

Run:
    python examples/pyg_multi_graph_sketch.py
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from dgad import GND

DATA_DIR = Path(__file__).resolve().parent / "data" / "TUDataset"
NUM_EPOCHS = 50
F_DIFF = 16


def graph_embedding(gnd, data):
    """Node embeddings -> one vector per graph (mean pool)."""
    (_, _), z = gnd((data.x, data.edge_index))
    return z.mean(dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUDataset(root=str(DATA_DIR), name="MUTAG")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    gnd = GND(
        num_features=F_DIFF,
        num_heads=2,
        num_steps=3,
        time_increment=0.2,
        encoder=[num_features, F_DIFF],
    ).to(device)
    head = torch.nn.Linear(F_DIFF, num_classes).to(device)
    optimizer = torch.optim.Adam(
        list(gnd.parameters()) + list(head.parameters()),
        lr=1e-3,
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        gnd.train()
        head.train()
        total_loss = 0.0
        correct = 0

        for batch in loader:
            graph = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            graph_vec = graph_embedding(gnd, graph)
            logits = head(graph_vec.unsqueeze(0))
            loss = F.cross_entropy(logits, graph.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += int(logits.argmax(dim=-1).eq(graph.y).sum())

        if epoch % 10 == 0 or epoch == NUM_EPOCHS:
            acc = correct / len(dataset)
            print(
                f"Epoch {epoch:03d}  loss={total_loss / len(loader):.4f}  "
                f"train_acc={acc:.3f}"
            )


if __name__ == "__main__":
    main()
