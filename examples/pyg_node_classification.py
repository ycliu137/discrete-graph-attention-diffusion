"""
Node classification with PyG + GND (single graph, full-batch training).

Uses the Cora citation graph: one forward pass over all nodes each epoch.
GND is unchanged; this script only wires PyG data into (x, edge_index).

Install:
    pip install -e ".[pyg]"

Run:
    python examples/pyg_node_classification.py
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from dgad import GND

DATA_DIR = Path(__file__).resolve().parent / "data" / "Planetoid"
NUM_EPOCHS = 200
LOG_EVERY = 20
F_DIFF = 32


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid(
        root=str(DATA_DIR),
        name="Cora",
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)

    gnd = GND(
        num_features=F_DIFF,
        num_heads=4,
        num_steps=4,
        time_increment=0.2,
        attention_type="sum",
        encoder=[data.num_features, F_DIFF],
    ).to(device)
    head = torch.nn.Linear(F_DIFF, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(
        list(gnd.parameters()) + list(head.parameters()),
        lr=0.01,
        weight_decay=5e-4,
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        gnd.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)

        (_, _), z = gnd((data.x, data.edge_index))
        loss = F.cross_entropy(z[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == NUM_EPOCHS:
            gnd.eval()
            head.eval()
            with torch.no_grad():
                (_, _), z = gnd((data.x, data.edge_index))
                pred = z.argmax(dim=-1)
                train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).float().mean()
                val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).float().mean()
                test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean()
            print(
                f"Epoch {epoch:03d}  loss={loss.item():.4f}  "
                f"train={train_acc:.3f}  val={val_acc:.3f}  test={test_acc:.3f}"
            )


if __name__ == "__main__":
    main()
