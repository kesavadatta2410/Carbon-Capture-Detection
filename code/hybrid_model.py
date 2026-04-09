#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tasks 4.2-4.6, 4.8: Hybrid GNN + Chemical + Quantum Model.

Architecture:
  - 4.2: GNN Pathway   -> SchNet (4 layers, 128 hidden, global mean pool -> 128-dim)
  - 4.3: Chemical Path  -> MLP (145 -> 64 -> 32)
  - 4.4: Quantum Path   -> MLP (9 -> 16 -> 8)
  - 4.5: Fusion Module  -> Concat(128+32+8=168) -> 256 -> 128 -> 64 -> 1 (residual)
  - 4.6: Full Hybrid    -> Assembles all pathways
  - 4.8: MC Dropout     -> 20 stochastic forward passes at inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SchNet, global_mean_pool
from torch_geometric.data import Data, Batch


# =============================================================================
# 4.2 GNN Pathway -- SchNet
# =============================================================================

class GNNPathway(nn.Module):
    """
    SchNet-based pathway for learning from molecular graphs.
    Input:  graph (z, pos, edge_index, edge_attr, batch)
    Output: 128-dim graph-level embedding
    """

    def __init__(self, hidden_channels=128, num_filters=64, num_interactions=4,
                 num_gaussians=50, cutoff=10.0, output_dim=128):
        super().__init__()
        self.cutoff = cutoff

        # Node embedding: 3 raw features -> hidden_channels
        self.node_embed = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # SchNet interaction layers (manually implemented for flexibility)
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(
                SchNetInteraction(hidden_channels, num_filters, num_gaussians, cutoff)
            )

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, z, pos, edge_index, edge_attr, batch):
        """
        Args:
            z: Node features (N, 3) -- [atomic_num, electronegativity, cov_radius]
            pos: Not used directly (distances pre-computed as edge_attr)
            edge_index: (2, E)
            edge_attr: (E,) -- pairwise distances in Angstrom
            batch: (N,) -- batch assignment
        """
        # Initial node embedding
        h = self.node_embed(z)

        # Interaction blocks
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)

        # Global mean pooling -> graph-level embedding
        h = global_mean_pool(h, batch)

        return self.readout(h)


class SchNetInteraction(nn.Module):
    """Single SchNet continuous-filter convolution interaction block."""

    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super().__init__()
        self.cutoff = cutoff

        # Gaussian RBF expansion for distances
        self.rbf = GaussianRBF(num_gaussians, cutoff)

        # Continuous filter network: distance -> filter weights
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )

        # Message: element-wise multiply node features with filters
        self.node_to_filter = nn.Linear(hidden_channels, num_filters)
        self.filter_to_node = nn.Sequential(
            nn.Linear(num_filters, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index

        # Expand distances to Gaussian RBF
        rbf = self.rbf(edge_attr)
        W = self.filter_net(rbf)  # (E, num_filters)

        # Message passing
        h_j = self.node_to_filter(h[col])  # (E, num_filters)
        msg = h_j * W  # Element-wise product

        # Aggregate messages
        agg = torch.zeros_like(h[:, :msg.size(1)])
        agg = agg.scatter_add(0, row.unsqueeze(1).expand_as(msg), msg)

        return self.filter_to_node(agg)


class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion for distance encoding."""

    def __init__(self, num_gaussians=50, cutoff=10.0):
        super().__init__()
        offset = torch.linspace(0, cutoff, num_gaussians)
        self.register_buffer("offset", offset)
        self.width = (offset[1] - offset[0]).item()

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return torch.exp(-0.5 * ((dist - self.offset) / self.width) ** 2)


# =============================================================================
# 4.3 Chemical Pathway
# =============================================================================

class ChemicalPathway(nn.Module):
    """
    MLP encoder for chemical composition features.
    Input:  145-dim Magpie-inspired features
    Output: 32-dim embedding
    """

    def __init__(self, input_dim=145, hidden_dim=64, output_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 4.4 Quantum Pathway
# =============================================================================

class QuantumPathway(nn.Module):
    """
    Small MLP encoder for sparse QMOF quantum features.
    Input:  9-dim (bandgap, energy, density, volume, etc. -- mostly zeros)
    Output: 8-dim embedding
    """

    def __init__(self, input_dim=9, hidden_dim=16, output_dim=8, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 4.5 Fusion Module (with residual connection)
# =============================================================================

class FusionModule(nn.Module):
    """
    Fuses GNN + Chemical + Quantum embeddings with residual connection.
    Input:  concat(128 + 32 + 8) = 168
    Output: scalar prediction

    Architecture:
      168 -> 256 -> 128 -> 64 -> 1
      with residual skip: 256-dim projected to 64-dim added before final layer
    """

    def __init__(self, gnn_dim=128, chem_dim=32, quantum_dim=8, dropout=0.1):
        super().__init__()
        concat_dim = gnn_dim + chem_dim + quantum_dim  # 168

        self.fc1 = nn.Linear(concat_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        # Residual: project 256 -> 64 for skip connection
        self.residual = nn.Linear(256, 64)

        self.fc_out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gnn_emb, chem_emb, quantum_emb):
        x = torch.cat([gnn_emb, chem_emb, quantum_emb], dim=-1)

        # Layer 1: 168 -> 256
        h1 = self.dropout(F.relu(self.bn1(self.fc1(x))))

        # Layer 2: 256 -> 128
        h2 = self.dropout(F.relu(self.bn2(self.fc2(h1))))

        # Layer 3: 128 -> 64
        h3 = self.dropout(F.relu(self.bn3(self.fc3(h2))))

        # Residual skip: 256 -> 64
        res = self.residual(h1)
        h3 = h3 + res

        return self.fc_out(h3).squeeze(-1)


# =============================================================================
# 4.6 Full Hybrid Model
# =============================================================================

class HybridMOFModel(nn.Module):
    """
    Full hybrid model: GNN + Chemical + Quantum -> Fusion -> Prediction.

    Input per sample:
      - Graph: node_features (N,3), edge_index (2,E), edge_dist (E,)
      - Chemical: 145-dim vector
      - Quantum: 9-dim vector (sparse, mostly zeros)
    """

    def __init__(
        self,
        # GNN params
        gnn_hidden=128, gnn_filters=64, gnn_layers=4,
        gnn_gaussians=50, gnn_cutoff=10.0, gnn_output=128,
        # Chemical params
        chem_input=145, chem_hidden=64, chem_output=32,
        # Quantum params
        quantum_input=9, quantum_hidden=16, quantum_output=8,
        # Shared
        dropout=0.1,
    ):
        super().__init__()

        self.gnn = GNNPathway(
            hidden_channels=gnn_hidden, num_filters=gnn_filters,
            num_interactions=gnn_layers, num_gaussians=gnn_gaussians,
            cutoff=gnn_cutoff, output_dim=gnn_output,
        )
        self.chem = ChemicalPathway(chem_input, chem_hidden, chem_output, dropout)
        self.quantum = QuantumPathway(quantum_input, quantum_hidden, quantum_output, dropout)
        self.fusion = FusionModule(gnn_output, chem_output, quantum_output, dropout)

    def forward(self, graph_batch, chem_features, quantum_features):
        """
        Args:
            graph_batch: PyG Batch with .x, .edge_index, .edge_attr, .batch
            chem_features: (B, 145)
            quantum_features: (B, 9)
        """
        gnn_emb = self.gnn(
            graph_batch.x, None,
            graph_batch.edge_index,
            graph_batch.edge_attr,
            graph_batch.batch,
        )
        chem_emb = self.chem(chem_features)
        quantum_emb = self.quantum(quantum_features)

        return self.fusion(gnn_emb, chem_emb, quantum_emb)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 4.8 MC Dropout Uncertainty
# =============================================================================

def enable_mc_dropout(model):
    """Enable dropout layers during inference for MC sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_with_uncertainty(model, graph_batch, chem_features, quantum_features,
                             n_samples=20):
    """
    MC Dropout uncertainty estimation.
    Returns: mean prediction, std (uncertainty), all samples.
    """
    model.eval()
    enable_mc_dropout(model)

    predictions = []
    for _ in range(n_samples):
        pred = model(graph_batch, chem_features, quantum_features)
        predictions.append(pred.cpu())

    predictions = torch.stack(predictions, dim=0)  # (n_samples, batch)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    model.eval()  # Disable dropout again
    return mean, std, predictions


# =============================================================================
# Test / Verification
# =============================================================================

def verify_model():
    """Quick smoke test with random data."""
    print("=" * 80)
    print("Hybrid Model Architecture Verification")
    print("=" * 80)

    model = HybridMOFModel()
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Count per module
    for name, child in [("GNN", model.gnn), ("Chemical", model.chem),
                         ("Quantum", model.quantum), ("Fusion", model.fusion)]:
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:12s}: {n:>8,} params")

    # Create fake batch of 4 graphs
    graphs = []
    for _ in range(4):
        n_atoms = torch.randint(10, 50, (1,)).item()
        n_edges = torch.randint(20, 100, (1,)).item()
        g = Data(
            x=torch.randn(n_atoms, 3),
            edge_index=torch.randint(0, n_atoms, (2, n_edges)),
            edge_attr=torch.rand(n_edges) * 5.0,
        )
        graphs.append(g)
    batch = Batch.from_data_list(graphs)

    chem = torch.randn(4, 145)
    quantum = torch.randn(4, 9)

    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(batch, chem, quantum)
    print(f"\nForward pass OK -- output shape: {pred.shape}")
    print(f"  Predictions: {pred.numpy()}")

    # MC Dropout
    mean, std, samples = predict_with_uncertainty(model, batch, chem, quantum, n_samples=10)
    print(f"\nMC Dropout (10 samples):")
    print(f"  Mean: {mean.numpy()}")
    print(f"  Std:  {std.numpy()}")

    print(f"\n{'=' * 80}")
    print("MODEL VERIFICATION COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    verify_model()
