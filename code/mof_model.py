"""
mof_model.py — Consolidated Hybrid GNN Model Architecture
==========================================================
Single-file model definition combining hybrid_model.py and hybrid_model_v2.py
with all architectural improvements.

Key Components:
- GNNBranch: SchNet-style message passing with attention pooling
- ChemicalBranch: MLP for 145-dim Magpie features  
- QuantumBranch: MLP with learnable missing-data embedding
- CrossAttentionFusion: Chemical queries attend to GNN keys/values
- HybridMOFModel: Full multi-branch architecture
- GNNOnlyModel: Stripped-down GNN for ablation studies

Usage:
    from mof_model import HybridMOFModel, GNNOnlyModel
    model = HybridMOFModel(cutoff=8.0, num_tasks=1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax as pyg_softmax
from typing import Optional, Tuple


# =============================================================================
# 1. Gaussian RBF Edge-Distance Expansion
# =============================================================================

class GaussianRBF(nn.Module):
    """Expand scalar distances to Gaussian basis functions."""
    
    def __init__(self, num_bins: int = 50, r_min: float = 0.0,
                 r_max: float = 6.0, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma
        centers = torch.linspace(r_min, r_max, num_bins)
        self.register_buffer("centers", centers)
    
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff ** 2)


# =============================================================================
# 2. Global Attention Pooling (in-house implementation)
# =============================================================================

class GlobalAttentionPool(nn.Module):
    """Soft-attention graph pooling: h_graph = sum_i(softmax(gate(h_i)) * h_i)"""
    
    def __init__(self, gate_nn: nn.Module):
        super().__init__()
        self.gate_nn = gate_nn
    
    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        gate = self.gate_nn(h)
        gate_max = global_max_pool(gate, batch)[batch]
        gate_exp = torch.exp(gate - gate_max)
        gate_sum = global_add_pool(gate_exp, batch)[batch] + 1e-8
        alpha = gate_exp / gate_sum
        weighted = alpha * h
        return global_add_pool(weighted, batch)


# =============================================================================
# 3. SchNet Interaction Block
# =============================================================================

class SchNetInteraction(MessagePassing):
    """SchNet continuous-filter convolution: h_i <- h_i + MLP(h_j) ⊙ W(||rij||)"""
    
    def __init__(self, hidden: int, rbf_bins: int = 50):
        super().__init__(aggr="add")
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_bins, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.lin = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_rbf: torch.Tensor) -> torch.Tensor:
        W = self.filter_net(edge_rbf)
        return self.act(self.lin(self.propagate(edge_index, h=h, W=W)))
    
    def message(self, h_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return h_j * W


# =============================================================================
# 4. GNN Branch
# =============================================================================

class GNNBranch(nn.Module):
    """
    SchNet-based GNN with attention pooling.
    
    Expected input graph attributes:
      - node_features: (N, 3) — [atomic_Z, electronegativity, covalent_radius]
      - edge_index: (2, E)
      - edge_dist: (E,) — distances in Å
      - batch: (N,) — batch assignment
    """
    
    def __init__(self, hidden: int = 128, num_interactions: int = 3,
                 rbf_bins: int = 50, out_dim: int = 128):
        super().__init__()
        
        self.node_embed = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
        )
        
        self.rbf = GaussianRBF(num_bins=rbf_bins)
        
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden, rbf_bins)
            for _ in range(num_interactions)
        ])
        
        gate_nn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.pool = GlobalAttentionPool(gate_nn)
        
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
    
    def forward(self, batch) -> torch.Tensor:
        h = self.node_embed(batch.node_features)
        edge_dist = batch.edge_dist
        edge_index = batch.edge_index.long()
        
        for interaction in self.interactions:
            edge_rbf = self.rbf(edge_dist)
            delta = interaction(h, edge_index, edge_rbf)
            h = h + delta
        
        h_graph = self.pool(h, batch.batch)
        return self.proj(h_graph)


# =============================================================================
# 5. Chemical Branch
# =============================================================================

class ChemicalBranch(nn.Module):
    """MLP encoder for 145-dim Magpie chemical features."""
    
    def __init__(self, in_dim: int = 145, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 6. Quantum Branch (with missing data handling)
# =============================================================================

class QuantumBranch(nn.Module):
    """MLP for quantum features with learnable embedding for missing data."""
    
    def __init__(self, in_dim: int = 8, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )
        self.missing_embed = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_dim) — NaN rows already replaced with 0s
        mask: (B,) bool — True where quantum data is available
        """
        out = self.missing_embed.unsqueeze(0).expand(x.size(0), -1).clone()
        if mask.any():
            out[mask] = self.net(x[mask])
        return out


# =============================================================================
# 7. Cross-Attention Fusion
# =============================================================================

class CrossAttentionFusion(nn.Module):
    """Chemical queries attend over GNN key/values."""
    
    def __init__(self, gnn_dim: int = 128, chem_dim: int = 128,
                 num_heads: int = 4, out_dim: int = 128):
        super().__init__()
        assert gnn_dim % num_heads == 0, "gnn_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = gnn_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(chem_dim, gnn_dim)
        self.k_proj = nn.Linear(gnn_dim, gnn_dim)
        self.v_proj = nn.Linear(gnn_dim, gnn_dim)
        self.out_proj = nn.Linear(gnn_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, gnn_emb: torch.Tensor, chem_emb: torch.Tensor) -> torch.Tensor:
        B = gnn_emb.size(0)
        H, D = self.num_heads, self.head_dim
        
        q = self.q_proj(chem_emb).view(B, 1, H, D).transpose(1, 2)
        k = self.k_proj(gnn_emb).view(B, 1, H, D).transpose(1, 2)
        v = self.v_proj(gnn_emb).view(B, 1, H, D).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        fused = (attn @ v).transpose(1, 2).contiguous().view(B, -1)
        
        out = self.out_proj(fused)
        if chem_emb.size(-1) == out.size(-1):
            out = out + chem_emb
        return self.norm(out)


# =============================================================================
# 8. Prediction Head
# =============================================================================

class PredictionHead(nn.Module):
    """Final prediction head combining fused + quantum features."""
    
    def __init__(self, in_dim: int = 128, quantum_dim: int = 64, num_tasks: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + quantum_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_tasks),
        )
    
    def forward(self, fused: torch.Tensor, quantum: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([fused, quantum], dim=-1)).squeeze(-1)


# =============================================================================
# 9. Full Hybrid Model
# =============================================================================

class HybridMOFModel(nn.Module):
    """
    Full hybrid model: GNN + Chemical + Quantum -> Fusion -> Prediction.
    
    Forward signature:
    ------------------
    graph_batch: PyG Batch (node_features, edge_index, edge_dist, batch)
    chemical_x: (B, 145) — Magpie features
    quantum_x: (B, 8) — quantum features (NaN rows replaced with 0)
    quantum_mask: (B,) bool — True where quantum data exists
    
    Returns: (B,) or (B, num_tasks) predictions
    """
    
    def __init__(
        self,
        gnn_hidden: int = 128,
        gnn_interactions: int = 3,
        rbf_bins: int = 50,
        gnn_out: int = 128,
        chem_in: int = 145,
        chem_out: int = 128,
        quantum_in: int = 8,
        quantum_out: int = 64,
        fusion_heads: int = 4,
        fusion_out: int = 128,
        cutoff: float = 8.0,
        num_tasks: int = 1,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        
        self.gnn = GNNBranch(gnn_hidden, gnn_interactions, rbf_bins, gnn_out)
        self.chem = ChemicalBranch(chem_in, chem_out)
        self.quantum = QuantumBranch(quantum_in, quantum_out)
        self.fusion = CrossAttentionFusion(gnn_out, chem_out, fusion_heads, fusion_out)
        self.head = PredictionHead(fusion_out, quantum_out, num_tasks)
    
    def forward(self, graph_batch, chemical_x: torch.Tensor,
                quantum_x: torch.Tensor, quantum_mask: torch.Tensor,
                mof_desc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through all branches."""
        g = self.gnn(graph_batch)
        c = self.chem(chemical_x)
        q = self.quantum(quantum_x, quantum_mask)
        
        fused = self.fusion(g, c)
        preds = self.head(fused, q)
        
        return preds.squeeze(-1) if self.num_tasks == 1 else preds
    
    def parameter_groups(self, gnn_lr: float = 1e-4,
                         tabular_lr: float = 1e-3) -> list:
        """Return parameter groups with different learning rates."""
        return [
            {"params": self.gnn.parameters(), "lr": gnn_lr, "name": "gnn"},
            {"params": self.chem.parameters(), "lr": tabular_lr, "name": "chemical"},
            {"params": self.quantum.parameters(), "lr": tabular_lr, "name": "quantum"},
            {"params": self.fusion.parameters(), "lr": tabular_lr, "name": "fusion"},
            {"params": self.head.parameters(), "lr": tabular_lr, "name": "head"},
        ]
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 10. GNN-Only Model (for ablation studies)
# =============================================================================

class GNNOnlyModel(nn.Module):
    """Stripped-down GNN for diagnosis and ablation."""
    
    def __init__(self, hidden: int = 128, num_interactions: int = 3,
                 rbf_bins: int = 50, cutoff: float = 8.0):
        super().__init__()
        out_dim = 128
        self.gnn = GNNBranch(hidden, num_interactions, rbf_bins, out_dim=out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    
    def forward(self, graph_batch) -> torch.Tensor:
        return self.head(self.gnn(graph_batch)).squeeze(-1)


# =============================================================================
# Quick smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MOF Model Architecture Verification")
    print("=" * 70)
    
    # Test Hybrid Model
    B = 4
    graphs = []
    for _ in range(B):
        N = torch.randint(5, 15, (1,)).item()
        E = N * 3
        graphs.append(Data(
            node_features=torch.rand(N, 3),
            edge_index=torch.randint(0, N, (2, E)),
            edge_dist=torch.rand(E) * 5.0,
        ))
    batch = Batch.from_data_list(graphs)
    
    model = HybridMOFModel()
    print(f"\nHybridMOFModel parameters: {model.num_parameters:,}")
    
    chemical_x = torch.rand(B, 145)
    quantum_x = torch.rand(B, 8)
    q_mask = torch.tensor([True, False, True, True])
    
    preds = model(batch, chemical_x, quantum_x, q_mask)
    print(f"Hybrid output shape: {preds.shape}")
    
    # Test GNN-Only Model
    gnn_only = GNNOnlyModel()
    preds2 = gnn_only(batch)
    print(f"GNN-only output shape: {preds2.shape}")
    
    print("\n" + "=" * 70)
    print("All forward passes OK!")
    print("=" * 70)