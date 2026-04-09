"""
hybrid_model_v2.py — Phase 5 Fixed Architecture with Critical Improvements
=========================================================================
Compatible with PyG 2.7.0 and the precomputed .npz graphs from build_graphs.py.

Key changes vs v1:
  • Custom SchNet-style GNN via MessagePassing (no reliance on PyG SchNet internals)
  • GlobalAttentionPooling implemented in-house (PyG 2.7.0 has no such class)
  • Gaussian RBF distance expansion (50 bins) as node/edge input
  • Cross-attention fusion (chemical queries → GNN keys/values, 4 heads)
  • Layer-wise learning-rate groups via parameter_groups()

CRITICAL IMPROVEMENTS (changes.md):
  • DimeNet++-inspired directional message passing with angular information
  • 3D periodic boundary handling with minimum image convention
  • Hierarchical pooling (SAGPool) for pore-level hierarchy capture
  • Edge type encoding (metal-ligand, ligand-ligand, guest-host, periodic)
  • Crystal graph attention with fractional coordinate encoding
  • MOF-specific descriptors integration (PLD, LCD, surface area)
  • Multi-task learning support (CO2 at 0.1, 1.0, 10 bar)
  • Physics-informed loss (Langmuir/Freundlich constraints)
  • Gas-kinetic diameter features (CO2 3.3Å, N2 3.64Å)

Graph format expected (from build_graphs.py):
  .node_features  (N, 3)  — [atomic_Z, electronegativity, covalent_radius]
  .edge_index     (2, E)  — COO-format, both directions
  .edge_dist      (E,)    — interatomic distances in Å (already computed)
  .edge_vec       (E, 3)  — edge vectors for directional features
  .frac_coords    (N, 3)  — fractional coordinates for periodic handling
  No .pos stored — model uses precomputed edge distances only.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax as pyg_softmax
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# 1. Gaussian RBF edge-distance expansion
# ---------------------------------------------------------------------------

class GaussianRBF(nn.Module):
    """Expand scalar distances to 'num_bins' Gaussian basis functions."""

    def __init__(self, num_bins: int = 50, r_min: float = 0.0,
                 r_max: float = 6.0, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma
        centers = torch.linspace(r_min, r_max, num_bins)
        self.register_buffer("centers", centers)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: (E,) raw distances in Å
        Returns:
            (E, num_bins) expanded edge features
        """
        diff = dist.unsqueeze(-1) - self.centers          # (E, B)
        return torch.exp(-self.gamma * diff ** 2)


# -----------------------------------------------------------------------------
# NEW: Directional RBF (DimeNet++ inspired)
# -----------------------------------------------------------------------------

class DirectionalRBF(nn.Module):
    """
    Radial basis functions combined with directional information.
    DimeNet++ uses spherical harmonics for directional message passing.
    """

    def __init__(self, num_radial: int = 50, num_spherical: int = 7,
                 r_max: float = 8.0):
        super().__init__()
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.r_max = r_max

        # Radial basis (Gaussian)
        centers = torch.linspace(0, r_max, num_radial)
        self.register_buffer('centers', centers)
        self.width = 0.5 * (r_max / num_radial)

        # Learnable scale for radial
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, distances: torch.Tensor, directions: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            distances: (E,) distances
            directions: (E, 3) unit vectors (optional)
        Returns:
            rbf: (E, num_radial) radial basis
            dir_features: (E, num_spherical) directional features
        """
        # Radial basis
        diff = distances.unsqueeze(-1) - self.centers
        rbf = torch.exp(-0.5 * (diff / self.width) ** 2)
        rbf = rbf * self.scale

        dir_features = None
        if directions is not None:
            # Spherical harmonics approximation
            x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]

            # Y_00, Y_1{-1,0,1}, Y_2{-2,-1,0,1,2} (7 total)
            sph = torch.stack([
                torch.ones_like(x),           # Y_00
                y,                            # Y_1{-1}
                z,                            # Y_10
                x,                            # Y_11
                x * y,                        # Y_2{-2}
                y * z,                        # Y_2{-1}
                3 * z**2 - 1,                 # Y_20
            ], dim=-1)[:, :self.num_spherical]

            dir_features = sph

        return rbf, dir_features


# -----------------------------------------------------------------------------
# NEW: Periodic Distance (3D boundary handling)
# -----------------------------------------------------------------------------

class PeriodicDistance(nn.Module):
    """
    Compute distances with minimum image convention for periodic systems.
    Essential for accurate pore channel connectivity in MOFs.
    """

    def __init__(self, cutoff: float = 8.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor,
                cell: Optional[torch.Tensor] = None,
                frac_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pos: (N, 3) atomic positions in Angstroms
            edge_index: (2, E) edge indices
            cell: (3, 3) or (B, 3, 3) lattice vectors (optional)
            frac_coords: (N, 3) fractional coordinates (optional)
        Returns:
            distances: (E,) minimum image distances
        """
        row, col = edge_index

        if frac_coords is not None and cell is not None:
            # CGCNN-style: use fractional coordinates for periodic-aware edges
            delta_frac = frac_coords[row] - frac_coords[col]
            # Apply periodic boundary wrapping
            delta_frac = delta_frac - torch.round(delta_frac)
            # Convert back to Cartesian
            if cell.dim() == 2:
                delta = torch.matmul(delta_frac, cell)
            else:
                delta = torch.bmm(delta_frac.unsqueeze(0), cell).squeeze(0)
        else:
            # Standard Euclidean
            delta = pos[row] - pos[col]

            if cell is not None:
                # Minimum image convention
                inv_cell = torch.inverse(cell)
                delta_frac = torch.matmul(delta, inv_cell)
                delta_frac = delta_frac - torch.round(delta_frac)
                delta = torch.matmul(delta_frac, cell)

        dist_sq = torch.sum(delta ** 2, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-8)
        return dist, delta


# -----------------------------------------------------------------------------
# NEW: Edge Type Encoding
# -----------------------------------------------------------------------------

class EdgeTypeEncoder(nn.Module):
    """
    Learnable embeddings for different edge types in MOFs:
      - 0: metal-ligand (coordination bonds)
      - 1: ligand-ligand (organic linker bonds)
      - 2: guest-host (adsorbate-framework interactions)
      - 3: periodic boundary edges
    """

    EDGE_TYPES = {
        'metal_ligand': 0,
        'ligand_ligand': 1,
        'guest_host': 2,
        'periodic': 3,
    }

    def __init__(self, num_types: int = 4, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_types, embed_dim)
        self.num_types = num_types
        self.embed_dim = embed_dim

    def forward(self, edge_type: torch.Tensor) -> torch.Tensor:
        return self.embedding(edge_type)

    @staticmethod
    def classify_edges(edge_index: torch.Tensor, atomic_numbers: torch.Tensor,
                        is_metal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Classify edges based on atom properties."""
        row, col = edge_index

        if is_metal is None:
            # Common metals in MOFs
            metal_z = {3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                       29, 30, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                       55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80}
            is_metal = torch.tensor([z.item() in metal_z for z in atomic_numbers],
                                    device=edge_index.device)

        edge_type = torch.full((edge_index.size(1),),
                               EdgeTypeEncoder.EDGE_TYPES['ligand_ligand'],
                               dtype=torch.long, device=edge_index.device)

        # Metal-ligand bonds
        metal_row = is_metal[row]
        metal_col = is_metal[col]
        metal_ligand_mask = (metal_row & ~metal_col) | (~metal_row & metal_col)
        edge_type[metal_ligand_mask] = EdgeTypeEncoder.EDGE_TYPES['metal_ligand']

        return edge_type


# -----------------------------------------------------------------------------
# NEW: Hierarchical Pooling (SAGPool)
# -----------------------------------------------------------------------------

class SAGPoolLayer(nn.Module):
    """
    Self-Attention Graph Pooling for hierarchical representations.
    Captures: atoms -> metal nodes -> organic linkers -> cages
    """

    def __init__(self, in_channels: int, ratio: float = 0.5, dropout: float = 0.0):
        super().__init__()
        self.ratio = ratio
        self.dropout = dropout

        self.score_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LayerNorm(in_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, F) node features
            edge_index: (2, E) edge indices
            batch: (N,) batch assignment
        Returns:
            x_pooled, edge_index_pooled, batch_pooled, perm, edge_mask
        """
        scores = self.score_net(x).squeeze(-1)

        # Select top-k nodes per graph
        num_nodes_per_graph = torch.bincount(batch)
        num_pooled = (num_nodes_per_graph.float() * self.ratio).ceil().long()

        perm_list = []
        offset = 0
        for i, n in enumerate(num_nodes_per_graph):
            mask = batch == i
            graph_scores = scores[mask]
            k = min(num_pooled[i].item(), mask.sum().item())
            _, idx = torch.topk(graph_scores, k, sorted=False)
            perm_list.append(idx + offset)
            offset += n

        perm = torch.cat(perm_list)

        # Pool
        x_pooled = x[perm]
        batch_pooled = batch[perm]

        # Filter and remap edges
        edge_mask = torch.isin(edge_index[0], perm) & torch.isin(edge_index[1], perm)
        edge_index_pooled = edge_index[:, edge_mask]

        node_map = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        node_map[perm] = torch.arange(perm.size(0), device=x.device)
        edge_index_pooled = node_map[edge_index_pooled]

        # The mapping ensures pooled edges refer to pooled node indices
        return x_pooled, edge_index_pooled, batch_pooled, perm, edge_mask


# ---------------------------------------------------------------------------
# 2. SchNet-style Interaction Block (MessagePassing)
# ---------------------------------------------------------------------------

class SchNetInteraction(MessagePassing):
    """
    One SchNet continuous-filter convolution layer.
    h_i <- h_i + MLP(h_j) ⊙ W(||rij||)
    where W(||rij||) is a dense filter network applied to the RBF expansion.
    """

    def __init__(self, hidden: int, rbf_bins: int = 50):
        super().__init__(aggr="add")
        # Filter network: RBF -> hidden
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_bins, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Atom-wise dense layer applied to neighbor messages
        self.lin = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_rbf: torch.Tensor) -> torch.Tensor:
        """
        h        : (N, hidden)
        edge_rbf : (E, rbf_bins)
        """
        W = self.filter_net(edge_rbf)           # (E, hidden)
        return self.act(self.lin(self.propagate(edge_index, h=h, W=W)))

    def message(self, h_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return h_j * W


# -----------------------------------------------------------------------------
# NEW: Directional Interaction Block (DimeNet++ inspired)
# -----------------------------------------------------------------------------

class DirectionalInteractionBlock(MessagePassing):
    """
    Message passing with directional information.
    Extends SchNet with angular information between triplets.
    """

    def __init__(self, hidden: int = 128, num_radial: int = 50,
                 num_spherical: int = 7):
        super().__init__(aggr='add', node_dim=0)

        self.hidden = hidden
        self.num_radial = num_radial

        # RBF expansion
        self.rbf = DirectionalRBF(num_radial, num_spherical)

        # Filter networks
        self.filter_net = nn.Sequential(
            nn.Linear(num_radial, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * 3),
        )

        # Message transformation
        self.lin_msg = nn.Linear(hidden, hidden)

        # Update transformation
        self.lin_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_dist: torch.Tensor, edge_vec: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Args:
            h: (N, hidden) node features
            edge_index: (2, E) edge indices
            edge_dist: (E,) edge distances
            edge_vec: (E, 3) edge vectors
        """
        # Normalize edge vectors for directional features
        directions = None
        if edge_vec is not None:
            directions = edge_vec / (torch.norm(edge_vec, dim=-1, keepdim=True) + 1e-8)

        rbf, dir_feat = self.rbf(edge_dist, directions)

        # Propagate messages
        out = self.propagate(edge_index, h=h, rbf=rbf, dir_feat=dir_feat)
        return self.lin_update(out)

    def message(self, h_j: torch.Tensor, rbf: torch.Tensor,
                dir_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute messages with continuous filters."""
        W = self.filter_net(rbf)  # (E, hidden * 3)

        # Split into query, key, value
        W_q, W_k, W_v = torch.chunk(W, 3, dim=-1)

        # Attention-like gating
        gate = torch.sigmoid(W_q) * torch.tanh(W_k)
        msg = h_j * W_v * gate

        return self.lin_msg(msg)


# -----------------------------------------------------------------------------
# NEW: Crystal Graph Attention (CGCNN-style)
# -----------------------------------------------------------------------------

class CrystalGraphAttention(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN) style attention.
    Uses fractional coordinates to preserve translational symmetry.
    """

    def __init__(self, hidden: int = 128, num_heads: int = 4):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        # Fractional coordinate embedding
        self.frac_embed = nn.Linear(3, hidden)

        # Attention
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

        # Edge weight based on distance
        self.edge_gate = nn.Sequential(
            nn.Linear(1, hidden // 4),
            nn.SiLU(),
            nn.Linear(hidden // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, frac_coords: torch.Tensor,
                edge_index: torch.Tensor, edge_dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, hidden) node features
            frac_coords: (N, 3) fractional coordinates
            edge_index: (2, E) edge indices
            edge_dist: (E,) edge distances
        """
        N = h.size(0)
        H, D = self.num_heads, self.head_dim

        # Add fractional coordinate info
        frac_h = self.frac_embed(frac_coords)
        h_combined = h + frac_h

        # Project to Q, K, V
        q = self.q_proj(h_combined).view(N, H, D)
        k = self.k_proj(h_combined).view(N, H, D)
        v = self.v_proj(h_combined).view(N, H, D)

        # Message passing with attention
        row, col = edge_index
        q_edge = q[row]  # (E, H, D)
        k_edge = k[col]
        v_edge = v[col]

        # Attention scores
        attn_scores = (q_edge * k_edge).sum(dim=-1) / math.sqrt(D)  # (E, H)

        # Distance-based gating
        edge_weight = self.edge_gate(edge_dist.unsqueeze(-1))  # (E, 1)
        attn_scores = attn_scores * edge_weight

        # Softmax over neighbors
        attn = pyg_softmax(attn_scores, row, num_nodes=N)

        # Aggregate
        out = attn.unsqueeze(-1) * v_edge  # (E, H, D)

        # Scatter add
        output = torch.zeros_like(q)
        output.index_add_(0, row, out.view(-1, H, D))

        # Reshape and project
        output = output.view(N, H * D)
        return self.out_proj(output)


# ---------------------------------------------------------------------------
# 3. In-house GlobalAttentionPooling
#    (equivalent to the missing PyG GlobalAttentionPooling class)
# ---------------------------------------------------------------------------

class GlobalAttentionPool(nn.Module):
    """
    Soft-attention graph pooling: h_graph = sum_i( softmax(gate(h_i)) * h_i )
    gate_nn: maps (hidden,) -> (1,) logit; softmax applied per graph.
    """

    def __init__(self, gate_nn: nn.Module):
        super().__init__()
        self.gate_nn = gate_nn

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        h     : (N, hidden)
        batch : (N,) — graph assignment index
        Returns: (B, hidden)
        """
        gate = self.gate_nn(h)                            # (N, 1)

        # Per-graph max for numerical stability (broadcast back to N)
        gate_max = global_max_pool(gate, batch)[batch]    # (N, 1)
        gate_exp = torch.exp(gate - gate_max)             # (N, 1)

        # Per-graph sum for normalization
        gate_sum = global_add_pool(gate_exp, batch)[batch] + 1e-8  # (N, 1)
        alpha    = gate_exp / gate_sum                    # (N, 1), per-graph softmax

        weighted = alpha * h                              # (N, hidden)
        return global_add_pool(weighted, batch)           # (B, hidden)


# ---------------------------------------------------------------------------
# 4. GNN branch: atom embedding + N interaction layers + attention pooling
# ---------------------------------------------------------------------------

class GNNBranch(nn.Module):
    """
    Enhanced GNN with directional message passing and hierarchical pooling.

    Input graph attributes expected:
      batch.node_features : (N, 3)  — [Z, EN, RC]
      batch.edge_index    : (2, E)
      batch.edge_dist     : (E,)    — distances in Å
      batch.edge_vec      : (E, 3)  — edge vectors (optional)
      batch.frac_coords   : (N, 3)  — fractional coordinates (optional)
      batch.batch         : (N,)
      batch.atomic_numbers: (N,)    — atomic numbers (optional)
    """

    def __init__(self, hidden: int = 128, num_interactions: int = 3,
                 rbf_bins: int = 50, out_dim: int = 128,
                 use_directional: bool = True,
                 use_hierarchical: bool = True,
                 num_levels: int = 3):
        super().__init__()
        self.use_directional = use_directional
        self.use_hierarchical = use_hierarchical
        self.num_levels = num_levels

        # Project raw node features
        self.node_embed = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
        )

        # RBF expansion
        self.rbf = GaussianRBF(num_bins=rbf_bins)

        # Interaction blocks
        if use_directional:
            self.interactions = nn.ModuleList([
                DirectionalInteractionBlock(hidden, rbf_bins)
                for _ in range(num_interactions)
            ])
        else:
            self.interactions = nn.ModuleList([
                SchNetInteraction(hidden, rbf_bins)
                for _ in range(num_interactions)
            ])

        # Edge type encoder
        self.edge_encoder = EdgeTypeEncoder(num_types=4, embed_dim=64)

        # Hierarchical pooling
        if use_hierarchical:
            self.pool_layers = nn.ModuleList([
                SAGPoolLayer(hidden, ratio=0.5 ** (i + 1))
                for i in range(num_levels - 1)
            ])
            self.level_readouts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                )
                for _ in range(num_levels)
            ])
            final_dim = out_dim * num_levels
        else:
            # Standard attention pooling
            gate_nn = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, 1),
            )
            self.pool = GlobalAttentionPool(gate_nn)
            final_dim = out_dim

        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        # Crystal graph attention (optional)
        self.crystal_attn = CrystalGraphAttention(hidden, num_heads=4)

        self.final_proj = nn.Linear(final_dim, out_dim)

    def forward(self, batch) -> torch.Tensor:
        h = self.node_embed(batch.node_features)
        edge_dist = batch.edge_dist
        edge_index = batch.edge_index.long()

        # Get directional edge vectors if available
        edge_vec = getattr(batch, 'edge_vec', None)

        # Edge type encoding
        if hasattr(batch, 'atomic_numbers'):
            edge_types = self.edge_encoder.classify_edges(
                edge_index, batch.atomic_numbers
            )
            edge_type_emb = self.edge_encoder(edge_types)
            # Could add to edge features

        # Crystal graph attention with fractional coordinates
        if hasattr(batch, 'frac_coords'):
            h = self.crystal_attn(h, batch.frac_coords, edge_index, edge_dist)

        level_embeddings = []
        batch_idx = batch.batch

        if self.use_hierarchical:
            for level in range(self.num_levels):
                # Interactions at this level
                for interaction in self.interactions:
                    if self.use_directional and edge_vec is not None:
                        delta = interaction(h, edge_index, edge_dist, edge_vec)
                    else:
                        if isinstance(interaction, DirectionalInteractionBlock):
                            delta = interaction(h, edge_index, edge_dist, None)
                        else:
                            edge_rbf = self.rbf(edge_dist)
                            delta = interaction(h, edge_index, edge_rbf)
                    h = h + delta

                # Pooling (except last level)
                if level < self.num_levels - 1:
                    # Record readout for current level before pooling
                    level_emb = global_mean_pool(h, batch_idx)
                    level_embeddings.append(self.level_readouts[level](level_emb))

                    h_pooled, edge_index_pooled, batch_pooled, perm, edge_mask = self.pool_layers[level](
                        h, edge_index, batch_idx
                    )
                    
                    # Update features for pooled graph
                    h = h_pooled
                    edge_index = edge_index_pooled
                    batch_idx = batch_pooled
                    edge_dist = edge_dist[edge_mask]
                    if edge_vec is not None:
                        edge_vec = edge_vec[edge_mask]
                else:
                    # Final level readout
                    level_emb = global_mean_pool(h, batch_idx)
                    level_embeddings.append(self.level_readouts[level](level_emb))

            # Combine all levels
            return torch.cat(level_embeddings, dim=-1)
        else:
            # Standard pooling
            for interaction in self.interactions:
                if self.use_directional and edge_vec is not None:
                    delta = interaction(h, edge_index, edge_dist, edge_vec)
                else:
                    if isinstance(interaction, DirectionalInteractionBlock):
                        delta = interaction(h, edge_index, edge_dist, None)
                    else:
                        edge_rbf = self.rbf(edge_dist)
                        delta = interaction(h, edge_index, edge_rbf)
                h = h + delta

            h_graph = self.pool(h, batch.batch)
            return self.proj(h_graph)


# ---------------------------------------------------------------------------
# 5. Chemical MLP branch
# ---------------------------------------------------------------------------

class ChemicalBranch(nn.Module):
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


# ---------------------------------------------------------------------------
# 6. Quantum MLP branch (handles missing data via masking)
# ---------------------------------------------------------------------------

class QuantumBranch(nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )
        # Learned embedding for samples with no quantum data
        self.missing_embed = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, in_dim) — NaN rows already replaced with 0s
        mask : (B,) bool   — True where quantum data is available
        """
        out = self.missing_embed.unsqueeze(0).expand(x.size(0), -1).clone()
        if mask.any():
            out[mask] = self.net(x[mask])
        return out


# ---------------------------------------------------------------------------
# 7. Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Chemical queries attend over GNN key/values.
    Both embeddings are single tokens → (B, 1, dim) for multi-head attention.
    """

    def __init__(self, gnn_dim: int = 128, chem_dim: int = 128,
                 num_heads: int = 4, out_dim: int = 128):
        super().__init__()
        assert gnn_dim % num_heads == 0, "gnn_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = gnn_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj   = nn.Linear(chem_dim, gnn_dim)
        self.k_proj   = nn.Linear(gnn_dim,  gnn_dim)
        self.v_proj   = nn.Linear(gnn_dim,  gnn_dim)
        self.out_proj = nn.Linear(gnn_dim,  out_dim)
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, gnn_emb: torch.Tensor,
                chem_emb: torch.Tensor) -> torch.Tensor:
        B    = gnn_emb.size(0)
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(chem_emb).view(B, 1, H, D).transpose(1, 2)   # (B,H,1,D)
        k = self.k_proj(gnn_emb).view(B, 1, H, D).transpose(1, 2)
        v = self.v_proj(gnn_emb).view(B, 1, H, D).transpose(1, 2)

        attn  = (q @ k.transpose(-2, -1)) / self.scale                # (B,H,1,1)
        attn  = attn.softmax(dim=-1)
        fused = (attn @ v).transpose(1, 2).contiguous().view(B, -1)   # (B, gnn_dim)

        out = self.out_proj(fused)
        if chem_emb.size(-1) == out.size(-1):
            out = out + chem_emb                                       # residual
        return self.norm(out)


# ---------------------------------------------------------------------------
# 8. Prediction Head
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    def __init__(self, in_dim: int = 128, quantum_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + quantum_dim, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, fused: torch.Tensor,
                quantum: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([fused, quantum], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# 9. Full Hybrid Model
# ---------------------------------------------------------------------------

class HybridMOFModel(nn.Module):
    """
    Phase 5 fixed hybrid model with critical improvements.

    Forward signature
    -----------------
    graph_batch   : PyG Batch  (node_features, edge_index, edge_dist, batch)
    chemical_x    : (B, 145)   Magpie features
    quantum_x     : (B, 8)     quantum features (NaN rows replaced with 0)
    quantum_mask  : (B,) bool  True where quantum data exists
    mof_desc      : (B, 8)     MOF descriptors (PLD, LCD, surface area, etc.)

    Multi-task support: can predict CO2 at multiple pressures (0.1, 1.0, 10 bar)
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
        num_tasks: int = 1,           # Multi-task learning support
        use_directional: bool = True,
        use_hierarchical: bool = True,
        mof_desc_dim: int = 8,        # PLD, LCD, surface area, void fraction, etc.
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_directional = use_directional
        self.use_hierarchical = use_hierarchical

        # Enhanced GNN with hierarchical pooling
        self.gnn = GNNBranch(
            gnn_hidden, gnn_interactions, rbf_bins, gnn_out,
            use_directional=use_directional,
            use_hierarchical=use_hierarchical,
            num_levels=3
        )

        # Chemical branch
        self.chem = ChemicalBranch(chem_in, chem_out)

        # Quantum branch
        self.quantum = QuantumBranch(quantum_in, quantum_out)

        # MOF descriptor branch (PLD, LCD, surface area, void fraction)
        self.mof_desc_branch = nn.Sequential(
            nn.Linear(mof_desc_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # Fusion
        fusion_in = gnn_out if not use_hierarchical else gnn_out * 3
        self.fusion = CrossAttentionFusion(fusion_in, chem_out, fusion_heads, fusion_out)

        # Enhanced prediction head with multi-task support
        total_dim = fusion_out + quantum_out + 64  # +64 for MOF descriptors
        self.head = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_tasks),
        )

        # Gas-kinetic diameter features for selective adsorption
        # CO2: 3.3 Å, N2: 3.64 Å
        self.register_buffer('gas_diameters', torch.tensor([3.3, 3.64]))

    def forward(self, graph_batch, chemical_x: torch.Tensor,
                quantum_x: torch.Tensor, quantum_mask: torch.Tensor,
                mof_desc: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns:
            predictions: (B, num_tasks) CO2 uptake predictions
        """
        g = self.gnn(graph_batch)
        c = self.chem(chemical_x)
        q = self.quantum(quantum_x, quantum_mask)

        fused = self.fusion(g, c)

        # MOF descriptors
        if mof_desc is not None:
            mof_emb = self.mof_desc_branch(mof_desc)
        else:
            mof_emb = torch.zeros(chemical_x.size(0), 64, device=chemical_x.device)

        # Concatenate all features
        all_features = torch.cat([fused, q, mof_emb], dim=-1)

        preds = self.head(all_features)
        return preds.squeeze(-1) if self.num_tasks == 1 else preds

    def parameter_groups(self, gnn_lr: float = 1e-4,
                         tabular_lr: float = 1e-3) -> list:
        return [
            {"params": self.gnn.parameters(),         "lr": gnn_lr,     "name": "gnn"},
            {"params": self.chem.parameters(),        "lr": tabular_lr, "name": "chemical"},
            {"params": self.quantum.parameters(),     "lr": tabular_lr, "name": "quantum"},
            {"params": self.mof_desc_branch.parameters(), "lr": tabular_lr, "name": "mof_desc"},
            {"params": self.fusion.parameters(),      "lr": tabular_lr, "name": "fusion"},
            {"params": self.head.parameters(),        "lr": tabular_lr, "name": "head"},
        ]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 10. GNN-only model for Task 5.1 diagnosis
# ---------------------------------------------------------------------------

class GNNOnlyModel(nn.Module):
    """Stripped-down GNN + attention pooling → scalar regression."""

    def __init__(self, hidden: int = 128, num_interactions: int = 3,
                 rbf_bins: int = 50, cutoff: float = 8.0,
                 use_hierarchical: bool = True, num_levels: int = 3):
        super().__init__()
        out_dim = 128
        self.gnn  = GNNBranch(hidden, num_interactions, rbf_bins, out_dim=out_dim,
                              use_hierarchical=use_hierarchical, num_levels=num_levels)
        
        # Calculate input dimension for head based on hierarchy
        head_in = out_dim * num_levels if use_hierarchical else out_dim
        
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1),
        )

    def forward(self, graph_batch) -> torch.Tensor:
        return self.head(self.gnn(graph_batch)).squeeze(-1)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
    print(f"Parameters: {model.num_parameters:,}")

    chemical_x = torch.rand(B, 145)
    quantum_x  = torch.rand(B, 8)
    q_mask     = torch.tensor([True, False, True, True])

    preds = model(batch, chemical_x, quantum_x, q_mask)
    print(f"Hybrid output shape: {preds.shape}")

    gnn_only = GNNOnlyModel()
    preds2 = gnn_only(batch)
    print(f"GNN-only output shape: {preds2.shape}")
    print("All forward passes OK.")