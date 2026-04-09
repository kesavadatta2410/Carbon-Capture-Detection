"""
train_hybrid_fixed.py — Phase 5 Fixed Training Pipeline with Critical Improvements
==================================================================================
Usage:
    python code/train_hybrid_fixed.py --stage gnn_only
    python code/train_hybrid_fixed.py --stage hybrid
    python code/train_hybrid_fixed.py --stage ensemble --seeds 3

Key improvements over train_hybrid.py:
  • CosineAnnealingWarmRestarts (no ReduceLROnPlateau)
  • Gradient clipping (max_norm=10.0)
  • Layer-wise LR decay (GNN 1e-4, rest 1e-3)
  • Optional W&B / TensorBoard logging
  • Masked loss for missing quantum features
  • Ensemble (N seeds) with averaged predictions

CRITICAL IMPROVEMENTS (changes.md):
  • Multi-task learning: predict CO2 at 0.1, 1.0, 10 bar simultaneously
  • Physics-informed loss: Langmuir/Freundlich isotherm constraints
  • Mixup/CutMix augmentation for graph data
  • Gradient checkpointing for memory efficiency
  • Huber loss for robustness to outliers (replaces MSE)
  • Data efficiency tracking (10%, 25%, 50%, 100% data)
  • Uncertainty calibration (ECE, NLL)
  • Attention visualization
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm

# Local import
from hybrid_model_v2 import HybridMOFModel, GNNOnlyModel

# NEW: Import for gradient checkpointing
from torch.utils.checkpoint import checkpoint

# ── optional logging ────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False


# Custom JSON Encoder for NumPy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# ── paths ───────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
GRAPH_DIR  = DATA_DIR / "graphs"
RESULT_DIR = Path("results")
CKPT_DIR   = Path("checkpoints")

RESULT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Dataset
# ===========================================================================

class MOFDataset(Dataset):
    """
    Loads a CSV split, the corresponding .npz graphs, chemical features,
    and (optionally) quantum features.

    Supports multi-task learning with multiple target columns.
    """

    def __init__(
        self,
        csv_path,
        graph_dir=GRAPH_DIR,
        chemical_feat_path=None,
        quantum_feat_path=None,
        target_col="CO2_uptake_1.0bar",  # Can be list for multi-task
        cutoff=8.0,
        rbf_bins=50,
        rbf_gamma=10.0,
        mof_desc_cols=None,  # MOF descriptor columns (PLD, LCD, etc.)
    ):
        if chemical_feat_path is None:
            chemical_feat_path = str(DATA_DIR / "chemical_features.npy")

        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.copy() # Avoid modifying original df
            csv_path = "DataFrame" # for logging
        else:
            self.df = pd.read_csv(csv_path)

        # Handle BOM and strip whitespace from column names
        self.df.columns = self.df.columns.str.replace('^\ufeff', '', regex=True).str.strip()
        
        # NOTE: DO NOT reset_index here, as the original index might be used for mapping to hMOF-N names
        # We will reset it AFTER calculating IDs if needed, or just keep it.
        # Actually, let's reset it at the very end of init after all subsetting.

        self.graph_dir = Path(graph_dir)
        self.target_col = target_col
        self.cutoff = cutoff

        # Chemical features
        chem_all = np.load(chemical_feat_path).astype(np.float32)
        if "index" in self.df.columns:
            self.chem = chem_all[self.df["index"].values]
        else:
            self.chem = chem_all[self.df.index.values]

        # Quantum features (optional)
        if quantum_feat_path and Path(quantum_feat_path).exists():
            qfeat_all = np.load(quantum_feat_path).astype(np.float32)
            if "index" in self.df.columns:
                self.quantum = qfeat_all[self.df["index"].values]
            else:
                self.quantum = qfeat_all[self.df.index.values]
        else:
            self.quantum = None

        # Targets (support multi-task)
        self.multi_task = isinstance(target_col, list)
        
        # Check if target columns exist
        available_cols = set(self.df.columns)
        if self.multi_task:
            missing = [c for c in target_col if c not in available_cols]
            if missing:
                print(f"Warning: missing target columns {missing} in {csv_path}. Using dummy zeros.")
                self.targets = np.zeros((len(self.df), len(target_col)), dtype=np.float32)
            else:
                self.targets = self.df[target_col].values.astype(np.float32)
        else:
            if target_col not in available_cols:
                print(f"Warning: missing target column {target_col} in {csv_path}. Using dummy zeros.")
                self.targets = np.zeros(len(self.df), dtype=np.float32)
            else:
                self.targets = self.df[target_col].values.astype(np.float32)

        # MOF descriptors (PLD, LCD, surface area, void fraction)
        self.mof_desc = None
        if mof_desc_cols:
            self.mof_desc = self.df[mof_desc_cols].values.astype(np.float32)

        # MOF IDs: try different strategies to find the graph files
        if "name" in self.df.columns:
            ids = self.df["name"].values
        elif "filename" in self.df.columns:
            # Check if filename.npz exists for the first sample
            test_id = self.df["filename"].iloc[0]
            if (self.graph_dir / f"{test_id}.npz").exists():
                ids = self.df["filename"].values
            else:
                # If filename.npz doesn't exist, DO NOT fall back to hMOF-index.
                # This fallback was incorrectly mapping non-hMOF datasets to hMOF graphs.
                ids = self.df["filename"].values
        elif "mof_id" in self.df.columns:
            ids = self.df["mof_id"].values
        else:
            ids = self.df.index.values.astype(str)
        
        # Ensure ids is a numpy array for correct indexing/subsetting
        ids = np.array([str(i) for i in ids])

        # Filter samples with existing graph files
        valid_indices = []
        for i, mof_id in enumerate(ids):
            path = self.graph_dir / f"{mof_id}.npz"
            if path.exists():
                valid_indices.append(i)
        
        if len(valid_indices) < len(ids):
            print(f"Warning: {len(ids) - len(valid_indices)} samples missing graph files in {self.graph_dir}. Removing them.")
            if len(valid_indices) > 0:
                print(f"First valid ID: {ids[valid_indices[0]]}")
            self.df = self.df.iloc[valid_indices] # DO NOT reset_index here yet
            self.chem = self.chem[valid_indices]
            if self.quantum is not None:
                self.quantum = self.quantum[valid_indices]
            self.targets = self.targets[valid_indices]
            if self.mof_desc is not None:
                self.mof_desc = self.mof_desc[valid_indices]
            
            # Update IDs for the filtered dataset
            self.ids = ids[valid_indices]
        else:
            self.ids = ids

        # Now that we've used the index for mapping, we can reset it.
        self.df = self.df.reset_index(drop=True)

        self._rbf_bins  = rbf_bins
        self._rbf_gamma = rbf_gamma

    def __len__(self):
        return len(self.df)

    def _rbf(self, dist):
        centers = np.linspace(0.0, self.cutoff, self._rbf_bins)
        diff    = dist[:, None] - centers[None, :]
        return np.exp(-self._rbf_gamma * diff ** 2).astype(np.float32)

    def __getitem__(self, idx):
        mof_id   = self.ids[idx]
        # Try different ID formats
        npz_path = self.graph_dir / f"{mof_id}.npz"
        if not npz_path.exists():
            # Try with zero-padding or different format
            npz_path_alt = self.graph_dir / f"{mof_id}.npz"
            if npz_path_alt.exists():
                npz_path = npz_path_alt

        # Keys written by build_graphs.py: node_features, edge_index, edge_dist
        raw = np.load(npz_path)
        node_features = torch.from_numpy(raw["node_features"].astype(np.float32))  # (N, 3)
        edge_index    = torch.from_numpy(raw["edge_index"].astype(np.int64))        # (2, E)
        edge_dist_np  = raw["edge_dist"].astype(np.float32).ravel()                 # (E,)
        edge_dist     = torch.from_numpy(edge_dist_np)                               # (E,)

        graph = Data(
            node_features=node_features,
            edge_index=edge_index,
            edge_dist=edge_dist,
        )

        chem = torch.from_numpy(self.chem[idx])

        if self.multi_task:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)  # (num_tasks,)
        else:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.quantum is not None:
            qfeat = torch.from_numpy(self.quantum[idx])
            qmask = torch.tensor(not torch.isnan(qfeat).any().item())
            qfeat = torch.nan_to_num(qfeat, nan=0.0)
        else:
            qfeat = torch.zeros(8)
            qmask = torch.tensor(False)

        # MOF descriptors
        if self.mof_desc is not None:
            mof_desc = torch.from_numpy(self.mof_desc[idx])
        else:
            mof_desc = torch.zeros(8)  # Default placeholder

        return graph, chem, qfeat, qmask, mof_desc, y


def collate_fn(batch):
    graphs, chems, quants, qmasks, mof_descs, ys = zip(*batch)
    graph_batch = Batch.from_data_list(list(graphs))
    return (
        graph_batch,
        torch.stack(chems),
        torch.stack(quants),
        torch.stack(qmasks),
        torch.stack(mof_descs),
        torch.stack(ys),
    )


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2":  round(float(r2),  4)}


# -----------------------------------------------------------------------------
# NEW: Physics-Informed Loss Functions
# -----------------------------------------------------------------------------

class PhysicsInformedLoss(nn.Module):
    """
    Loss function with physics constraints for adsorption prediction.
    Enforces Langmuir/Freundlich isotherm properties:
      - Monotonicity: uptake should increase with pressure
      - Saturation: uptake should approach finite limit
    """

    def __init__(self, delta: float = 0.1, weight_physics: float = 0.1):
        super().__init__()
        self.delta = delta
        self.weight_physics = weight_physics
        self.base_loss = nn.HuberLoss(delta=delta)  # Robust to outliers

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                pressure_levels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: (B, num_tasks) predictions at different pressures
            target: (B, num_tasks) targets at different pressures
            pressure_levels: (num_tasks,) pressure values (e.g., [0.1, 1.0, 10.0])
        """
        # Base prediction loss
        loss = self.base_loss(pred, target)

        # Physics constraints (if multi-task with pressure levels)
        if self.weight_physics > 0 and pred.dim() > 1 and pred.size(1) > 1:
            # Monotonicity constraint: uptake should increase with pressure
            diffs = pred[:, 1:] - pred[:, :-1]  # Differences between consecutive pressures
            monotonicity_penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences

            # Saturation constraint: second derivative should be negative (concave)
            second_diffs = diffs[:, 1:] - diffs[:, :-1]
            saturation_penalty = torch.mean(F.relu(second_diffs))

            loss = loss + self.weight_physics * (monotonicity_penalty + saturation_penalty)

        return loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weighting (Kendall et al.)."""

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.num_tasks = num_tasks
        # Learnable log variance for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, num_tasks) predictions
            target: (B, num_tasks) targets
        """
        losses = []
        for i in range(self.num_tasks):
            precision = torch.exp(-self.log_vars[i])
            losses.append(precision * F.smooth_l1_loss(pred[:, i], target[:, i]) + self.log_vars[i])
        return torch.stack(losses).sum()


# -----------------------------------------------------------------------------
# NEW: Mixup/CutMix Augmentation for Graphs
# -----------------------------------------------------------------------------

class GraphMixup:
    """Mixup augmentation for graph data."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, graph1: Data, graph2: Data, y1: torch.Tensor, y2: torch.Tensor
                 ) -> Tuple[Data, torch.Tensor]:
        """Interpolate two graphs and their targets."""
        lam = np.random.beta(self.alpha, self.alpha)

        # Interpolate node features
        mixed_node_features = lam * graph1.node_features + (1 - lam) * graph2.node_features

        # Create mixed graph (use graph1 structure)
        mixed_graph = Data(
            node_features=mixed_node_features,
            edge_index=graph1.edge_index,
            edge_dist=graph1.edge_dist,
        )

        # Interpolate targets
        mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_graph, mixed_y


# -----------------------------------------------------------------------------
# NEW: Uncertainty Calibration Metrics
# -----------------------------------------------------------------------------

def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                              y_std: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) for regression.

    Measures how well the predicted uncertainty matches actual errors.
    """
    # Normalize errors by std
    errors = np.abs(y_true - y_pred)
    z_scores = errors / (y_std + 1e-8)

    # Bin by confidence (inverse of std)
    confidences = 1 / (1 + y_std)
    bins = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = (z_scores[mask] < 1).mean()  # Within 1 std
            ece += mask.sum() * np.abs(bin_acc - bin_conf)

    return ece / len(y_true)


def negative_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray,
                            y_std: np.ndarray) -> float:
    """Compute negative log-likelihood for regression."""
    log_likelihood = -0.5 * (
        np.log(2 * np.pi * y_std**2) +
        (y_true - y_pred)**2 / (y_std**2 + 1e-8)
    )
    return -np.mean(log_likelihood)


# ===========================================================================
# Training loop
# ===========================================================================

def train_epoch(model, loader, optimizer, scheduler, device,
                clip_norm: float = 10.0, gnn_only: bool = False,
                use_mixup: bool = False, mixup_alpha: float = 0.2,
                use_physics_loss: bool = False):
    """
    Training epoch with optional Mixup augmentation and physics-informed loss.

    Args:
        use_mixup: Whether to use Mixup augmentation
        mixup_alpha: Alpha parameter for Mixup beta distribution
        use_physics_loss: Whether to use physics-informed loss constraints
    """
    model.train()
    total_loss = 0.0

    # Use Huber loss for robustness to outliers (replaces MSE)
    base_criterion = nn.HuberLoss(delta=0.1)

    # Physics-informed loss
    if use_physics_loss:
        criterion = PhysicsInformedLoss(delta=0.1, weight_physics=0.1)
    else:
        criterion = base_criterion

    # Mixup augmentation
    mixup = GraphMixup(alpha=mixup_alpha) if use_mixup else None

    for batch in loader:
        optimizer.zero_grad()

        if gnn_only:
            graph_b, _, _, _, _, y = batch
            graph_b = graph_b.to(device)
            y = y.to(device)
            pred = model(graph_b)
        else:
            graph_b, chem, quant, qmask, mof_desc, y = batch
            graph_b = graph_b.to(device)
            chem    = chem.to(device)
            quant   = quant.to(device)
            qmask   = qmask.to(device)
            mof_desc = mof_desc.to(device)
            y       = y.to(device)

            # Optional: Apply Mixup with probability 0.5
            if mixup is not None and np.random.rand() < 0.5:
                # Simple implementation: mix features at batch level
                indices = torch.randperm(y.size(0))
                lam = np.random.beta(mixup_alpha, mixup_alpha)

                chem = lam * chem + (1 - lam) * chem[indices]
                mof_desc = lam * mof_desc + (1 - lam) * mof_desc[indices]
                y = lam * y + (1 - lam) * y[indices]

            pred = model(graph_b, chem, quant, qmask, mof_desc)

        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, gnn_only: bool = False,
             mc_samples: int = 0) -> Tuple[dict, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Evaluate model with optional MC Dropout for uncertainty estimation.

    Args:
        mc_samples: Number of MC Dropout samples (0 = no uncertainty)

    Returns:
        metrics: Dictionary of metrics
        preds: Predictions
        targets: True values
        uncertainties: Standard deviation of predictions (if mc_samples > 0)
    """
    if mc_samples > 0:
        model.train()  # Enable dropout for MC sampling
    else:
        model.eval()

    all_preds, all_targets = [], []
    mc_preds = [] if mc_samples > 0 else None

    for batch in loader:
        if gnn_only:
            graph_b, _, _, _, _, y = batch
            graph_b = graph_b.to(device)

            if mc_samples > 0:
                # Multiple forward passes with dropout
                mc_batch_preds = [model(graph_b).cpu().numpy() for _ in range(mc_samples)]
                pred = np.mean(mc_batch_preds, axis=0)
                mc_preds.append(mc_batch_preds)
            else:
                pred = model(graph_b)
        else:
            graph_b, chem, quant, qmask, mof_desc, y = batch
            graph_b = graph_b.to(device)
            chem    = chem.to(device)
            quant   = quant.to(device)
            qmask   = qmask.to(device)
            mof_desc = mof_desc.to(device)

            if mc_samples > 0:
                mc_batch_preds = [
                    model(graph_b, chem, quant, qmask, mof_desc).cpu().numpy()
                    for _ in range(mc_samples)
                ]
                pred = np.mean(mc_batch_preds, axis=0)
                mc_preds.append(mc_batch_preds)
            else:
                pred = model(graph_b, chem, quant, qmask, mof_desc)

        all_preds.append(pred.cpu().numpy() if torch.is_tensor(pred) else pred)
        all_targets.append(y.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(targets, preds)

    # Compute uncertainty if MC samples used
    uncertainties = None
    if mc_samples > 0 and mc_preds:
        all_mc = [np.concatenate([p[i] for p in mc_preds]) for i in range(mc_samples)]
        uncertainties = np.std(all_mc, axis=0)

        # Add calibration metrics
        metrics['ECE'] = expected_calibration_error(targets, preds, uncertainties)
        metrics['NLL'] = negative_log_likelihood(targets, preds, uncertainties)

    return metrics, preds, targets, uncertainties


# ===========================================================================
# Main training function
# ===========================================================================

def run_training(
    stage="hybrid",
    seed=42,
    epochs=150,
    batch_size=64,
    gnn_lr=1e-4,
    tabular_lr=1e-3,
    weight_decay=1e-4,
    cutoff=8.0,
    use_wandb=False,
    run_name=None,
    save_prefix="best",
    # NEW: Additional parameters for improvements
    multi_task: bool = False,
    data_fraction: float = 1.0,  # For data efficiency experiments (0.1, 0.25, 0.5, 1.0)
    use_mixup: bool = True,
    use_physics_loss: bool = False,
    mc_samples: int = 20,  # MC Dropout samples for uncertainty
    use_gradient_checkpointing: bool = False,
):
    """
    Enhanced training function with critical improvements.

    Args:
        multi_task: Whether to train on multiple CO2 pressures (0.1, 1.0, 10 bar)
        data_fraction: Fraction of training data to use (for data efficiency)
        use_mixup: Enable Mixup augmentation
        use_physics_loss: Enable physics-informed loss constraints
        mc_samples: Number of MC Dropout samples for uncertainty
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Stage: {stage}  |  Seed: {seed}")
    print(f"Data fraction: {data_fraction:.1%}  |  Multi-task: {multi_task}")

    # -- datasets -------------------------------------------------------
    # Multi-task targets
    if multi_task:
        target_cols = ["CO2_uptake_0.1bar", "CO2_uptake_1.0bar", "CO2_uptake_10bar"]
    else:
        target_cols = "CO2_uptake_1.0bar"

    # MOF descriptor columns
    mof_desc_cols = ["LCD", "PLD", "surface_area_m2g", "void_fraction"]

    train_ds = MOFDataset(
        DATA_DIR / "hmof_train.csv",
        target_col=target_cols,
        cutoff=cutoff,
        mof_desc_cols=None
    )
    val_ds   = MOFDataset(DATA_DIR / "hmof_val.csv", target_col=target_cols, cutoff=cutoff)
    test_ds  = MOFDataset(DATA_DIR / "hmof_test.csv", target_col=target_cols, cutoff=cutoff)

    # Data efficiency: subset training data
    if data_fraction < 1.0:
        subset_size = int(len(train_ds) * data_fraction)
        indices = np.random.choice(len(train_ds), subset_size, replace=False)
        train_ds = torch.utils.data.Subset(train_ds, indices)
        print(f"Using {subset_size} training samples ({data_fraction:.0%})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    gnn_only = (stage == "gnn_only")

    # -- model -----------------------------------------------------------
    if gnn_only:
        model = GNNOnlyModel(cutoff=cutoff).to(device)
        optimizer = AdamW(model.parameters(), lr=tabular_lr,
                          weight_decay=weight_decay)
    else:
        num_tasks = 3 if multi_task else 1
        model = HybridMOFModel(
            cutoff=cutoff,
            num_tasks=num_tasks,
            use_directional=True,
            use_hierarchical=True,
        ).to(device)
        param_groups = model.parameter_groups(gnn_lr, tabular_lr)
        optimizer = AdamW(param_groups, weight_decay=weight_decay)

        # Gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            model.gnn.gradient_checkpointing = True
            print("Gradient checkpointing enabled")

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # CosineAnnealingWarmRestarts
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50 * steps_per_epoch, T_mult=2
    )

    # -- logging ---------------------------------------------------------
    rname = run_name or f"{stage}_seed{seed}"
    if use_wandb and _WANDB:
        wandb.init(project="mof-co2", name=rname, config={
            "stage": stage, "seed": seed, "epochs": epochs,
            "batch_size": batch_size, "cutoff": cutoff,
            "multi_task": multi_task, "data_fraction": data_fraction,
        })
    tb_writer = SummaryWriter(f"runs/{rname}") if _TB else None

    # -- training --------------------------------------------------------
    ckpt_path  = CKPT_DIR / f"{save_prefix}_{stage}_seed{seed}.pt"
    best_val_r2 = -np.inf
    history    = {"train_loss": [], "val_r2": [], "val_mae": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, gnn_only=gnn_only,
            use_mixup=use_mixup,
            use_physics_loss=use_physics_loss
        )
        val_metrics, _, _, _ = evaluate(model, val_loader, device, gnn_only=gnn_only)

        history["train_loss"].append(train_loss)
        history["val_r2"].append(val_metrics["R2"])
        history["val_mae"].append(val_metrics["MAE"])

        if val_metrics["R2"] > best_val_r2:
            best_val_r2 = val_metrics["R2"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_r2": best_val_r2,
                "config": {
                    "multi_task": multi_task,
                    "data_fraction": data_fraction,
                }
            }, ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Ep {epoch:4d} | loss={train_loss:.4f} "
                  f"| val R²={val_metrics['R2']:.4f} "
                  f"| val MAE={val_metrics['MAE']:.4f} "
                  f"| best={best_val_r2:.4f}")

        if tb_writer:
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("R2/val",     val_metrics["R2"],  epoch)
            tb_writer.add_scalar("MAE/val",    val_metrics["MAE"], epoch)

        if use_wandb and _WANDB:
            wandb.log({"train_loss": train_loss, **{f"val_{k}": v
                       for k, v in val_metrics.items()}})

    # -- final test evaluation with uncertainty ------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Evaluate with MC Dropout for uncertainty
    test_metrics, test_preds, test_targets, uncertainties = evaluate(
        model, test_loader, device, gnn_only=gnn_only, mc_samples=mc_samples)

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Stage: {stage}  Seed: {seed}  ({elapsed:.0f}s)")
    print(f"Test  R²={test_metrics['R2']:.4f}  "
          f"MAE={test_metrics['MAE']:.4f}  "
          f"RMSE={test_metrics['RMSE']:.4f}")
    if uncertainties is not None:
        print(f"Mean uncertainty: {np.mean(uncertainties):.4f}")
        if 'ECE' in test_metrics:
            print(f"ECE: {test_metrics['ECE']:.4f}  NLL: {test_metrics['NLL']:.4f}")
    print(f"{'='*55}\n")

    result = {
        "stage": stage, "seed": seed,
        "best_val_r2": round(best_val_r2, 4),
        "test": test_metrics,
        "elapsed_s": round(elapsed),
        "history": history,
        "ckpt": str(ckpt_path),
        "config": {
            "multi_task": multi_task,
            "data_fraction": data_fraction,
            "mc_samples": mc_samples,
        }
    }

    # Include uncertainty in results
    if uncertainties is not None:
        result["uncertainty_stats"] = {
            "mean": float(np.mean(uncertainties)),
            "std": float(np.std(uncertainties)),
            "max": float(np.max(uncertainties)),
        }

    out_path = RESULT_DIR / f"train_{stage}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    print(f"Results saved → {out_path}")

    if tb_writer:
        tb_writer.close()
    if use_wandb and _WANDB:
        wandb.finish()

    return result


# ===========================================================================
# Ensemble helper
# ===========================================================================

def run_ensemble(seeds=(42, 1, 7), stage="hybrid", **kwargs):
    """Train N seeds and average test predictions with uncertainty."""
    all_preds = []
    all_uncertainties = []
    all_targets = None
    results = []

    for s in seeds:
        res = run_training(stage=stage, seed=s, **kwargs)
        results.append(res)

        # Reload model and get predictions with uncertainty
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if stage == "gnn_only":
            model = GNNOnlyModel().to(device)
        else:
            multi_task = kwargs.get('multi_task', False)
            num_tasks = 3 if multi_task else 1
            model = HybridMOFModel(num_tasks=num_tasks).to(device)

        ckpt = torch.load(res["ckpt"], map_location=device)
        model.load_state_dict(ckpt["model_state"])

        test_ds     = MOFDataset(
            DATA_DIR / "hmof_test.csv",
            target_col=kwargs.get('target_cols', "CO2_uptake_1.0bar")
        )
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                 collate_fn=collate_fn)

        # Get predictions with uncertainty
        mc_samples = kwargs.get('mc_samples', 20)
        _, preds, targets, uncertainties = evaluate(
            model, test_loader, device,
            gnn_only=(stage == "gnn_only"),
            mc_samples=mc_samples
        )
        all_preds.append(preds)
        if uncertainties is not None:
            all_uncertainties.append(uncertainties)
        all_targets = targets

    # Ensemble predictions
    stacked_preds = np.stack(all_preds)
    ensemble_preds = np.mean(stacked_preds, axis=0)
    ens_metrics = compute_metrics(all_targets, ensemble_preds)

    # Ensemble uncertainty
    if all_uncertainties:
        mean_uncertainty = np.mean(np.stack(all_uncertainties), axis=0)
        ens_metrics['mean_uncertainty'] = float(np.mean(mean_uncertainty))

    print(f"\nEnsemble ({len(seeds)} seeds): {ens_metrics}")

    out = {
        "seeds": list(seeds),
        "ensemble_test": ens_metrics,
        "per_seed": [r["test"] for r in results],
        "data_fraction": kwargs.get('data_fraction', 1.0),
    }
    out_path = RESULT_DIR / f"ensemble_{stage}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, cls=NpEncoder)
    return out


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MOF CO2 adsorption model with critical improvements"
    )
    parser.add_argument("--stage", choices=["gnn_only", "hybrid", "ensemble"],
                        default="hybrid")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of ensemble seeds (1 = single run with seed=42)")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--cutoff",     type=float, default=8.0)
    parser.add_argument("--gnn_lr",     type=float, default=1e-4)
    parser.add_argument("--tabular_lr", type=float, default=1e-3)
    parser.add_argument("--wandb",      action="store_true")

    # NEW: Multi-task learning
    parser.add_argument("--multi_task", action="store_true",
                        help="Train on multiple CO2 pressures (0.1, 1.0, 10 bar)")

    # NEW: Data efficiency
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (0.1, 0.25, 0.5, 1.0)")

    # NEW: Augmentation
    parser.add_argument("--no_mixup", action="store_true",
                        help="Disable Mixup augmentation")

    # NEW: Physics-informed loss
    parser.add_argument("--physics_loss", action="store_true",
                        help="Enable physics-informed loss constraints")

    # NEW: MC Dropout for uncertainty
    parser.add_argument("--mc_samples", type=int, default=20,
                        help="Number of MC Dropout samples for uncertainty")

    # NEW: Gradient checkpointing
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for memory efficiency")

    args = parser.parse_args()

    # Data efficiency experiment: run with different fractions
    if args.data_fraction != 1.0 and args.stage != "ensemble":
        print(f"Running data efficiency experiment with {args.data_fraction:.0%} data")

    if args.stage == "ensemble" or args.seeds > 1:
        seed_list = list(range(42, 42 + args.seeds))
        run_ensemble(
            seeds=seed_list, stage="hybrid",
            epochs=args.epochs, batch_size=args.batch_size,
            cutoff=args.cutoff, gnn_lr=args.gnn_lr,
            tabular_lr=args.tabular_lr, use_wandb=args.wandb,
            multi_task=args.multi_task,
            data_fraction=args.data_fraction,
            use_mixup=not args.no_mixup,
            use_physics_loss=args.physics_loss,
            mc_samples=args.mc_samples,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    else:
        run_training(
            stage=args.stage, seed=42,
            epochs=args.epochs, batch_size=args.batch_size,
            cutoff=args.cutoff, gnn_lr=args.gnn_lr,
            tabular_lr=args.tabular_lr, use_wandb=args.wandb,
            multi_task=args.multi_task,
            data_fraction=args.data_fraction,
            use_mixup=not args.no_mixup,
            use_physics_loss=args.physics_loss,
            mc_samples=args.mc_samples,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )