"""
mof_train.py — Consolidated Training Pipeline
==============================================
Single-file training script combining train_hybrid.py and train_hybrid_fixed.py
with all improvements: CosineAnnealingWarmRestarts, gradient clipping, layer-wise LR.

Usage:
    # Train GNN-only (diagnostic)
    python mof_train.py --stage gnn_only --epochs 100
    
    # Train full hybrid model
    python mof_train.py --stage hybrid --epochs 150
    
    # Ensemble with multiple seeds
    python mof_train.py --stage ensemble --seeds 3
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

from mof_model import HybridMOFModel, GNNOnlyModel


# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
GRAPH_DIR = DATA_DIR / "graphs"
RESULT_DIR = Path("results")
CKPT_DIR = Path("checkpoints")

RESULT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)


# ── Custom JSON Encoder ─────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# =============================================================================
# Dataset
# =============================================================================

class MOFDataset(Dataset):
    """
    Loads CSV split with corresponding .npz graphs and chemical features.
    
    Args:
        csv_path: Path to CSV file with MOF data
        graph_dir: Directory containing .npz graph files
        chemical_feat_path: Path to chemical_features.npy
        target_col: Target column name(s)
        cutoff: Distance cutoff for edges
    """
    
    def __init__(
        self,
        csv_path,
        graph_dir=GRAPH_DIR,
        chemical_feat_path=None,
        target_col="CO2_uptake_1.0bar",
        cutoff=8.0,
        rbf_bins=50,
        rbf_gamma=10.0,
    ):
        if chemical_feat_path is None:
            chemical_feat_path = str(DATA_DIR / "chemical_features.npy")
        
        if isinstance(csv_path, pd.DataFrame):
            self.df = csv_path.copy()
            csv_path = "DataFrame"
        else:
            self.df = pd.read_csv(csv_path)
        
        # Handle BOM and strip whitespace from column names
        self.df.columns = self.df.columns.str.replace('^\ufeff', '', regex=True).str.strip()
        
        self.graph_dir = Path(graph_dir)
        self.target_col = target_col
        self.cutoff = cutoff
        
        # Load chemical features
        chem_all = np.load(chemical_feat_path).astype(np.float32)
        if "index" in self.df.columns:
            self.chem = chem_all[self.df["index"].values]
        else:
            self.chem = chem_all[self.df.index.values]
        
        # Load targets
        self.multi_task = isinstance(target_col, list)
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
        
        # Get MOF IDs
        if "name" in self.df.columns:
            ids = self.df["name"].values
        elif "filename" in self.df.columns:
            ids = self.df["filename"].values
        elif "mof_id" in self.df.columns:
            ids = self.df["mof_id"].values
        else:
            ids = self.df.index.values.astype(str)
        
        ids = np.array([str(i) for i in ids])
        
        # Filter samples with existing graph files
        valid_indices = []
        for i, mof_id in enumerate(ids):
            path = self.graph_dir / f"{mof_id}.npz"
            if path.exists():
                valid_indices.append(i)
        
        if len(valid_indices) < len(ids):
            print(f"Warning: {len(ids) - len(valid_indices)} samples missing graph files. Removing them.")
            if len(valid_indices) > 0:
                print(f"First valid ID: {ids[valid_indices[0]]}")
            self.df = self.df.iloc[valid_indices]
            self.chem = self.chem[valid_indices]
            self.targets = self.targets[valid_indices]
            self.ids = ids[valid_indices]
        else:
            self.ids = ids
        
        self.df = self.df.reset_index(drop=True)
        self._rbf_bins = rbf_bins
        self._rbf_gamma = rbf_gamma
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        mof_id = self.ids[idx]
        npz_path = self.graph_dir / f"{mof_id}.npz"
        
        # Load graph
        raw = np.load(npz_path)
        node_features = torch.from_numpy(raw["node_features"].astype(np.float32))
        edge_index = torch.from_numpy(raw["edge_index"].astype(np.int64))
        edge_dist = torch.from_numpy(raw["edge_dist"].astype(np.float32).ravel())
        
        graph = Data(
            node_features=node_features,
            edge_index=edge_index,
            edge_dist=edge_dist,
        )
        
        chem = torch.from_numpy(self.chem[idx])
        
        if self.multi_task:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        # Quantum features (placeholder - not used in basic training)
        qfeat = torch.zeros(8)
        qmask = torch.tensor(False)
        
        # MOF descriptors (placeholder)
        mof_desc = torch.zeros(8)
        
        return graph, chem, qfeat, qmask, mof_desc, y


def collate_fn(batch):
    """Collate function for DataLoader."""
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


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4)
    }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, device,
                clip_norm: float = 10.0, gnn_only: bool = False):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    criterion = nn.HuberLoss(delta=0.1)
    
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
            chem = chem.to(device)
            quant = quant.to(device)
            qmask = qmask.to(device)
            y = y.to(device)
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
def evaluate(model, loader, device, gnn_only: bool = False) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset."""
    model.eval()
    all_preds, all_targets = [], []
    
    for batch in loader:
        if gnn_only:
            graph_b, _, _, _, _, y = batch
            graph_b = graph_b.to(device)
            pred = model(graph_b)
        else:
            graph_b, chem, quant, qmask, mof_desc, y = batch
            graph_b = graph_b.to(device)
            chem = chem.to(device)
            quant = quant.to(device)
            qmask = qmask.to(device)
            pred = model(graph_b, chem, quant, qmask, mof_desc)
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(targets, preds)
    
    return metrics, preds, targets


# =============================================================================
# Main Training Function
# =============================================================================

def run_training(
    stage="hybrid",
    seed=42,
    epochs=150,
    batch_size=64,
    gnn_lr=1e-4,
    tabular_lr=1e-3,
    weight_decay=1e-4,
    cutoff=8.0,
    save_prefix="best",
):
    """
    Main training function.
    
    Args:
        stage: "gnn_only" or "hybrid"
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        gnn_lr: Learning rate for GNN branch
        tabular_lr: Learning rate for tabular branches
        weight_decay: Weight decay for optimizer
        cutoff: Distance cutoff for graphs
        save_prefix: Prefix for checkpoint filename
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Stage: {stage} | Seed: {seed}")
    
    # Load datasets
    target_col = "CO2_uptake_1.0bar"
    
    train_ds = MOFDataset(DATA_DIR / "hmof_train.csv", target_col=target_col, cutoff=cutoff)
    val_ds = MOFDataset(DATA_DIR / "hmof_val.csv", target_col=target_col, cutoff=cutoff)
    test_ds = MOFDataset(DATA_DIR / "hmof_test.csv", target_col=target_col, cutoff=cutoff)
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )
    
    gnn_only = (stage == "gnn_only")
    
    # Initialize model
    if gnn_only:
        model = GNNOnlyModel(cutoff=cutoff).to(device)
        optimizer = AdamW(model.parameters(), lr=tabular_lr, weight_decay=weight_decay)
    else:
        model = HybridMOFModel(cutoff=cutoff, num_tasks=1).to(device)
        param_groups = model.parameter_groups(gnn_lr, tabular_lr)
        optimizer = AdamW(param_groups, weight_decay=weight_decay)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # CosineAnnealingWarmRestarts scheduler
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50 * steps_per_epoch, T_mult=2
    )
    
    # Training loop
    ckpt_path = CKPT_DIR / f"{save_prefix}_{stage}_seed{seed}.pt"
    best_val_r2 = -np.inf
    history = {"train_loss": [], "val_r2": [], "val_mae": []}
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, gnn_only=gnn_only
        )
        val_metrics, _, _ = evaluate(model, val_loader, device, gnn_only=gnn_only)
        
        history["train_loss"].append(train_loss)
        history["val_r2"].append(val_metrics["R2"])
        history["val_mae"].append(val_metrics["MAE"])
        
        if val_metrics["R2"] > best_val_r2:
            best_val_r2 = val_metrics["R2"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_r2": best_val_r2,
            }, ckpt_path)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Ep {epoch:4d} | loss={train_loss:.4f} | val R²={val_metrics['R2']:.4f} | best={best_val_r2:.4f}")
    
    # Final evaluation
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, device, gnn_only=gnn_only
    )
    
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Stage: {stage} Seed: {seed} ({elapsed:.0f}s)")
    print(f"Test R²={test_metrics['R2']:.4f} MAE={test_metrics['MAE']:.4f} RMSE={test_metrics['RMSE']:.4f}")
    print(f"{'='*55}\n")
    
    result = {
        "stage": stage,
        "seed": seed,
        "best_val_r2": round(best_val_r2, 4),
        "test": test_metrics,
        "elapsed_s": round(elapsed),
        "history": history,
        "ckpt": str(ckpt_path),
    }
    
    out_path = RESULT_DIR / f"train_{stage}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NpEncoder)
    print(f"Results saved → {out_path}")
    
    return result


def run_ensemble(seeds=(42, 43, 44), stage="hybrid", **kwargs):
    """Train multiple seeds and ensemble predictions."""
    all_preds = []
    all_targets = None
    results = []
    
    for s in seeds:
        res = run_training(stage=stage, seed=s, **kwargs)
        results.append(res)
        
        # Reload and predict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if stage == "gnn_only":
            model = GNNOnlyModel().to(device)
        else:
            model = HybridMOFModel().to(device)
        
        ckpt = torch.load(res["ckpt"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        
        test_ds = MOFDataset(DATA_DIR / "hmof_test.csv", target_col="CO2_uptake_1.0bar")
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        _, preds, targets = evaluate(
            model, test_loader, device,
            gnn_only=(stage == "gnn_only")
        )
        all_preds.append(preds)
        all_targets = targets
    
    # Ensemble
    stacked_preds = np.stack(all_preds)
    ensemble_preds = np.mean(stacked_preds, axis=0)
    ens_metrics = compute_metrics(all_targets, ensemble_preds)
    
    print(f"\nEnsemble ({len(seeds)} seeds): R²={ens_metrics['R2']:.4f}")
    
    out = {
        "seeds": list(seeds),
        "ensemble_test": ens_metrics,
        "per_seed": [r["test"] for r in results],
    }
    out_path = RESULT_DIR / f"ensemble_{stage}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, cls=NpEncoder)
    return out


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MOF CO2 adsorption model")
    parser.add_argument("--stage", choices=["gnn_only", "hybrid", "ensemble"], default="hybrid")
    parser.add_argument("--seeds", type=int, default=1, help="Number of ensemble seeds")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--gnn_lr", type=float, default=1e-4)
    parser.add_argument("--tabular_lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    if args.stage == "ensemble" or args.seeds > 1:
        seed_list = list(range(42, 42 + args.seeds))
        run_ensemble(
            seeds=seed_list,
            stage="hybrid",
            epochs=args.epochs,
            batch_size=args.batch_size,
            cutoff=args.cutoff,
            gnn_lr=args.gnn_lr,
            tabular_lr=args.tabular_lr,
        )
    else:
        run_training(
            stage=args.stage,
            seed=42,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cutoff=args.cutoff,
            gnn_lr=args.gnn_lr,
            tabular_lr=args.tabular_lr,
        )