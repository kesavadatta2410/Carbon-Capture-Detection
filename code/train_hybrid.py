#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4.7: Training loop for the Hybrid MOF Model.

- MSE loss
- Adam optimizer (lr=0.001, weight_decay=1e-5)
- ReduceLROnPlateau (patience=10, factor=0.5)
- Early stopping (patience=25)
- Batch size 32
- Target: CO2_uptake_1.0bar
"""

import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import our model
import sys
sys.path.insert(0, os.path.dirname(__file__))
from hybrid_model import HybridMOFModel, predict_with_uncertainty

# -- Paths ------------------------------------------------------------------
DATA_DIR = "data"
GRAPH_DIR = os.path.join(DATA_DIR, "graphs")
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
TARGET_COL = "CO2_uptake_1.0bar"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -- Training Config --------------------------------------------------------
CONFIG = {
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "scheduler_patience": 10,
    "scheduler_factor": 0.5,
    "early_stop_patience": 25,
    "max_epochs": 200,
    "mc_samples": 20,
    "target": TARGET_COL,
}


# =============================================================================
# Custom Dataset -- loads graph + tabular features per sample
# =============================================================================

class MOFDataset:
    """
    Loads PyG Data objects with graph + chemical + quantum features attached.
    """

    def __init__(self, df, chem_all, name_to_idx, qmof_cols, feature_cols):
        self.df = df.reset_index(drop=True)
        self.chem_all = chem_all
        self.name_to_idx = name_to_idx
        self.qmof_cols = qmof_cols
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["name"]

        # Load graph
        graph_path = os.path.join(GRAPH_DIR, f"{name}.npz")
        g = np.load(graph_path)
        node_features = torch.tensor(g["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(g["edge_dist"], dtype=torch.float32)

        # Chemical features
        chem_idx = self.name_to_idx[name]
        chem = torch.tensor(self.chem_all[chem_idx], dtype=torch.float32)

        # Quantum features (fill NaN with 0)
        quantum = torch.tensor(
            row[self.qmof_cols].values.astype(np.float32),
            dtype=torch.float32
        )
        quantum = torch.nan_to_num(quantum, nan=0.0)

        # Target
        y = torch.tensor(row[TARGET_COL], dtype=torch.float32)

        # Create Data object with extra attributes
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            chem=chem,
            quantum=quantum,
        )
        return data


def collate_fn(data_list):
    """Custom collate to separate graph batch from tabular tensors."""
    batch = Batch.from_data_list(data_list)

    # Stack tabular features
    chem = torch.stack([d.chem for d in data_list], dim=0)
    quantum = torch.stack([d.quantum for d in data_list], dim=0)
    y = torch.stack([d.y for d in data_list], dim=0)

    return batch, chem, quantum, y


# =============================================================================
# Data Loading
# =============================================================================

def load_datasets():
    """Load train/val/test DataLoaders."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "hmof_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "hmof_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "hmof_test.csv"))

    with open(os.path.join(DATA_DIR, "preprocess_config.json")) as f:
        cfg = json.load(f)
    feature_cols = cfg["feature_columns"]

    full_df = pd.read_csv(os.path.join(DATA_DIR, "hmof_enhanced.csv"))
    chem_all = np.load(os.path.join(DATA_DIR, "chemical_features.npy"))
    name_to_idx = {name: i for i, name in enumerate(full_df["name"])}

    # Identify QMOF columns
    qmof_cols = [c for c in train_df.columns if "qmof" in c]

    print(f"  QMOF columns: {len(qmof_cols)}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Chemical dim: {chem_all.shape[1]}")

    use_cuda = torch.cuda.is_available()

    def make_loader(df, shuffle):
        ds = MOFDataset(df, chem_all, name_to_idx, qmof_cols, feature_cols)
        return torch.utils.data.DataLoader(
            ds, batch_size=CONFIG["batch_size"], shuffle=shuffle,
            collate_fn=collate_fn, num_workers=2 if use_cuda else 0,
            pin_memory=use_cuda, persistent_workers=True if use_cuda else False,
        )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    return train_loader, val_loader, test_loader, len(qmof_cols)


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_samples = 0

    for batch, chem, quantum, y in loader:
        batch = batch.to(device)
        chem = chem.to(device)
        quantum = quantum.to(device)
        y = y.to(device)

        pred = model(batch, chem, quantum)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n_samples += y.size(0)

    return total_loss / n_samples


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    all_pred = []
    all_true = []

    for batch, chem, quantum, y in loader:
        batch = batch.to(device)
        chem = chem.to(device)
        quantum = quantum.to(device)
        y = y.to(device)

        pred = model(batch, chem, quantum)
        loss = criterion(pred, y)

        total_loss += loss.item() * y.size(0)
        n_samples += y.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    avg_loss = total_loss / n_samples
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)

    return avg_loss, mae, rmse, r2


# =============================================================================
# Uncertainty Evaluation
# =============================================================================

def evaluate_with_uncertainty(model, loader, device, n_samples=20):
    """Run MC Dropout on test set."""
    from hybrid_model import enable_mc_dropout

    model.eval()
    enable_mc_dropout(model)

    all_means = []
    all_stds = []
    all_true = []

    for batch, chem, quantum, y in loader:
        batch = batch.to(device)
        chem = chem.to(device)
        quantum = quantum.to(device)

        preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = model(batch, chem, quantum)
            preds.append(pred.cpu())

        preds = torch.stack(preds, dim=0)
        all_means.append(preds.mean(0).numpy())
        all_stds.append(preds.std(0).numpy())
        all_true.append(y.numpy())

    model.eval()

    means = np.concatenate(all_means)
    stds = np.concatenate(all_stds)
    trues = np.concatenate(all_true)

    return means, stds, trues


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Task 4.7: Train Hybrid MOF Model")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\nDevice: {device} ({torch.cuda.get_device_name(0)})")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"\nDevice: {device}")
        print("  WARNING: No GPU detected, training will be slow!")

    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, n_qmof = load_datasets()

    # Build model
    model = HybridMOFModel(quantum_input=n_qmof).to(device)
    n_params = model.count_parameters()
    print(f"\nModel parameters: {n_params:,}")

    for name, child in [("GNN", model.gnn), ("Chemical", model.chem),
                         ("Quantum", model.quantum), ("Fusion", model.fusion)]:
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:12s}: {n:>8,}")

    # Optimizer, loss, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"],
                                  weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=CONFIG["scheduler_patience"],
        factor=CONFIG["scheduler_factor"]
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_r2": [], "lr": []}

    print(f"\n{'-' * 70}")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Val MAE':>10} {'Val R2':>10} {'LR':>10}")
    print(f"{'-' * 70}")

    t0 = time.time()

    for epoch in range(CONFIG["max_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_rmse, val_r2 = eval_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_mae"].append(round(val_mae, 4))
        history["val_r2"].append(round(val_r2, 4))
        history["lr"].append(lr)

        # Print every 5 epochs or on improvement
        if (epoch + 1) % 5 == 0 or val_loss < best_val_loss:
            marker = " *" if val_loss < best_val_loss else ""
            print(f"{epoch+1:5d} {train_loss:12.6f} {val_loss:12.6f} {val_mae:10.4f} {val_r2:10.4f} {lr:10.1e}{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "config": CONFIG,
            }, os.path.join(CHECKPOINT_DIR, "hybrid_best.pt"))
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["early_stop_patience"]:
            print(f"\n  Early stopping at epoch {epoch+1} (patience={CONFIG['early_stop_patience']})")
            break

    total_time = time.time() - t0
    print(f"\n  Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Load best model
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "hybrid_best.pt"), weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best model from epoch {ckpt['epoch']+1}")

    # -- Final Evaluation -----------------------------------------------------
    print(f"\n{'=' * 70}")
    print("FINAL EVALUATION")
    print(f"{'=' * 70}")

    _, val_mae, val_rmse, val_r2 = eval_epoch(model, val_loader, criterion, device)
    _, test_mae, test_rmse, test_r2 = eval_epoch(model, test_loader, criterion, device)

    print(f"\n  {'Set':10s}  {'MAE':>8s}  {'RMSE':>8s}  {'R2':>8s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Val':10s}  {val_mae:8.4f}  {val_rmse:8.4f}  {val_r2:8.4f}")
    print(f"  {'Test':10s}  {test_mae:8.4f}  {test_rmse:8.4f}  {test_r2:8.4f}")

    # -- Uncertainty (MC Dropout) ---------------------------------------------
    print(f"\n{'-' * 70}")
    print(f"MC Dropout Uncertainty ({CONFIG['mc_samples']} samples)")
    print(f"{'-' * 70}")

    means, stds, trues = evaluate_with_uncertainty(
        model, test_loader, device, n_samples=CONFIG["mc_samples"]
    )
    unc_mae = mean_absolute_error(trues, means)
    print(f"  MC-MAE:          {unc_mae:.4f}")
    print(f"  Mean uncertainty: {stds.mean():.4f}")
    print(f"  Max uncertainty:  {stds.max():.4f}")

    # -- Save Results ---------------------------------------------------------
    results = {
        "config": CONFIG,
        "n_params": n_params,
        "train_time_s": round(total_time, 1),
        "best_epoch": ckpt["epoch"] + 1,
        "val": {"mae": round(val_mae, 4), "rmse": round(val_rmse, 4), "r2": round(val_r2, 4)},
        "test": {"mae": round(test_mae, 4), "rmse": round(test_rmse, 4), "r2": round(test_r2, 4)},
        "uncertainty": {
            "mc_mae": round(float(unc_mae), 4),
            "mean_std": round(float(stds.mean()), 4),
            "max_std": round(float(stds.max()), 4),
        },
        "history": history,
    }

    out_path = os.path.join(RESULTS_DIR, "hybrid_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Best checkpoint: {CHECKPOINT_DIR}/hybrid_best.pt")

    print(f"\n{'=' * 80}")
    print("HYBRID MODEL TRAINING COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
