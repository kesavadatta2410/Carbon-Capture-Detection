"""
ablation_study.py — Phase 6: Full Ablation Study (Experiments A–L)
==================================================================
Runs all ablation experiments with critical improvements and
saves per-experiment JSON results + a summary CSV.

Usage:
    python code/ablation_study.py                   # all experiments
    python code/ablation_study.py --exps C D G      # specific experiments
    python code/ablation_study.py --skip_existing   # resume interrupted run
    python code/ablation_study.py --data_efficiency # data efficiency experiments

Original experiment map:
  A  No pre-training      – train from scratch on CoRE-MOF 10% only
  B  XGBoost baseline     – re-run for fair comparison
  C  GNN-only             – SchNet, no chemical/quantum
  D  Chemical-only        – 145-dim Magpie MLP
  E  No quantum           – GNN + Chemical (no quantum branch)
  F  No attention         – Concatenate + MLP (no cross-attention)
  G  Full hybrid          – best model with attention
  H  Unified training     – train on hMOF + CoRE-MOF together
  I  Ensemble (GNN+XGB)   – average GNN and XGBoost predictions

NEW experiments (changes.md):
  J  Data efficiency      – 10%, 25%, 50%, 100% training data
  K  Uncertainty calibration – ECE, NLL metrics with MC Dropout
  L  Attention visualization – map attention weights to structures
  M  Physics-informed     – with Langmuir/Freundlich constraints
  N  Multi-task learning  – predict CO2 at 0.1, 1.0, 10 bar
  O  Stratified by topology – test generalization across MOF families
  P  Directional GNN      – DimeNet++-inspired angular features
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch_geometric.data import Batch
from tqdm import tqdm

# Local imports
from mof_model import HybridMOFModel, GNNOnlyModel
from mof_train import (
    MOFDataset, collate_fn, evaluate,
    train_epoch, compute_metrics, run_training,
)

DATA_DIR   = Path("data")
RESULT_DIR = Path("results")
CKPT_DIR   = Path("checkpoints")
RESULT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

SEED    = 42
EPOCHS  = 150   # same for all exps
BATCH   = 64
CUTOFF  = 8.0
GNN_LR  = 1e-4
TAB_LR  = 1e-3


# ===========================================================================
# Chemical-only MLP  (Experiment D)
# ===========================================================================

class ChemicalOnlyModel(nn.Module):
    """145-dim Magpie → scalar regression (no GNN, no quantum)."""

    def __init__(self, in_dim: int = 145):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128),   nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(128, 64),    nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, graph_b, chem, quant=None, qmask=None):
        return self.net(chem).squeeze(-1)


# ===========================================================================
# No-attention Hybrid  (Experiment F)
# ===========================================================================

class HybridNoAttentionModel(nn.Module):
    """Concatenate GNN + Chem + Quantum → MLP (no cross-attention)."""

    def __init__(self, cutoff: float = 8.0):
        super().__init__()
        from hybrid_model_v2 import GNNBranch, ChemicalBranch, QuantumBranch
        self.gnn     = GNNBranch(128, 64, 3, cutoff, out_dim=128)
        self.chem    = ChemicalBranch(145, 128)
        self.quantum = QuantumBranch(8, 64)
        concat_dim   = 128 + 128 + 64  # 320
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 64),         nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, graph_b, chem, quant, qmask):
        g = self.gnn(graph_b)
        c = self.chem(chem)
        q = self.quantum(quant, qmask)
        x = torch.cat([g, c, q], dim=-1)
        return self.head(x).squeeze(-1)


# ===========================================================================
# No-quantum Hybrid  (Experiment E)
# ===========================================================================

class HybridNoQuantumModel(nn.Module):
    """GNN + Chemical with cross-attention, no quantum branch."""

    def __init__(self, cutoff: float = 8.0):
        super().__init__()
        from hybrid_model_v2 import GNNBranch, ChemicalBranch, CrossAttentionFusion
        self.gnn     = GNNBranch(128, 64, 3, cutoff, out_dim=128)
        self.chem    = ChemicalBranch(145, 128)
        self.fusion  = CrossAttentionFusion(128, 128, 4, 128)
        self.head    = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, graph_b, chem, quant=None, qmask=None):
        g = self.gnn(graph_b)
        c = self.chem(chem)
        f = self.fusion(g, c)
        return self.head(f).squeeze(-1)


# ===========================================================================
# Generic training runner
# ===========================================================================

def _train_model(model, train_loader, val_loader, test_loader,
                 gnn_only: bool = False,
                 chem_only: bool = False,
                 epochs: int = EPOCHS,
                 lr: float = TAB_LR,
                 ckpt_path: Path = None) -> dict:
    """Train any model with the fixed protocol."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50 * steps, T_mult=2)

    best_val_r2 = -np.inf
    criterion   = nn.MSELoss()
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            graph_b, chem, quant, qmask, y = batch
            graph_b = graph_b.to(device)
            y = y.to(device)

            if gnn_only:
                pred = model(graph_b)
            elif chem_only:
                pred = model(graph_b, chem.to(device))
            else:
                pred = model(graph_b, chem.to(device),
                             quant.to(device), qmask.to(device))

            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

        # val
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for batch in val_loader:
                graph_b, chem, quant, qmask, y = batch
                graph_b = graph_b.to(device)
                if gnn_only:
                    pred = model(graph_b)
                elif chem_only:
                    pred = model(graph_b, chem.to(device))
                else:
                    pred = model(graph_b, chem.to(device),
                                 quant.to(device), qmask.to(device))
                all_p.append(pred.cpu().numpy())
                all_t.append(y.numpy())
        val_r2 = r2_score(np.concatenate(all_t), np.concatenate(all_p))

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)

        if epoch % 25 == 0:
            print(f"  ep {epoch:3d}  val_r2={val_r2:.4f}  best={best_val_r2:.4f}")

    # Test
    if ckpt_path and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            graph_b, chem, quant, qmask, y = batch
            graph_b = graph_b.to(device)
            if gnn_only:
                pred = model(graph_b)
            elif chem_only:
                pred = model(graph_b, chem.to(device))
            else:
                pred = model(graph_b, chem.to(device),
                             quant.to(device), qmask.to(device))
            all_p.append(pred.cpu().numpy())
            all_t.append(y.numpy())

    test_preds   = np.concatenate(all_p)
    test_targets = np.concatenate(all_t)
    metrics = compute_metrics(test_targets, test_preds)
    metrics["elapsed_s"] = round(time.time() - t0)
    return metrics, test_preds, test_targets


# ===========================================================================
# Standard data loaders helper
# ===========================================================================

def _loaders(train_csv, val_csv=None, test_csv=None,
             batch: int = BATCH, cutoff: float = CUTOFF):
    train_ds = MOFDataset(train_csv, cutoff=cutoff)
    val_ds   = MOFDataset(val_csv or DATA_DIR / "hmof_val.csv",   cutoff=cutoff)
    test_ds  = MOFDataset(test_csv or DATA_DIR / "hmof_test.csv", cutoff=cutoff)
    kw = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True)
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch, shuffle=False, **kw),
    )


# ===========================================================================
# Experiment runners
# ===========================================================================

def exp_A():
    """A — No pre-training: train from scratch on CoRE-MOF 10%."""
    print("\n── Experiment A: No pre-training (CoRE-MOF 10%) ──")
    core_train = DATA_DIR / "core_mof_finetune.csv"
    core_test  = DATA_DIR / "core_mof_test.csv"
    if not core_train.exists():
        return {"error": f"{core_train} not found"}

    train_l, val_l, test_l = _loaders(core_train, test_csv=core_test)
    model = HybridMOFModel(cutoff=CUTOFF)
    optimizer = AdamW(model.parameter_groups(GNN_LR, TAB_LR), weight_decay=1e-4)
    metrics, preds, targets = _train_model(
        model, train_l, val_l, test_l,
        ckpt_path=CKPT_DIR / "ablation_A.pt",
        lr=TAB_LR, epochs=EPOCHS,
    )
    return metrics


def exp_B():
    """B — XGBoost baseline re-run."""
    print("\n── Experiment B: XGBoost baseline ──")
    try:
        import joblib, xgboost as xgb
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        train_df = pd.read_csv(DATA_DIR / "hmof_train.csv")
        test_df  = pd.read_csv(DATA_DIR / "hmof_test.csv")
        feature_cols = [c for c in train_df.columns
                        if c not in ("CO2_uptake_1.0bar", "mof_id", "index")]
        X_tr = train_df[feature_cols].values
        y_tr = train_df["CO2_uptake_1.0bar"].values
        X_te = test_df[feature_cols].values
        y_te = test_df["CO2_uptake_1.0bar"].values

        t0 = time.time()
        clf = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                n_jobs=-1, random_state=SEED)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        metrics = compute_metrics(y_te, preds)
        metrics["elapsed_s"] = round(time.time() - t0)
        return metrics
    except ImportError:
        return {"error": "xgboost not installed"}


def exp_C():
    """C — GNN-only."""
    print("\n── Experiment C: GNN-only ──")
    train_l, val_l, test_l = _loaders(DATA_DIR / "hmof_train.csv")
    model = GNNOnlyModel(cutoff=CUTOFF)
    metrics, _, _ = _train_model(
        model, train_l, val_l, test_l,
        gnn_only=True,
        ckpt_path=CKPT_DIR / "ablation_C.pt",
        lr=GNN_LR, epochs=EPOCHS,
    )
    return metrics


def exp_D():
    """D — Chemical-only MLP."""
    print("\n── Experiment D: Chemical-only ──")
    train_l, val_l, test_l = _loaders(DATA_DIR / "hmof_train.csv")
    model = ChemicalOnlyModel()
    metrics, _, _ = _train_model(
        model, train_l, val_l, test_l,
        chem_only=True,
        ckpt_path=CKPT_DIR / "ablation_D.pt",
        lr=TAB_LR, epochs=EPOCHS,
    )
    return metrics


def exp_E():
    """E — No quantum (GNN + Chemical only)."""
    print("\n── Experiment E: No quantum ──")
    train_l, val_l, test_l = _loaders(DATA_DIR / "hmof_train.csv")
    model = HybridNoQuantumModel(cutoff=CUTOFF)
    metrics, _, _ = _train_model(
        model, train_l, val_l, test_l,
        ckpt_path=CKPT_DIR / "ablation_E.pt",
        lr=TAB_LR, epochs=EPOCHS,
    )
    return metrics


def exp_F():
    """F — No cross-attention (concatenation)."""
    print("\n── Experiment F: No attention ──")
    train_l, val_l, test_l = _loaders(DATA_DIR / "hmof_train.csv")
    model = HybridNoAttentionModel(cutoff=CUTOFF)
    metrics, _, _ = _train_model(
        model, train_l, val_l, test_l,
        ckpt_path=CKPT_DIR / "ablation_F.pt",
        lr=TAB_LR, epochs=EPOCHS,
    )
    return metrics


def exp_G():
    """G — Full hybrid (best model)."""
    print("\n── Experiment G: Full hybrid ──")
    train_l, val_l, test_l = _loaders(DATA_DIR / "hmof_train.csv")
    model = HybridMOFModel(cutoff=CUTOFF)
    metrics, _, _ = _train_model(
        model, train_l, val_l, test_l,
        ckpt_path=CKPT_DIR / "ablation_G.pt",
        lr=TAB_LR, epochs=EPOCHS,
    )
    return metrics


def exp_H():
    """H — Unified training (hMOF + CoRE-MOF together)."""
    print("\n── Experiment H: Unified training ──")
    core_train = DATA_DIR / "core_mof_finetune.csv"
    core_test  = DATA_DIR / "core_mof_test.csv"

    hmof_train_ds = MOFDataset(DATA_DIR / "hmof_train.csv",  cutoff=CUTOFF)
    hmof_val_ds   = MOFDataset(DATA_DIR / "hmof_val.csv",    cutoff=CUTOFF)

    if core_train.exists():
        core_ds   = MOFDataset(core_train, cutoff=CUTOFF)
        unified   = ConcatDataset([hmof_train_ds, core_ds])
    else:
        print("  Warning: core_mof_finetune.csv not found, training on hMOF only")
        unified = hmof_train_ds

    test_csv = core_test if core_test.exists() else DATA_DIR / "hmof_test.csv"
    test_ds  = MOFDataset(test_csv, cutoff=CUTOFF)

    kw = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True)
    train_l = DataLoader(unified,    batch_size=BATCH, shuffle=True,  **kw)
    val_l   = DataLoader(hmof_val_ds, batch_size=BATCH, shuffle=False, **kw)
    test_l  = DataLoader(test_ds,    batch_size=BATCH, shuffle=False, **kw)

    model = HybridMOFModel(cutoff=CUTOFF)
    metrics, _, _ = _train_model(
        model, train_l, val_l, test_l,
        ckpt_path=CKPT_DIR / "ablation_H.pt",
        lr=TAB_LR, epochs=EPOCHS,
    )
    return metrics


def exp_I():
    """I — Ensemble GNN + XGBoost (average predictions)."""
    print("\n── Experiment I: Ensemble GNN + XGBoost ──")
    try:
        import xgboost as xgb

        # Load trained XGBoost
        xgb_path = Path("model/xgboost.pkl")
        if xgb_path.exists():
            import joblib
            xgb_model = joblib.load(xgb_path)
        else:
            return {"error": "model/xgboost.pkl not found; run exp B first"}

        test_df  = pd.read_csv(DATA_DIR / "hmof_test.csv")
        feature_cols = [c for c in test_df.columns
                        if c not in ("CO2_uptake_1.0bar", "mof_id", "index")]
        X_te    = test_df[feature_cols].values
        y_te    = test_df["CO2_uptake_1.0bar"].values
        xgb_pred = xgb_model.predict(X_te)

        # Load best hybrid GNN (exp G)
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gnn_ckpt = CKPT_DIR / "ablation_G.pt"
        if not gnn_ckpt.exists():
            return {"error": "Run experiment G first to get the GNN checkpoint"}

        model = HybridMOFModel(cutoff=CUTOFF).to(device)
        model.load_state_dict(torch.load(gnn_ckpt, map_location=device))
        model.eval()

        test_ds = MOFDataset(DATA_DIR / "hmof_test.csv", cutoff=CUTOFF)
        test_l  = DataLoader(test_ds, batch_size=BATCH, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

        all_p, all_t = [], []
        with torch.no_grad():
            for batch in test_l:
                # Updated batch format
                graph_b, chem, quant, qmask, mof_desc, y = batch
                pred = model(graph_b.to(device), chem.to(device),
                             quant.to(device), qmask.to(device),
                             mof_desc.to(device))
                all_p.append(pred.cpu().numpy())
                all_t.append(y.numpy())

        gnn_pred = np.concatenate(all_p)
        y_te2    = np.concatenate(all_t)

        # Ensemble: simple average
        ens_pred = (gnn_pred + xgb_pred) / 2.0
        metrics  = compute_metrics(y_te2, ens_pred)
        return metrics

    except ImportError:
        return {"error": "xgboost not installed"}


# ===========================================================================
# NEW: Data Efficiency Experiments (J)
# ===========================================================================

def exp_J():
    """J — Data efficiency: test performance with 10%, 25%, 50%, 100% data."""
    print("\n── Experiment J: Data Efficiency ──")
    results = {}

    for frac in [0.1, 0.25, 0.5, 1.0]:
        print(f"  Training with {frac:.0%} of data...")
        result = run_training(
            stage="hybrid", seed=SEED, epochs=EPOCHS, batch_size=BATCH,
            cutoff=CUTOFF, gnn_lr=GNN_LR, tabular_lr=TAB_LR,
            data_fraction=frac, multi_task=False,
            use_wandb=False, run_name=f"data_efficiency_{frac:.0%}",
        )
        results[f"fraction_{frac}"] = result

    # Summary metrics
    summary = {
        "10%": results["fraction_0.1"]["test"],
        "25%": results["fraction_0.25"]["test"],
        "50%": results["fraction_0.5"]["test"],
        "100%": results["fraction_1.0"]["test"],
    }
    return {"data_efficiency_results": summary}


# ===========================================================================
# NEW: Uncertainty Calibration (K)
# ===========================================================================

def exp_K():
    """K — Uncertainty calibration with MC Dropout and ECE/NLL metrics."""
    print("\n── Experiment K: Uncertainty Calibration ──")

    # First train a model
    train_l, val_l, test_l = _loaders(DATA_DIR / "hmof_train.csv")
    model = HybridMOFModel(cutoff=CUTOFF)

    # Train briefly
    optimizer = AdamW(model.parameter_groups(GNN_LR, TAB_LR), weight_decay=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    steps = len(train_l)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50 * steps, T_mult=2)
    criterion = nn.MSELoss()

    print("  Training model...")
    for epoch in range(1, 51):  # Train for 50 epochs
        model.train()
        for batch in train_l:
            optimizer.zero_grad()
            graph_b, chem, quant, qmask, mof_desc, y = batch
            graph_b = graph_b.to(device)
            y = y.to(device)
            pred = model(graph_b, chem.to(device), quant.to(device),
                        qmask.to(device), mof_desc.to(device))
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()

    # Evaluate with MC Dropout
    print("  Evaluating with MC Dropout...")
    test_metrics, preds, targets, uncertainties = evaluate(
        model, test_l, device, gnn_only=False, mc_samples=50
    )

    if uncertainties is not None:
        # Compute additional calibration metrics
        ece = expected_calibration_error(targets, preds, uncertainties)
        nll = negative_log_likelihood(targets, preds, uncertainties)

        return {
            **test_metrics,
            "ECE": ece,
            "NLL": nll,
            "mean_uncertainty": float(np.mean(uncertainties)),
            "max_uncertainty": float(np.max(uncertainties)),
        }
    else:
        return {"error": "Uncertainty evaluation failed"}


# ===========================================================================
# NEW: Attention Visualization (L)
# ===========================================================================

def exp_L():
    """L — Attention visualization: extract and save cross-attention weights."""
    print("\n── Experiment L: Attention Visualization ──")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridMOFModel(cutoff=CUTOFF).to(device)

    # Load checkpoint if available
    ckpt_path = CKPT_DIR / "ablation_G.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Loaded checkpoint from {ckpt_path}")

    model.eval()

    # Get a few test samples
    test_ds = MOFDataset(DATA_DIR / "hmof_test.csv", cutoff=CUTOFF)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    attention_maps = []
    sample_ids = []

    print("  Extracting attention weights...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # Just first 10 samples
                break

            graph_b, chem, quant, qmask, mof_desc, y = batch
            graph_b = graph_b.to(device)
            chem = chem.to(device)
            quant = quant.to(device)
            qmask = qmask.to(device)

            # Forward pass through fusion layer to get attention
            g = model.gnn(graph_b)
            c = model.chem(chem)

            # Get attention weights from fusion layer
            fused, attn_weights = model.fusion(g, c, return_attention=True)

            attention_maps.append(attn_weights.cpu().numpy())
            sample_ids.append(test_ds.ids[i])

    # Save attention maps
    out_path = RESULT_DIR / "attention_visualization.npz"
    np.savez(out_path,
             attention_maps=attention_maps,
             sample_ids=sample_ids)

    return {
        "attention_saved": str(out_path),
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
    }


# ===========================================================================
# NEW: Physics-Informed Training (M)
# ===========================================================================

def exp_M():
    """M — Physics-informed training with Langmuir/Freundlich constraints."""
    print("\n── Experiment M: Physics-Informed Training ──")

    result = run_training(
        stage="hybrid", seed=SEED, epochs=EPOCHS, batch_size=BATCH,
        cutoff=CUTOFF, gnn_lr=GNN_LR, tabular_lr=TAB_LR,
        use_physics_loss=True,
        use_wandb=False, run_name="physics_informed",
    )

    return result["test"]


# ===========================================================================
# NEW: Multi-Task Learning (N)
# ===========================================================================

def exp_N():
    """N — Multi-task learning: predict CO2 at 0.1, 1.0, 10 bar simultaneously."""
    print("\n── Experiment N: Multi-Task Learning ──")

    result = run_training(
        stage="hybrid", seed=SEED, epochs=EPOCHS, batch_size=BATCH,
        cutoff=CUTOFF, gnn_lr=GNN_LR, tabular_lr=TAB_LR,
        multi_task=True,
        use_wandb=False, run_name="multitask",
    )

    return result["test"]


# ===========================================================================
# NEW: Stratified by Topology (O)
# ===========================================================================

def exp_O():
    """O — Stratified evaluation by MOF topology."""
    print("\n── Experiment O: Stratified by Topology ──")

    # Load test data with topology info if available
    test_csv = DATA_DIR / "hmof_test.csv"
    test_df = pd.read_csv(test_csv)

    # Check if topology column exists
    if "topology" not in test_df.columns:
        # Try to infer from mof_id or use placeholder
        print("  Note: topology column not found, using placeholder")
        test_df["topology"] = "unknown"

    # Common MOF topologies
    topologies = test_df["topology"].unique() if "topology" in test_df.columns else ["pcu", "dia", "fcu"]

    results_by_topology = {}

    for topo in topologies[:5]:  # Limit to first 5
        print(f"  Evaluating topology: {topo}")
        topo_mask = test_df["topology"] == topo
        if topo_mask.sum() < 10:
            continue

        # Would need to create stratified dataset here
        # For now, return placeholder
        results_by_topology[topo] = {
            "num_samples": int(topo_mask.sum()),
            "note": "Stratified evaluation requires topology labels",
        }

    return {"stratified_by_topology": results_by_topology}


# ===========================================================================
# NEW: Directional GNN (P)
# ===========================================================================

def exp_P():
    """P — Directional GNN with DimeNet++-inspired angular features."""
    print("\n── Experiment P: Directional GNN ──")

    # Train with directional features enabled
    from hybrid_model_v2 import GNNBranch

    # This experiment uses the enhanced GNN with directional interactions
    train_l, val_l, test_l = _loaders(DATA_DIR / "hmof_train.csv")

    # Create model with directional features
    class DirectionalGNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gnn = GNNBranch(128, 3, 50, 128, use_directional=True, use_hierarchical=False)
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1),
            )

        def forward(self, graph_b):
            return self.head(self.gnn(graph_b)).squeeze(-1)

    model = DirectionalGNNModel()
    metrics, _, _ = _train_model(
        model, train_l, val_l, test_l,
        gnn_only=True,
        ckpt_path=CKPT_DIR / "ablation_P.pt",
        lr=GNN_LR, epochs=EPOCHS,
    )

    return metrics


# ===========================================================================
# Orchestrator
# ===========================================================================

EXP_REGISTRY = {
    "A": ("No pre-training",           "~0.60", exp_A),
    "B": ("XGBoost baseline",          "0.93",  exp_B),
    "C": ("GNN-only",                  ">0.75", exp_C),
    "D": ("Chemical-only",             "~0.85", exp_D),
    "E": ("No quantum",                "~0.88", exp_E),
    "F": ("No attention",              "~0.82", exp_F),
    "G": ("Full hybrid",               ">0.90", exp_G),
    "H": ("Unified training",          "~0.91", exp_H),
    "I": ("Ensemble (GNN+XGB)",        ">0.94", exp_I),
    # NEW experiments from changes.md
    "J": ("Data efficiency",           "varies", exp_J),
    "K": ("Uncertainty calibration",   "N/A",  exp_K),
    "L": ("Attention visualization",   "N/A",  exp_L),
    "M": ("Physics-informed",          ">0.90", exp_M),
    "N": ("Multi-task learning",       ">0.88", exp_N),
    "O": ("Stratified by topology",    "N/A",  exp_O),
    "P": ("Directional GNN",           ">0.78", exp_P),
}


def run_all(exps: list[str], skip_existing: bool = False) -> pd.DataFrame:
    summary_rows = []

    for key in exps:
        name, expected, fn = EXP_REGISTRY[key]
        out_path = RESULT_DIR / f"ablation_{key}.json"

        if skip_existing and out_path.exists():
            print(f"  Skipping {key} (result exists)")
            with open(out_path) as f:
                result = json.load(f)
        else:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            try:
                result = fn()
            except Exception as e:
                result = {"error": str(e)}
                print(f"  !! Exp {key} failed: {e}")

            result["exp"]      = key
            result["name"]     = name
            result["expected"] = expected
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  → saved {out_path}")

        summary_rows.append({
            "Exp":      key,
            "Name":     name,
            "Expected R²": expected,
            "MAE":      result.get("MAE",  result.get("error", "—")),
            "RMSE":     result.get("RMSE", "—"),
            "R²":       result.get("R2",   "—"),
            "Time(s)":  result.get("elapsed_s", "—"),
        })

    df = pd.DataFrame(summary_rows)
    csv_path = RESULT_DIR / "ablation_summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAblation summary → {csv_path}")
    print(df.to_string(index=False))
    return df


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for MOF CO2 adsorption prediction"
    )
    parser.add_argument("--exps", nargs="*", default=list(EXP_REGISTRY.keys()),
                        help="Which experiments to run (default: all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments that already have result JSON")
    parser.add_argument("--data_efficiency", action="store_true",
                        help="Run only data efficiency experiments (J)")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Run only uncertainty calibration experiments (K)")
    parser.add_argument("--original", action="store_true",
                        help="Run only original A-I experiments")
    args = parser.parse_args()

    # Handle shortcuts
    if args.data_efficiency:
        args.exps = ["J"]
    elif args.uncertainty:
        args.exps = ["K"]
    elif args.original:
        args.exps = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

    # Validate
    invalid = [e for e in args.exps if e not in EXP_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown experiments: {invalid}. Choose from {list(EXP_REGISTRY.keys())}")

    run_all(args.exps, skip_existing=args.skip_existing)