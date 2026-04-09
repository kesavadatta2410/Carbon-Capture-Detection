#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4.1: Train 3 baseline models on tabular + chemical features.
  - XGBoost
  - Random Forest
  - MLP (PyTorch)

Target: CO2_uptake_1.0bar
Input:  8 structural features + 145 chemical features = 153 dims
"""

import json
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -- Paths ------------------------------------------------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"
MODEL_DIR = "model"
TARGET_COL = "CO2_uptake_1.0bar"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def load_data():
    """Load train/val/test + chemical features and return aligned X, y arrays."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "hmof_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "hmof_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "hmof_test.csv"))

    with open(os.path.join(DATA_DIR, "preprocess_config.json")) as f:
        cfg = json.load(f)
    feature_cols = cfg["feature_columns"]

    # Safety check: ensure target is NOT in features
    if TARGET_COL in feature_cols:
        print(f"  WARNING: Removing target '{TARGET_COL}' from feature columns!")
        feature_cols = [c for c in feature_cols if c != TARGET_COL]

    # Remove any other CO2/gas uptake columns (prevent leakage)
    leak_keywords = ["uptake", "CH4_", "H2_", "N2_"]
    leaky = [c for c in feature_cols if any(k in c for k in leak_keywords)]
    if leaky:
        print(f"  WARNING: Removing {len(leaky)} leaky columns: {leaky}")
        feature_cols = [c for c in feature_cols if c not in leaky]

    # Load full chemical features and build name->index map
    full_df = pd.read_csv(os.path.join(DATA_DIR, "hmof_enhanced.csv"))
    chem_all = np.load(os.path.join(DATA_DIR, "chemical_features.npy"))
    name_to_idx = {name: i for i, name in enumerate(full_df["name"])}

    def build_xy(df):
        tab = df[feature_cols].values.astype(np.float32)
        # Align chemical features by name
        chem_idx = [name_to_idx[n] for n in df["name"]]
        chem = chem_all[chem_idx].astype(np.float32)
        X = np.hstack([tab, chem])
        y = df[TARGET_COL].values.astype(np.float32)
        # Replace NaN with 0
        X = np.nan_to_num(X, nan=0.0)
        return X, y

    X_train, y_train = build_xy(train_df)
    X_val, y_val = build_xy(val_df)
    X_test, y_test = build_xy(test_df)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Features: {len(feature_cols)} tabular + {chem_all.shape[1]} chemical = {X_train.shape[1]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:15s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


# =============================================================================
# Model 1: XGBoost
# =============================================================================

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n-- XGBoost -------------------------------------------------------")
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s (best iteration: {model.best_iteration})")

    val_metrics = evaluate(y_val, model.predict(X_val), "Val")
    test_metrics = evaluate(y_test, model.predict(X_test), "Test")

    # Save
    with open(os.path.join(MODEL_DIR, "xgboost.pkl"), "wb") as f:
        pickle.dump(model, f)

    return {"val": val_metrics, "test": test_metrics, "train_time": round(elapsed, 1),
            "best_iteration": model.best_iteration}


# =============================================================================
# Model 2: Random Forest
# =============================================================================

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n-- Random Forest -------------------------------------------------")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    val_metrics = evaluate(y_val, model.predict(X_val), "Val")
    test_metrics = evaluate(y_test, model.predict(X_test), "Test")

    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(model, f)

    return {"val": val_metrics, "test": test_metrics, "train_time": round(elapsed, 1)}


# =============================================================================
# Model 3: MLP (PyTorch)
# =============================================================================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n-- MLP -----------------------------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  Device: {device}")

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, pin_memory=use_cuda)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, pin_memory=use_cuda)

    model = MLPRegressor(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 25
    best_state = None
    t0 = time.time()

    for epoch in range(200):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or patience_counter == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.1e}")

        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    # Evaluate
    def predict(loader):
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb.to(device)).cpu().numpy())
        return np.concatenate(preds)

    val_pred = predict(val_loader)
    test_pred = predict(test_loader)

    val_metrics = evaluate(y_val, val_pred, "Val")
    test_metrics = evaluate(y_test, test_pred, "Test")

    torch.save(best_state, os.path.join(MODEL_DIR, "mlp_baseline.pt"))

    return {"val": val_metrics, "test": test_metrics, "train_time": round(elapsed, 1),
            "n_params": n_params, "epochs_trained": epoch + 1}


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Task 4.1: Train 3 Baseline Models")
    print("=" * 80)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    results = {}
    results["xgboost"] = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
    results["random_forest"] = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    results["mlp"] = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test)

    # -- Summary --------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("BASELINE COMPARISON (Test Set)")
    print(f"{'=' * 80}")
    print(f"  {'Model':20s}  {'MAE':>8s}  {'RMSE':>8s}  {'R2':>8s}  {'Time':>6s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for name, res in results.items():
        t = res["test"]
        print(f"  {name:20s}  {t['mae']:8.4f}  {t['rmse']:8.4f}  {t['r2']:8.4f}  {res['train_time']:5.1f}s")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Models saved to: {MODEL_DIR}/")

    print(f"\n{'=' * 80}")
    print("BASELINE TRAINING COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
