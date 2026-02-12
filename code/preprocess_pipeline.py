#!/usr/bin/env python3
"""
Tasks 3.4–3.6: Preprocessing pipeline for hMOF ML dataset.
  3.4: Normalize/scale features (StandardScaler + log transforms for skewed)
  3.5: Handle outliers (IQR clipping on adsorption values)
  3.6: Create train/val/test splits (80/10/10)
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import skew

# ── Paths ──────────────────────────────────────────────────────────────────────
ENHANCED_PATH = "data/hmof_enhanced.csv"
CHEM_FEATURES_PATH = "data/chemical_features.npy"
CHEM_NAMES_PATH = "data/chemical_feature_names.json"

OUTPUT_DIR = "data"
SCALER_PATH = "data/scaler.pkl"
CONFIG_PATH = "data/preprocess_config.json"
TRAIN_PATH = "data/hmof_train.csv"
VAL_PATH = "data/hmof_val.csv"
TEST_PATH = "data/hmof_test.csv"
RESULTS_PATH = "results/preprocessing_summary.txt"

# ── Feature columns ───────────────────────────────────────────────────────────
# Adsorption target columns (to apply IQR outlier clipping)
ADSORPTION_COLS = [
    "CO2_uptake_0.01bar", "CO2_uptake_0.05bar", "CO2_uptake_0.1bar",
    "CO2_uptake_0.5bar", "CO2_uptake_1.0bar", "CO2_uptake_2.5bar",
    "CH4_uptake_max", "H2_uptake_max", "N2_uptake_max",
]

# Structural feature columns (numeric, for scaling)
STRUCTURAL_COLS = [
    "num_elements", "num_atoms", "cell_volume",
    "LCD", "PLD", "surface_area_m2g", "void_fraction",
]

# Columns to exclude from ML features
EXCLUDE_COLS = [
    "name", "mofid", "mofkey", "database", "elements", "dataset",
    "CO2_units", "CH4_units", "H2_units", "N2_units",
    "CO2_num_points", "CH4_num_points", "H2_num_points", "N2_num_points",
    "CO2_pressure_max", "CH4_pressure_max", "H2_pressure_max", "N2_pressure_max",
]

# QMOF columns (sparse, will keep but not use as primary features)
QMOF_COLS = [
    "qmof_bandgap_eV", "qmof_energy_total_eV", "qmof_energy_vdw_eV",
    "qmof_net_magmom", "qmof_natoms", "qmof_pld", "qmof_lcd",
    "qmof_density", "qmof_volume",
]

RANDOM_STATE = 42
SKEW_THRESHOLD = 2.0  # Apply log1p if skewness > this


def iqr_clip(series, factor=1.5):
    """Clip values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    n_clipped = ((series < lower) | (series > upper)).sum()
    clipped = series.clip(lower=lower, upper=upper)
    return clipped, n_clipped, lower, upper


def main():
    print("=" * 80)
    print("Tasks 3.4–3.6: Preprocessing Pipeline")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(ENHANCED_PATH)
    print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Load chemical features
    chem_features = np.load(CHEM_FEATURES_PATH)
    with open(CHEM_NAMES_PATH) as f:
        chem_names = json.load(f)
    print(f"Chemical features: {chem_features.shape}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Task 3.5: Handle Outliers (IQR clipping)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("TASK 3.5: Outlier Handling (IQR Clipping)")
    print(f"{'─' * 60}")

    clip_report = {}
    for col in ADSORPTION_COLS:
        if col in df.columns:
            original = df[col].copy()
            df[col], n_clipped, lower, upper = iqr_clip(df[col])
            clip_report[col] = {
                "n_clipped": int(n_clipped),
                "pct_clipped": round(100 * n_clipped / len(df), 2),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
            }
            print(f"  {col}: clipped {n_clipped} ({100*n_clipped/len(df):.1f}%) "
                  f"→ [{lower:.4f}, {upper:.4f}]")

    # ═══════════════════════════════════════════════════════════════════════════
    # Task 3.4: Normalize/Scale Features
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("TASK 3.4: Normalization & Scaling")
    print(f"{'─' * 60}")

    # Identify all numeric feature columns
    all_cols = list(df.columns)
    exclude = set(EXCLUDE_COLS + QMOF_COLS)
    feature_cols = [c for c in all_cols if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32']]
    print(f"\n  Numeric feature columns: {len(feature_cols)}")

    # Check skewness and apply log1p to highly skewed features
    log_transformed = []
    skewness_report = {}
    for col in feature_cols:
        s = skew(df[col].dropna())
        skewness_report[col] = round(s, 3)
        if abs(s) > SKEW_THRESHOLD:
            df[col] = np.log1p(df[col].clip(lower=0))  # log1p needs non-negative
            log_transformed.append(col)

    print(f"  Log-transformed (|skew| > {SKEW_THRESHOLD}): {len(log_transformed)}")
    for col in log_transformed:
        print(f"    {col}: skew={skewness_report[col]}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Task 3.6: Create Train/Val/Test Splits (80/10/10)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("TASK 3.6: Train/Val/Test Splits (80/10/10)")
    print(f"{'─' * 60}")

    # Stratify on binned CO2 uptake at 1 bar
    target_col = "CO2_uptake_1.0bar"
    n_bins = 10
    df["_strat_bin"] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates="drop")

    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE,
        stratify=df["_strat_bin"]
    )

    # Second split: 50/50 of temp → 10% val, 10% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE,
        stratify=temp_df["_strat_bin"]
    )

    # Remove stratification helper column
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["_strat_bin"], inplace=True)
    df.drop(columns=["_strat_bin"], inplace=True)

    print(f"\n  Train: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    print(f"  Val:   {len(val_df)} ({100*len(val_df)/len(df):.1f}%)")
    print(f"  Test:  {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")
    print(f"  Total: {len(train_df)+len(val_df)+len(test_df)} (should be {len(df)})")

    # Verify no overlap
    train_names = set(train_df["name"])
    val_names = set(val_df["name"])
    test_names = set(test_df["name"])
    assert len(train_names & val_names) == 0, "Train/Val overlap!"
    assert len(train_names & test_names) == 0, "Train/Test overlap!"
    assert len(val_names & test_names) == 0, "Val/Test overlap!"
    print("  ✓ No overlap between splits")

    # Check target distribution similarity
    print(f"\n  {target_col} distribution across splits:")
    print(f"    Train mean: {train_df[target_col].mean():.4f}")
    print(f"    Val mean:   {val_df[target_col].mean():.4f}")
    print(f"    Test mean:  {test_df[target_col].mean():.4f}")

    # ── Fit StandardScaler on TRAIN only ───────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    print(f"\n  StandardScaler fit on train ({len(feature_cols)} features)")

    # Apply to all splits
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    # ── Save outputs ───────────────────────────────────────────────────────────
    train_scaled.to_csv(TRAIN_PATH, index=False)
    val_scaled.to_csv(VAL_PATH, index=False)
    test_scaled.to_csv(TEST_PATH, index=False)
    print(f"\n  Saved: {TRAIN_PATH} ({len(train_scaled)} rows)")
    print(f"  Saved: {VAL_PATH} ({len(val_scaled)} rows)")
    print(f"  Saved: {TEST_PATH} ({len(test_scaled)} rows)")

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved: {SCALER_PATH}")

    # Save preprocessing config
    config = {
        "feature_columns": feature_cols,
        "log_transformed_columns": log_transformed,
        "skewness": skewness_report,
        "outlier_clipping": clip_report,
        "split_sizes": {
            "train": len(train_scaled),
            "val": len(val_scaled),
            "test": len(test_scaled),
        },
        "random_state": RANDOM_STATE,
        "skew_threshold": SKEW_THRESHOLD,
        "target_column": target_col,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {CONFIG_PATH}")

    # ── Summary report ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write("Preprocessing Pipeline Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("=== Outlier Clipping (IQR) ===\n")
        for col, info in clip_report.items():
            f.write(f"  {col}: {info['n_clipped']} clipped ({info['pct_clipped']}%)\n")
            f.write(f"    Bounds: [{info['lower_bound']}, {info['upper_bound']}]\n")

        f.write(f"\n=== Log Transforms (|skew| > {SKEW_THRESHOLD}) ===\n")
        for col in log_transformed:
            f.write(f"  {col}: skew={skewness_report[col]}\n")

        f.write(f"\n=== Train/Val/Test Splits ===\n")
        f.write(f"  Train: {len(train_scaled)} (80%)\n")
        f.write(f"  Val:   {len(val_scaled)} (10%)\n")
        f.write(f"  Test:  {len(test_scaled)} (10%)\n")

        f.write(f"\n=== Feature Columns ({len(feature_cols)}) ===\n")
        for col in feature_cols:
            f.write(f"  {col}\n")

    print(f"\n  Saved: {RESULTS_PATH}")

    print(f"\n{'=' * 80}")
    print("PREPROCESSING PIPELINE COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
