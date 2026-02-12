#!/usr/bin/env python3
"""
Task 3.1: Integrate QMOF quantum properties into hMOF dataset.
Merges bandgap and formation energy from QMOF database via MOF-ID matching.
Note: Only ~24 structures overlap (hMOFs are hypothetical, QMOF is experimental).
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
HMOF_PATH = "data/hmof_properties.csv"
QMOF_PATH = "Database/qmof_database/qmof.csv"
OUTPUT_PATH = "data/hmof_enhanced.csv"

# ── QMOF columns to extract ───────────────────────────────────────────────────
QMOF_COLS = [
    "info.mofid.mofid",
    "info.mofid.mofkey",
    "outputs.pbe.bandgap",
    "outputs.pbe.energy_total",
    "outputs.pbe.energy_vdw",
    "outputs.pbe.net_magmom",
    "info.natoms",
    "info.pld",
    "info.lcd",
    "info.density",
    "info.volume",
]

# Rename map for cleaner column names
RENAME_MAP = {
    "outputs.pbe.bandgap": "qmof_bandgap_eV",
    "outputs.pbe.energy_total": "qmof_energy_total_eV",
    "outputs.pbe.energy_vdw": "qmof_energy_vdw_eV",
    "outputs.pbe.net_magmom": "qmof_net_magmom",
    "info.natoms": "qmof_natoms",
    "info.pld": "qmof_pld",
    "info.lcd": "qmof_lcd",
    "info.density": "qmof_density",
    "info.volume": "qmof_volume",
}


def main():
    print("=" * 80)
    print("Task 3.1: Integrate QMOF Quantum Properties")
    print("=" * 80)

    # ── Load hMOF dataset ──────────────────────────────────────────────────────
    hmof = pd.read_csv(HMOF_PATH)
    print(f"\nhMOF dataset: {len(hmof)} structures, {len(hmof.columns)} columns")

    # ── Load QMOF dataset (selected columns only) ─────────────────────────────
    print(f"\nLoading QMOF from {QMOF_PATH}...")
    qmof = pd.read_csv(QMOF_PATH, usecols=QMOF_COLS, low_memory=False)
    print(f"QMOF dataset: {len(qmof)} structures")
    print(f"  With MOF-ID: {qmof['info.mofid.mofid'].notna().sum()}")
    print(f"  With bandgap: {qmof['outputs.pbe.bandgap'].notna().sum()}")
    print(f"  With energy: {qmof['outputs.pbe.energy_total'].notna().sum()}")

    # ── Rename columns ─────────────────────────────────────────────────────────
    qmof = qmof.rename(columns=RENAME_MAP)

    # ── Deduplicate QMOF on MOF-ID (keep first) to avoid many-to-many ─────────
    qmof_merge = qmof.drop(columns=["info.mofid.mofkey"], errors="ignore")
    qmof_merge = qmof_merge.dropna(subset=["info.mofid.mofid"])
    qmof_merge = qmof_merge.drop_duplicates(subset=["info.mofid.mofid"], keep="first")
    print(f"  QMOF after dedup on mofid: {len(qmof_merge)}")

    # ── Merge on MOF-ID (left join) ────────────────────────────────────────────
    merged = pd.merge(
        hmof,
        qmof_merge,
        left_on="mofid",
        right_on="info.mofid.mofid",
        how="left",
        validate="many_to_one",  # ensure no explosion
    )
    merged = merged.drop(columns=["info.mofid.mofid"], errors="ignore")

    # ── Report overlap ─────────────────────────────────────────────────────────
    qmof_new_cols = [c for c in RENAME_MAP.values()]
    n_matched = merged[qmof_new_cols[0]].notna().sum()
    print(f"\n{'─' * 60}")
    print(f"MERGE RESULTS:")
    print(f"  Structures matched: {n_matched} / {len(hmof)} ({100*n_matched/len(hmof):.2f}%)")
    print(f"  New columns added: {len(qmof_new_cols)}")
    for col in qmof_new_cols:
        n_valid = merged[col].notna().sum()
        if n_valid > 0:
            print(f"    {col}: {n_valid} values, mean={merged[col].mean():.4f}")
        else:
            print(f"    {col}: all NaN")

    # ── Save enhanced dataset ──────────────────────────────────────────────────
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved enhanced dataset: {OUTPUT_PATH}")
    print(f"  Shape: {merged.shape}")
    print(f"  Columns: {list(merged.columns)}")
    print(f"\n{'=' * 80}")
    print("QMOF INTEGRATION COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
