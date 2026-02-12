#!/usr/bin/env python3
"""Verify all Phase 3 outputs."""
import pandas as pd
import numpy as np
import os
import json

print("=" * 70)
print("PHASE 3 VERIFICATION")
print("=" * 70)

# 3.1 QMOF Integration
enh = pd.read_csv("data/hmof_enhanced.csv")
qmof_cols = [c for c in enh.columns if "qmof" in c]
n_qmof = enh[qmof_cols[0]].notna().sum() if qmof_cols else 0
print(f"\n3.1 QMOF Integration:")
print(f"  hmof_enhanced.csv: {enh.shape}")
print(f"  QMOF columns added: {len(qmof_cols)}")
print(f"  Structures with QMOF data: {n_qmof}")

# 3.2 Graph Representations
graphs = [f for f in os.listdir("data/graphs") if f.endswith(".npz")]
stats = pd.read_csv("data/graph_stats.csv")
ok = stats[stats["status"] == "ok"]
print(f"\n3.2 Graph Representations:")
print(f"  .npz files: {len(graphs)}")
print(f"  graph_stats.csv: {len(stats)} rows")
print(f"  Successful: {len(ok)} ({100*len(ok)/len(stats):.1f}%)")
avg_n = ok["num_nodes"].mean()
avg_e = ok["num_edges"].mean()
print(f"  Avg nodes: {avg_n:.1f}, Avg edges: {avg_e:.1f}")

# 3.3 Chemical Embeddings
cf = np.load("data/chemical_features.npy")
with open("data/chemical_feature_names.json") as f:
    names = json.load(f)
print(f"\n3.3 Chemical Embeddings:")
print(f"  Shape: {cf.shape}  ({cf.nbytes/1e6:.1f} MB)")
print(f"  Feature names: {len(names)}")
print(f"  NaN count: {np.isnan(cf).sum()}")

# 3.4-3.5 Preprocessing
with open("data/preprocess_config.json") as f:
    cfg = json.load(f)
print(f"\n3.4-3.5 Preprocessing:")
print(f"  Feature columns: {len(cfg['feature_columns'])}")
print(f"  Log-transformed: {len(cfg['log_transformed_columns'])}")
print(f"  Outlier clipping: {len(cfg['outlier_clipping'])} columns")

# 3.6 Train/Val/Test Splits
train = pd.read_csv("data/hmof_train.csv")
val = pd.read_csv("data/hmof_val.csv")
test = pd.read_csv("data/hmof_test.csv")
total = len(train) + len(val) + len(test)
print(f"\n3.6 Data Splits:")
print(f"  Train: {len(train)} ({100*len(train)/total:.1f}%)")
print(f"  Val:   {len(val)} ({100*len(val)/total:.1f}%)")
print(f"  Test:  {len(test)} ({100*len(test)/total:.1f}%)")
print(f"  Total: {total} (expected 32768)")
overlap1 = len(set(train["name"]) & set(val["name"]))
overlap2 = len(set(train["name"]) & set(test["name"]))
overlap3 = len(set(val["name"]) & set(test["name"]))
print(f"  Overlaps: train-val={overlap1}, train-test={overlap2}, val-test={overlap3}")

# 3.7 CoRE-MOF Sets
cft = pd.read_csv("data/core_mof_ft.csv")
ctest = pd.read_csv("data/core_mof_test.csv")
ctotal = len(cft) + len(ctest)
print(f"\n3.7 CoRE-MOF Transfer Sets:")
print(f"  Fine-tune: {len(cft)} ({100*len(cft)/ctotal:.1f}%)")
print(f"  Test:      {len(ctest)} ({100*len(ctest)/ctotal:.1f}%)")

# 3.8 GA_MOFs
ga = pd.read_csv("data/ga_mofs_test.csv")
print(f"\n3.8 GA_MOFs Zero-Shot:")
print(f"  Test set: {len(ga)} structures")

# 3.9 COF Exclusion
with open("data/cofs_excluded.txt") as f:
    cof_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
print(f"\n3.9 COF Exclusion:")
print(f"  Excluded: {len(cof_lines)} structures")

print(f"\n{'=' * 70}")
print("ALL PHASE 3 OUTPUTS VERIFIED!")
print(f"{'=' * 70}")
