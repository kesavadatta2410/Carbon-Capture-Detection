# MOF CO₂ Adsorption Prediction — Consolidated Pipeline

**Senior Researcher Analysis & Fix Guide**

---

## Executive Summary

This repository contains a consolidated 3-file ML pipeline for predicting CO₂ adsorption in Metal-Organic Frameworks (MOFs). The original codebase had **25+ files** with scattered functionality. This version consolidates everything into **3 clean files** with a critical fix for the transfer learning R²=0.0000 error.

---

## The R²=0.0000 Error — Root Cause & Fix

### Problem Statement

When running transfer learning from hMOF (simulated) to CoRE-MOF (experimental), you see:

```
Warning: missing target column CO2_uptake_1.0bar in data\_tmp_ft_val.csv. Using dummy zeros.
Zero-shot: R²=0.0000  MAE=0.3696
Frozen-GNN test: R²=0.0000
Full fine-tune test: R²=0.0000
```

### Root Cause Analysis

| Issue | Explanation |
|-------|-------------|
| **Column Name Mismatch** | hMOF uses `CO2_uptake_1.0bar` (simulated data) |
| | CoRE-MOF uses different column names (experimental data) |
| **Dummy Targets** | When target column not found, script uses zeros |
| **Zero Variance** | Predicting zeros → R² = 0 (no variance explained) |

### The Fix (in `mof_transfer.py`)

```python
# Automatic target column detection
CORE_MOF_TARGET_COLUMNS = [
    "CO2_uptake_1.0bar",      # hMOF-style
    "CO2_wt_percent",         # Weight percent
    "CO2_wt%",
    "CO2_mol_kg",             # mol/kg
    "CO2_uptake",             # Generic
    "Uptake_CO2",
    # ... more columns
]

def find_target_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect CO2 uptake column with fallback strategies."""
    # 1. Exact match
    # 2. Case-insensitive match
    # 3. Partial match (contains 'co2' or 'uptake')
```

### Immediate Actions to Take

1. **Debug Mode** — Check what columns exist:
   ```bash
   python mof_transfer.py --debug
   ```

2. **Inspect CoRE-MOF CSV** — Find the actual column name:
   ```python
   import pandas as pd
   df = pd.read_csv("data/core_mof_ft.csv")
   print(df.columns.tolist())
   ```

3. **Specify Target Column** — If auto-detection fails:
   ```bash
   python mof_transfer.py --target_col "your_actual_column_name"
   ```

4. **Add to List** — Permanently add your column to `mof_transfer.py`:
   ```python
   CORE_MOF_TARGET_COLUMNS.append("your_column_name")
   ```

---

## File Structure (3 Files Only)

```
carbon/
├── mof_model.py      # Model architecture (Hybrid GNN)
├── mof_train.py      # Training pipeline
├── mof_transfer.py   # Transfer learning with CoRE-MOF fix
└── README.md         # This file
```

### What Got Consolidated

| Original Files | Consolidated Into |
|----------------|-------------------|
| `hybrid_model.py` | `mof_model.py` |
| `hybrid_model_v2.py` | `mof_model.py` |
| `train_hybrid.py` | `mof_train.py` |
| `train_hybrid_fixed.py` | `mof_train.py` |
| `transfer_learning.py` | `mof_transfer.py` |
| `baselines.py` | Use XGBoost directly |
| `ablation_study.py` | Commented sections in `mof_train.py` |

---

## Installation & Setup

### Requirements

```bash
pip install torch>=2.0.0 torch_geometric scikit-learn xgboost
pip install pandas numpy matplotlib seaborn tqdm
pip install pymatgen matminer
```

### Directory Structure Expected

```
carbon/
├── data/
│   ├── graphs/              # .npz graph files (from build_graphs.py)
│   ├── chemical_features.npy
│   ├── hmof_train.csv
│   ├── hmof_val.csv
│   ├── hmof_test.csv
│   ├── core_mof_ft.csv      # CoRE-MOF fine-tune (10%)
│   └── core_mof_test.csv    # CoRE-MOF test (90%)
├── checkpoints/             # Saved models
└── results/                 # JSON metrics
```

---

## Usage Guide

### 1. Train on hMOF (Source Domain)

```bash
# Train GNN-only (diagnostic - check if GNN learns)
python mof_train.py --stage gnn_only --epochs 100

# Train full hybrid model
python mof_train.py --stage hybrid --epochs 150

# Ensemble with multiple seeds
python mof_train.py --stage ensemble --seeds 3
```

### 2. Transfer to CoRE-MOF (Target Domain)

```bash
# Debug mode - check data first
python mof_transfer.py --debug

# Standard transfer learning
python mof_transfer.py

# With custom hyperparameters
python mof_transfer.py \
    --frozen_epochs 50 \
    --full_finetune_epochs 100 \
    --full_finetune_lr 1e-5

# Specify target column if auto-detection fails
python mof_transfer.py --target_col "CO2_wt_percent"
```

---

## Architecture Overview

### HybridMOFModel

```
Input Graph (N atoms)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ GNN Branch (SchNet-style)                                    │
│   - Node embedding: 3 → 128                                  │
│   - 3 Interaction blocks with RBF expansion                  │
│   - Global Attention Pooling → 128-dim                       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Chemical Branch (Magpie features)                            │
│   - 145 → 256 → 128 (LayerNorm + GELU)                       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Cross-Attention Fusion                                       │
│   - Chemical queries attend to GNN keys/values (4 heads)     │
│   - Residual connection                                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Prediction Head                                              │
│   - Fused(128) + Quantum(64) → 256 → 128 → 1                 │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: CO₂ uptake (mol/kg)
```

### Key Improvements Over v1

| Feature | v1 | v2 (This) |
|---------|-----|-----------|
| GNN | Custom SchNet | MessagePassing-based |
| Pooling | Global Mean | Global Attention |
| Fusion | Concat + MLP | Cross-Attention |
| Quantum | Hard zeros | Learnable missing embed |
| Scheduler | ReduceLROnPlateau | CosineAnnealingWarmRestarts |
| Gradient Clipping | No | Yes (max_norm=10) |

---

## Troubleshooting Guide

### Issue: R² = 0.0000 on Transfer Learning

**Diagnosis Steps:**

1. Check if target column exists:
   ```python
   df = pd.read_csv("data/core_mof_ft.csv")
   print("CO2_uptake_1.0bar" in df.columns)  # Likely False
   print(df.columns.tolist())  # See actual columns
   ```

2. Run debug mode:
   ```bash
   python mof_transfer.py --debug
   ```

3. Check for common CoRE-MOF column names:
   - `CO2_wt_percent`
   - `CO2_wt%`
   - `CO2_uptake`
   - `Uptake_CO2`
   - `co2_uptake`

**Fix:**

```python
# In mof_transfer.py, add your column:
CORE_MOF_TARGET_COLUMNS = [
    "your_actual_column_name",  # Add this
    "CO2_uptake_1.0bar",
    # ...
]
```

### Issue: "Missing graph files"

**Cause:** CoRE-MOF structures don't have pre-computed graphs.

**Fix:** You need to build graphs for CoRE-MOF:

```python
# Run build_graphs.py on CoRE-MOF CIF files
# Or use the original script:
python code/build_graphs.py --input data/core_mof_ft.csv
```

### Issue: Low R² on hMOF Test Set

**Expected Baselines:**

| Model | Expected R² |
|-------|-------------|
| XGBoost | ~0.93 |
| Random Forest | ~0.85 |
| MLP | ~0.42 |
| Hybrid GNN | >0.75 (target) |

**If GNN-only R² < 0.5:**
- Check graph quality: `data/graphs/*.npz`
- Verify edge distances are in Å (not normalized)
- Check node features: [atomic_Z, electronegativity, covalent_radius]

### Issue: CUDA Out of Memory

```bash
# Reduce batch size
python mof_train.py --batch_size 32  # or 16

# Use gradient checkpointing (if implemented)
```

---

## Performance Benchmarks

### hMOF Test Set (32,768 structures)

| Model | R² | MAE | Training Time |
|-------|-----|-----|---------------|
| XGBoost | 0.930 | 0.193 | 12s |
| Hybrid GNN | 0.448* | 0.598 | 90min |

*Note: GNN underperforms XGBoost on tabular features alone. Consider:
- Training longer (300+ epochs)
- Increasing GNN layers (3 → 5)
- Using attention pooling (implemented)

### CoRE-MOF Transfer Learning

| Stage | Expected R² | Notes |
|-------|-------------|-------|
| Zero-shot | 0.3-0.5 | No adaptation |
| Frozen-GNN | 0.6-0.7 | Fast adaptation |
| Full fine-tune | 0.7-0.8 | Best performance |

---

## Advanced Usage

### Custom Target Columns

```python
# In mof_train.py or mof_transfer.py
target_col = "CO2_uptake_0.1bar"  # Different pressure
target_cols = ["CO2_uptake_0.1bar", "CO2_uptake_1.0bar"]  # Multi-task
```

### Layer-wise Learning Rates

```python
# In model.parameter_groups()
param_groups = [
    {"params": model.gnn.parameters(), "lr": 1e-4},      # Slower
    {"params": model.chem.parameters(), "lr": 1e-3},     # Faster
    {"params": model.head.parameters(), "lr": 1e-3},     # Faster
]
```

### MC Dropout for Uncertainty

```python
# Enable during evaluation
model.train()  # Keep dropout active
predictions = [model(batch) for _ in range(20)]
mean = np.mean(predictions, axis=0)
std = np.std(predictions, axis=0)  # Uncertainty
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{mof_co2_prediction,
  title={MOF CO₂ Adsorption Prediction Pipeline},
  author={Your Name},
  year={2024},
  note={Consolidated 3-file pipeline with transfer learning fixes}
}
```

---

## Contact & Support

For issues with:
- **Code bugs**: Check troubleshooting section above
- **Data issues**: Verify CSV columns match expected format
- **Model performance**: Compare against baseline benchmarks

---

## Changelog

### v2.0 (Consolidated)
- Combined 25+ files into 3 clean files
- Fixed CoRE-MOF target column auto-detection
- Added debug mode for data inspection
- Improved error messages

### v1.0 (Original)
- Initial implementation with scattered files
- Basic transfer learning (broken for CoRE-MOF)

---

**End of README**
