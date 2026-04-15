<div align="center">

# 🔬 Carbon Capture Detection
### Hybrid Graph Neural Network Pipeline for CO₂ Adsorption Prediction in Metal-Organic Frameworks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4+-3c9cd7.svg)](https://pyg.org/)

</div>

---

## 📖 Overview

This repository implements a full end-to-end machine learning pipeline for predicting **CO₂ adsorption capacity** (mol/kg at 1.0 bar) in Metal-Organic Frameworks (MOFs). The pipeline integrates **seven large MOF/COF databases**, engineers multi-modal features (graph topology, Magpie chemical embeddings, QMOF quantum descriptors), trains baseline and hybrid deep learning models, and evaluates generalization via **transfer learning** to experimental CoRE-MOF data.

**Key results (test set, hMOF 3,277 structures):**

| Model | R² | MAE | RMSE |
|---|---|---|---|
| XGBoost (best baseline) | **0.930** | 0.193 | 0.264 |
| Random Forest | 0.851 | 0.282 | 0.386 |
| MLP Baseline | 0.909 | 0.227 | 0.302 |
| GNN-Only (SchNet) | 0.858 | 0.262 | 0.377 |
| **Hybrid GNN** (ours) | **0.863** | 0.255 | 0.369 |

---

## 🏗️ Architecture

The **Hybrid MOF Model** fuses three parallel branches via cross-attention:

```
 Crystal Structure (CIF)
         │
    [Graph Builder]                    [Magpie]                    [QMOF DFT]
    8Å radius cutoff              145-dim embeddings              8-dim features
    Gaussian RBF (50 bins)             │                              │
         │                       [Chem Branch]              [Quantum Branch]
    [GNN Branch]                  MLP 145→256→128            MLP 8→64 + mask
  SchNet × 3 layers                    │                              │
  GlobalAttentionPool                  │                              │
         │ (128-dim)                   │ (128-dim)                    │ (64-dim)
         └──────────── [Cross-Attention Fusion] ────────────────────┘
                          Chemical queries → GNN keys/values
                                    │ (128-dim fused)
                             [Prediction Head]
                              128+64 → 256 → 128 → 1
                                    │
                            CO₂ uptake (mol/kg)
```

**Training:** Huber loss (δ=0.1), AdamW with layer-wise LR (GNN: 1e-4, tabular: 1e-3), CosineAnnealingWarmRestarts, gradient clipping (max_norm=10).

---

## 📂 Project Structure

```
Carbon-Capture-Detection/
├── code/                          # All pipeline scripts
│   ├── database_eda.py            # Phase 1: Multi-database EDA
│   ├── extract_hmof_properties.py # Phase 2: Property extraction
│   ├── integrate_qmof.py          # Task 3.1: QMOF quantum features
│   ├── build_graphs.py            # Task 3.2: Crystal graph construction
│   ├── chemical_embeddings.py     # Task 3.3: Magpie feature vectors
│   ├── preprocess_pipeline.py     # Tasks 3.4–3.6: Normalization & splits
│   ├── prepare_transfer_sets.py   # Tasks 3.7–3.9: CoRE-MOF & GA-MOF sets
│   ├── verify_graphs.py           # Phase 3 sanity checks
│   ├── baselines.py               # Phase 4: XGBoost, RF, MLP baselines
│   ├── mof_model.py               # Model architecture (GNN + Hybrid)
│   ├── mof_train.py               # Training pipeline (CLI)
│   ├── mof_optimized.py           # Optimized training variant
│   ├── mof_ensemble_final.py      # Ensemble inference
│   ├── transfer_learning.py       # Phase 5: Transfer to CoRE-MOF
│   └── ablation_study.py          # Phase 6: Ablation experiments A–P
├── data/                          # Processed datasets (see Download Guide)
│   ├── chemical_feature_names.json
│   ├── preprocess_config.json
│   ├── scaler.pkl
│   ├── structural_features.csv
│   ├── core_mof_ft.csv
│   ├── cofs_excluded.txt
│   └── graphs/                    # Per-MOF .npz graph files (~GB)
├── checkpoints/                   # Saved model checkpoints
├── model/                         # Final trained model artifacts
│   ├── xgboost.pkl
│   ├── mlp_baseline.pt
│   └── random_forest.pkl
├── results/                       # EDA plots & model evaluation outputs
├── docs/                          # Project proposal & dataset download guide
├── paper/                         # IEEE LaTeX paper source
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── CONTRIBUTING.md
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Kesavadatta2410/Carbon-Capture-Detection.git
cd Carbon-Capture-Detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Databases

Follow the step-by-step instructions in `docs/Dataset_Download_Guide.docx` to download:
- **hMOF-mofdb** (32,768 structures, ~3 GB) → `Database/hMOF-mofdb/`
- **CoRE-MOF** (12,020 structures) → `Database/CoRE-MOF/`
- **GA_MOFs** (51,163 structures) → `Database/GA_MOFs/`
- **QMOF** (~20,000 structures) → `Database/qmof_database/`
- **CURATED-COFs** (~600 structures) → `Database/CURATED-COFs/`

### 3. Run the Full Pipeline

```bash
# ── Phase 1: Database EDA ──────────────────────────────────────────────────
python code/database_eda.py

# ── Phase 2: Property Extraction ──────────────────────────────────────────
python code/extract_hmof_properties.py

# ── Phase 3: Feature Engineering & Splits ─────────────────────────────────
python code/integrate_qmof.py
python code/build_graphs.py            # builds data/graphs/*.npz
python code/chemical_embeddings.py     # builds data/chemical_features.npy
python code/preprocess_pipeline.py     # creates train/val/test splits
python code/prepare_transfer_sets.py   # creates CoRE-MOF fine-tune sets
python code/verify_graphs.py           # sanity checks

# ── Phase 4: Baseline Models ──────────────────────────────────────────────
python code/baselines.py

# ── Phase 5: Hybrid GNN Training ──────────────────────────────────────────
python code/mof_train.py --stage gnn_only --epochs 100     # GNN diagnosis
python code/mof_train.py --stage hybrid --epochs 150        # Full hybrid
python code/mof_train.py --stage ensemble --seeds 3         # 3-seed ensemble

# ── Phase 5: Transfer Learning ────────────────────────────────────────────
python code/transfer_learning.py

# ── Phase 6: Ablation Study ───────────────────────────────────────────────
python code/ablation_study.py                   # all experiments A–P
python code/ablation_study.py --exps C D G      # specific experiments
python code/ablation_study.py --skip_existing   # resume interrupted run
```

---

## 📊 Databases

| Database | Structures | Description |
|---|---|---|
| **hMOF-mofdb** | 32,768 | Hypothetical MOFs — CO₂, CH₄, H₂, N₂ uptakes at multiple pressures |
| **CoRE-MOF** | 12,020 | Computation-ready experimental MOFs with geometric descriptors |
| **GA_MOFs** | 51,163 | Genetically assembled MOFs via evolutionary algorithms |
| **QMOF** | ~20,000 | DFT-computed electronic properties (band gap, DDEC charges) |
| **CURATED-COFs** | ~600 | Curated Covalent Organic Frameworks |
| **CoRE-COFs** | 1,242 | COF exclusion manifest (v7.0) |
| **MOF_database** | ~500 | Supplementary reference MOFs |

---

## 🧪 Ablation Study

| Exp | Configuration | Test R² |
|---|---|---|
| A | No pre-training (CoRE-MOF 10% only) | ~0.60 |
| B | XGBoost baseline | 0.930 |
| C | GNN-only (SchNet) | 0.858 |
| D | Chemical-only (Magpie MLP) | ~0.85 |
| E | No quantum branch | ~0.88 |
| F | No cross-attention (concatenation) | ~0.82 |
| **G** | **Full hybrid (ours)** | **0.863** |
| H | Unified training (hMOF + CoRE-MOF) | ~0.91 |
| I | Ensemble: Hybrid GNN + XGBoost | >0.94 |

---

## 📦 Model Architecture Details

| Component | Details |
|---|---|
| **GNN Branch** | SchNet-style, 3 interaction layers, hidden=128, 8Å cutoff, 50-bin Gaussian RBF |
| **Chemical Branch** | Magpie 145→256→128, LayerNorm + GELU + Dropout(0.15) |
| **Quantum Branch** | QMOF 8→64, learnable missing-data embedding |
| **Fusion** | Cross-attention (chem queries → GNN keys/values), 4 heads, 128-dim output |
| **Head** | (128+64)→256→128→1, LayerNorm + GELU + Dropout(0.10) |
| **Loss** | Huber (δ=0.10) |
| **Optimizer** | AdamW, weight_decay=1e-4, gradient clipping max_norm=10 |
| **LR Schedule** | CosineAnnealingWarmRestarts (T₀=50 epochs, T_mult=2) |
| **Total Parameters** | ~1.5M (HybridMOFModel) |

---

## 📝 Citation

If you use this code or the associated paper in your research, please cite:

```bibtex
@article{cc-hybrid-gnn-2024,
  title   = {Hybrid Graph Neural Networks with Cross-Attention Fusion for
             CO₂ Adsorption Prediction in Metal-Organic Frameworks},
  author  = {Kesavadatta},
  journal = {IEEE},
  year    = {2024}
}
```

See `CITATION.cff` for the full citation metadata.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **hMOF database**: Wilmer et al., *Nature Chemistry*, 2012
- **CoRE-MOF 2019**: Chung et al., *J. Chem. Eng. Data*, 2019
- **QMOF database**: Rosen et al., *Matter*, 2021
- **SchNet**: Schütt et al., *J. Chem. Phys.*, 2018
- **PyTorch Geometric**: Fey & Lenssen, *ICLR Workshop*, 2019
