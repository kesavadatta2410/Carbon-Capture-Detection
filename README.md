# Carbon Capture Detection — ML Pipeline for MOF CO₂ Adsorption Prediction

A machine learning pipeline for predicting CO₂ adsorption capacity in Metal-Organic Frameworks (MOFs). The project integrates seven MOF/COF databases, engineers graph and chemical features, trains ML baselines, and builds a hybrid GNN + tabular deep learning model with transfer learning and full ablation study.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Database/](#database)
- [code/ — Pipeline Scripts](#code--pipeline-scripts)
- [data/ — Processed Datasets](#data--processed-datasets)
- [results/ — EDA & Model Outputs](#results--eda--model-outputs)
- [model/ — Saved Artifacts](#model--saved-artifacts)
- [docs/](#docs)
- [Pipeline Overview](#pipeline-overview)
- [Results So Far](#results-so-far)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Project Structure

```
carbon/
├── Database/                  # Raw MOF/COF databases (7 sources)
├── carbon/                    # Python virtual environment
├── checkpoints/               # Saved model checkpoints during training
├── code/                      # All pipeline scripts (EDA → modeling)
├── data/                      # Processed datasets, features, splits
├── docs/                      # Project proposal and download guides
├── model/                     # Final trained model artifacts
├── results/                   # EDA reports, plots, and evaluation summaries
├── preprocessing.log          # Full log from the preprocessing pipeline run
└── README.md                  # This file
```

---

## Database/

Raw source databases downloaded for the project. Each subfolder contains the original files (CIF crystal structures, CSV property tables, JSON metadata) from the respective database.

| Folder                      | # Structures  | Description |
|-----------------------------|:-------------:|-------------|
| `hMOF-mofdb`                | 32,768        | Hypothetical MOF database with JSON property files (CO₂, CH₄, H₂, N₂ uptakes at multiple pressures) |
| `CoRE-MOF`                  | 12,020        | Computation-Ready Experimental MOF database with geometric descriptors and metal-site annotations |
| `GA_MOFs`                   | 51,163        | Genetically Assembled MOFs — diverse hypothetical structures generated via evolutionary algorithms |
| `qmof_database`             | ~20,000+      | Quantum MOF database with DFT-computed electronic properties (band gap, partial charges, DDEC charges) |
| `CURATED-COFs`              | ~600          | Curated Covalent Organic Frameworks collection with structural CIFs |
| `CoRE-COFs_1242-v7.0`      | 1,242         | CoRE-COF v7.0 — used as the COF exclusion manifest to filter non-MOF structures |
| `MOF_database`              | ~500+         | Supplementary reference MOF database with CIF structures |

---

## code/ — Pipeline Scripts

All Python scripts for the ML pipeline, organized in execution order from exploratory analysis through model training.

---

### `database_eda.py` — Phase 1: Multi-Database Exploratory Analysis

**Purpose:** Scans all 7 databases, parses CIF crystal structure files, and generates a comprehensive comparative EDA report.

**Key operations:**
- Reads and parses CIF files across all database folders (up to 500 sampled per database for efficiency)
- Extracts structural descriptors: atom count, cell volume, density, element composition
- Computes metal content (fraction of structures containing metal atoms)
- Generates correlation heatmaps, density vs. volume scatter plots, element frequency charts, and dataset comparison bar charts
- Writes `results/complete_database_eda_report.md` and all visualization PNGs

**Outputs:** `results/complete_database_eda_report.md`, `results/dataset_comparison.csv`, `results/all_database_features.csv`, and 6 PNG plots

---

### `extract_hmof_properties.py` — Phase 2: Property Extraction

**Purpose:** Parses all 32,768 hMOF-mofdb JSON property files and consolidates them into a single ML-ready CSV.

**Key operations:**
- Reads individual JSON files, one per MOF structure, containing adsorption isotherms (CO₂ at 0.01, 0.05, 0.1, 0.5, 1.0, 2.5 bar; CH₄, H₂, N₂)
- Extracts pore descriptors: LCD (Largest Cavity Diameter), PLD (Pore Limiting Diameter), surface area (m²/g and m²/cm³), void fraction
- Extracts structural metadata: number of atoms, cell volume, unique elements, embedded CIF
- Generates `results/hmof_property_summary.txt` with per-property statistics
- Writes master property table to `data/hmof_properties.csv` and a cleaned ML-ready copy to `data/hmof_ml_dataset.csv`

**Outputs:** `data/hmof_properties.csv` (32,768 × 43 columns), `data/hmof_ml_dataset.csv`, `results/hmof_property_summary.txt`, `results/hmof_property_distributions.png`, `results/hmof_correlation_heatmap.png`

---

### `integrate_qmof.py` — Task 3.1: QMOF Quantum Feature Integration

**Purpose:** Merges DFT-computed quantum-chemical descriptors from the QMOF database into the hMOF property table via structural matching.

**Key operations:**
- Loads QMOF CSV (band gap, metal partial charges, DDEC6 charges, magnetic moments)
- Performs structure-name matching between hMOF and QMOF entries
- Left-joins quantum features onto `hmof_ml_dataset.csv`, filling unmatched rows with NaN
- Writes the enriched table to `data/hmof_enhanced.csv`

**Outputs:** `data/hmof_enhanced.csv` (32,768 × 43 columns with quantum features appended)

---

### `build_graphs.py` — Task 3.2: Graph Construction

**Purpose:** Converts embedded CIF crystal structures into atom-bond graph representations stored as `.npz` files for use with PyTorch Geometric.

**Key operations:**
- Reads the embedded CIF string from each row of `hmof_enhanced.csv`
- Parses the crystal unit cell and atomic positions using pymatgen's `CifParser`
- Builds a radius-based neighbor graph (cutoff ~6 Å) for each structure
- Encodes node features per atom: atomic number Z, Pauling electronegativity, covalent radius, row/group in periodic table (9 features total)
- Encodes edge features: interatomic distance
- Saves each graph as a compressed `.npz` file in `data/graphs/`
- Records per-structure success/failure and node/edge counts in `data/graph_stats.csv`

**Outputs:** `data/graphs/*.npz` (32,768 files), `data/graph_stats.csv`

---

### `chemical_embeddings.py` — Task 3.3: Chemical Feature Vectors

**Purpose:** Generates 145-dimensional Magpie-style chemical feature vectors from MOF elemental compositions.

**Key operations:**
- Parses elemental formulas from the hMOF dataset
- Computes element-weighted statistics (mean, std, min, max, range) over 29 Magpie elemental properties (electronegativity, atomic radius, formation enthalpy, valence electrons, cohesive energy, etc.)
- Generates an additional set of composition-level features (element diversity, metal fraction, etc.)
- Validates for NaN values and reports coverage
- Saves the embedding matrix as `data/chemical_features.npy` (shape: 32,768 × 145)
- Saves feature names to `data/chemical_feature_names.json`
- Writes summary statistics to `data/chemical_features_summary.csv`

**Outputs:** `data/chemical_features.npy`, `data/chemical_feature_names.json`, `data/chemical_features_summary.csv`

---

### `preprocess_pipeline.py` — Tasks 3.4–3.6: Normalization, Clipping & Splits

**Purpose:** Applies outlier clipping, feature normalization, and stratified train/val/test splitting to produce the final ML datasets.

**Key operations:**
- Loads `hmof_enhanced.csv`; selects the 17 feature columns used for modeling
- Applies IQR-based outlier clipping (1.5×IQR rule) per column to reduce extreme-value influence
- Fits a `StandardScaler` on the training data only and applies it to all splits (prevents data leakage)
- Performs stratified 80/10/10 split using CO₂ uptake bins for even label coverage
- Saves the fitted scaler to `data/scaler.pkl`
- Saves preprocessing configuration (bounds, column list) to `data/preprocess_config.json`
- Writes `data/hmof_train.csv`, `data/hmof_val.csv`, `data/hmof_test.csv`
- Generates `results/preprocessing_summary.txt`

**Outputs:** `data/hmof_train.csv` (26,214), `data/hmof_val.csv` (3,277), `data/hmof_test.csv` (3,277), `data/scaler.pkl`, `data/preprocess_config.json`, `results/preprocessing_summary.txt`

---

### `prepare_transfer_sets.py` — Tasks 3.7–3.9: Transfer & Zero-Shot Sets

**Purpose:** Creates transfer learning and zero-shot evaluation datasets from CoRE-MOF, GA_MOFs, and COF databases.

**Key operations:**
- **CoRE-MOF split:** Reads CoRE-MOF 2019-ASR CSV (12,020 structures) and performs a 10/90 split → fine-tuning set and held-out test set
- **GA_MOFs zero-shot set:** Parses all 51,163 GA_MOF CIF filenames and extracts basic structural metadata (atom count, volume) to create a zero-shot evaluation catalog
- **COF exclusion manifest:** Merges structure IDs from CoRE-COFs (1,242) and CURATED-COFs to build an exclusion list preventing COF contamination in MOF-only evaluations
- Writes `results/transfer_sets_summary.txt`

**Outputs:** `data/core_mof_ft.csv` (1,202), `data/core_mof_test.csv` (10,818), `data/ga_mofs_test.csv` (51,163), `data/cofs_excluded.txt` (1,242 IDs), `results/transfer_sets_summary.txt`

---

### `verify_phase3.py` — Phase 3 Verification

**Purpose:** Automated sanity-check script that validates all Phase 3 outputs before proceeding to modeling.

**Checks performed:**
- Row counts match expected totals (26,214 / 3,277 / 3,277 for train/val/test)
- No overlap between train, val, and test splits (by MOF ID)
- `chemical_features.npy` shape is (32,768, 145) with zero NaN values
- `data/graphs/` directory contains 32,768 `.npz` files
- `graph_stats.csv` success rate reported
- `scaler.pkl` loads correctly and has expected `mean_` / `scale_` attributes

**Output:** Console pass/fail report; exits with error code on any failure

---

### `baselines.py` — Task 4.5: Baseline Models

**Purpose:** Trains and evaluates three baseline models (XGBoost, Random Forest, MLP) on the tabular chemical features to establish benchmarks before the hybrid model.

**Key operations:**
- Loads `hmof_train/val/test.csv` (17 structural/chemical features) and the target `CO2_uptake_1.0bar`
- **XGBoost:** gradient-boosted trees with 500 estimators, max_depth 6, early stopping on validation
- **Random Forest:** 100 trees, max_depth 20, scikit-learn implementation
- **MLP baseline:** 3-layer fully connected network (256→128→64), ReLU activations, Adam optimizer, early stopping at 35 epochs
- Reports MAE, RMSE, and R² on validation and test sets
- Saves trained model artifacts to `model/xgboost.pkl`, `model/random_forest.pkl`, `model/mlp_baseline.pt`
- Writes all metrics to `results/baseline_results.json`

**Outputs:** `model/xgboost.pkl`, `model/random_forest.pkl`, `model/mlp_baseline.pt`, `results/baseline_results.json`

---

### `hybrid_model.py` — Task 4.6: Hybrid Model Architecture

**Purpose:** Defines the full multi-branch hybrid architecture that combines graph-based structural learning with tabular chemical and quantum feature branches.

**Architecture components:**
- **GNN Branch (SchNet-style):** 4 interaction layers, each with a continuous-filter convolutional operator using radial basis function (RBF) edge features; skip/residual connections; `GlobalMeanPooling` over atom representations → 256-dim graph embedding
- **Chemical MLP Branch:** 3-layer MLP (145 → 256 → 128) for the Magpie chemical feature vectors
- **Quantum MLP Branch:** 2-layer MLP for QMOF quantum features (when available)
- **Residual Fusion Block:** Concatenates all branch outputs; 2 fully connected layers with batch normalization, GELU activations, and residual skip connection → scalar CO₂ uptake prediction
- **MC Dropout:** Dropout applied at inference time (20 forward passes) to produce prediction mean and uncertainty (std) estimates
- **Total parameters:** 325,881

**Key classes:** `SchNetInteraction`, `GNNBranch`, `ChemicalMLP`, `HybridMOFModel`

---

### `train_hybrid.py` — Task 4.7: Training Loop (v1)

**Purpose:** End-to-end training script for the original hybrid model (v1) with full evaluation and uncertainty quantification.

**Key operations:**
- Builds `torch_geometric.data.Dataset` from `data/graphs/*.npz`, `data/chemical_features.npy`, and the train/val/test CSVs
- Training loop: MSE loss, Adam optimizer (lr=1e-3, weight_decay=1e-5), `ReduceLROnPlateau` (patience=10, factor=0.5), early stopping (patience=25)
- Evaluates MAE, RMSE, R² on validation set after each epoch
- Runs 20-sample MC Dropout inference on the test set to produce uncertainty estimates
- Saves the best checkpoint to `checkpoints/best_hybrid.pt`
- Saves full training history (train/val loss curves, MAE, R², LR schedule) and final metrics to `results/hybrid_results.json`

**Outputs:** `checkpoints/best_hybrid.pt`, `results/hybrid_results.json`

---

### `hybrid_model_v2.py` — Phase 5: Fixed Architecture

**Purpose:** Redesigned hybrid architecture addressing the weaknesses of v1 (custom SchNet, mean pooling, no attention fusion).

**Key improvements over v1:**
- Uses PyG's built-in `SchNet` (not a custom reimplementation) for battle-tested interaction layers
- `GlobalAttentionPooling` (gate network selects important atoms) instead of naive mean pooling
- `GaussianRBF` edge-distance expansion (50 Gaussian bins, γ=10) for richer bond-level features
- **Cross-attention fusion** (`CrossAttentionFusion`): chemical features act as queries attending to GNN key/values across 4 heads, enabling the model to selectively weight structural information using chemical context
- Quantum branch now uses a learnable **zero-fill embedding** for missing samples rather than hard zeros
- `parameter_groups()` helper exposes layer-wise LR groups for the optimizer (GNN gets lower LR)

**Architecture summary:**

| Component | Details |
|---|---|
| `GNNBranch` | SchNet (3 interactions, hidden=128, filters=64) + `GlobalAttentionPooling` → 128-dim |
| `ChemicalBranch` | 145 → 256 → 128 (LayerNorm + GELU + Dropout) |
| `QuantumBranch` | 8 → 64, learnable missing embedding |
| `CrossAttentionFusion` | chem-queries → GNN-keys/values, 4 heads → 128-dim fused |
| `PredictionHead` | fused(128) + quantum(64) → 64 → 1 |
| `GNNOnlyModel` | Stripped SchNet for Task 5.1 diagnosis (no tabular branches) |

**Outputs:** Architecture class definitions only (no training); imported by `train_hybrid_fixed.py` and `ablation_study.py`

---

### `train_hybrid_fixed.py` — Phase 5: Fixed Training Pipeline

**Purpose:** Replacement training script for `hybrid_model_v2.py` with improved optimization, logging, and multi-seed ensemble support.

**Key improvements over `train_hybrid.py`:**
- **`CosineAnnealingWarmRestarts`** (T₀=50 epochs, T_mult=2) instead of `ReduceLROnPlateau` — avoids LR plateau stagnation
- **Gradient clipping** (`max_norm=10.0`) to prevent GNN exploding gradients
- **Layer-wise LR decay**: GNN branch gets `gnn_lr=1e-4`, tabular branches + fusion head get `tabular_lr=1e-3`
- **RBF edge features** pre-computed per sample in `MOFDataset.__getitem__` (50 Gaussian bins)
- Masked quantum loss — `QuantumBranch` handles missing rows by mask rather than all-zero imputation
- Optional **W&B** and **TensorBoard** logging (auto-detected on import)
- `run_ensemble(seeds)` helper trains N independent seeds and averages test predictions

**CLI usage:**
```bash
python code/train_hybrid_fixed.py --stage gnn_only   # Task 5.1 diagnosis
python code/train_hybrid_fixed.py --stage hybrid      # full hybrid
python code/train_hybrid_fixed.py --stage ensemble --seeds 3
```

**Outputs:** `checkpoints/best_hybrid_stage_seed<N>.pt`, `results/train_<stage>_seed<N>.json`, (optionally) `results/ensemble_<stage>.json`

---

### `ablation_study.py` — Phase 6: Full Ablation (Experiments A–I)

**Purpose:** Runs all 9 ablation experiments with identical training protocols and saves results to a summary CSV for analysis.

**Experiments:**

| ID | Name | Description | Expected R² |
|----|------|-------------|:-----------:|
| A | No pre-training | Train from scratch on CoRE-MOF 10% only (no hMOF transfer) | ~0.60 |
| B | XGBoost baseline | Re-run for fair apples-to-apples comparison | 0.93 |
| C | GNN-only | SchNet + attention pooling, no chemical/quantum | >0.75 |
| D | Chemical-only | 145-dim Magpie MLP, no graph input | ~0.85 |
| E | No quantum | GNN + Chemical with cross-attention, no QuantumBranch | ~0.88 |
| F | No attention | Concatenate all branches, plain MLP head (no cross-attention) | ~0.82 |
| G | Full hybrid | Complete `HybridMOFModel` v2 | >0.90 |
| H | Unified training | Train on hMOF + CoRE-MOF 10% combined | ~0.91 |
| I | Ensemble (GNN+XGB) | Average predictions from exp G + XGBoost | >0.94 |

**Key classes defined:** `ChemicalOnlyModel` (Exp D), `HybridNoAttentionModel` (Exp F), `HybridNoQuantumModel` (Exp E)

**CLI usage:**
```bash
python code/ablation_study.py                     # all experiments
python code/ablation_study.py --exps C D G        # specific experiments
python code/ablation_study.py --skip_existing     # resume interrupted run
```

**Outputs:** `results/ablation_<X>.json` per experiment, `results/ablation_summary_table.csv`

---

### `transfer_learning.py` — Phase 5 Task 5.5: Transfer Learning

**Purpose:** Evaluates and applies the pre-trained hMOF model to the CoRE-MOF experimental database via three protocols: zero-shot, frozen-GNN fine-tune, and full fine-tune.

**Pipeline:**
1. **Zero-shot evaluation** — loads best hMOF checkpoint and evaluates directly on CoRE-MOF test set (10,818 structures) with no adaptation
2. **Frozen-GNN fine-tune** — freezes all `model.gnn` parameters; trains only `ChemicalBranch` + `CrossAttentionFusion` + `PredictionHead` on CoRE-MOF 10% (1,202 structures) for 30 epochs at lr=1e-3
3. **Full fine-tune (low LR)** — unfreezes all layers and trains with layer-wise LRs (GNN: 10× lower) on CoRE-MOF 10% for 50 epochs at lr=5e-5

**Helper functions:** `load_pretrained()`, `freeze_gnn()`, `unfreeze_all()`, `evaluate_model()`, `finetune()`

**CLI usage:**
```bash
python code/transfer_learning.py
python code/transfer_learning.py --frozen_epochs 50 --full_finetune_lr 1e-5
```

**Outputs:** `checkpoints/transfer_frozen_gnn.pt`, `checkpoints/transfer_full_finetune.pt`, `results/transfer_learning_results.json`

---

## data/ — Processed Datasets

All files generated by the pipeline scripts and consumed by the models.

| File / Folder                    | Size        | Description |
|----------------------------------|:-----------:|-------------|
| `catalog.csv`                    | 34 MB       | Master catalog of all hMOF structures with metadata |
| `hmof_properties.csv`            | 13 MB       | Raw extracted properties from hMOF JSON files (CO₂/CH₄/H₂/N₂ uptakes, pore descriptors) |
| `hmof_ml_dataset.csv`            | 13 MB       | Cleaned version of `hmof_properties.csv` ready for ML |
| `hmof_enhanced.csv`              | 13 MB       | `hmof_ml_dataset.csv` + QMOF quantum features merged (32,768 × 43) |
| `hmof_train.csv`                 | 15 MB       | Training split — 26,214 structures (80%) after IQR clipping + normalization |
| `hmof_val.csv`                   | 1.9 MB      | Validation split — 3,277 structures (10%) |
| `hmof_test.csv`                  | 1.9 MB      | Test split — 3,277 structures (10%) |
| `chemical_features.npy`          | 18 MB       | 145-dim Magpie chemical embedding matrix, shape (32,768 × 145), zero NaN |
| `chemical_feature_names.json`    | 3.3 KB      | Human-readable labels for each of the 145 chemical embedding dimensions |
| `chemical_features_summary.csv`  | 2.7 MB      | Per-feature mean, std, min, max statistics |
| `preprocess_config.json`         | 661 B       | IQR clip bounds and feature column list saved from preprocessing |
| `scaler.pkl`                     | 1.2 KB      | Fitted `StandardScaler` (train-set fit only) for inference-time normalization |
| `structural_features.csv`        | 26 KB       | Structural descriptors extracted during Phase 1 EDA |
| `core_mof_ft.csv`                | 172 KB      | CoRE-MOF fine-tuning set — 1,202 structures (10% of CoRE-MOF 2019-ASR) |
| `core_mof_test.csv`              | 1.5 MB      | CoRE-MOF held-out test set — 10,818 structures (90%) |
| `ga_mofs_test.csv`               | 4.3 MB      | GA_MOFs zero-shot test catalog — 51,163 structures |
| `cofs_excluded.txt`              | 35 KB       | COF exclusion manifest — 1,242 structure IDs filtered from MOF evaluations |
| `graph_stats.csv`                | 928 KB      | Per-structure node count, edge count, and parse success flags |
| `graphs/`                        | ~GB         | 32,768 individual `.npz` graph files (node features, edge index, edge distances) |

---

## results/ — EDA & Model Outputs

All report files, visualizations, and model performance summaries.

| File                                   | Description |
|----------------------------------------|-------------|
| `complete_database_eda_report.md`      | Phase 1 EDA report: 3,500-structure sample across all 7 databases with structural statistics and element distribution |
| `all_database_features.csv`            | Aggregated structural feature table from all database EDA samples |
| `dataset_comparison.csv`              | Side-by-side statistics (count, avg atoms, avg density, avg volume, metal content) per database |
| `dataset_distribution.png`            | Bar chart of structure counts per database |
| `feature_distributions_by_dataset.png`| Histogram grid of density, volume, and atom count grouped by database source |
| `density_vs_volume_by_dataset.png`    | Scatter plot of density vs. unit cell volume colored by database |
| `element_frequency_all.png`           | Element frequency distribution across all 3,500 sampled structures (62 unique elements) |
| `metal_content_by_dataset.png`        | Bar chart comparing metal fraction across databases |
| `hmof_correlation_heatmap.png`        | Pearson correlation heatmap of all 43 extracted hMOF numerical properties |
| `hmof_property_distributions.png`     | Distribution plots for all hMOF properties (CO₂/CH₄/H₂/N₂ uptakes, pore descriptors) |
| `hmof_property_summary.txt`           | Statistical summary of all hMOF extracted properties (mean, std, min, max, coverage) |
| `preprocessing_summary.txt`           | IQR clip bounds per column, number of clipped values, and train/val/test split counts |
| `transfer_sets_summary.txt`           | Split sizes for CoRE-MOF fine-tune/test sets and GA_MOF zero-shot catalog |
| `baseline_results.json`               | MAE, RMSE, R² for XGBoost, Random Forest, and MLP baseline on val and test sets |
| `hybrid_results.json`                 | Full training config, per-epoch train/val loss, MAE, R², LR schedule, and final test metrics for the hybrid GNN model |

---

## model/ — Saved Artifacts

| File                  | Size    | Description |
|-----------------------|:-------:|-------------|
| `xgboost.pkl`         | 6.5 MB  | Trained XGBoost model (500 trees, max_depth 6) — best baseline, R²=0.93 on test |
| `random_forest.pkl`   | 154 MB  | Trained Random Forest (100 trees, max_depth 20) — R²=0.85 on test |
| `mlp_baseline.pt`     | 329 KB  | Trained MLP baseline (256→128→64) — R²=0.42 on test |

---

## docs/

| File                                        | Description |
|---------------------------------------------|-------------|
| `Dataset_Download_Guide.docx`               | Step-by-step guide for downloading all 7 source databases (links, authentication, directory structure) |
| `ML_PPN_CO2_Capture_Project_Proposal.docx`  | Original project proposal document: research objectives, methodology, and evaluation criteria |

---

## Pipeline Overview

```
Phase 1: Database EDA
  └── database_eda.py → results/ (EDA report + 6 plots)

Phase 2: Data Extraction
  └── extract_hmof_properties.py → data/hmof_properties.csv + data/hmof_ml_dataset.csv

Phase 3: Feature Engineering & Splits
  ├── integrate_qmof.py         → data/hmof_enhanced.csv         (+ quantum features)
  ├── build_graphs.py            → data/graphs/*.npz               (32,768 graph files)
  ├── chemical_embeddings.py     → data/chemical_features.npy      (32,768 × 145)
  ├── preprocess_pipeline.py     → data/hmof_train/val/test.csv   (26,214 / 3,277 / 3,277)
  ├── prepare_transfer_sets.py   → data/core_mof_*.csv, ga_mofs_test.csv, cofs_excluded.txt
  └── verify_phase3.py           → Console validation report

Phase 4: Modeling (v1 — baseline)
  ├── baselines.py               → model/xgboost.pkl, rf.pkl, mlp.pt + results/baseline_results.json
  ├── hybrid_model.py            → Architecture definition v1 (325,881 params)
  └── train_hybrid.py            → checkpoints/best_hybrid.pt + results/hybrid_results.json

Phase 5: Fixed Architecture & Transfer Learning
  ├── hybrid_model_v2.py         → Architecture v2 (SchNet + GlobalAttentionPooling + CrossAttentionFusion)
  ├── train_hybrid_fixed.py      → Fixed training (CosineAnnealing, grad clip, layer-wise LR, ensemble)
  └── transfer_learning.py       → Zero-shot → frozen-GNN → full fine-tune on CoRE-MOF

Phase 6: Ablation Study
  └── ablation_study.py          → Experiments A–I → results/ablation_summary_table.csv
```

---

## Results So Far

### Phase 1 — EDA Highlights

| Database              | Structures Sampled | Avg Atoms | Avg Density (g/cm³) | Metal Content |
|-----------------------|:-----------------:|:---------:|:-------------------:|:------------:|
| hMOF-mofdb            | 500               | 124.3     | 0.944               | 100%         |
| CoRE-MOF              | —                 | —         | —                   | —            |
| GA_MOFs               | 500               | 68.1      | 1.065               | 100%         |
| qmof_database         | 500 + 500         | 106.9 / 102.2 | 1.725 / 0.773   | 86% / 100%   |
| CoRE-COFs             | 500               | 242.9     | 0.589               | 8.6%         |
| CURATED-COFs          | 500               | 325.8     | 0.596               | 7.4%         |
| MOF_database          | 500               | 75.7      | 0.975               | 100%         |

- **62 unique elements** found; C, H, O, N, Zn most prevalent
- Density range: 0.045 – 3.753 g/cm³; unit cell volume range: 152 – 288,235 Å³

### Phase 2 — hMOF Property Extraction (32,768 structures, 100% coverage)

| Property              | Mean    | Std     | Min    | Max     |
|-----------------------|:-------:|:-------:|:------:|:-------:|
| CO₂ uptake @ 1.0 bar  | 2.928   | 1.668   | 0.000  | 9.765   |
| CO₂ uptake @ 0.1 bar  | 0.707   | 0.747   | 0.000  | 6.459   |
| Surface Area (m²/g)   | 2439.3  | —       | 0.00   | 6741.7  |
| Void Fraction         | 0.62    | —       | 0.00   | 0.93    |
| LCD (Å)               | 8.51    | —       | 0.00   | 24.25   |

### Phase 3 — Preprocessing

| Stage          | Details |
|----------------|---------|
| Outlier Clipping | IQR-based; CO₂@0.01bar: 11.81% clipped; CO₂@1.0bar: 0.68% clipped |
| Splits         | Train 26,214 (80%) / Val 3,277 (10%) / Test 3,277 (10%) |
| Features       | 17 columns: structural + adsorption isotherms |
| Chemical embeds| 145-dim Magpie vectors, 32,768 structures, 0 NaN |
| Graphs built   | 32,768 `.npz` files with atom-bond graphs |
| CoRE-MOF       | Fine-tune: 1,202 • Test: 10,818 |
| GA_MOFs        | Zero-shot catalog: 51,163 structures |
| COFs excluded  | 1,242 structure IDs |

### Phase 4 — Model Performance

**Target:** `CO2_uptake_1.0bar` (mol/kg · mean 2.928, std 1.668)

#### Baseline Models

| Model           | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² | Train Time |
|-----------------|:-------:|:--------:|:------:|:--------:|:---------:|:-------:|:----------:|
| **XGBoost**     | 0.1946  | 0.2679   | **0.928** | 0.1926 | 0.2642 | **0.930** | 12.6 s |
| Random Forest   | 0.2870  | 0.3888   | 0.849  | 0.2824   | 0.3855    | 0.851   | 3.2 s      |
| MLP Baseline    | 0.6063  | 0.7631   | 0.419  | 0.6055   | 0.7570    | 0.425   | 69.1 s     |

#### Hybrid GNN Model (SchNet + Chemical MLP + Quantum MLP + Residual Fusion)

| Setting         | Value |
|-----------------|-------|
| Parameters      | 325,881 |
| Best Epoch      | 115 / 140 trained |
| Train Time      | 5,414 s (~90 min) |
| LR Schedule     | 1e-3 → decayed to 6.25e-5 via ReduceLROnPlateau |

| Split      | MAE    | RMSE   | R²     |
|------------|:------:|:------:|:------:|
| Validation | 0.6079 | 0.7474 | 0.4424 |
| **Test**   | **0.5983** | **0.7418** | **0.4479** |

**MC Dropout Uncertainty (20 samples, test set):**
- MC MAE: 0.5985
- Mean predictive std: 0.0494 mol/kg
- Max predictive std: 0.1884 mol/kg

#### Summary & Gap Analysis

| Model        | Test R² | Test MAE |
|--------------|:-------:|:--------:|
| XGBoost ⭐   | **0.930** | 0.193  |
| Random Forest| 0.851   | 0.282    |
| MLP Baseline | 0.425   | 0.606    |
| Hybrid GNN   | 0.448   | 0.598    |

> **Key finding:** XGBoost dominates on the tabular structural features alone (R²=0.93), outperforming the hybrid GNN (R²=0.45). The GNN has not yet learned to exploit graph topology as effectively as gradient boosting on the 17 pre-computed features. Next steps: GNN architecture tuning, longer training, ensemble XGBoost + GNN predictions.

---

## Requirements

```
Python 3.9+
torch >= 2.0.0
torch_geometric
scikit-learn
xgboost
pandas, numpy, matplotlib, seaborn
pymatgen
matminer
tqdm
```

---

## Usage

```bash
# 1. Activate virtual environment
.\carbon\Scripts\activate

# 2. Phase 1 — Database EDA
python code/database_eda.py

# 3. Phase 2 — Extract hMOF properties
python code/extract_hmof_properties.py

# 4. Phase 3 — Feature engineering
python code/integrate_qmof.py
python code/build_graphs.py
python code/chemical_embeddings.py
python code/preprocess_pipeline.py
python code/prepare_transfer_sets.py
python code/verify_phase3.py           # sanity checks

# 5. Phase 4 — Baseline models
python code/baselines.py

# 6. Phase 4 — Hybrid GNN model v1 (original)
python code/train_hybrid.py

# 7. Phase 5 — Fixed architecture (GNN diagnosis first, then full hybrid)
python code/train_hybrid_fixed.py --stage gnn_only
python code/train_hybrid_fixed.py --stage hybrid
python code/train_hybrid_fixed.py --stage ensemble --seeds 3

# 8. Phase 5 — Transfer learning to CoRE-MOF
python code/transfer_learning.py

# 9. Phase 6 — Full ablation study
python code/ablation_study.py                   # all A–I
python code/ablation_study.py --exps C D G      # specific experiments
python code/ablation_study.py --skip_existing   # resume
```
