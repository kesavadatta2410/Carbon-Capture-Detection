# 🧪 Carbon Capture ML Pipeline

> Machine learning pipeline for predicting CO₂ adsorption in hypothetical Metal-Organic Frameworks (hMOFs), integrating multi-database features, graph representations, and transfer learning.

---

## 📁 Project Structure

```
carbon/
├── Database/           # Raw source databases (read-only)
├── code/               # All processing & ML scripts
├── data/               # Processed datasets, features, and splits
│   └── graphs/         # Per-structure graph representations
├── results/            # EDA reports, plots, and summaries
├── docs/               # Project documentation
├── model/              # Trained model artifacts (empty — Phase 4)
├── checkpoints/        # Training checkpoints (empty — Phase 4)
├── carbon/             # Python virtual environment
├── preprocessing.log   # Log from preprocessing runs
└── README.md           # This file
```

---

## 🗄️ Database/

Raw source databases used for feature extraction and transfer learning.

| Folder | Description |
|--------|-------------|
| `hMOF-mofdb/` | 32,768 hypothetical MOF JSON files with embedded CIF data, adsorption isotherms (CO₂, CH₄, H₂, N₂), and pore descriptors. **Primary dataset.** |
| `qmof_database/` | QMOF database (~24K entries) with DFT-computed quantum properties (bandgap, formation energy). Used for sparse property integration. |
| `CoRE-MOF/` | Computation-Ready Experimental MOF database (ASR & FSR subsets, ~12K structures). Used for transfer learning splits. |
| `GA_MOFs/` | 51,163 genetically-assembled MOF CIF files. Used as a zero-shot test set (no adsorption labels). |
| `MOF_database/` | Large general MOF database (~324K files). Reference collection. |
| `CoRE-COFs_1242-v7.0/` | 1,242 Covalent Organic Frameworks. **Excluded** from MOF training for domain-gap analysis. |
| `CURATED-COFs/` | ~994 curated COF structures. **Excluded** alongside CoRE-COFs. |

---

## 💻 code/

All Python scripts for the pipeline, organized by phase.

| File | Phase | Description |
|------|-------|-------------|
| `database_eda.py` | 1 | Exploratory data analysis across all 7 databases. Generates summary statistics, distribution plots, correlation heatmaps, and a markdown report. |
| `extract_hmof_properties.py` | 2 | Parses all 32,768 hMOF JSON files to extract CO₂/CH₄/H₂/N₂ adsorption isotherms, pore descriptors (LCD, PLD, surface area, void fraction), and structural features. Produces the base ML dataset. |
| `integrate_qmof.py` | 3 | Merges quantum properties (bandgap, total energy, etc.) from QMOF into the hMOF dataset via MOF-ID matching. Handles deduplication and sparse left-join. |
| `chemical_embeddings.py` | 3 | Generates 145-dimensional Magpie-inspired chemical embeddings from elemental composition. Computes 5 statistics (mean, std, min, max, range) over 29 elemental properties. |
| `build_graphs.py` | 3 | Creates atom-bond graph representations from CIF data. Nodes carry atomic features (Z, electronegativity, covalent radius); edges are distance-based bonds. Saves one `.npz` per structure. |
| `preprocess_pipeline.py` | 3 | End-to-end preprocessing: IQR outlier clipping on adsorption values, log-transforms for skewed features, StandardScaler (fitted on train only), and 80/10/10 stratified train/val/test splitting. |
| `prepare_transfer_sets.py` | 3 | Prepares transfer learning data: 10/90 fine-tune/test split for CoRE-MOF, zero-shot test set from GA_MOFs (CIF parsing), and COF exclusion manifest. |
| `verify_phase3.py` | 3 | Verification script that checks all Phase 3 outputs — file existence, shapes, split ratios, overlap checks, and summary statistics. |

---

## 📊 data/

Processed datasets, feature matrices, and ML-ready splits.

| File | Size | Description |
|------|------|-------------|
| `hmof_properties.csv` | 12.5 MB | Base hMOF dataset (32,768 × 34) with adsorption isotherms, pore descriptors, and structural features. |
| `hmof_ml_dataset.csv` | 12.5 MB | Alias of the base ML-ready dataset. |
| `hmof_enhanced.csv` | 12.7 MB | hMOF dataset augmented with 9 QMOF quantum property columns (32,768 × 43). |
| `hmof_train.csv` | 15.2 MB | Training split — 26,214 structures (80%). |
| `hmof_val.csv` | 1.9 MB | Validation split — 3,277 structures (10%). |
| `hmof_test.csv` | 1.9 MB | Test split — 3,277 structures (10%). |
| `chemical_features.npy` | 19.0 MB | Chemical embedding matrix (32,768 × 145), float32, zero NaN. |
| `chemical_feature_names.json` | 3.4 KB | Ordered list of 145 chemical feature names. |
| `chemical_features_summary.csv` | 2.7 MB | Summary CSV with structure names and top-10 chemical features. |
| `scaler.pkl` | 1.2 KB | Fitted StandardScaler (trained on training set only). |
| `preprocess_config.json` | 2.5 KB | Preprocessing configuration: feature columns, log-transformed columns, outlier bounds. |
| `graph_stats.csv` | 927 KB | Per-structure graph statistics (num_nodes, num_edges, density, status). |
| `core_mof_ft.csv` | 171 KB | CoRE-MOF fine-tuning set — 1,202 structures (10%). |
| `core_mof_test.csv` | 1.5 MB | CoRE-MOF test set — 10,818 structures (90%). |
| `ga_mofs_test.csv` | 4.3 MB | GA_MOFs zero-shot test set — 51,163 structures (no adsorption labels). |
| `cofs_excluded.txt` | 35.6 KB | Manifest of 1,242 COF structures excluded from training. |
| `structural_features.csv` | 26.3 KB | Supplementary structural feature summary. |
| `catalog.csv` | 32.9 MB | Full database catalog from EDA phase. |

### data/graphs/

Contains **32,768** compressed `.npz` files (one per hMOF structure). Each file stores:

- `node_features` — (N × 3) array: atomic number, electronegativity, covalent radius
- `edge_index` — (2 × E) array: source and destination atom indices
- `edge_dist` — (E,) array: bond distances in Ångströms

Average graph size: **120 nodes, 261 edges**.

---

## 📈 results/

EDA outputs, visualizations, and processing summaries.

| File | Description |
|------|-------------|
| `complete_database_eda_report.md` | Full EDA report covering all 7 databases with statistics and findings. |
| `all_database_features.csv` | Aggregated feature table across all databases (3,500 sampled structures). |
| `dataset_comparison.csv` | Side-by-side comparison of database characteristics. |
| `dataset_distribution.png` | Bar chart of structure counts per database. |
| `feature_distributions_by_dataset.png` | Violin/box plots of key features across databases. |
| `density_vs_volume_by_dataset.png` | Scatter plot of density vs. cell volume colored by database. |
| `element_frequency_all.png` | Element frequency distribution across all databases. |
| `metal_content_by_dataset.png` | Metal atom prevalence comparison by database. |
| `hmof_property_distributions.png` | Distribution plots for all hMOF adsorption and pore properties. |
| `hmof_correlation_heatmap.png` | Pearson correlation heatmap of hMOF numeric features. |
| `hmof_property_summary.txt` | Text summary of hMOF property statistics. |
| `preprocessing_summary.txt` | Summary of outlier clipping, scaling, and split details. |
| `transfer_sets_summary.txt` | Summary of CoRE-MOF, GA_MOFs, and COF processing results. |

---

## 📄 docs/

| File | Description |
|------|-------------|
| `Dataset_Download_Guide.docx` | Instructions for downloading all source databases. |
| `ML_PPN_CO2_Capture_Project_Proposal.docx` | Original project proposal for ML-driven CO₂ capture prediction. |

---

## 🔬 Pipeline Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** — Database EDA | ✅ Complete | Exploratory analysis of 7 MOF/COF databases, 3,500 sampled structures. |
| **Phase 2** — Property Extraction | ✅ Complete | Parsed 32,768 hMOF JSONs → 34-column ML dataset with full adsorption coverage. |
| **Phase 3** — Feature Engineering | ✅ Complete | Graph representations, chemical embeddings, normalization, outlier handling, stratified splits, transfer sets, COF exclusion. |
| **Phase 4** — Advanced Analysis | 🔜 Upcoming | Model training, hyperparameter tuning, transfer learning experiments. |

---

## ⚙️ Dependencies

- Python 3.9+
- pandas, numpy, scikit-learn, scipy, tqdm, matplotlib, seaborn

## 🚀 Quick Start

```bash
# Activate virtual environment
.\carbon\Scripts\activate

# Run full preprocessing pipeline
python code/extract_hmof_properties.py    # Phase 2: Extract properties
python code/integrate_qmof.py             # Phase 3.1: QMOF integration
python code/chemical_embeddings.py        # Phase 3.3: Chemical features
python code/build_graphs.py               # Phase 3.2: Graph representations
python code/preprocess_pipeline.py        # Phase 3.4-3.6: Preprocess & split
python code/prepare_transfer_sets.py      # Phase 3.7-3.9: Transfer sets

# Verify all outputs
python code/verify_phase3.py
```
