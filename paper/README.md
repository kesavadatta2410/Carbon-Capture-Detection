# IEEE Paper — Compilation Guide

## Requirements

- **LaTeX distribution**: TeX Live 2022+ or MiKTeX 22+
- **Required packages** (all in standard distributions):
  `IEEEtran`, `amsmath`, `amssymb`, `algorithmic`, `algorithm`,
  `graphicx`, `booktabs`, `multirow`, `xcolor`, `url`, `hyperref`,
  `tikz`, `pgfplots`, `cite`, `balance`

## Compile Instructions

### Option 1: Command Line (Recommended)

```bash
cd paper/

# Full compilation (required for bibliography + cross-references)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 2: latexmk (Automatic)

```bash
cd paper/
latexmk -pdf main.tex
```

### Option 3: Overleaf

1. Upload `main.tex` and `references.bib` to a new Overleaf project
2. Set compiler to **pdfLaTeX**
3. Click **Recompile**

## Paper Structure

| Section | Content |
|---|---|
| Abstract | Overview of method and results |
| I. Introduction | CCS motivation, contributions |
| II. Related Work | SchNet, CGCNN, MEGNet, QMOF, transfer learning |
| III. Datasets | 7 databases, hMOF statistics (Tables I–II) |
| IV. Methodology | Graph construction (Algorithm 1), Magpie embeddings, architecture (Fig. 1), training (Algorithm 2) |
| V. Experiments | Baselines (Table III), ablation (Table IV), transfer (Table V), convergence (Fig. 2) |
| VI. Discussion | XGBoost gap analysis, attention benefit, limitations |
| VII. Conclusion | Summary and future directions |
| References | 20 post-2018 citations |

## Key Figures and Tables

| Label | Description |
|---|---|
| `tab:databases` | 7-database statistics |
| `tab:target_stats` | hMOF CO₂ uptake distribution |
| `tab:baselines` | Model comparison (RF, MLP, XGB, GNN, Hybrid) |
| `tab:ablation` | 9-experiment ablation study |
| `tab:transfer` | Transfer learning protocols |
| `fig:arch` | TikZ architecture block diagram |
| `fig:curves` | Validation R² convergence (pgfplots) |
| `alg:graph` | Graph construction algorithm |
| `alg:train` | HybridMOF-Net training algorithm |

## Equations

| Label | Description |
|---|---|
| `eq:node` | Node feature vector definition |
| `eq:rbf` | Gaussian RBF edge expansion |
| `eq:schnet` | SchNet message passing update |
| `eq:attnpool` | Global attention pooling |
| `eq:chemb` | Chemical branch encoding |
| `eq:quantum` | Quantum branch with missing-data handling |
| `eq:q`,`eq:k`,`eq:v`,`eq:cross-attn` | Cross-attention mechanism |
| `eq:head` | Prediction head |
| `eq:huber` | Huber loss function |
