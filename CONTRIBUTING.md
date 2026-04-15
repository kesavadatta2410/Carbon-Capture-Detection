# Contributing to Carbon Capture Detection

Thank you for your interest in contributing! This project is a research pipeline for predicting CO₂ adsorption in Metal-Organic Frameworks using hybrid GNN deep learning.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/Carbon-Capture-Detection.git
   cd Carbon-Capture-Detection
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

- Create a feature branch: `git checkout -b feature/your-feature-name`
- Make changes following the code style below
- Run tests before submitting
- Submit a **Pull Request** with a clear description

## Code Style

- Follow **PEP 8** for all Python files
- Use **type hints** for function signatures
- All modules must have a module-level docstring
- Use descriptive variable names; avoid single-letter names except for loop indices
- Maximum line length: 100 characters

## Project Structure

```
code/           → All pipeline scripts (EDA → modeling → evaluation)
data/           → Processed datasets and features (not committed to git)
checkpoints/    → Model checkpoints (not committed to git)
results/        → Evaluation outputs and visualizations
model/          → Final trained model artifacts
docs/           → Project documentation
paper/          → IEEE LaTeX paper source
```

## Adding New Experiments

1. Add experiment function to `code/ablation_study.py` following the `exp_*()` pattern
2. Register it in `EXP_REGISTRY` with name, expected R², and function
3. Document results in `results/ablation_<X>.json`

## Reporting Issues

Please use GitHub Issues and include:
- Python and library versions (`pip list`)
- Operating system
- Full error traceback
- Steps to reproduce

## Citation

If you use this code in your research, please cite:
```
@article{carbon-capture-detection-2024,
  title={Hybrid Graph Neural Networks for CO₂ Adsorption Prediction in Metal-Organic Frameworks},
  year={2024}
}
```
