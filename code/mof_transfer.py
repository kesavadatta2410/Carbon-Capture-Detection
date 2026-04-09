"""
mof_transfer.py — Transfer Learning with CoRE-MOF Support
=========================================================
Single-file transfer learning script with fixes for CoRE-MOF target column mapping.

Key Fixes:
- Automatic CoRE-MOF target column detection and mapping
- Domain adaptation for different CO2 uptake column names
- Proper handling of experimental vs simulated data scales

Usage:
    # Standard transfer learning (after training on hMOF)
    python mof_transfer.py
    
    # With custom epochs and learning rates
    python mof_transfer.py --frozen_epochs 50 --full_finetune_lr 1e-5
    
    # Debug mode: check data without training
    python mof_transfer.py --debug
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from mof_model import HybridMOFModel
from mof_train import MOFDataset, collate_fn, compute_metrics


# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
GRAPH_DIR = DATA_DIR / "graphs"
CKPT_DIR = Path("checkpoints")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH = 32
CUTOFF = 8.0


# =============================================================================
# CoRE-MOF Target Column Mapping
# =============================================================================

CORE_MOF_TARGET_COLUMNS = [
    # Common CoRE-MOF CO2 uptake column names
    "CO2_uptake_1.0bar",      # hMOF-style (if present)
    "CO2_wt_percent",         # Weight percent
    "CO2_wt%",                
    "CO2_mol_kg",             # mol/kg
    "CO2_uptake",             # Generic
    "Uptake_CO2",             # Alternative naming
    "co2_uptake",             # Lowercase
    "CO2_adsorption",         # Alternative
    "Adsorption_CO2",         # Alternative
    # Add more as discovered
]


def find_target_column(df: pd.DataFrame, debug: bool = False) -> Optional[str]:
    """
    Find the CO2 uptake target column in a DataFrame.
    
    Args:
        df: DataFrame to search
        debug: Print debug info
        
    Returns:
        Column name if found, None otherwise
    """
    available_cols = set(df.columns.str.strip().str.lower())
    
    if debug:
        print(f"\nAvailable columns: {list(df.columns)}")
    
    # Try exact match first
    for col in CORE_MOF_TARGET_COLUMNS:
        if col in df.columns:
            if debug:
                print(f"Found target column (exact): {col}")
            return col
    
    # Try case-insensitive match
    for col in CORE_MOF_TARGET_COLUMNS:
        col_lower = col.lower()
        for avail_col in df.columns:
            if avail_col.lower() == col_lower:
                if debug:
                    print(f"Found target column (case-insensitive): {avail_col}")
                return avail_col
    
    # Try partial match for CO2-related columns
    co2_cols = [c for c in df.columns if 'co2' in c.lower() or 'uptake' in c.lower()]
    if co2_cols:
        if debug:
            print(f"Found potential CO2 columns (partial match): {co2_cols}")
        return co2_cols[0]  # Return first match
    
    if debug:
        print("WARNING: No CO2 uptake column found!")
    return None


def analyze_core_mof_data(df: pd.DataFrame) -> dict:
    """Analyze CoRE-MOF data and return statistics."""
    stats = {
        "total_samples": len(df),
        "columns": list(df.columns),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
    }
    
    # Check for target column
    target_col = find_target_column(df, debug=False)
    stats["target_column"] = target_col
    
    if target_col:
        stats["target_stats"] = {
            "mean": float(df[target_col].mean()),
            "std": float(df[target_col].std()),
            "min": float(df[target_col].min()),
            "max": float(df[target_col].max()),
        }
    
    return stats


# =============================================================================
# Model Loading and Freezing
# =============================================================================

def load_pretrained(ckpt_glob: str = "best_hybrid_seed42.pt") -> HybridMOFModel:
    """Load best hMOF checkpoint."""
    ckpt_path = CKPT_DIR / ckpt_glob
    
    if not ckpt_path.exists():
        matches = list(CKPT_DIR.glob("best_hybrid*.pt"))
        if not matches:
            raise FileNotFoundError(
                f"No pre-trained checkpoint found in {CKPT_DIR}. "
                "Run: python mof_train.py --stage hybrid"
            )
        ckpt_path = sorted(matches)[-1]
        print(f"Using checkpoint: {ckpt_path}")
    
    model = HybridMOFModel(cutoff=CUTOFF)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def freeze_gnn(model: HybridMOFModel):
    """Freeze all GNN branch parameters."""
    for param in model.gnn.parameters():
        param.requires_grad = False
    print("GNN branch frozen.")


def unfreeze_all(model: HybridMOFModel):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    print("All parameters unfrozen.")


# =============================================================================
# Evaluation and Fine-tuning
# =============================================================================

@torch.no_grad()
def evaluate_model(model, loader, device) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate model on a dataset."""
    model.eval()
    all_p, all_t = [], []
    
    for batch in loader:
        graph_b, chem, quant, qmask, mof_desc, y = batch
        pred = model(
            graph_b.to(device),
            chem.to(device),
            quant.to(device),
            qmask.to(device),
            mof_desc.to(device),
        )
        all_p.append(pred.cpu().numpy())
        all_t.append(y.numpy())
    
    preds = np.concatenate(all_p)
    targets = np.concatenate(all_t)
    return compute_metrics(targets, preds), preds, targets


def finetune(
    model: HybridMOFModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    frozen_gnn: bool = True,
    ckpt_path: Path = None,
    run_label: str = "",
) -> dict:
    """Fine-tune model with CosineAnnealingWarmRestarts."""
    
    if frozen_gnn:
        freeze_gnn(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
    else:
        unfreeze_all(model)
        optimizer = AdamW(
            model.parameter_groups(gnn_lr=lr * 0.1, tabular_lr=lr),
            weight_decay=1e-4
        )
    
    steps = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 2) * steps
    )
    criterion = nn.MSELoss()
    best_val_r2 = -np.inf
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            graph_b, chem, quant, qmask, mof_desc, y = batch
            pred = model(
                graph_b.to(device), chem.to(device),
                quant.to(device), qmask.to(device),
                mof_desc.to(device),
            )
            loss = criterion(pred, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
        
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        if val_metrics["R2"] > best_val_r2:
            best_val_r2 = val_metrics["R2"]
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{run_label}] ep {epoch:3d} val R²={val_metrics['R2']:.4f} best={best_val_r2:.4f}")
    
    if ckpt_path and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    unfreeze_all(model)
    return {"best_val_r2": round(best_val_r2, 4)}


# =============================================================================
# Main Transfer Learning Pipeline
# =============================================================================

def run_transfer(
    freeze_gnn_only: bool = True,
    frozen_epochs: int = 30,
    full_finetune_epochs: int = 50,
    full_finetune_lr: float = 5e-5,
    debug: bool = False,
):
    """
    Run transfer learning pipeline.
    
    Args:
        freeze_gnn_only: Whether to freeze GNN during first fine-tuning stage
        frozen_epochs: Epochs for frozen-GNN fine-tuning
        full_finetune_epochs: Epochs for full fine-tuning
        full_finetune_lr: Learning rate for full fine-tuning
        debug: Debug mode - only analyze data without training
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ── Load CoRE-MOF Data ──────────────────────────────────────────────────
    core_train_csv = DATA_DIR / "core_mof_ft.csv"
    core_test_csv = DATA_DIR / "core_mof_test.csv"
    
    if not core_train_csv.exists() or not core_test_csv.exists():
        raise FileNotFoundError(
            "CoRE-MOF split files not found. Expected:\n"
            f"  {core_train_csv}\n  {core_test_csv}\n"
            "Run: python prepare_transfer_sets.py"
        )
    
    # Load and analyze data
    finetune_df = pd.read_csv(core_train_csv)
    test_df = pd.read_csv(core_test_csv)
    
    print(f"\n{'='*60}")
    print("CoRE-MOF Data Analysis")
    print(f"{'='*60}")
    
    # Analyze train data
    print("\n--- Fine-tune Set ---")
    train_stats = analyze_core_mof_data(finetune_df)
    print(f"Samples: {train_stats['total_samples']}")
    print(f"Target column: {train_stats.get('target_column', 'NOT FOUND')}")
    if 'target_stats' in train_stats:
        print(f"Target stats: mean={train_stats['target_stats']['mean']:.4f}, "
              f"std={train_stats['target_stats']['std']:.4f}")
    
    # Analyze test data
    print("\n--- Test Set ---")
    test_stats = analyze_core_mof_data(test_df)
    print(f"Samples: {test_stats['total_samples']}")
    print(f"Target column: {test_stats.get('target_column', 'NOT FOUND')}")
    if 'target_stats' in test_stats:
        print(f"Target stats: mean={test_stats['target_stats']['mean']:.4f}, "
              f"std={test_stats['target_stats']['std']:.4f}")
    
    # Find target column
    target_col = train_stats.get('target_column') or test_stats.get('target_column')
    
    if target_col is None:
        print("\n" + "!"*60)
        print("ERROR: No CO2 uptake target column found in CoRE-MOF data!")
        print("!"*60)
        print("\nPossible solutions:")
        print("1. Check if CoRE-MOF data has CO2 uptake columns with different names")
        print("2. Add the correct column name to CORE_MOF_TARGET_COLUMNS in this script")
        print("3. Manually specify the target column using --target_col argument")
        print(f"\nAvailable columns in train: {list(finetune_df.columns)}")
        print(f"Available columns in test: {list(test_df.columns)}")
        return None
    
    if debug:
        print("\nDebug mode - skipping training.")
        return None
    
    # ── Create Data Loaders ────────────────────────────────────────────────
    # Use a small validation split from the fine-tune set (20%)
    val_size = max(1, int(0.2 * len(finetune_df)))
    val_df = finetune_df.iloc[:val_size]
    train_df = finetune_df.iloc[val_size:]
    
    tmp_train = DATA_DIR / "_tmp_ft_train.csv"
    tmp_val = DATA_DIR / "_tmp_ft_val.csv"
    train_df.to_csv(tmp_train, index=False)
    val_df.to_csv(tmp_val, index=False)
    
    kw = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    ft_train_loader = DataLoader(
        MOFDataset(tmp_train, target_col=target_col, cutoff=CUTOFF),
        batch_size=BATCH, shuffle=True, **kw
    )
    ft_val_loader = DataLoader(
        MOFDataset(tmp_val, target_col=target_col, cutoff=CUTOFF),
        batch_size=BATCH, shuffle=False, **kw
    )
    test_loader = DataLoader(
        MOFDataset(core_test_csv, target_col=target_col, cutoff=CUTOFF),
        batch_size=64, shuffle=False, **kw
    )
    
    print(f"\n{'='*60}")
    print("Transfer Learning Pipeline")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Target column: {target_col}")
    
    # ── Step 1: Zero-shot evaluation ────────────────────────────────────────
    print("\n[1/4] Zero-shot evaluation on CoRE-MOF test set ...")
    pretrained = load_pretrained().to(device)
    zero_metrics, zero_preds, _ = evaluate_model(pretrained, test_loader, device)
    print(f"  Zero-shot: R²={zero_metrics['R2']:.4f} MAE={zero_metrics['MAE']:.4f}")
    
    # ── Step 2: Fine-tune with frozen GNN ───────────────────────────────────
    print(f"\n[2/4] Frozen-GNN fine-tune ({frozen_epochs} epochs) ...")
    frozen_model = load_pretrained().to(device)
    t0 = time.time()
    frozen_info = finetune(
        frozen_model, ft_train_loader, ft_val_loader,
        device=device, epochs=frozen_epochs,
        lr=1e-3, frozen_gnn=True,
        ckpt_path=CKPT_DIR / "transfer_frozen_gnn.pt",
        run_label="frozen",
    )
    frozen_test_metrics, _, _ = evaluate_model(frozen_model, test_loader, device)
    frozen_time = round(time.time() - t0)
    print(f"  Frozen-GNN test: R²={frozen_test_metrics['R2']:.4f} ({frozen_time}s)")
    
    # ── Step 3: Full fine-tune (low LR) ─────────────────────────────────────
    print(f"\n[3/4] Full fine-tune ({full_finetune_epochs} epochs, lr={full_finetune_lr}) ...")
    full_model = load_pretrained().to(device)
    t0 = time.time()
    full_info = finetune(
        full_model, ft_train_loader, ft_val_loader,
        device=device, epochs=full_finetune_epochs,
        lr=full_finetune_lr, frozen_gnn=False,
        ckpt_path=CKPT_DIR / "transfer_full_finetune.pt",
        run_label="full_ft",
    )
    full_test_metrics, _, _ = evaluate_model(full_model, test_loader, device)
    full_time = round(time.time() - t0)
    print(f"  Full fine-tune test: R²={full_test_metrics['R2']:.4f} ({full_time}s)")
    
    # ── Step 4: Summary ──────────────────────────────────────────────────────
    print("\n[4/4] Transfer Learning Summary")
    print("─" * 55)
    print(f"  Zero-shot (no fine-tune):    R²={zero_metrics['R2']:.4f} MAE={zero_metrics['MAE']:.4f}")
    print(f"  Frozen-GNN fine-tune:        R²={frozen_test_metrics['R2']:.4f} MAE={frozen_test_metrics['MAE']:.4f}")
    print(f"  Full fine-tune (low LR):     R²={full_test_metrics['R2']:.4f} MAE={full_test_metrics['MAE']:.4f}")
    print("─" * 55)
    
    results = {
        "finetune_samples": len(train_df),
        "test_samples": len(test_df),
        "target_column": target_col,
        "zero_shot": zero_metrics,
        "frozen_gnn": {
            "val_best_r2": frozen_info["best_val_r2"],
            "test": frozen_test_metrics,
            "elapsed_s": frozen_time,
            "epochs": frozen_epochs,
        },
        "full_finetune": {
            "val_best_r2": full_info["best_val_r2"],
            "test": full_test_metrics,
            "elapsed_s": full_time,
            "epochs": full_finetune_epochs,
            "lr": full_finetune_lr,
        },
    }
    
    out_path = RESULT_DIR / "transfer_learning_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    
    # Clean up tmp files
    for p in (tmp_train, tmp_val):
        p.unlink(missing_ok=True)
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning for MOF CO2 adsorption")
    parser.add_argument("--frozen_epochs", type=int, default=30)
    parser.add_argument("--full_finetune_epochs", type=int, default=50)
    parser.add_argument("--full_finetune_lr", type=float, default=5e-5)
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: analyze data without training")
    parser.add_argument("--target_col", type=str, default=None,
                        help="Manually specify target column name")
    
    args = parser.parse_args()
    
    # If target column specified, add it to the list
    if args.target_col:
        CORE_MOF_TARGET_COLUMNS.insert(0, args.target_col)
    
    run_transfer(
        frozen_epochs=args.frozen_epochs,
        full_finetune_epochs=args.full_finetune_epochs,
        full_finetune_lr=args.full_finetune_lr,
        debug=args.debug,
    )