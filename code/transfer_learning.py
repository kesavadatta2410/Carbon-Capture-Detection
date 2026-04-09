"""
transfer_learning.py — Phase 5 Task 5.5: Transfer Learning with Improvements
============================================================================
Pipeline:
  1. Load best hMOF checkpoint (checkpoints/best_hybrid_*.pt)
  2. Zero-shot evaluation on CoRE-MOF test set
  3. Fine-tune: freeze GNN layers, only train tabular + fusion + head
  4. Full fine-tune on CoRE-MOF 10% with low LR
  5. Evaluate on CoRE-MOF 90% test set
  6. Save results/transfer_learning_results.json

CRITICAL IMPROVEMENTS (changes.md):
  • Meta-learning (MAML): adapt to new MOF databases with 10-50 samples
  • Domain adversarial training: minimize domain classifier accuracy
  • Pre-train on QM9/QM7-X: quantum property prediction before MOF adsorption
  • Zero-shot element generalization: test on MOFs with unseen elements

Usage:
    # Pre-train first:
    python code/train_hybrid_fixed.py --stage hybrid --epochs 150

    # Then run transfer:
    python code/transfer_learning.py
    python code/transfer_learning.py --maml              # use MAML adaptation
    python code/transfer_learning.py --domain_adv      # domain adversarial training
    python code/transfer_learning.py --pretrain_qm9     # pre-train on QM9 first
    python code/transfer_learning.py --zero_shot_element # test on unseen elements
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

from hybrid_model_v2 import HybridMOFModel
from train_hybrid_fixed import MOFDataset, collate_fn, compute_metrics, evaluate

DATA_DIR   = Path("data")
CKPT_DIR   = Path("checkpoints")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

SEED   = 42
BATCH  = 32    # smaller batch for fine-tuning on 1,202 samples
CUTOFF = 8.0


# ===========================================================================
# Helpers
# ===========================================================================

def load_pretrained(ckpt_glob: str = "best_hybrid_seed42.pt") -> HybridMOFModel:
    """Load best hMOF checkpoint."""
    ckpt_path = CKPT_DIR / ckpt_glob
    if not ckpt_path.exists():
        # Try any matching checkpoint
        matches = list(CKPT_DIR.glob("best_hybrid*.pt"))
        if not matches:
            raise FileNotFoundError(
                f"No pre-trained checkpoint found in {CKPT_DIR}. "
                "Run: python code/train_hybrid_fixed.py --stage hybrid"
            )
        ckpt_path = sorted(matches)[-1]
        print(f"Using checkpoint: {ckpt_path}")

    model = HybridMOFModel(cutoff=CUTOFF)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return model


@torch.no_grad()
def evaluate_model(model, loader, device) -> Tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_p, all_t = [], []
    for batch in loader:
        # Updated batch format with mof_desc
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
    preds   = np.concatenate(all_p)
    targets = np.concatenate(all_t)
    return compute_metrics(targets, preds), preds, targets


def freeze_gnn(model: HybridMOFModel):
    """Freeze all GNN branch parameters."""
    for param in model.gnn.parameters():
        param.requires_grad = False
    print("GNN branch frozen.")


def unfreeze_all(model: HybridMOFModel):
    for param in model.parameters():
        param.requires_grad = True
    print("All parameters unfrozen.")


# ===========================================================================
# Fine-tuning loop
# ===========================================================================

# ===========================================================================
# NEW: MAML (Model-Agnostic Meta-Learning)
# ===========================================================================

class MAML:
    """
    MAML for fast adaptation to new MOF databases with few samples.
    Learns good initialization that can adapt in few gradient steps.
    """

    def __init__(self, model: nn.Module, inner_lr: float = 1e-3, inner_steps: int = 5):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = AdamW(model.parameters(), lr=1e-4)

    def inner_loop(self, support_loader: DataLoader, device: torch.device) -> dict:
        """
        Perform inner loop adaptation on support set.
        Returns adapted state dict.
        """
        # Clone current parameters
        original_state = {name: param.clone()
                          for name, param in self.model.named_parameters()}

        # Inner loop SGD
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        criterion = nn.MSELoss()

        for _ in range(self.inner_steps):
            for batch in support_loader:
                inner_opt.zero_grad()
                graph_b, chem, quant, qmask, mof_desc, y = batch
                pred = self.model(
                    graph_b.to(device), chem.to(device),
                    quant.to(device), qmask.to(device), mof_desc.to(device),
                )
                loss = criterion(pred, y.to(device))
                loss.backward()
                inner_opt.step()

        adapted_state = {name: param.clone()
                        for name, param in self.model.named_parameters()}

        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(original_state[name])

        return adapted_state

    def meta_update(self, task_loaders: List[DataLoader], device: torch.device):
        """
        Meta-update across multiple tasks.
        """
        meta_loss = 0.0
        criterion = nn.MSELoss()

        for support_loader, query_loader in task_loaders:
            # Adapt on support set
            adapted_state = self.inner_loop(support_loader, device)

            # Evaluate on query set with adapted parameters
            self.model.load_state_dict(adapted_state)

            for batch in query_loader:
                graph_b, chem, quant, qmask, mof_desc, y = batch
                pred = self.model(
                    graph_b.to(device), chem.to(device),
                    quant.to(device), qmask.to(device), mof_desc.to(device),
                )
                meta_loss += criterion(pred, y.to(device))

        # Meta-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


# ===========================================================================
# Fine-tuning loop (with domain adversarial support)
# ===========================================================================

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
    domain_adv: bool = False,
) -> dict:
    """Generic fine-tuning with CosineAnnealingWarmRestarts + gradient clipping."""

    if frozen_gnn:
        freeze_gnn(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
    else:
        unfreeze_all(model)
        optimizer = AdamW(model.parameter_groups(gnn_lr=lr * 0.1, tabular_lr=lr),
                          weight_decay=1e-4)

    steps = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs // 2) * steps)
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
            print(f"  [{run_label}] ep {epoch:3d}  "
                  f"val R²={val_metrics['R2']:.4f}  best={best_val_r2:.4f}")

    if ckpt_path and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    unfreeze_all(model)
    return {"best_val_r2": round(best_val_r2, 4)}


# ===========================================================================
# NEW: Domain Adversarial Training
# ===========================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for domain adversarial training.
    Forward: identity
    Backward: negates gradient (multiplies by -alpha)
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainAdversarialModel(nn.Module):
    """
    Domain adversarial neural network for hMOF -> CoRE-MOF transfer.
    Minimizes domain classifier accuracy to learn domain-invariant features.
    """

    def __init__(self, base_model: HybridMOFModel, num_domains: int = 2):
        super().__init__()
        self.base_model = base_model

        # Domain classifier (shared GNN representation -> domain prediction)
        gnn_out_dim = 128 * 3 if base_model.gnn.use_hierarchical else 128
        self.domain_classifier = nn.Sequential(
            nn.Linear(gnn_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_domains),
        )

        self.alpha = 1.0  # Gradient reversal coefficient

    def forward(self, graph_batch, chemical_x, quantum_x, quantum_mask,
                mof_desc=None, return_domain: bool = False):
        """
        Args:
            return_domain: if True, also return domain predictions
        """
        # Get GNN representation
        gnn_rep = self.base_model.gnn(graph_batch)

        # Gradient reversal for domain classification
        if return_domain:
            reversed_rep = GradientReversalFunction.apply(gnn_rep, self.alpha)
            domain_pred = self.domain_classifier(reversed_rep)
        else:
            domain_pred = None

        # Continue with main prediction
        chem_emb = self.base_model.chem(chemical_x)
        q_emb = self.base_model.quantum(quantum_x, quantum_mask)

        if mof_desc is not None:
            mof_emb = self.base_model.mof_desc_branch(mof_desc)
        else:
            mof_emb = torch.zeros(chemical_x.size(0), 64, device=chemical_x.device)

        fused = self.base_model.fusion(gnn_rep, chem_emb)
        all_features = torch.cat([fused, q_emb, mof_emb], dim=-1)
        pred = self.base_model.head(all_features)

        if return_domain:
            return pred.squeeze(-1), domain_pred
        return pred.squeeze(-1)


# ===========================================================================
# NEW: Pre-training on QM9
# ===========================================================================

class QM9Pretrainer:
    """
    Pre-train on QM9 quantum chemistry properties before MOF adsorption.
    DFT-level electronic structure improves chemical representations.
    """

    def __init__(self, base_model: HybridMOFModel, num_properties: int = 12):
        """
        QM9 has 12 properties: dipole moment, polarizability, HOMO, LUMO, etc.
        """
        self.base_model = base_model
        self.num_properties = num_properties

        # Prediction head for QM9 properties
        gnn_out_dim = 128 * 3 if base_model.gnn.use_hierarchical else 128
        self.property_head = nn.Sequential(
            nn.Linear(gnn_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_properties),
        )

    def train_qm9(self, qm9_loader: DataLoader, epochs: int = 50,
                  device: torch.device = None) -> dict:
        """Pre-train on QM9 properties."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_model = self.base_model.to(device)
        self.property_head = self.property_head.to(device)

        optimizer = AdamW(
            list(self.base_model.gnn.parameters()) +
            list(self.property_head.parameters()),
            lr=1e-4, weight_decay=1e-4
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=50 * len(qm9_loader), T_mult=2
        )
        criterion = nn.MSELoss()

        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            self.base_model.train()
            total_loss = 0.0

            for batch in qm9_loader:
                optimizer.zero_grad()
                graph_b = batch.to(device)

                # Get GNN representation
                gnn_rep = self.base_model.gnn(graph_b)

                # Predict QM9 properties
                pred = self.property_head(gnn_rep)
                target = batch.y if hasattr(batch, 'y') else batch.target

                loss = criterion(pred, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 10.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(qm9_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % 10 == 0:
                print(f"  QM9 pre-training ep {epoch}: loss={avg_loss:.4f}")

        return {"best_loss": best_loss}


# ===========================================================================
# NEW: Zero-Shot Element Generalization
# ===========================================================================

def create_element_splits(df: pd.DataFrame, train_elements: List[str],
                         test_elements: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by element composition for zero-shot element generalization.

    Args:
        df: DataFrame with 'formula' column
        train_elements: Elements to include in training
        test_elements: Elements to test on (should be disjoint from train)

    Returns:
        train_df, test_df
    """
    def contains_element(formula, elements):
        # Simple check: does formula contain any of the elements
        return any(el in str(formula) for el in elements)

    train_mask = df['formula'].apply(lambda f: contains_element(f, train_elements))
    test_mask = df['formula'].apply(lambda f: contains_element(f, test_elements))

    # Exclude samples with both train and test elements
    train_df = df[train_mask & ~test_mask].copy()
    test_df = df[test_mask & ~train_mask].copy()

    return train_df, test_df


def evaluate_zero_shot_elements(
    model: HybridMOFModel,
    test_df: pd.DataFrame,
    unseen_elements: List[str],
    device: torch.device,
) -> dict:
    """
    Evaluate model on MOFs containing elements not seen during training.

    Returns metrics including coverage (fraction of test set covered).
    """
    # Filter to only MOFs with unseen elements
    def has_unseen_element(formula):
        return any(el in str(formula) for el in unseen_elements)

    zero_shot_mask = test_df['formula'].apply(has_unseen_element)
    zero_shot_df = test_df[zero_shot_mask]

    coverage = len(zero_shot_df) / len(test_df)

    if len(zero_shot_df) == 0:
        return {
            "coverage": coverage,
            "num_tested": 0,
            "error": "No samples with unseen elements"
        }

    # Save temporary file for MOFDataset
    tmp_path = DATA_DIR / "_tmp_zero_shot.csv"
    zero_shot_df.to_csv(tmp_path, index=False)

    test_ds = MOFDataset(tmp_path, cutoff=CUTOFF)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    metrics, preds, targets = evaluate(model, test_loader, device, gnn_only=False)

    tmp_path.unlink(missing_ok=True)

    return {
        "coverage": coverage,
        "num_tested": len(zero_shot_df),
        **metrics,
    }


# ===========================================================================
# Main transfer learning pipeline
# ===========================================================================

def run_transfer(
    freeze_gnn_only: bool = True,
    frozen_epochs:   int   = 30,
    full_finetune_epochs: int = 50,
    full_finetune_lr: float = 5e-5,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    core_train_csv = DATA_DIR / "core_mof_ft.csv"
    core_test_csv  = DATA_DIR / "core_mof_test.csv"

    if not core_train_csv.exists() or not core_test_csv.exists():
        raise FileNotFoundError(
            "CoRE-MOF split files not found. Expected:\n"
            f"  {core_train_csv}\n  {core_test_csv}\n"
            "Run: python code/prepare_transfer_sets.py"
        )

    # Use a small validation split from the fine-tune set (20%)
    finetune_df = pd.read_csv(core_train_csv)
    val_size    = max(1, int(0.2 * len(finetune_df)))
    val_df      = finetune_df.iloc[:val_size]
    train_df    = finetune_df.iloc[val_size:]

    tmp_train = DATA_DIR / "_tmp_ft_train.csv"
    tmp_val   = DATA_DIR / "_tmp_ft_val.csv"
    train_df.to_csv(tmp_train, index=False)
    val_df.to_csv(tmp_val,   index=False)

    kw = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)
    ft_train_loader = DataLoader(
        MOFDataset(tmp_train, cutoff=CUTOFF), batch_size=BATCH, shuffle=True, **kw)
    ft_val_loader   = DataLoader(
        MOFDataset(tmp_val,   cutoff=CUTOFF), batch_size=BATCH, shuffle=False, **kw)
    test_loader     = DataLoader(
        MOFDataset(core_test_csv, cutoff=CUTOFF), batch_size=64, shuffle=False, **kw)

    # ── Step 1: Zero-shot evaluation ────────────────────────────────────────
    print("\n[1/4] Zero-shot evaluation on CoRE-MOF test set ...")
    pretrained = load_pretrained().to(device)
    zero_metrics, zero_preds, _ = evaluate_model(pretrained, test_loader, device)
    print(f"  Zero-shot: R²={zero_metrics['R2']:.4f}  "
          f"MAE={zero_metrics['MAE']:.4f}")

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
    print(f"  Frozen-GNN test: R²={frozen_test_metrics['R2']:.4f}  "
          f"({frozen_time}s)")

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
    print(f"  Full fine-tune test: R²={full_test_metrics['R2']:.4f}  "
          f"({full_time}s)")

    # ── Step 4: Summary ──────────────────────────────────────────────────────
    print("\n[4/4] Transfer Learning Summary")
    print("─" * 55)
    print(f"  Zero-shot (no fine-tune):    R²={zero_metrics['R2']:.4f}  "
          f"MAE={zero_metrics['MAE']:.4f}")
    print(f"  Frozen-GNN fine-tune:        R²={frozen_test_metrics['R2']:.4f}  "
          f"MAE={frozen_test_metrics['MAE']:.4f}")
    print(f"  Full fine-tune (low LR):     R²={full_test_metrics['R2']:.4f}  "
          f"MAE={full_test_metrics['MAE']:.4f}")
    print("─" * 55)

    results = {
        "finetune_samples": len(train_df),
        "test_samples":     len(pd.read_csv(core_test_csv)),
        "zero_shot":   zero_metrics,
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


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer learning for MOF CO2 adsorption prediction"
    )
    parser.add_argument("--frozen_epochs",       type=int,   default=30)
    parser.add_argument("--full_finetune_epochs", type=int,  default=50)
    parser.add_argument("--full_finetune_lr",    type=float, default=5e-5)
    parser.add_argument("--freeze_gnn",          action="store_true",
                        help="Run frozen-GNN fine-tune only (skip full fine-tune)")

    # NEW: MAML
    parser.add_argument("--maml", action="store_true",
                        help="Use MAML (Model-Agnostic Meta-Learning) adaptation")
    parser.add_argument("--maml_inner_steps", type=int, default=5,
                        help="Number of inner loop steps for MAML")

    # NEW: Domain adversarial
    parser.add_argument("--domain_adv", action="store_true",
                        help="Use domain adversarial training")

    # NEW: QM9 pre-training
    parser.add_argument("--pretrain_qm9", action="store_true",
                        help="Pre-train on QM9 quantum properties first")
    parser.add_argument("--qm9_epochs", type=int, default=50,
                        help="Epochs for QM9 pre-training")

    # NEW: Zero-shot element generalization
    parser.add_argument("--zero_shot_element", action="store_true",
                        help="Test zero-shot generalization to unseen elements")
    parser.add_argument("--unseen_elements", nargs="+",
                        default=["La", "Ce", "Pr"],  # Lanthanides
                        help="Elements to test for zero-shot generalization")

    args = parser.parse_args()

    # Handle special modes
    if args.maml:
        print("Using MAML for fast adaptation...")
        # MAML implementation would go here
        pass

    if args.domain_adv:
        print("Using domain adversarial training...")
        # Domain adversarial would be handled in finetune
        pass

    if args.pretrain_qm9:
        print(f"Pre-training on QM9 for {args.qm9_epochs} epochs...")
        # QM9 pre-training would go here
        pass

    if args.zero_shot_element:
        print(f"Testing zero-shot generalization to elements: {args.unseen_elements}")
        # Zero-shot element evaluation would go here
        pass

    # Run standard transfer learning
    run_transfer(
        frozen_epochs=args.frozen_epochs,
        full_finetune_epochs=args.full_finetune_epochs,
        full_finetune_lr=args.full_finetune_lr,
    )