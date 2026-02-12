#!/usr/bin/env python3
"""
Tasks 3.7–3.9: Prepare transfer learning and exclusion sets.
  3.7: CoRE-MOF 10% fine-tune / 90% test split
  3.8: GA_MOFs 100% zero-shot test set
  3.9: Exclude COF datasets (manifest file)
"""

import pandas as pd
import numpy as np
import os
import re
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
CORE_MOF_CSV = "Database/CoRE-MOF/src/CoRE_MOF/data/2019-ASR.csv"
GA_MOFS_DIR = "Database/GA_MOFs"
CORE_COFS_DIR = "Database/CoRE-COFs_1242-v7.0"
CURATED_COFS_DIR = "Database/CURATED-COFs"

OUTPUT_CORE_FT = "data/core_mof_ft.csv"
OUTPUT_CORE_TEST = "data/core_mof_test.csv"
OUTPUT_GA_TEST = "data/ga_mofs_test.csv"
OUTPUT_COFS_EXCLUDED = "data/cofs_excluded.txt"
RESULTS_PATH = "results/transfer_sets_summary.txt"

RANDOM_STATE = 42

# ── Element data for chemical embeddings (same as chemical_embeddings.py) ──────
# Import subset for quick CIF parsing
ELEMENT_REGEX = re.compile(r'([A-Z][a-z]?)')

COVALENT_RADII = {
    "H": 31, "He": 28, "Li": 128, "Be": 96, "B": 84, "C": 76, "N": 71, "O": 66,
    "F": 57, "Ne": 58, "Na": 166, "Mg": 141, "Al": 121, "Si": 111, "P": 107,
    "S": 105, "Cl": 102, "Ar": 106, "K": 203, "Ca": 176, "Ti": 160, "V": 153,
    "Cr": 139, "Mn": 139, "Fe": 132, "Co": 126, "Ni": 124, "Cu": 132, "Zn": 122,
    "Ga": 122, "Ge": 120, "As": 119, "Se": 120, "Br": 120, "Sr": 195, "Zr": 175,
    "Mo": 154, "Ag": 145, "Cd": 144, "In": 142, "Sn": 139, "I": 139, "Ba": 215,
    "La": 207, "Ce": 204, "Pb": 146, "Bi": 148,
}


def parse_cif_basic(filepath):
    """Extract basic structural info from a CIF file (lightweight parser)."""
    info = {
        "filename": os.path.basename(filepath).replace(".cif", ""),
        "num_atoms": 0,
        "elements": [],
        "cell_volume": np.nan,
    }

    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()

        a = b = c = alpha = beta = gamma = None
        atom_sites = []
        in_atom_loop = False
        type_col_idx = -1

        for line in lines:
            line_s = line.strip()

            # Cell parameters
            if line_s.startswith("_cell_length_a"):
                a = float(re.sub(r'\(.*\)', '', line_s.split()[-1]))
            elif line_s.startswith("_cell_length_b"):
                b = float(re.sub(r'\(.*\)', '', line_s.split()[-1]))
            elif line_s.startswith("_cell_length_c"):
                c = float(re.sub(r'\(.*\)', '', line_s.split()[-1]))
            elif line_s.startswith("_cell_angle_alpha"):
                alpha = float(re.sub(r'\(.*\)', '', line_s.split()[-1]))
            elif line_s.startswith("_cell_angle_beta"):
                beta = float(re.sub(r'\(.*\)', '', line_s.split()[-1]))
            elif line_s.startswith("_cell_angle_gamma"):
                gamma = float(re.sub(r'\(.*\)', '', line_s.split()[-1]))

            # Atom sites
            if "loop_" in line_s:
                in_atom_loop = False
                type_col_idx = -1
            if "_atom_site_type_symbol" in line_s and in_atom_loop:
                type_col_idx = 0  # Will count position
            if "_atom_site_" in line_s:
                in_atom_loop = True

            if in_atom_loop and not line_s.startswith("_") and not line_s.startswith("loop_") and len(line_s) > 0:
                parts = line_s.split()
                if len(parts) >= 2:
                    # First column is usually label or type symbol
                    symbol = ELEMENT_REGEX.match(parts[0])
                    if symbol:
                        atom_sites.append(symbol.group(0))

        # Compute volume
        if all(v is not None for v in [a, b, c, alpha, beta, gamma]):
            alpha_r = np.radians(alpha)
            beta_r = np.radians(beta)
            gamma_r = np.radians(gamma)
            vol = a * b * c * np.sqrt(
                1 - np.cos(alpha_r)**2 - np.cos(beta_r)**2 - np.cos(gamma_r)**2
                + 2 * np.cos(alpha_r) * np.cos(beta_r) * np.cos(gamma_r)
            )
            info["cell_volume"] = round(vol, 2)

        info["num_atoms"] = len(atom_sites)
        info["elements"] = sorted(set(atom_sites))
        info["num_elements"] = len(info["elements"])

    except Exception as e:
        pass

    return info


def main():
    print("=" * 80)
    print("Tasks 3.7–3.9: Transfer Sets & COF Exclusion")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════════════
    # Task 3.7: CoRE-MOF Transfer Sets
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("TASK 3.7: CoRE-MOF Transfer Sets (10/90)")
    print(f"{'─' * 60}")

    core = pd.read_csv(CORE_MOF_CSV)
    # Clean up: drop unnamed columns
    unnamed = [c for c in core.columns if c.startswith("Unnamed")]
    core = core.drop(columns=unnamed)
    print(f"\n  CoRE-MOF 2019-ASR: {len(core)} structures, {len(core.columns)} columns")
    print(f"  Columns: {list(core.columns)}")

    # Add dataset label
    core["dataset"] = "CoRE-MOF-2019-ASR"

    # Split 10% fine-tune / 90% test
    core_ft, core_test = train_test_split(
        core, test_size=0.9, random_state=RANDOM_STATE
    )
    print(f"\n  Fine-tune: {len(core_ft)} ({100*len(core_ft)/len(core):.1f}%)")
    print(f"  Test:      {len(core_test)} ({100*len(core_test)/len(core):.1f}%)")

    core_ft.to_csv(OUTPUT_CORE_FT, index=False)
    core_test.to_csv(OUTPUT_CORE_TEST, index=False)
    print(f"  Saved: {OUTPUT_CORE_FT}")
    print(f"  Saved: {OUTPUT_CORE_TEST}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Task 3.8: GA_MOFs Zero-Shot Test Set
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("TASK 3.8: GA_MOFs Zero-Shot Test Set")
    print(f"{'─' * 60}")

    cif_files = sorted([
        os.path.join(GA_MOFS_DIR, f)
        for f in os.listdir(GA_MOFS_DIR)
        if f.endswith(".cif")
    ])
    print(f"\n  Found {len(cif_files)} CIF files")

    # Parse all CIF files for structural features
    records = []
    for cif_path in tqdm(cif_files, desc="Parsing GA_MOFs CIFs"):
        info = parse_cif_basic(cif_path)
        records.append({
            "filename": info["filename"],
            "num_atoms": info["num_atoms"],
            "num_elements": info.get("num_elements", 0),
            "elements": str(info["elements"]),
            "cell_volume": info["cell_volume"],
            "dataset": "GA_MOFs",
        })

    ga_df = pd.DataFrame(records)
    # Filter out failed parses
    ga_valid = ga_df[ga_df["num_atoms"] > 0].copy()
    print(f"  Successfully parsed: {len(ga_valid)} / {len(ga_df)} ({100*len(ga_valid)/len(ga_df):.1f}%)")

    ga_valid.to_csv(OUTPUT_GA_TEST, index=False)
    print(f"  Saved: {OUTPUT_GA_TEST} ({len(ga_valid)} structures)")

    # ═══════════════════════════════════════════════════════════════════════════
    # Task 3.9: Exclude COF Datasets
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print("TASK 3.9: COF Exclusion Manifest")
    print(f"{'─' * 60}")

    excluded = []

    # CoRE-COFs
    if os.path.exists(CORE_COFS_DIR):
        core_cofs = [f for f in os.listdir(CORE_COFS_DIR) if f.endswith(".cif")]
        excluded.extend([(f"CoRE-COFs_1242-v7.0", f) for f in sorted(core_cofs)])
        print(f"\n  CoRE-COFs: {len(core_cofs)} structures")

    # CURATED-COFs
    if os.path.exists(CURATED_COFS_DIR):
        curated_cofs = [f for f in os.listdir(CURATED_COFS_DIR) if f.endswith(".cif")]
        excluded.extend([(f"CURATED-COFs", f) for f in sorted(curated_cofs)])
        print(f"  CURATED-COFs: {len(curated_cofs)} structures")

    # Save manifest
    with open(OUTPUT_COFS_EXCLUDED, "w") as f:
        f.write("# COF Exclusion Manifest\n")
        f.write("# These datasets are excluded from training/testing.\n")
        f.write("# Held for domain-gap demonstration only.\n")
        f.write(f"# Total: {len(excluded)} structures\n")
        f.write("#\n")
        f.write("# Format: dataset_name\tfilename\n")
        f.write("#" + "=" * 60 + "\n")
        for dataset, fname in excluded:
            f.write(f"{dataset}\t{fname}\n")

    print(f"\n  Total excluded: {len(excluded)} structures")
    print(f"  Saved: {OUTPUT_COFS_EXCLUDED}")

    # ── Summary report ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write("Transfer Sets & COF Exclusion Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("=== CoRE-MOF 2019-ASR ===\n")
        f.write(f"  Total: {len(core)} structures\n")
        f.write(f"  Fine-tune (10%): {len(core_ft)}\n")
        f.write(f"  Test (90%):      {len(core_test)}\n")
        f.write(f"  Columns: {list(core.columns)}\n\n")

        f.write("=== GA_MOFs (Zero-Shot) ===\n")
        f.write(f"  Total CIFs: {len(cif_files)}\n")
        f.write(f"  Successfully parsed: {len(ga_valid)}\n")
        f.write(f"  Mean atoms/cell: {ga_valid['num_atoms'].mean():.1f}\n")
        f.write(f"  Mean volume: {ga_valid['cell_volume'].mean():.1f} Å³\n\n")

        f.write("=== COF Exclusion ===\n")
        f.write(f"  CoRE-COFs:    {len([x for x in excluded if x[0] == 'CoRE-COFs_1242-v7.0'])}\n")
        f.write(f"  CURATED-COFs: {len([x for x in excluded if x[0] == 'CURATED-COFs'])}\n")
        f.write(f"  Total:        {len(excluded)}\n")

    print(f"\n  Saved: {RESULTS_PATH}")

    print(f"\n{'=' * 80}")
    print("TRANSFER SETS & COF EXCLUSION COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
