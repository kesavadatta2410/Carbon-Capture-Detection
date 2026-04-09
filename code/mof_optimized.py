#!/usr/bin/env python3
"""
build_graphs_fixed.py — CORRECTED Graph Construction with 8.0 Å Radius Cutoff
===========================================================================
This fixes the critical bug where edges were only created for covalent bonds.
Now captures pore structure with proper radius-based neighbor graph.

Key change: Replaced bond-based threshold with fixed 8.0 Å radius cutoff.

Usage:
    python build_graphs_fixed.py
"""

import json
import numpy as np
import os
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
JSON_DIR = "Database/hMOF-mofdb"
OUTPUT_DIR = "data/graphs"
STATS_PATH = "data/graph_stats_fixed.csv"

# ── Configuration ──────────────────────────────────────────────────────────────
CUTOFF = 8.0  # Å — THIS IS THE CRITICAL FIX! Was using bond distances (~1.5 Å)
MAX_ATOMS = 700  # Skip structures with more atoms (memory safety)

# ── Elemental properties for node features ─────────────────────────────────────
ELEM_PROPS = {
    "H":  {"Z": 1,  "en": 2.20, "rc": 31},
    "He": {"Z": 2,  "en": 0.00, "rc": 28},
    "Li": {"Z": 3,  "en": 0.98, "rc": 128},
    "Be": {"Z": 4,  "en": 1.57, "rc": 96},
    "B":  {"Z": 5,  "en": 2.04, "rc": 84},
    "C":  {"Z": 6,  "en": 2.55, "rc": 76},
    "N":  {"Z": 7,  "en": 3.04, "rc": 71},
    "O":  {"Z": 8,  "en": 3.44, "rc": 66},
    "F":  {"Z": 9,  "en": 3.98, "rc": 57},
    "Ne": {"Z": 10, "en": 0.00, "rc": 58},
    "Na": {"Z": 11, "en": 0.93, "rc": 166},
    "Mg": {"Z": 12, "en": 1.31, "rc": 141},
    "Al": {"Z": 13, "en": 1.61, "rc": 121},
    "Si": {"Z": 14, "en": 1.90, "rc": 111},
    "P":  {"Z": 15, "en": 2.19, "rc": 107},
    "S":  {"Z": 16, "en": 2.58, "rc": 105},
    "Cl": {"Z": 17, "en": 3.16, "rc": 102},
    "Ar": {"Z": 18, "en": 0.00, "rc": 106},
    "K":  {"Z": 19, "en": 0.82, "rc": 203},
    "Ca": {"Z": 20, "en": 1.00, "rc": 176},
    "Sc": {"Z": 21, "en": 1.36, "rc": 170},
    "Ti": {"Z": 22, "en": 1.54, "rc": 160},
    "V":  {"Z": 23, "en": 1.63, "rc": 153},
    "Cr": {"Z": 24, "en": 1.66, "rc": 139},
    "Mn": {"Z": 25, "en": 1.55, "rc": 139},
    "Fe": {"Z": 26, "en": 1.83, "rc": 132},
    "Co": {"Z": 27, "en": 1.88, "rc": 126},
    "Ni": {"Z": 28, "en": 1.91, "rc": 124},
    "Cu": {"Z": 29, "en": 1.90, "rc": 132},
    "Zn": {"Z": 30, "en": 1.65, "rc": 122},
    "Ga": {"Z": 31, "en": 1.81, "rc": 122},
    "Ge": {"Z": 32, "en": 2.01, "rc": 120},
    "As": {"Z": 33, "en": 2.18, "rc": 119},
    "Se": {"Z": 34, "en": 2.55, "rc": 120},
    "Br": {"Z": 35, "en": 2.96, "rc": 120},
    "Kr": {"Z": 36, "en": 3.00, "rc": 120},
    "Rb": {"Z": 37, "en": 0.82, "rc": 220},
    "Sr": {"Z": 38, "en": 0.95, "rc": 195},
    "Y":  {"Z": 39, "en": 1.22, "rc": 190},
    "Zr": {"Z": 40, "en": 1.33, "rc": 175},
    "Nb": {"Z": 41, "en": 1.6,  "rc": 164},
    "Mo": {"Z": 42, "en": 2.16, "rc": 154},
    "Tc": {"Z": 43, "en": 1.9,  "rc": 147},
    "Ru": {"Z": 44, "en": 2.2,  "rc": 146},
    "Rh": {"Z": 45, "en": 2.28, "rc": 142},
    "Pd": {"Z": 46, "en": 2.20, "rc": 139},
    "Ag": {"Z": 47, "en": 1.93, "rc": 145},
    "Cd": {"Z": 48, "en": 1.69, "rc": 144},
    "In": {"Z": 49, "en": 1.78, "rc": 142},
    "Sn": {"Z": 50, "en": 1.96, "rc": 139},
    "Sb": {"Z": 51, "en": 2.05, "rc": 139},
    "Te": {"Z": 52, "en": 2.1,  "rc": 138},
    "I":  {"Z": 53, "en": 2.66, "rc": 139},
    "Xe": {"Z": 54, "en": 2.6,  "rc": 140},
    "Cs": {"Z": 55, "en": 0.79, "rc": 244},
    "Ba": {"Z": 56, "en": 0.89, "rc": 215},
    "La": {"Z": 57, "en": 1.10, "rc": 207},
    "Ce": {"Z": 58, "en": 1.12, "rc": 204},
    "Pr": {"Z": 59, "en": 1.13, "rc": 203},
    "Nd": {"Z": 60, "en": 1.14, "rc": 201},
    "Pm": {"Z": 61, "en": 1.13, "rc": 199},
    "Sm": {"Z": 62, "en": 1.17, "rc": 198},
    "Eu": {"Z": 63, "en": 1.2,  "rc": 198},
    "Gd": {"Z": 64, "en": 1.20, "rc": 196},
    "Tb": {"Z": 65, "en": 1.2,  "rc": 194},
    "Dy": {"Z": 66, "en": 1.22, "rc": 192},
    "Ho": {"Z": 67, "en": 1.23, "rc": 192},
    "Er": {"Z": 68, "en": 1.24, "rc": 189},
    "Tm": {"Z": 69, "en": 1.25, "rc": 190},
    "Yb": {"Z": 70, "en": 1.1,  "rc": 187},
    "Lu": {"Z": 71, "en": 1.27, "rc": 187},
    "Hf": {"Z": 72, "en": 1.3,  "rc": 175},
    "Ta": {"Z": 73, "en": 1.5,  "rc": 170},
    "W":  {"Z": 74, "en": 2.36, "rc": 162},
    "Re": {"Z": 75, "en": 1.9,  "rc": 151},
    "Os": {"Z": 76, "en": 2.2,  "rc": 144},
    "Ir": {"Z": 77, "en": 2.20, "rc": 141},
    "Pt": {"Z": 78, "en": 2.28, "rc": 136},
    "Au": {"Z": 79, "en": 2.54, "rc": 136},
    "Hg": {"Z": 80, "en": 2.00, "rc": 132},
    "Tl": {"Z": 81, "en": 1.62, "rc": 145},
    "Pb": {"Z": 82, "en": 2.33, "rc": 146},
    "Bi": {"Z": 83, "en": 2.02, "rc": 148},
    "Po": {"Z": 84, "en": 2.0,  "rc": 140},
    "At": {"Z": 85, "en": 2.2,  "rc": 150},
    "Rn": {"Z": 86, "en": 2.2,  "rc": 150},
}


def get_node_features(symbol):
    """Return [atomic_number, electronegativity, covalent_radius] for an element."""
    if symbol in ELEM_PROPS:
        p = ELEM_PROPS[symbol]
        return [p["Z"], p["en"], p["rc"]]
    # Default for unknown elements
    return [0, 0, 0]


def parse_cif_from_json(json_data):
    """
    Extract atom positions from CIF data embedded in the JSON.
    Returns: symbols (list), frac_coords (Nx3 array), cell params (a,b,c,alpha,beta,gamma)
    """
    cif_text = json_data.get("cif", "")
    if not cif_text:
        return None, None, None

    lines = cif_text.split('\n')
    a = b = c = alpha = beta = gamma = None
    symbols = []
    frac_coords = []

    for line in lines:
        line_s = line.strip()
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

    if not all(v is not None for v in [a, b, c, alpha, beta, gamma]):
        return None, None, None

    # Parse atom sites
    in_atom_loop = False
    col_names = []

    for line in lines:
        line_s = line.strip()
        if line_s == "loop_":
            in_atom_loop = False
            col_names = []
        if "_atom_site_" in line_s and not any(c.isdigit() and not line_s.startswith("_atom_site") for c in line_s):
            if "_atom_site_" in line_s and len(line_s.split()) == 1:
                in_atom_loop = True
                col_names.append(line_s)
                continue

        if in_atom_loop and line_s.startswith("_"):
            col_names.append(line_s)
            continue

        if in_atom_loop and line_s and not line_s.startswith("_") and not line_s.startswith("loop_") and not line_s.startswith("#"):
            parts = line_s.split()
            if len(parts) >= len(col_names) and len(col_names) >= 4:
                # Find column indices
                try:
                    label_idx = next(i for i, c in enumerate(col_names) if "label" in c)
                except StopIteration:
                    label_idx = 0

                try:
                    x_idx = next(i for i, c in enumerate(col_names) if "fract_x" in c)
                    y_idx = next(i for i, c in enumerate(col_names) if "fract_y" in c)
                    z_idx = next(i for i, c in enumerate(col_names) if "fract_z" in c)
                except StopIteration:
                    continue

                # Extract symbol from label (e.g., "C1" -> "C", "Zn1" -> "Zn")
                label = parts[label_idx]
                match = re.match(r'([A-Z][a-z]?)', label)
                if match:
                    sym = match.group(1)
                    try:
                        x = float(parts[x_idx])
                        y = float(parts[y_idx])
                        z = float(parts[z_idx])
                        symbols.append(sym)
                        frac_coords.append([x, y, z])
                    except (ValueError, IndexError):
                        pass

    if len(symbols) == 0:
        return None, None, None

    cell_params = (a, b, c, alpha, beta, gamma)
    return symbols, np.array(frac_coords), cell_params


def frac_to_cart(frac_coords, a, b, c, alpha, beta, gamma):
    """Convert fractional to Cartesian coordinates."""
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    cos_a = np.cos(alpha_r)
    cos_b = np.cos(beta_r)
    cos_g = np.cos(gamma_r)
    sin_g = np.sin(gamma_r)

    # Transformation matrix
    vol_factor = np.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g)

    M = np.array([
        [a, b * cos_g, c * cos_b],
        [0, b * sin_g, c * (cos_a - cos_b * cos_g) / sin_g],
        [0, 0, c * vol_factor / sin_g]
    ])

    return frac_coords @ M.T


def build_radius_graph(symbols, cart_coords, cutoff=CUTOFF):
    """
    Build radius-based neighbor graph (NOT bond-based).
    
    THIS IS THE CRITICAL FIX:
    - OLD: Edge if distance < BOND_FACTOR * (rc_i + rc_j)  [~1.5 Å]
    - NEW: Edge if distance < CUTOFF  [8.0 Å]
    
    This captures pore structure, not just covalent bonds.
    
    Returns: node_features, edge_index, edge_dist
    """
    n_atoms = len(symbols)
    if n_atoms > MAX_ATOMS:
        return None, None, None

    # Node features: [Z, EN, RC] for each atom
    node_features = np.array([get_node_features(s) for s in symbols], dtype=np.float32)

    # Compute pairwise distances (Å)
    diff = cart_coords[:, np.newaxis, :] - cart_coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))

    # ═══════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Use fixed radius cutoff instead of bond thresholds
    # ═══════════════════════════════════════════════════════════════════════
    mask = (dists < cutoff) & (dists > 0.1)  # 0.1 Å min to avoid self-loops
    np.fill_diagonal(mask, False)

    # Get edge indices (both directions for undirected)
    src, dst = np.where(mask)

    # Edge features: distances
    edge_dist = dists[src, dst].astype(np.float32)

    return node_features, np.stack([src, dst], axis=0).astype(np.int32), edge_dist


def main():
    print("=" * 80)
    print("build_graphs_fixed.py — Graph Construction with 8.0 Å Radius Cutoff")
    print("=" * 80)
    print(f"\nCRITICAL FIX: Using radius cutoff = {CUTOFF} Å (was using bond distances)")
    print("This captures pore structure, not just covalent bonds!\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all JSON files
    json_files = sorted([
        os.path.join(JSON_DIR, f)
        for f in os.listdir(JSON_DIR)
        if f.endswith(".json")
    ])
    print(f"Found {len(json_files)} JSON files")

    stats = []
    n_success = 0
    n_skipped = 0
    n_too_large = 0
    
    # Track edge distance distribution
    all_edge_dists = []

    for json_path in tqdm(json_files, desc="Building graphs"):
        name = os.path.basename(json_path).replace(".json", "")

        try:
            with open(json_path, "r", errors="ignore") as f:
                data = json.load(f)

            # Parse CIF
            symbols, frac_coords, cell_params = parse_cif_from_json(data)
            if symbols is None or len(symbols) == 0:
                n_skipped += 1
                continue

            if len(symbols) > MAX_ATOMS:
                n_too_large += 1
                stats.append({
                    "name": name,
                    "num_nodes": len(symbols),
                    "num_edges": 0,
                    "edge_dist_min": 0,
                    "edge_dist_max": 0,
                    "status": "too_large",
                })
                continue

            # Convert to Cartesian
            a, b, c, alpha, beta, gamma = cell_params
            cart_coords = frac_to_cart(frac_coords, a, b, c, alpha, beta, gamma)

            # Build graph with RADIUS cutoff (not bond cutoff)
            node_features, edge_index, edge_dist = build_radius_graph(symbols, cart_coords)
            if node_features is None:
                n_skipped += 1
                continue

            n_edges = edge_index.shape[1] if edge_index.shape[1] > 0 else 0
            n_nodes = node_features.shape[0]
            
            # Track edge distances
            if len(edge_dist) > 0:
                all_edge_dists.extend(edge_dist.tolist())

            # Save as compressed npz
            np.savez_compressed(
                os.path.join(OUTPUT_DIR, f"{name}.npz"),
                node_features=node_features,
                edge_index=edge_index,
                edge_dist=edge_dist,
            )

            stats.append({
                "name": name,
                "num_nodes": n_nodes,
                "num_edges": n_edges,
                "edge_dist_min": round(float(edge_dist.min()), 2) if len(edge_dist) > 0 else 0,
                "edge_dist_max": round(float(edge_dist.max()), 2) if len(edge_dist) > 0 else 0,
                "status": "ok",
            })
            n_success += 1

        except Exception as e:
            n_skipped += 1
            stats.append({
                "name": name,
                "num_nodes": 0,
                "num_edges": 0,
                "edge_dist_min": 0,
                "edge_dist_max": 0,
                "status": f"error: {str(e)[:50]}",
            })

    # ── Save stats ─────────────────────────────────────────────────────────────
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(STATS_PATH, index=False)

    ok_df = stats_df[stats_df["status"] == "ok"]
    print(f"\n{'─' * 60}")
    print(f"GRAPH CONSTRUCTION RESULTS:")
    print(f"  Total:       {len(json_files)}")
    print(f"  Successful:  {n_success} ({100*n_success/len(json_files):.1f}%)")
    print(f"  Too large:   {n_too_large}")
    print(f"  Skipped:     {n_skipped}")
    
    if len(ok_df) > 0:
        print(f"\n  Avg nodes:   {ok_df['num_nodes'].mean():.1f}")
        print(f"  Avg edges:   {ok_df['num_edges'].mean():.1f}")
        
    if len(all_edge_dists) > 0:
        all_edge_dists = np.array(all_edge_dists)
        print(f"\n  EDGE DISTANCE DISTRIBUTION (FIXED!):")
        print(f"    Range: [{all_edge_dists.min():.2f}, {all_edge_dists.max():.2f}] Å")
        print(f"    Mean:  {all_edge_dists.mean():.2f} Å")
        print(f"    Std:   {all_edge_dists.std():.2f} Å")
        print(f"    95th percentile: {np.percentile(all_edge_dists, 95):.2f} Å")
        
        # Check if we captured pore structure
        edges_above_5A = (all_edge_dists > 5.0).sum()
        print(f"\n    Edges > 5 Å (pore structure): {edges_above_5A} ({100*edges_above_5A/len(all_edge_dists):.1f}%)")
        
        if all_edge_dists.max() < 3.0:
            print(f"\n  ⚠️  WARNING: Max edge distance < 3 Å — pore structure NOT captured!")
        elif np.percentile(all_edge_dists, 95) > 6.0:
            print(f"\n  ✅ SUCCESS: Pore structure captured!")
        
    print(f"\n  Saved: {OUTPUT_DIR}/ ({n_success} .npz files)")
    print(f"  Saved: {STATS_PATH}")

    print(f"\n{'=' * 80}")
    print("GRAPH REPRESENTATIONS COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()