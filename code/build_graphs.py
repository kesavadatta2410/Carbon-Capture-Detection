#!/usr/bin/env python3
"""
Task 3.2: Create graph representations from embedded CIF data in hMOF JSON files.
Nodes = atoms (features: atomic number, electronegativity, covalent radius)
Edges = bonds (distance < 1.2 × sum of covalent radii)
Output: One .npz file per structure in data/graphs/
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
STATS_PATH = "data/graph_stats.csv"

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
    "Na": {"Z": 11, "en": 0.93, "rc": 166},
    "Mg": {"Z": 12, "en": 1.31, "rc": 141},
    "Al": {"Z": 13, "en": 1.61, "rc": 121},
    "Si": {"Z": 14, "en": 1.90, "rc": 111},
    "P":  {"Z": 15, "en": 2.19, "rc": 107},
    "S":  {"Z": 16, "en": 2.58, "rc": 105},
    "Cl": {"Z": 17, "en": 3.16, "rc": 102},
    "K":  {"Z": 19, "en": 0.82, "rc": 203},
    "Ca": {"Z": 20, "en": 1.00, "rc": 176},
    "Ti": {"Z": 22, "en": 1.54, "rc": 160},
    "V":  {"Z": 23, "en": 1.63, "rc": 153},
    "Cr": {"Z": 24, "en": 1.66, "rc": 139},
    "Mn": {"Z": 25, "en": 1.55, "rc": 139},
    "Fe": {"Z": 26, "en": 1.83, "rc": 132},
    "Co": {"Z": 27, "en": 1.88, "rc": 126},
    "Ni": {"Z": 28, "en": 1.91, "rc": 124},
    "Cu": {"Z": 29, "en": 1.90, "rc": 132},
    "Zn": {"Z": 30, "en": 1.65, "rc": 122},
    "Br": {"Z": 35, "en": 2.96, "rc": 120},
    "Sr": {"Z": 38, "en": 0.95, "rc": 195},
    "Zr": {"Z": 40, "en": 1.33, "rc": 175},
    "Mo": {"Z": 42, "en": 2.16, "rc": 154},
    "Ag": {"Z": 47, "en": 1.93, "rc": 145},
    "Cd": {"Z": 48, "en": 1.69, "rc": 144},
    "In": {"Z": 49, "en": 1.78, "rc": 142},
    "Sn": {"Z": 50, "en": 1.96, "rc": 139},
    "I":  {"Z": 53, "en": 2.66, "rc": 139},
    "Ba": {"Z": 56, "en": 0.89, "rc": 215},
    "La": {"Z": 57, "en": 1.10, "rc": 207},
    "Ce": {"Z": 58, "en": 1.12, "rc": 204},
    "Pb": {"Z": 82, "en": 2.33, "rc": 146},
    "Bi": {"Z": 83, "en": 2.02, "rc": 148},
}

BOND_FACTOR = 1.2  # Bond if distance < factor * (rc_i + rc_j)
MAX_ATOMS = 700    # Skip structures with more atoms (memory safety)


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


def build_graph(symbols, cart_coords):
    """
    Build adjacency list from atom positions.
    Edge if distance < BOND_FACTOR * (rc_i + rc_j) in pm → convert to Å (/ 100).
    Returns: edge_src, edge_dst, edge_dist (bond distances)
    """
    n_atoms = len(symbols)
    if n_atoms > MAX_ATOMS:
        return None, None, None

    # Node features: [Z, EN, RC] for each atom
    node_features = np.array([get_node_features(s) for s in symbols], dtype=np.float32)

    # Compute pairwise distances (Å)
    # Use broadcasting for efficiency
    diff = cart_coords[:, np.newaxis, :] - cart_coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))

    # Bond threshold per pair: BOND_FACTOR * (rc_i + rc_j) / 100 (pm → Å)
    rc = np.array([ELEM_PROPS.get(s, {"rc": 150})["rc"] for s in symbols], dtype=np.float64)
    thresholds = BOND_FACTOR * (rc[:, np.newaxis] + rc[np.newaxis, :]) / 100.0

    # Find bonds (upper triangle only, avoid self-loops)
    mask = (dists < thresholds) & (dists > 0.1)  # min 0.1 Å to avoid self
    np.fill_diagonal(mask, False)

    # Get edge indices (both directions for undirected)
    src, dst = np.where(mask)

    # Edge features: distances
    edge_dist = dists[src, dst].astype(np.float32)

    return node_features, np.stack([src, dst], axis=0).astype(np.int32), edge_dist


def main():
    print("=" * 80)
    print("Task 3.2: Create Graph Representations from CIF Data")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all JSON files
    json_files = sorted([
        os.path.join(JSON_DIR, f)
        for f in os.listdir(JSON_DIR)
        if f.endswith(".json")
    ])
    print(f"\nFound {len(json_files)} JSON files")

    stats = []
    n_success = 0
    n_skipped = 0
    n_too_large = 0

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
                # Still save minimal info
                stats.append({
                    "name": name,
                    "num_nodes": len(symbols),
                    "num_edges": 0,
                    "graph_density": 0,
                    "status": "too_large",
                })
                continue

            # Convert to Cartesian
            a, b, c, alpha, beta, gamma = cell_params
            cart_coords = frac_to_cart(frac_coords, a, b, c, alpha, beta, gamma)

            # Build graph
            node_features, edge_index, edge_dist = build_graph(symbols, cart_coords)
            if node_features is None:
                n_skipped += 1
                continue

            n_edges = edge_index.shape[1] if edge_index.shape[1] > 0 else 0
            n_nodes = node_features.shape[0]
            max_possible_edges = n_nodes * (n_nodes - 1)
            density = n_edges / max_possible_edges if max_possible_edges > 0 else 0

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
                "graph_density": round(density, 4),
                "status": "ok",
            })
            n_success += 1

        except Exception as e:
            n_skipped += 1
            stats.append({
                "name": name,
                "num_nodes": 0,
                "num_edges": 0,
                "graph_density": 0,
                "status": f"error",
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
        print(f"  Avg density: {ok_df['graph_density'].mean():.4f}")
    print(f"\n  Saved: {OUTPUT_DIR}/ ({n_success} .npz files)")
    print(f"  Saved: {STATS_PATH}")

    print(f"\n{'=' * 80}")
    print("GRAPH REPRESENTATIONS COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
