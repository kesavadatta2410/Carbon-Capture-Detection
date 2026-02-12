"""
Extract CO2 adsorption isotherms and pore properties from hMOF-mofdb JSON files.

Processes ALL 32,767 hMOF-mofdb structures to extract:
- CO2 adsorption isotherms at multiple pressures (298K)
- H2, CH4, N2 adsorption data
- Pore descriptors: LCD, PLD, surface area, void fraction
- Structural info: elements, formula from embedded CIF
- Creates ML-ready dataset with features + targets
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# Configuration
DATABASE_DIR = "Database/hMOF-mofdb"
RESULTS_DIR = "results"
DATA_DIR = "data"

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Target pressures for CO2 uptake extraction (in bar)
CO2_TARGET_PRESSURES = [0.01, 0.05, 0.1, 0.5, 1.0, 2.5]


def extract_elements_from_json(data):
    elements_list = data.get('elements', [])
    element_symbols = sorted([e.get('symbol', '') for e in elements_list if e.get('symbol')])
    return element_symbols


def extract_atom_count_from_cif(cif_text):
    """Extract number of atoms from the embedded CIF string."""
    if not cif_text:
        return np.nan
    # Count atom_site entries (lines with element symbols after _atom_site_fract_z)
    atom_lines = re.findall(r'\n\w+\s+([A-Z][a-z]?)\s+[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+', cif_text)
    return len(atom_lines) if atom_lines else np.nan


def extract_cell_volume_from_cif(cif_text):
    """Extract unit cell volume from CIF lattice parameters."""
    if not cif_text:
        return np.nan
    try:
        a = float(re.search(r'_cell_length_a\s+([\d.]+)', cif_text).group(1))
        b = float(re.search(r'_cell_length_b\s+([\d.]+)', cif_text).group(1))
        c = float(re.search(r'_cell_length_c\s+([\d.]+)', cif_text).group(1))
        alpha = float(re.search(r'_cell_angle_alpha\s+([\d.]+)', cif_text).group(1))
        beta = float(re.search(r'_cell_angle_beta\s+([\d.]+)', cif_text).group(1))
        gamma = float(re.search(r'_cell_angle_gamma\s+([\d.]+)', cif_text).group(1))
        
        # Convert to radians
        alpha_r = np.radians(alpha)
        beta_r = np.radians(beta)
        gamma_r = np.radians(gamma)
        
        # Calculate volume
        cos_a, cos_b, cos_g = np.cos(alpha_r), np.cos(beta_r), np.cos(gamma_r)
        volume = a * b * c * np.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g)
        return volume
    except Exception:
        return np.nan


def extract_best_isotherm(isotherms_list, gas_formula, temperature, prefer_units='mol/kg'):
    """
    Extract best isotherm for a given gas at a given temperature.
    Prefers mol/kg units but falls back to other units.
    
    Returns:
        tuple: (pressures, uptakes, units) or (None, None, None)
    """
    candidates = []
    for iso in isotherms_list:
        temp = iso.get('temperature', 0)
        adsorbates = iso.get('adsorbates', [])
        has_gas = any(ads.get('formula') == gas_formula for ads in adsorbates)
        # Only single-component isotherms
        is_single = len(adsorbates) == 1
        
        if has_gas and is_single and temp == temperature and 'isotherm_data' in iso:
            candidates.append(iso)
    
    if not candidates:
        return None, None, None
    
    # Prefer mol/kg units
    best = None
    for iso in candidates:
        if iso.get('adsorptionUnits') == prefer_units:
            best = iso
            break
    if best is None:
        best = candidates[0]
    
    # Extract pressure-uptake pairs
    pressures = []
    uptakes = []
    for point in best['isotherm_data']:
        p = point.get('pressure', np.nan)
        u = point.get('total_adsorption', np.nan)
        if not np.isnan(p) and not np.isnan(u):
            pressures.append(p)
            uptakes.append(u)
    
    if len(pressures) < 1:
        return None, None, None
    
    return np.array(pressures), np.array(uptakes), best.get('adsorptionUnits', 'unknown')


def interpolate_at_pressures(pressures, uptakes, target_pressures):
    """Interpolate uptake values at target pressures."""
    results = {}
    if pressures is None or len(pressures) < 1:
        return {f'{tp}bar': np.nan for tp in target_pressures}
    
    # Sort
    idx = np.argsort(pressures)
    pressures = pressures[idx]
    uptakes = uptakes[idx]
    
    for tp in target_pressures:
        if len(pressures) == 1:
            # Only one data point - use it if it matches closely
            if np.isclose(pressures[0], tp, rtol=0.1):
                results[f'{tp}bar'] = float(uptakes[0])
            else:
                results[f'{tp}bar'] = np.nan
        elif tp >= pressures.min() and tp <= pressures.max():
            if tp in pressures:
                results[f'{tp}bar'] = float(uptakes[np.where(pressures == tp)[0][0]])
            else:
                interp_func = interp1d(pressures, uptakes, kind='linear')
                results[f'{tp}bar'] = float(interp_func(tp))
        else:
            results[f'{tp}bar'] = np.nan
    
    return results


def parse_json_file(json_path):
    """
    Parse a single hMOF JSON file and extract all relevant properties.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic info
        mof_name = data.get('name', os.path.basename(json_path).replace('.json', ''))
        mofid = data.get('mofid', '')
        mofkey = data.get('mofkey', '')
        database = data.get('database', 'hMOF')
        
        # Structural features from JSON
        elements = extract_elements_from_json(data)
        num_elements = len(elements)
        elements_str = ','.join(elements)
        
        # Extract from CIF
        cif_text = data.get('cif', '')
        num_atoms = extract_atom_count_from_cif(cif_text)
        cell_volume = extract_cell_volume_from_cif(cif_text)
        
        # Pore descriptors
        lcd = data.get('lcd', np.nan)
        pld = data.get('pld', np.nan)
        surface_area_m2g = data.get('surface_area_m2g', np.nan)
        surface_area_m2cm3 = data.get('surface_area_m2cm3', np.nan)
        void_fraction = data.get('void_fraction', np.nan)
        
        # Collect all isotherms from both 'isotherms' and 'heats' fields
        all_isotherms = []
        if 'isotherms' in data and data['isotherms']:
            all_isotherms.extend(data['isotherms'])
        if 'heats' in data and data['heats']:
            all_isotherms.extend(data['heats'])
        
        # --- CO2 at 298K ---
        co2_p, co2_u, co2_units = extract_best_isotherm(all_isotherms, 'CO2', 298)
        co2_uptakes = interpolate_at_pressures(co2_p, co2_u, CO2_TARGET_PRESSURES)
        co2_uptakes = {f'CO2_uptake_{k}': v for k, v in co2_uptakes.items()}
        co2_uptakes['CO2_units'] = co2_units if co2_p is not None else ''
        co2_uptakes['CO2_num_points'] = len(co2_p) if co2_p is not None else 0
        
        # --- CH4 at 298K ---
        ch4_p, ch4_u, ch4_units = extract_best_isotherm(all_isotherms, 'CH4', 298)
        ch4_data = {}
        if ch4_p is not None and len(ch4_p) > 0:
            ch4_data['CH4_uptake_max'] = float(np.max(ch4_u))
            ch4_data['CH4_pressure_max'] = float(ch4_p[np.argmax(ch4_u)])
            ch4_data['CH4_units'] = ch4_units
            ch4_data['CH4_num_points'] = len(ch4_p)
        
        # --- H2 at 77K ---
        h2_p, h2_u, h2_units = extract_best_isotherm(all_isotherms, 'H2', 77)
        h2_data = {}
        if h2_p is not None and len(h2_p) > 0:
            h2_data['H2_uptake_max'] = float(np.max(h2_u))
            h2_data['H2_pressure_max'] = float(h2_p[np.argmax(h2_u)])
            h2_data['H2_units'] = h2_units
            h2_data['H2_num_points'] = len(h2_p)
        
        # --- N2 at 298K ---
        n2_p, n2_u, n2_units = extract_best_isotherm(all_isotherms, 'N2', 298)
        n2_data = {}
        if n2_p is not None and len(n2_p) > 0:
            n2_data['N2_uptake_max'] = float(np.max(n2_u))
            n2_data['N2_pressure_max'] = float(n2_p[np.argmax(n2_u)])
            n2_data['N2_units'] = n2_units
            n2_data['N2_num_points'] = len(n2_p)
        
        # Build result
        result = {
            'name': mof_name,
            'mofid': mofid,
            'mofkey': mofkey,
            'database': database,
            'elements': elements_str,
            'num_elements': num_elements,
            'num_atoms': num_atoms,
            'cell_volume': cell_volume,
            'lcd': lcd,
            'pld': pld,
            'surface_area_m2g': surface_area_m2g,
            'surface_area_m2cm3': surface_area_m2cm3,
            'void_fraction': void_fraction,
            **co2_uptakes,
            **ch4_data,
            **h2_data,
            **n2_data,
        }
        
        return result
        
    except Exception as e:
        return None


def process_all_hmof_files():
    """Process ALL hMOF-mofdb JSON files."""
    print("\n" + "="*80)
    print("EXTRACTING PROPERTIES FROM ALL hMOF-mofdb JSON FILES")
    print("="*80)
    
    json_files = sorted(Path(DATABASE_DIR).glob("*.json"))
    print(f"\nFound {len(json_files)} JSON files in {DATABASE_DIR}")
    
    all_properties = []
    failed = 0
    
    for jf in tqdm(json_files, desc="Parsing JSON files"):
        result = parse_json_file(jf)
        if result is not None:
            all_properties.append(result)
        else:
            failed += 1
    
    print(f"\nSuccessfully parsed: {len(all_properties)}")
    print(f"Failed: {failed}")
    
    df = pd.DataFrame(all_properties)
    df['dataset'] = 'hMOF-mofdb'
    return df


def generate_summary_statistics(df):
    """Generate and print summary statistics."""
    print("\n" + "="*80)
    print("PROPERTY EXTRACTION SUMMARY")
    print("="*80)
    
    lines = []
    lines.append(f"Total structures: {len(df)}")
    
    # CO2 data availability
    co2_cols = sorted([c for c in df.columns if c.startswith('CO2_uptake_') and 'bar' in c])
    lines.append("\n=== CO2 Adsorption (298K) ===")
    has_any_co2 = df['CO2_num_points'].gt(0).sum() if 'CO2_num_points' in df.columns else 0
    lines.append(f"Structures with CO2 data: {has_any_co2} ({has_any_co2/len(df)*100:.1f}%)")
    
    for col in co2_cols:
        valid = df[col].notna().sum()
        if valid > 0:
            lines.append(f"\n  {col}:")
            lines.append(f"    Available: {valid} ({valid/len(df)*100:.1f}%)")
            lines.append(f"    Mean: {df[col].mean():.4f} | Std: {df[col].std():.4f}")
            lines.append(f"    Min: {df[col].min():.4f} | Max: {df[col].max():.4f}")
    
    # CH4 data
    if 'CH4_uptake_max' in df.columns:
        ch4_count = df['CH4_uptake_max'].notna().sum()
        lines.append(f"\n=== CH4 Adsorption (298K) ===")
        lines.append(f"Structures with CH4 data: {ch4_count} ({ch4_count/len(df)*100:.1f}%)")
        if ch4_count > 0:
            lines.append(f"  Max uptake - Mean: {df['CH4_uptake_max'].mean():.4f} | Range: [{df['CH4_uptake_max'].min():.4f}, {df['CH4_uptake_max'].max():.4f}]")
    
    # H2 data
    if 'H2_uptake_max' in df.columns:
        h2_count = df['H2_uptake_max'].notna().sum()
        lines.append(f"\n=== H2 Adsorption (77K) ===")
        lines.append(f"Structures with H2 data: {h2_count} ({h2_count/len(df)*100:.1f}%)")
        if h2_count > 0:
            lines.append(f"  Max uptake - Mean: {df['H2_uptake_max'].mean():.4f} | Range: [{df['H2_uptake_max'].min():.4f}, {df['H2_uptake_max'].max():.4f}]")
    
    # N2 data
    if 'N2_uptake_max' in df.columns:
        n2_count = df['N2_uptake_max'].notna().sum()
        lines.append(f"\n=== N2 Adsorption (298K) ===")
        lines.append(f"Structures with N2 data: {n2_count} ({n2_count/len(df)*100:.1f}%)")
    
    # Pore descriptors
    lines.append(f"\n=== Pore Descriptors ===")
    for col, label in [('lcd', 'LCD (A)'), ('pld', 'PLD (A)'), 
                       ('surface_area_m2g', 'Surface Area (m2/g)'), 
                       ('void_fraction', 'Void Fraction')]:
        if col in df.columns:
            valid = df[col].notna().sum()
            lines.append(f"  {label}: {valid} structures | Mean: {df[col].mean():.2f} | Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
    
    # Structural
    lines.append(f"\n=== Structural Features ===")
    if 'num_atoms' in df.columns:
        lines.append(f"  Atoms/cell: Mean {df['num_atoms'].mean():.1f} | Range: [{df['num_atoms'].min():.0f}, {df['num_atoms'].max():.0f}]")
    if 'cell_volume' in df.columns:
        lines.append(f"  Cell Volume (A3): Mean {df['cell_volume'].mean():.1f} | Range: [{df['cell_volume'].min():.1f}, {df['cell_volume'].max():.1f}]")
    if 'num_elements' in df.columns:
        lines.append(f"  Unique elements: Mean {df['num_elements'].mean():.1f} | Range: [{df['num_elements'].min():.0f}, {df['num_elements'].max():.0f}]")
    
    text = "\n".join(lines)
    print(text)
    
    # Save
    summary_file = os.path.join(RESULTS_DIR, "hmof_property_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("hMOF-mofdb Property Extraction Summary\n")
        f.write("="*80 + "\n\n")
        f.write(text)
    print(f"\nSaved: {summary_file}")


def generate_visualizations(df):
    """Generate property distribution visualizations."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('hMOF-mofdb Property Distributions (32,768 structures)', fontsize=14, fontweight='bold')
    
    # 1. CO2 uptake at 0.1 bar
    col = 'CO2_uptake_0.1bar'
    if col in df.columns:
        data = df[col].dropna()
        if len(data) > 0:
            axes[0,0].hist(data, bins=60, edgecolor='black', alpha=0.7, color='#2196F3')
            axes[0,0].axvline(data.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {data.mean():.3f}')
            axes[0,0].set_xlabel('CO2 Uptake at 0.1 bar (mol/kg)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_title(f'CO2 at 0.1 bar (n={len(data)})')
            axes[0,0].legend()
    
    # 2. CO2 uptake at 2.5 bar
    col = 'CO2_uptake_2.5bar'
    if col in df.columns:
        data = df[col].dropna()
        if len(data) > 0:
            axes[0,1].hist(data, bins=60, edgecolor='black', alpha=0.7, color='#4CAF50')
            axes[0,1].axvline(data.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {data.mean():.3f}')
            axes[0,1].set_xlabel('CO2 Uptake at 2.5 bar (mol/kg)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title(f'CO2 at 2.5 bar (n={len(data)})')
            axes[0,1].legend()
    
    # 3. LCD vs PLD
    if 'lcd' in df.columns and 'pld' in df.columns:
        valid = df[['lcd','pld']].dropna()
        axes[0,2].scatter(valid['pld'], valid['lcd'], alpha=0.3, s=8, color='#FF9800')
        axes[0,2].plot([0,max(valid['lcd'].max(), valid['pld'].max())], 
                      [0,max(valid['lcd'].max(), valid['pld'].max())], 'r--', lw=1, label='LCD=PLD')
        axes[0,2].set_xlabel('PLD (Angstrom)')
        axes[0,2].set_ylabel('LCD (Angstrom)')
        axes[0,2].set_title(f'Pore Sizes (n={len(valid)})')
        axes[0,2].legend()
    
    # 4. Surface area vs CO2
    if 'surface_area_m2g' in df.columns and 'CO2_uptake_0.1bar' in df.columns:
        valid = df[['surface_area_m2g','CO2_uptake_0.1bar']].dropna()
        if len(valid) > 0:
            axes[1,0].scatter(valid['surface_area_m2g'], valid['CO2_uptake_0.1bar'], alpha=0.3, s=8, color='#9C27B0')
            corr = valid.corr().iloc[0,1]
            axes[1,0].set_xlabel('Surface Area (m2/g)')
            axes[1,0].set_ylabel('CO2 Uptake at 0.1 bar')
            axes[1,0].set_title(f'Surface Area vs CO2 (r={corr:.3f})')
    
    # 5. Void fraction vs CO2
    if 'void_fraction' in df.columns and 'CO2_uptake_0.1bar' in df.columns:
        valid = df[['void_fraction','CO2_uptake_0.1bar']].dropna()
        if len(valid) > 0:
            axes[1,1].scatter(valid['void_fraction'], valid['CO2_uptake_0.1bar'], alpha=0.3, s=8, color='#F44336')
            corr = valid.corr().iloc[0,1]
            axes[1,1].set_xlabel('Void Fraction')
            axes[1,1].set_ylabel('CO2 Uptake at 0.1 bar')
            axes[1,1].set_title(f'Void Fraction vs CO2 (r={corr:.3f})')
    
    # 6. Surface area distribution
    if 'surface_area_m2g' in df.columns:
        data = df['surface_area_m2g'].dropna()
        if len(data) > 0:
            axes[1,2].hist(data, bins=60, edgecolor='black', alpha=0.7, color='#009688')
            axes[1,2].axvline(data.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {data.mean():.0f}')
            axes[1,2].set_xlabel('Surface Area (m2/g)')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].set_title(f'Surface Area Distribution (n={len(data)})')
            axes[1,2].legend()
    
    plt.tight_layout()
    viz_path = os.path.join(RESULTS_DIR, "hmof_property_distributions.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {viz_path}")
    
    # --- Correlation heatmap ---
    numeric_cols = ['lcd', 'pld', 'surface_area_m2g', 'surface_area_m2cm3', 'void_fraction',
                    'num_atoms', 'cell_volume', 'num_elements']
    co2_cols = [c for c in df.columns if c.startswith('CO2_uptake_') and 'bar' in c]
    numeric_cols.extend(co2_cols)
    
    available_cols = [c for c in numeric_cols if c in df.columns]
    corr_data = df[available_cols].dropna(how='all')
    
    if len(corr_data) > 100:
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        corr_matrix = corr_data.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   ax=ax2, square=True, linewidths=0.5)
        ax2.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        corr_path = os.path.join(RESULTS_DIR, "hmof_correlation_heatmap.png")
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {corr_path}")


def main():
    """Main pipeline."""
    print("\n" + "="*80)
    print("hMOF-mofdb COMPLETE PROPERTY EXTRACTION PIPELINE")
    print("Processing ALL structures from Database/hMOF-mofdb/")
    print("="*80)
    
    # Step 1: Extract properties
    df = process_all_hmof_files()
    
    # Step 2: Save full dataset
    print("\n" + "="*80)
    print("SAVING DATASETS")
    print("="*80)
    
    full_file = os.path.join(DATA_DIR, "hmof_properties.csv")
    df.to_csv(full_file, index=False)
    print(f"Full properties dataset: {full_file} ({len(df)} structures)")
    
    # Step 3: Create ML-ready dataset (only structures with CO2 data in mol/kg)
    co2_cols = [c for c in df.columns if c.startswith('CO2_uptake_') and 'bar' in c]
    has_co2 = df[co2_cols].notna().any(axis=1)
    
    # Filter for mol/kg units (clean ML target)
    if 'CO2_units' in df.columns:
        molkg_mask = df['CO2_units'] == 'mol/kg'
        ml_df = df[has_co2 & molkg_mask].copy()
    else:
        ml_df = df[has_co2].copy()
    
    ml_file = os.path.join(DATA_DIR, "hmof_ml_dataset.csv")
    ml_df.to_csv(ml_file, index=False)
    print(f"ML dataset (CO2 mol/kg only): {ml_file} ({len(ml_df)} structures)")
    
    # Step 4: Summary stats
    generate_summary_statistics(df)
    
    # Step 5: Visualize
    generate_visualizations(df)
    
    # Final
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {full_file} ({len(df)} structures, all properties)")
    print(f"  - {ml_file} ({len(ml_df)} structures, CO2 in mol/kg)")
    print(f"  - {os.path.join(RESULTS_DIR, 'hmof_property_summary.txt')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'hmof_property_distributions.png')}")
    print(f"  - {os.path.join(RESULTS_DIR, 'hmof_correlation_heatmap.png')}")
    print("="*80)


if __name__ == "__main__":
    main()
