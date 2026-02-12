"""
Comprehensive Exploratory Data Analysis for Entire MOF/COF Database
Analyzes all datasets: CURATED-COFs, QMOF, GA_MOFs, and additional CIF collections
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Structure
from tqdm import tqdm
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

class ComprehensiveDatabaseEDA:
    """Complete EDA across all MOF/COF datasets"""
    
    def __init__(self, database_dir="Database", results_dir="results"):
        self.database_dir = database_dir
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.all_data = []
        self.df = None
        self.dataset_stats = {}
        
    def scan_all_datasets(self):
        """Scan all datasets and count CIF files"""
        print("=" * 90)
        print("SCANNING ENTIRE DATABASE")
        print("=" * 90)
        
        datasets = {}
        for root, dirs, files in os.walk(self.database_dir):
            cif_files = [f for f in files if f.endswith('.cif')]
            if cif_files:
                rel_path = os.path.relpath(root, self.database_dir)
                datasets[rel_path] = len(cif_files)
        
        print(f"\nFound {len(datasets)} directories with CIF files:")
        total_cifs = 0
        for dataset, count in sorted(datasets.items(), key=lambda x: x[1], reverse=True):
            print(f"  {dataset:50s} : {count:>8,} files")
            total_cifs += count
        
        print(f"\n{'TOTAL':50s} : {total_cifs:>8,} CIF files")
        print("=" * 90)
        
        return datasets
    
    def parse_all_cif_files(self, max_files_per_dataset=None):
        """
        Parse CIF files from all datasets
        
        Args:
            max_files_per_dataset: Limit files per dataset (None for all)
        """
        print("\n" + "=" * 90)
        print("PARSING CIF FILES FROM ALL DATASETS")
        print("=" * 90)
        
        total_parsed = 0
        total_failed = 0
        
        for root, dirs, files in os.walk(self.database_dir):
            cif_files = [f for f in files if f.endswith('.cif')]
            
            if not cif_files:
                continue
            
            dataset_name = os.path.relpath(root, self.database_dir)
            
            if max_files_per_dataset:
                cif_files = cif_files[:max_files_per_dataset]
            
            print(f"\nDataset: {dataset_name}")
            print(f"Processing {len(cif_files):,} files...")
            
            failed_files = []
            parsed_count = 0
            
            for cif_file in tqdm(cif_files, desc=f"Parsing {dataset_name}"):
                try:
                    filepath = os.path.join(root, cif_file)
                    structure = Structure.from_file(filepath)
                    
                    # Extract features
                    features = self._extract_features(structure, cif_file, dataset_name)
                    self.all_data.append(features)
                    parsed_count += 1
                    
                except Exception as e:
                    failed_files.append({'dataset': dataset_name, 'file': cif_file, 'error': str(e)})
            
            total_parsed += parsed_count
            total_failed += len(failed_files)
            
            print(f"✓ Parsed: {parsed_count:,} | ✗ Failed: {len(failed_files):,}")
            
            # Save dataset-specific failures
            if failed_files:
                fail_df = pd.DataFrame(failed_files)
                fail_path = os.path.join(self.results_dir, f'failed_{dataset_name.replace(os.sep, "_")}.csv')
                fail_df.to_csv(fail_path, index=False)
        
        print("\n" + "=" * 90)
        print(f"PARSING COMPLETE")
        print(f"✓ Total successfully parsed: {total_parsed:,}")
        print(f"✗ Total failed: {total_failed:,}")
        print("=" * 90)
        
        # Create master DataFrame
        self.df = pd.DataFrame(self.all_data)
        
        # Save features
        features_path = os.path.join(self.results_dir, 'all_database_features.csv')
        self.df.to_csv(features_path, index=False)
        print(f"\n✓ Saved all features to: {features_path}")
        
    def _extract_features(self, structure, filename, dataset_name):
        """Extract comprehensive structural features from a structure"""
        
        # Basic properties
        composition = structure.composition
        
        # Unit cell properties
        lattice = structure.lattice
        volume = lattice.volume
        density = structure.density
        
        # Calculate number of atoms
        num_atoms = len(structure.sites)
        
        # Element analysis
        elements = composition.elements
        num_elements = len(elements)
        element_symbols = sorted([str(el) for el in elements])
        
        # Metal detection (common MOF/COF metals)
        metals = ['Zn', 'Cu', 'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'V', 'Ti', 'Al', 
                 'Mg', 'Ca', 'Zr', 'Sc', 'Mo', 'W', 'Ru', 'Rh', 'Pd', 'Ag', 
                 'Cd', 'In', 'Sn', 'La', 'Ce', 'Nd', 'Gd', 'Dy', 'Er', 'Yb']
        metal_content = sum([composition.get_atomic_fraction(el) for el in elements if str(el) in metals])
        has_metal = metal_content > 0
        
        # Organic content (C, H, O, N)
        organic_elements = ['C', 'H', 'O', 'N']
        organic_content = sum([composition.get_atomic_fraction(el) for el in elements if str(el) in organic_elements])
        
        # Halogen content (F, Cl, Br, I)
        halogen_elements = ['F', 'Cl', 'Br', 'I']
        halogen_content = sum([composition.get_atomic_fraction(el) for el in elements if str(el) in halogen_elements])
        
        # Void fraction estimate (simplified)
        expected_dense_density = 2.0  # Typical dense material density (g/cm³)
        void_fraction = max(0, 1 - (density / expected_dense_density))
        
        # Cell parameters
        a, b, c = lattice.abc
        alpha, beta, gamma = lattice.angles
        
        # Cell shape metrics
        is_cubic = abs(a - b) < 0.1 and abs(b - c) < 0.1 and abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 90) < 1
        is_orthorhombic = abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 90) < 1
        
        features = {
            'dataset': dataset_name,
            'filename': filename,
            'formula': composition.formula,
            'reduced_formula': composition.reduced_formula,
            'num_atoms': num_atoms,
            'num_elements': num_elements,
            'elements': '_'.join(element_symbols),
            'density_g_cm3': density,
            'volume_A3': volume,
            'cell_a': a,
            'cell_b': b,
            'cell_c': c,
            'cell_alpha': alpha,
            'cell_beta': beta,
            'cell_gamma': gamma,
            'is_cubic': is_cubic,
            'is_orthorhombic': is_orthorhombic,
            'metal_fraction': metal_content,
            'has_metal': has_metal,
            'organic_fraction': organic_content,
            'halogen_fraction': halogen_content,
            'void_fraction_estimate': void_fraction,
            'mass_amu': composition.weight,
        }
        
        # Add individual element fractions for common elements
        common_elements = ['C', 'H', 'O', 'N', 'Zn', 'Cu', 'Fe', 'Co', 'Ni', 'F', 'Cl', 'Br']
        for el_symbol in common_elements:
            frac = sum([composition.get_atomic_fraction(el) for el in elements if str(el) == el_symbol])
            features[f'frac_{el_symbol}'] = frac
        
        return features
    
    def generate_dataset_comparison(self):
        """Compare statistics across datasets"""
        print("\n" + "=" * 90)
        print("DATASET COMPARISON")
        print("=" * 90)
        
        comp_data = []
        for dataset in self.df['dataset'].unique():
            subset = self.df[self.df['dataset'] == dataset]
            comp_data.append({
                'Dataset': dataset,
                'Count': len(subset),
                'Avg Atoms': subset['num_atoms'].mean(),
                'Avg Density': subset['density_g_cm3'].mean(),
                'Avg Volume': subset['volume_A3'].mean(),
                'Metal %': (subset['has_metal'].sum() / len(subset)) * 100,
                'Avg Void Frac': subset['void_fraction_estimate'].mean()
            })
        
        comp_df = pd.DataFrame(comp_data).sort_values('Count', ascending=False)
        
        # Save comparison
        comp_path = os.path.join(self.results_dir, 'dataset_comparison.csv')
        comp_df.to_csv(comp_path, index=False)
        
        print("\nDataset Statistics:")
        print(comp_df.to_string(index=False))
        print(f"\n✓ Saved comparison to: {comp_path}")
        
        return comp_df
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "=" * 90)
        print("GENERATING VISUALIZATIONS")
        print("=" * 90)
        
        # 1. Dataset distribution pie chart
        plt.figure(figsize=(12, 8))
        dataset_counts = self.df['dataset'].value_counts()
        colors = plt.cm.Set3(range(len(dataset_counts)))
        plt.pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Distribution of Structures Across Datasets', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: dataset_distribution.png")
        plt.close()
        
        # 2. Feature distributions by dataset
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Distributions Across Datasets', fontsize=16, fontweight='bold')
        
        features = ['density_g_cm3', 'volume_A3', 'num_atoms', 'void_fraction_estimate']
        titles = ['Density (g/cm³)', 'Volume (Ų)', 'Number of Atoms', 'Void Fraction']
        
        for idx, (feat, title) in enumerate(zip(features, titles)):
            ax = axes[idx // 2, idx % 2]
            for dataset in self.df['dataset'].unique():
                subset = self.df[self.df['dataset'] == dataset]
                ax.hist(subset[feat], bins=30, alpha=0.5, label=dataset)
            ax.set_xlabel(title)
            ax.set_ylabel('Frequency')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'feature_distributions_by_dataset.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: feature_distributions_by_dataset.png")
        plt.close()
        
        # 3. Density vs Volume scatter (color by dataset)
        plt.figure(figsize=(12, 8))
        for dataset in self.df['dataset'].unique():
            subset = self.df[self.df['dataset'] == dataset]
            plt.scatter(subset['volume_A3'], subset['density_g_cm3'], 
                       label=dataset, alpha=0.6, s=20)
        plt.xlabel('Unit Cell Volume (Ų)', fontsize=12)
        plt.ylabel('Density (g/cm³)', fontsize=12)
        plt.title('Density vs Volume by Dataset', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'density_vs_volume_by_dataset.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: density_vs_volume_by_dataset.png")
        plt.close()
        
        # 4. Element frequency across all datasets
        all_elements = []
        for elements_str in self.df['elements']:
            all_elements.extend(elements_str.split('_'))
        element_counts = pd.Series(all_elements).value_counts().head(25)
        
        plt.figure(figsize=(14, 8))
        element_counts.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Top 25 Most Frequent Elements Across All Datasets', fontsize=16, fontweight='bold')
        plt.xlabel('Element', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'element_frequency_all.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: element_frequency_all.png")
        plt.close()
        
        # 5. Metal content comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        metal_data = []
        for dataset in self.df['dataset'].unique():
            subset = self.df[self.df['dataset'] == dataset]
            metal_pct = (subset['has_metal'].sum() / len(subset)) * 100
            metal_data.append({'Dataset': dataset, 'Metal %': metal_pct})
        
        metal_df = pd.DataFrame(metal_data).sort_values('Metal %', ascending=False)
        ax.bar(range(len(metal_df)), metal_df['Metal %'], color='coral', edgecolor='black')
        ax.set_xticks(range(len(metal_df)))
        ax.set_xticklabels(metal_df['Dataset'], rotation=45, ha='right')
        ax.set_ylabel('Percentage with Metals (%)', fontsize=12)
        ax.set_title('Metal Content by Dataset', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'metal_content_by_dataset.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: metal_content_by_dataset.png")
        plt.close()
        
        print(f"\n✓ All visualizations saved to: {self.results_dir}/")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive markdown report"""
        print("\n" + "=" * 90)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 90)
        
        report_path = os.path.join(self.results_dir, 'complete_database_eda_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Complete MOF/COF Database - Exploratory Data Analysis\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Structures Analyzed**: {len(self.df):,}\n")
            f.write(f"- **Number of Datasets**: {self.df['dataset'].nunique()}\n")
            f.write(f"- **Unique Elements**: {len(set([e for el_str in self.df['elements'] for e in el_str.split('_')]))}\n")
            f.write(f"- **Structures with Metals**: {self.df['has_metal'].sum():,} ({self.df['has_metal'].mean()*100:.1f}%)\n\n")
            
            f.write("## Dataset Breakdown\n\n")
            for dataset in self.df['dataset'].unique():
                subset = self.df[self.df['dataset'] == dataset]
                f.write(f"### {dataset}\n\n")
                f.write(f"- **Count**: {len(subset):,} structures\n")
                f.write(f"- **Avg Atoms**: {subset['num_atoms'].mean():.1f}\n")
                f.write(f"- **Avg Density**: {subset['density_g_cm3'].mean():.3f} g/cm³\n")
                f.write(f"- **Avg Volume**: {subset['volume_A3'].mean():.1f} Ų\n")
                f.write(f"- **Metal Content**: {(subset['has_metal'].sum()/len(subset))*100:.1f}%\n\n")
            
            f.write("## Overall Statistics\n\n")
            f.write(f"**Density (g/cm³)**:\n")
            f.write(f"- Range: {self.df['density_g_cm3'].min():.3f} - {self.df['density_g_cm3'].max():.3f}\n")
            f.write(f"- Mean ± Std: {self.df['density_g_cm3'].mean():.3f} ± {self.df['density_g_cm3'].std():.3f}\n\n")
            
            f.write(f"**Unit Cell Volume (Ų)**:\n")
            f.write(f"- Range: {self.df['volume_A3'].min():.1f} - {self.df['volume_A3'].max():.1f}\n")
            f.write(f"- Mean ± Std: {self.df['volume_A3'].mean():.1f} ± {self.df['volume_A3'].std():.1f}\n\n")
            
            f.write(f"**Number of Atoms**:\n")
            f.write(f"- Range: {self.df['num_atoms'].min():.0f} - {self.df['num_atoms'].max():.0f}\n")
            f.write(f"- Mean ± Std: {self.df['num_atoms'].mean():.1f} ± {self.df['num_atoms'].std():.1f}\n\n")
            
            # Element distribution
            f.write("## Element Distribution\n\n")
            all_elements = []
            for elements_str in self.df['elements']:
                all_elements.extend(elements_str.split('_'))
            element_counts = pd.Series(all_elements).value_counts().head(20)
            
            f.write("**Top 20 Most Common Elements**:\n\n")
            for el, count in element_counts.items():
                f.write(f"- {el}: {count:,} occurrences\n")
            
            # Visualizations
            f.write("\n## Visualizations\n\n")
            f.write("### Dataset Distribution\n")
            f.write("![Dataset Distribution](dataset_distribution.png)\n\n")
            
            f.write("### Feature Distributions by Dataset\n")
            f.write("![Feature Distributions](feature_distributions_by_dataset.png)\n\n")
            
            f.write("### Density vs Volume\n")
            f.write("![Density vs Volume](density_vs_volume_by_dataset.png)\n\n")
            
            f.write("### Element Frequency\n")
            f.write("![Element Frequency](element_frequency_all.png)\n\n")
            
            f.write("### Metal Content Comparison\n")
            f.write("![Metal Content](metal_content_by_dataset.png)\n\n")
            
            # Recommendations
            f.write("## Key Findings & Recommendations\n\n")
            f.write("1. **Dataset Diversity**: The database contains diverse MOF/COF structures across multiple datasets\n")
            f.write("2. **Metal Content**: Majority of structures contain metal centers, typical for MOFs\n")
            f.write("3. **Property Range**: Wide range in density, volume, and atom counts indicates structural diversity\n")
            f.write("4. **Next Steps**:\n")
            f.write("   - Perform clustering analysis to identify structure families\n")
            f.write("   - Calculate additional geometric descriptors (pore size, surface area)\n")
            f.write("   - Integrate with property databases for supervised learning\n")
            f.write("   - Consider molecular simulation for property prediction\n\n")
            
            f.write("---\n")
            f.write(f"\n*Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        print(f"✓ Comprehensive report saved to: {report_path}")
    
    def run_complete_eda(self, max_files_per_dataset=None):
        """Execute complete EDA pipeline across all datasets"""
        print("\n" + "=" * 90)
        print("COMPLETE DATABASE EXPLORATORY DATA ANALYSIS")
        print("=" * 90)
        
        # Step 1: Scan datasets
        self.scan_all_datasets()
        
        # Step 2: Parse all CIF files
        self.parse_all_cif_files(max_files_per_dataset=max_files_per_dataset)
        
        # Step 3: Dataset comparison
        self.generate_dataset_comparison()
        
        # Step 4: Create visualizations
        self.create_visualizations()
        
        # Step 5: Generate report
        self.generate_comprehensive_report()
        
        print("\n" + "=" * 90)
        print("COMPLETE DATABASE EDA FINISHED!")
        print("=" * 90)
        print(f"\nAll results saved to: {os.path.abspath(self.results_dir)}/")
        print("\nGenerated files:")
        print("  - all_database_features.csv (all structural features)")
        print("  - dataset_comparison.csv (dataset statistics)")
        print("  - complete_database_eda_report.md (comprehensive report)")
        print("  - dataset_distribution.png")
        print("  - feature_distributions_by_dataset.png")
        print("  - density_vs_volume_by_dataset.png")
        print("  - element_frequency_all.png")
        print("  - metal_content_by_dataset.png")
        print("\n" + "=" * 90)


if __name__ == "__main__":
    # Run complete database EDA
    eda = ComprehensiveDatabaseEDA(database_dir="Database", results_dir="results")
    
    # For initial testing, limit to 500 files per dataset
    # Remove max_files_per_dataset=500 to process ALL files in the database
    print("\n⚠️  RUNNING WITH SAMPLE: 500 files per dataset")
    print("To analyze the COMPLETE database, set max_files_per_dataset=None\n")
    
    eda.run_complete_eda(max_files_per_dataset=500)
    
    print("\n📊 To process ALL structures in the database:")
    print("   eda.run_complete_eda(max_files_per_dataset=None)")
