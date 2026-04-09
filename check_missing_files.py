"""
verify_graphs.py — Verify graph quality after rebuilding
=======================================================
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def verify_graphs(graph_dir="data/graphs", sample_size=100):
    """Verify that graphs have proper edge distance distribution."""
    
    graph_dir = Path(graph_dir)
    npz_files = list(graph_dir.glob("*.npz"))[:sample_size]
    
    if not npz_files:
        print(f"No .npz files found in {graph_dir}")
        return
    
    all_dists = []
    stats = []
    
    for npz_path in npz_files:
        data = np.load(npz_path)
        dists = data['edge_dist'].ravel()
        all_dists.extend(dists)
        
        stats.append({
            'file': npz_path.name,
            'num_atoms': data['node_features'].shape[0],
            'num_edges': len(dists),
            'dist_min': dists.min(),
            'dist_max': dists.max(),
            'dist_mean': dists.mean(),
        })
    
    all_dists = np.array(all_dists)
    
    print(f"\n{'='*60}")
    print("Graph Verification Report")
    print(f"{'='*60}")
    print(f"Sampled graphs: {len(npz_files)}")
    print(f"Total edges: {len(all_dists)}")
    
    print(f"\nEdge Distance Statistics:")
    print(f"  Range: [{all_dists.min():.2f}, {all_dists.max():.2f}] Å")
    print(f"  Mean: {all_dists.mean():.2f} Å")
    print(f"  Std: {all_dists.std():.2f} Å")
    print(f"\nPercentiles:")
    for p in [5, 25, 50, 75, 95, 99]:
        print(f"  {p}th: {np.percentile(all_dists, p):.2f} Å")
    
    # Check if cutoff is working
    edges_below_3A = (all_dists < 3.0).sum()
    edges_above_5A = (all_dists > 5.0).sum()
    
    print(f"\nEdge Distribution:")
    print(f"  < 3 Å (covalent): {edges_below_3A} ({100*edges_below_3A/len(all_dists):.1f}%)")
    print(f"  > 5 Å (pore): {edges_above_5A} ({100*edges_above_5A/len(all_dists):.1f}%)")
    
    # Quality assessment
    print(f"\n{'='*60}")
    print("Quality Assessment")
    print(f"{'='*60}")
    
    if all_dists.max() < 3.0:
        print("❌ FAIL: Max edge distance < 3 Å")
        print("   Graphs only capture covalent bonds!")
        print("   Run: python build_graphs_fixed.py --cutoff 8.0")
    elif all_dists.max() < 5.0:
        print("⚠️  WARNING: Max edge distance < 5 Å")
        print("   May miss large pore channels")
        print("   Consider increasing cutoff to 8.0 Å")
    elif np.percentile(all_dists, 95) > 6.0:
        print("✅ PASS: Good edge distance distribution")
        print("   Captures both bonds and pore structure")
    else:
        print("⚠️  UNCLEAR: Check distribution manually")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_dists, bins=100, range=(0, 10), edgecolor='black', alpha=0.7)
    plt.axvline(x=3.0, color='r', linestyle='--', label='Covalent bond limit')
    plt.axvline(x=8.0, color='g', linestyle='--', label='Expected cutoff')
    plt.xlabel('Edge Distance (Å)')
    plt.ylabel('Count')
    plt.title('Edge Distance Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    edge_counts = [s['num_edges'] for s in stats]
    plt.hist(edge_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Edges per Graph')
    plt.ylabel('Count')
    plt.title('Edge Count Distribution')
    
    plt.tight_layout()
    plt.savefig('graph_verification.png', dpi=150)
    print(f"\nPlot saved to: graph_verification.png")
    
    return stats

if __name__ == '__main__':
    import sys
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else "data/graphs"
    verify_graphs(graph_dir)