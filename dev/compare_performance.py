"""
Compare CPlantBox and GNN performance

Uses timing data from:
- CPlantBox: phloem_timing.csv (from sim_phloem_flow.py) - includes timing and graph size
- GNN: {data_prefix}_gnn_latency.csv (from benchmark_gnn.py with batch_size=1)

Usage:
    python compare_performance.py \
        --cplantbox-timing cplantbox/data/sim_08/phloem_timing.csv \
        --gnn-timing benchmarks/sim_08_gnn_latency.csv \
        --output-dir comparison \
        --plot

Note: Data prefix (e.g., "sim_08") is automatically detected from the input paths.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_cplantbox_timings(timing_file):
    """Load CPlantBox timing data with graph sizes."""
    df = pd.read_csv(timing_file)
    df['time_ms'] = df['time_s'] * 1000

    # Verify that graph size columns exist
    if 'nodes' not in df.columns or 'edges' not in df.columns:
        print("WARNING: Graph size data (nodes, edges) not found in CPlantBox timing CSV.")
        print("         The CSV should contain columns: step, time_s, nodes, edges")
        print("         Run sim_phloem_flow.py again to regenerate with graph size data.")
        df['nodes'] = np.nan
        df['edges'] = np.nan

    return df


def load_gnn_timings(timing_file):
    """Load GNN timing data."""
    return pd.read_csv(timing_file)


def print_comparison(cplantbox_df, gnn_df):
    """Print comparison summary."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: CPlantBox vs GNN")
    print("="*80)

    # CPlantBox stats
    print("\nCPlantBox Phloem Solver:")
    print("-" * 40)
    print(f"  Total steps:              {len(cplantbox_df)}")
    print(f"  Mean time per step:       {cplantbox_df['time_ms'].mean():.2f} ms")
    print(f"  Median time per step:     {cplantbox_df['time_ms'].median():.2f} ms")
    print(f"  Std dev:                  {cplantbox_df['time_ms'].std():.2f} ms")
    print(f"  Min:                      {cplantbox_df['time_ms'].min():.2f} ms")
    print(f"  Max:                      {cplantbox_df['time_ms'].max():.2f} ms")

    if not pd.isna(cplantbox_df['nodes'].iloc[0]):
        cb_mean_nodes = cplantbox_df['nodes'].mean()
        cb_mean_edges = cplantbox_df['edges'].mean()
        cb_time_per_node = cplantbox_df['time_ms'].mean() / cb_mean_nodes
        cb_time_per_edge = cplantbox_df['time_ms'].mean() / cb_mean_edges

        print(f"\n  Mean nodes:               {cb_mean_nodes:.1f}")
        print(f"  Mean edges:               {cb_mean_edges:.1f}")
        print(f"  Time per node:            {cb_time_per_node:.4f} ms")
        print(f"  Time per edge:            {cb_time_per_edge:.4f} ms")

    # GNN stats
    print("\nGNN Model:")
    print("-" * 40)
    print(f"  Total graphs:             {len(gnn_df)}")
    print(f"  Mean time per graph:      {gnn_df['time_ms'].mean():.2f} ms")
    print(f"  Median time per graph:    {gnn_df['time_ms'].median():.2f} ms")
    print(f"  Std dev:                  {gnn_df['time_ms'].std():.2f} ms")
    print(f"  Min:                      {gnn_df['time_ms'].min():.2f} ms")
    print(f"  Max:                      {gnn_df['time_ms'].max():.2f} ms")

    gnn_mean_nodes = gnn_df['nodes'].mean()
    gnn_mean_edges = gnn_df['edges'].mean()
    gnn_time_per_node = gnn_df['time_ms'].mean() / gnn_mean_nodes
    gnn_time_per_edge = gnn_df['time_ms'].mean() / gnn_mean_edges

    print(f"\n  Mean nodes:               {gnn_mean_nodes:.1f}")
    print(f"  Mean edges:               {gnn_mean_edges:.1f}")
    print(f"  Time per node:            {gnn_time_per_node:.4f} ms")
    print(f"  Time per edge:            {gnn_time_per_edge:.4f} ms")

    # Speedup analysis
    print("\nSpeedup Analysis:")
    print("-" * 40)
    speedup = cplantbox_df['time_ms'].mean() / gnn_df['time_ms'].mean()
    print(f"  GNN is {speedup:.1f}x faster (mean time)")

    speedup_median = cplantbox_df['time_ms'].median() / gnn_df['time_ms'].median()
    print(f"  GNN is {speedup_median:.1f}x faster (median time)")

    if not pd.isna(cplantbox_df['nodes'].iloc[0]):
        speedup_per_node = cb_time_per_node / gnn_time_per_node
        speedup_per_edge = cb_time_per_edge / gnn_time_per_edge
        print(f"  GNN is {speedup_per_node:.1f}x faster (per node)")
        print(f"  GNN is {speedup_per_edge:.1f}x faster (per edge)")

    print("\n" + "="*80)


def create_plots(cplantbox_df, gnn_df, output_dir, data_prefix):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Boxplot comparison
    ax = axes[0, 0]
    ax.boxplot([cplantbox_df['time_ms'], gnn_df['time_ms']],
               labels=['CPlantBox', 'GNN'],
               showmeans=True)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Time per Graph')
    ax.grid(True, alpha=0.3)

    # Plot 2: Time series
    ax = axes[0, 1]
    ax.plot(cplantbox_df.index, cplantbox_df['time_ms'],
            label='CPlantBox', alpha=0.7, marker='o', markersize=4)
    ax.plot(gnn_df.index, gnn_df['time_ms'],
            label='GNN', alpha=0.7, marker='s', markersize=4)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Inference Time Over Samples')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Histogram
    ax = axes[1, 0]
    ax.hist(cplantbox_df['time_ms'], bins=20, alpha=0.5, label='CPlantBox', edgecolor='black')
    ax.hist(gnn_df['time_ms'], bins=20, alpha=0.5, label='GNN', edgecolor='black')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Inference Times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Per-node comparison (if available)
    ax = axes[1, 1]
    if not pd.isna(cplantbox_df['nodes'].iloc[0]):
        cb_time_per_node = cplantbox_df['time_ms'] / cplantbox_df['nodes']
        gnn_time_per_node = gnn_df['time_ms'] / gnn_df['nodes']

        ax.boxplot([cb_time_per_node, gnn_time_per_node],
                   labels=['CPlantBox', 'GNN'],
                   showmeans=True)
        ax.set_ylabel('Time per Node (ms)')
        ax.set_title('Normalized Performance')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Graph size data\nnot available',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    output_path = Path(output_dir) / f'{data_prefix}_performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")
    plt.close()


def save_summary(cplantbox_df, gnn_df, output_dir, data_prefix):
    """Save summary statistics to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        'Method': ['CPlantBox', 'GNN'],
        'Mean_ms': [cplantbox_df['time_ms'].mean(), gnn_df['time_ms'].mean()],
        'Median_ms': [cplantbox_df['time_ms'].median(), gnn_df['time_ms'].median()],
        'Std_ms': [cplantbox_df['time_ms'].std(), gnn_df['time_ms'].std()],
        'Min_ms': [cplantbox_df['time_ms'].min(), gnn_df['time_ms'].min()],
        'Max_ms': [cplantbox_df['time_ms'].max(), gnn_df['time_ms'].max()],
    })

    output_path = output_dir / f'{data_prefix}_summary.csv'
    summary.to_csv(output_path, index=False)
    print(f"Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare CPlantBox and GNN performance")
    parser.add_argument('--cplantbox-timing', type=str, required=True,
                        help='Path to CPlantBox timing CSV (with nodes/edges columns)')
    parser.add_argument('--gnn-timing', type=str, required=True,
                        help='Path to GNN timing CSV (gnn_latency.csv)')
    parser.add_argument('--output-dir', type=str, default='logs/comparison',
                        help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')

    args = parser.parse_args()

    # Auto-detect data prefix from cplantbox timing path (e.g., "cplantbox/data/sim_08/phloem_timing.csv" -> "sim_08")
    cplantbox_path = Path(args.cplantbox_timing)
    if cplantbox_path.is_file():
        data_prefix = cplantbox_path.parent.name
    else:
        data_prefix = "comparison"
    print(f"Data prefix: {data_prefix}")

    # Load data
    print(f"Loading CPlantBox data from: {args.cplantbox_timing}")
    cplantbox_df = load_cplantbox_timings(args.cplantbox_timing)

    print(f"Loading GNN data from: {args.gnn_timing}")
    gnn_df = load_gnn_timings(args.gnn_timing)

    # Print comparison
    print_comparison(cplantbox_df, gnn_df)

    # Save summary
    save_summary(cplantbox_df, gnn_df, args.output_dir, data_prefix)

    # Create plots if requested
    if args.plot:
        create_plots(cplantbox_df, gnn_df, args.output_dir, data_prefix)


if __name__ == '__main__':
    main()
