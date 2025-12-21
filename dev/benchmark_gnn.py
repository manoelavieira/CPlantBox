"""
GNN benchmarking using the train.py infrastructure

This script measures both latency (batch_size=1) and throughput (batch_size>1).
Uses the benchmark_model_inference() function from train.py, which measures timing
around run_forward() - the same code path used during training and evaluation.

Usage:
    # Latency only (batch_size=1)
    python benchmark_gnn.py --model-path models/best_model.pt --data-path data/sim_07

    # Latency + throughput at different batch sizes
    python benchmark_gnn.py --model-path models/best_model.pt --data-path data/sim_07 \
        --batch-sizes 1 4 8 16 32
"""
import argparse
import torch
import pandas as pd
from pathlib import Path

from model.gnn_operator import PhloemOperatorGNN
from model.config import ModelConfig
from model.utils import Standardizer
from data.dataset_loader import load_graphs_from_file
from torch_geometric.loader import DataLoader
from training.train import benchmark_model_inference


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config and state_dict from checkpoint
    if 'cfg' in checkpoint:
        # Checkpoint contains full training state
        cfg_dict = checkpoint['cfg']
        state_dict = checkpoint['state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

        # Convert dict to ModelConfig object if needed
        if isinstance(cfg_dict, dict):
            cfg = ModelConfig()
            # Update config with saved parameters
            for key, value in cfg_dict.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        else:
            cfg = cfg_dict
    else:
        # Checkpoint is just the state_dict
        cfg = ModelConfig()
        state_dict = checkpoint

    # Create model with loaded config
    model = PhloemOperatorGNN(cfg)
    model = model.double()  # Convert to float64

    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load scalers from checkpoint if available
    def load_scaler(scaler_dict):
        """Reconstruct a Standardizer from saved dict."""
        scaler = Standardizer()
        if scaler_dict and 'mean' in scaler_dict and 'std' in scaler_dict:
            scaler.mean = scaler_dict['mean']
            scaler.std = scaler_dict['std']
            scaler.device = torch.device(scaler_dict.get('device', 'cpu'))
        return scaler

    if 'feature_scaler' in checkpoint:
        model.feature_scaler = load_scaler(checkpoint['feature_scaler'])
    if 'target_scaler' in checkpoint:
        model.target_scaler = load_scaler(checkpoint['target_scaler'])
    if 'time_scaler' in checkpoint:
        model.time_scaler = load_scaler(checkpoint['time_scaler'])
    if 'edge_scaler' in checkpoint:
        model.edge_scaler = load_scaler(checkpoint['edge_scaler'])

    print(f"Model loaded successfully")
    print(f"  - feature_scaler: {'✓' if hasattr(model, 'feature_scaler') and model.feature_scaler else '✗'}")
    print(f"  - target_scaler: {'✓' if hasattr(model, 'target_scaler') and model.target_scaler else '✗'}")
    print(f"  - time_scaler: {'✓' if hasattr(model, 'time_scaler') and model.time_scaler else '✗'}")
    print(f"  - edge_scaler: {'✓' if hasattr(model, 'edge_scaler') and model.edge_scaler else '✗'}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Benchmark GNN inference with multiple batch sizes")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to HDF5 file (.h5) containing simulation data')
    parser.add_argument('--output-dir', type=str, default='logs/benchmarks',
                        help='Directory to save timing results')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1],
                        help='Batch sizes to test (default: [1]). First value should be 1 for latency.')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup batches')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU execution')

    args = parser.parse_args()

    # Setup device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Validate batch sizes
    if 1 not in args.batch_sizes:
        print("WARNING: batch_size=1 not in list. Latency measurement requires batch_size=1.")
        print("Adding batch_size=1 to the beginning of the list.")
        args.batch_sizes = [1] + args.batch_sizes

    print(f"\n{'='*70}")
    print("GNN Performance Benchmark")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Batch sizes to test: {args.batch_sizes}")
    print(f"{'='*70}")

    # Load model
    model = load_model(args.model_path, device)

    # Load dataset
    print(f"\nLoading dataset...")
    path = Path(args.data_path)

    if not path.exists():
        raise RuntimeError(f"Path does not exist: {args.data_path}")

    if path.is_dir():
        raise RuntimeError(f"Path is a directory, not a file: {args.data_path}\n"
                         f"Please provide a single HDF5 file (.h5), not a directory.")

    if not path.is_file():
        raise RuntimeError(f"Path is not a valid file: {args.data_path}")

    if not str(path).endswith('.h5'):
        print(f"WARNING: File does not have .h5 extension: {args.data_path}")

    print(f"Loading data from file: {args.data_path}")
    graphs = load_graphs_from_file(args.data_path, None)

    if not graphs:
        raise RuntimeError("No valid graphs loaded")

    print(f"Successfully loaded {len(graphs)} graphs for benchmarking")

    # Extract data prefix from the data path (e.g., "sim_08" from "cplantbox/data/sim_08/phloem_simulation.h5")
    data_path_obj = Path(args.data_path)
    if data_path_obj.is_file():
        data_prefix = data_path_obj.parent.name  # Get parent directory name
    else:
        data_prefix = data_path_obj.name

    print(f"Data prefix: {data_prefix}")

    # Storage for results
    all_results = []
    latency_results = None
    throughput_results = []

    # Test each batch size
    for batch_size in args.batch_sizes:
        print(f"\n{'='*70}")
        print(f"Testing with batch_size = {batch_size}")
        print(f"{'='*70}")

        # Create dataloader
        test_loader = DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=False
        )

        # Determine output path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if batch_size == 1:
            # Latency measurement
            output_path = output_dir / f'{data_prefix}_gnn_latency.csv'
            print("Mode: LATENCY (per-graph timing)")
        else:
            # Throughput measurement
            output_path = output_dir / f'{data_prefix}_gnn_throughput_bs{batch_size}.csv'
            print("Mode: THROUGHPUT (batched processing)")

        # Run benchmark
        results = benchmark_model_inference(
            model=model,
            loader=test_loader,
            device=device,
            warmup=args.warmup,
            save_path=str(output_path)
        )

        # Store results
        results['batch_size'] = batch_size
        all_results.append(results)

        if batch_size == 1:
            latency_results = results
        else:
            throughput_results.append(results)

    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    if latency_results:
        print("\nLatency (batch_size=1):")
        print(f"  Mean time per graph:     {latency_results['latency_mean_ms']:.3f} ms")
        print(f"  Median time per graph:   {latency_results['latency_median_ms']:.3f} ms")
        print(f"  Std dev:                 {latency_results['latency_std_ms']:.3f} ms")
        print(f"  Time per node:           {latency_results['time_per_node_ms']:.4f} ms")

    if throughput_results:
        print("\nThroughput (batched processing):")
        print(f"  {'Batch Size':<12} {'Graphs/sec':<15} {'Time/graph (ms)':<18} {'Speedup vs BS=1'}")
        print("-" * 70)

        baseline_throughput = latency_results['throughput_graphs_per_sec'] if latency_results else 1.0

        for res in throughput_results:
            speedup = res['throughput_graphs_per_sec'] / baseline_throughput
            print(f"  {res['batch_size']:<12} "
                  f"{res['throughput_graphs_per_sec']:<15.2f} "
                  f"{res['latency_mean_ms']:<18.3f} "
                  f"{speedup:.2f}x")

    # Save consolidated summary
    summary_df = pd.DataFrame(all_results)
    summary_path = Path(args.output_dir) / f'{data_prefix}_benchmark_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nConsolidated summary saved to: {summary_path}")

    print(f"\n{'='*70}")
    print("Benchmark Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - {data_prefix}_gnn_latency.csv           (per-graph latency, batch_size=1)")

    if len(args.batch_sizes) > 1:
        print(f"  - {data_prefix}_gnn_throughput_bs*.csv    (throughput for each batch size)")

    print(f"  - {data_prefix}_benchmark_summary.csv     (consolidated results)")
    print(f"{'='*70}\n")

    return all_results


if __name__ == '__main__':
    main()
