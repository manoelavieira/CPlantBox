# GNN Training for Phloem Flow Simulation

This directory contains a Graph Neural Network (GNN) implementation for learning phloem flow dynamics in plant vascular systems.

## Quick Start

### Traditional Training (Single Split)

```bash
python train_gnn.py --data-path ./data/simulation.h5 --epochs 100
```

### K-Fold Cross-Validation (Recommended for Multiple Simulation Files)

**Note:** The number of folds is automatically determined by the number of `.h5` files in your data directory. All folds are trained automatically.

```bash
# With 5 .h5 files, automatically creates and trains all 5 folds
python train_gnn.py --data-path ./data/ --use-kfold --epochs 100
```

## Training Modes

### 1. Traditional Mode (Default)

Loads all graphs and splits them randomly or chronologically:

```bash
python train_gnn.py \
    --data-path ./data/ \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --split-method random  # or 'time'
```

**Use when:**
- Working with a single simulation file
- Quick prototyping
- Need custom train/val/test ratios

### 2. K-Fold Cross-Validation Mode

Keeps all graphs from the same simulation file together (prevents data leakage).

**The number of folds equals the number of `.h5` files** - each file is used exactly once for validation and once for testing.

```bash
python train_gnn.py \
    --data-path ./data/ \
    --use-kfold
```

**Use when:**
- Multiple simulation files (different runs of same plant)
- Publishing results (better scientific rigor)
- Need robust cross-validation metrics (mean ± std)

**How it works:**
- Each `.h5` file represents one simulation run
- Number of folds = number of files (e.g., 5 files → 5 folds automatically)
- Each fold uses: (n-2) files for train, 1 for val, 1 for test
- No graphs from the same file appear in different splits
- Each file is tested exactly once and validated exactly once

**Example with 5 files:**
```
Fold 0: train=[file0,file1,file2], val=[file3], test=[file4]
Fold 1: train=[file1,file2,file3], val=[file4], test=[file0]
Fold 2: train=[file2,file3,file4], val=[file0], test=[file1]
Fold 3: train=[file3,file4,file0], val=[file1], test=[file2]
Fold 4: train=[file4,file0,file1], val=[file2], test=[file3]
```

## Command-Line Arguments

### Data Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | Required | Path to HDF5 file or directory |
| `--batch-size` | 8 | Batch size |
| `--train-ratio` | 0.8 | Train ratio (traditional mode only) |
| `--val-ratio` | 0.1 | Val ratio (traditional mode only) |
| `--split-method` | `random` | Split method: `random` or `time` (traditional mode only) |

### K-Fold Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-kfold` | `False` | Enable k-fold cross-validation. Trains all folds automatically. Number of folds = number of .h5 files |

### Model Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | `operator` | Model: `nnconv` or `operator` |
| `--use-analytical-residual` | `False` | Use analytical physics residual |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Maximum epochs |
| `--lr` | 3e-3 | Learning rate |
| `--weight-decay` | 1e-5 | Weight decay |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed |

### Loss Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--loss-type` | `physics` | Loss: `data`, `physics`, or `combined` |
| `--lambda-data` | 1.0 | Weight for MSE term |
| `--lambda-phys` | 1.0 | Weight for physics residual |
| `--lambda-ic` | 1.0 | Weight for initial condition |
| `--lambda-bc` | 1.0 | Weight for boundary condition |

### Logging Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-physics-logging` | `False` | Enable physics debug logging |
| `--tensorboard-log-dir` | `results/tensorboard_logs` | TensorBoard directory |

## Examples

### Example 1: Quick Test with Physics Loss

```bash
python train_gnn.py \
    --data-path ./data/simulation.h5 \
    --epochs 50 \
    --loss-type physics
```

### Example 2: Full K-Fold Cross-Validation

```bash
# Number of folds determined automatically from number of .h5 files
python train_gnn.py \
    --data-path ./data/ \
    --use-kfold \
    --epochs 100 \
    --lr 3e-3 \
    --loss-type physics \
    --enable-physics-logging
```

### Example 3: Combined Loss with Custom Weights

```bash
python train_gnn.py \
    --data-path ./data/ \
    --use-kfold \
    --loss-type combined \
    --lambda-data 1.0 \
    --lambda-phys 0.5 \
    --lambda-ic 2.0 \
    --lambda-bc 1.5
```

## Output

### Traditional Mode

```
======================================================================
Training in Traditional Mode
======================================================================
Train batches: 45, Validation batches: 11, Test batches: 11
[Training progress...]
TensorBoard logs saved. To view: tensorboard --logdir=results/tensorboard_logs
```

### K-Fold Mode (All Folds)

Example output when training with `--data-path cplantbox/data/sim_00`:

```
======================================================================
K-Fold Cross-Validation Training (5 folds)
======================================================================

[Training each fold...]

======================================================================
K-Fold Cross-Validation Results Summary
======================================================================

Fold   Val Loss     Model Path
----------------------------------------------------------------------
0      4.12e-02     logs/model/sim_00_best_model_fold0.pt
1      4.32e-02     logs/model/sim_00_best_model_fold1.pt
2      3.98e-02     logs/model/sim_00_best_model_fold2.pt
3      4.24e-02     logs/model/sim_00_best_model_fold3.pt
4      4.09e-02     logs/model/sim_00_best_model_fold4.pt
----------------------------------------------------------------------

Aggregated Validation Results:
  Mean Val Loss: 4.15e-02 ± 1.12e-03
======================================================================

All fold TensorBoard logs: logs/tensorboard/sim_00_fold_*
To view: tensorboard --logdir=logs/tensorboard
```

## File Organization

### Traditional Mode

```
logs/
├── model/best_model.pt
├── metrics/metrics.csv
├── physics/debugs.txt
└── tensorboard_logs/
```

### K-Fold Mode

**Note:** When using k-fold mode, all output files are automatically prefixed with the data directory name. For example, if `--data-path` is `cplantbox/data/sim_00`, the prefix will be `sim_00`.

```
logs/
├── model/
│   ├── sim_00_best_model_fold0.pt
│   ├── sim_00_best_model_fold1.pt
│   └── ...
├── metrics/
│   └── sim_00_metrics.csv         # All folds logged to same CSV with fold column
├── physics/
│   ├── sim_00_debugs_fold0.txt
│   ├── sim_00_debugs_fold1.txt
│   └── ...
└── tensorboard_logs/
    ├── sim_00_fold_0/
    ├── sim_00_fold_1/
    └── ...
```

This prefix makes it easy to organize results when running k-fold training on multiple datasets.

**Metrics CSV format:**
The metrics CSV file includes a `fold` column and `epoch` column to track which fold and epoch each row corresponds to:
```csv
fold,epoch,learning_rate,train_loss,train_mse,...,val_loss,val_mse,...
0,1,0.003,0.042,0.0015,...,0.045,0.0018,...
0,2,0.003,0.038,0.0012,...,0.041,0.0015,...
1,1,0.003,0.044,0.0016,...,0.046,0.0019,...
...
```

## Directory Structure

```
dev/
├── train_gnn.py                    # Main training script
├── data/
│   ├── dataset_loader.py          # Data loading with k-fold support
│   └── __init__.py
├── model/
│   ├── model.py                   # GNN architecture
│   ├── config.py                  # Model configuration
│   ├── physics.py                 # Physics-based loss functions
│   └── utils.py                   # Model utilities
├── training/
│   ├── cli.py                     # Command-line argument parsing
│   ├── config.py                  # Training configuration
│   ├── setup.py                   # Training setup (model, optimizer, etc.)
│   ├── train.py                   # Training loop
│   ├── utils.py                   # Training utilities
│   └── logging.py                 # TensorBoard logging
└── README.md                      # This file
```

## Data Loading

The `dataset_loader.py` module provides:

1. **`load_phloem_data()`** - Traditional loading (mixes files)
   ```python
   train_loader, val_loader, test_loader = load_phloem_data(
       h5_path="./data/",
       batch_size=8,
       train_ratio=0.8,
       val_ratio=0.1,
       split_method="random"
   )
   ```

2. **`load_phloem_data_kfold()`** - K-fold loading (keeps files separate)
   ```python
   folds = load_phloem_data_kfold(
       h5_dir="./data/",
       batch_size=8
   )
   # Returns list of (train_loader, val_loader, test_loader) tuples
   # Number of folds = number of .h5 files in directory
   print(f"Created {len(folds)} folds from {len(folds)} .h5 files")
   ```

## Requirements

### For K-Fold:
- Minimum 3 `.h5` files in directory
- Number of folds is automatically set to the number of files
- All files should be from same plant type (same structure, different runs)

### For Traditional:
- Can use single file or directory
- `train_ratio + val_ratio < 1`

## Tips

1. **Use k-fold for final experiments** - Better for publication with exhaustive evaluation
2. **Set `--seed` for reproducibility**
3. **Enable `--enable-physics-logging`** when debugging physics
4. **View results in TensorBoard**: `tensorboard --logdir=results/tensorboard_logs`
5. **All folds trained automatically** - K-fold will train all folds sequentially

## TensorBoard

View training progress:

```bash
# Traditional mode
tensorboard --logdir=results/tensorboard_logs

# K-fold mode (view all folds together)
tensorboard --logdir=results/tensorboard_logs

# K-fold mode (specific fold with prefix, e.g., sim_00)
tensorboard --logdir=results/tensorboard_logs/sim_00_fold_0
```

## Troubleshooting

**"Need at least 3 files for k-fold CV"**
- Add more simulation files, or use traditional mode with `--train-ratio` instead

**Models have dataset prefix in filename**
- This is intentional! The prefix (e.g., `sim_00`) comes from your data directory name
- Makes it easy to organize results from multiple datasets

**Out of memory**
- Reduce `--batch-size`
- Reduce graph complexity in data generation

**K-fold takes too long**
- K-fold trains all folds sequentially, so it takes N times longer than traditional training
- This is expected and necessary for proper cross-validation

## Summary

| Mode | Command | Use Case |
|------|---------|----------|
| Traditional | `python train_gnn.py --data-path ./data/` | Single file, quick tests |
| K-Fold | `python train_gnn.py --data-path ./data/ --use-kfold` | Multiple files, exhaustive cross-validation |

---

**For more details on the implementation, see the source code in `training/` and `data/` directories.**

---

## Performance Benchmarking

### Overview

The benchmarking suite measures inference time to compare CPlantBox phloem solver with the GNN model. All output files follow a consistent naming convention with data prefixes (e.g., `sim_08_*`) for easy organization across multiple simulations.

### Quick Start Guide

#### 1. Run CPlantBox Simulation (timing automatic)

```bash
cd dev/cplantbox
python3 sim_phloem_flow.py --phloem-dir data/sim_08 --seed 42
```

**Output**:
- `data/sim_08/phloem_simulation.h5` - Full simulation data
- `data/sim_08/phloem_timing.csv` - Timing per step with graph size info

#### 2. Benchmark GNN Model

```bash
cd dev
python3 benchmark_gnn.py \
    --model-path logs/model/sim_08_best_model.pt \
    --data-path cplantbox/data/sim_08/phloem_simulation.h5 \
    --output-dir benchmarks
```

**Output** (automatically prefixed with `sim_08`):
- `benchmarks/sim_08_gnn_latency.csv` - Per-graph latency (batch_size=1)
- `benchmarks/sim_08_gnn_throughput_bs4.csv` - Throughput at batch_size=4
- `benchmarks/sim_08_benchmark_summary.csv` - Consolidated summary

#### 3. Compare Performance

```bash
python3 compare_performance.py \
    --cplantbox-timing cplantbox/data/sim_08/phloem_timing.csv \
    --gnn-timing benchmarks/sim_08_gnn_latency.csv \
    --output-dir comparison \
    --plot
```

**Output** (auto-detects prefix from input paths):
- `comparison/sim_08_summary.csv` - Performance summary
- `comparison/sim_08_performance_comparison.png` - Visualization plots

### File Naming Convention

All output files use consistent data prefix naming for easy organization:

**Training/Metrics:**
```
logs/
├── model/sim_08_best_model.pt
├── metrics/
│   ├── sim_08_metrics.csv          # Epoch-level metrics
│   └── sim_08_batch_metrics.csv    # Batch-level metrics
└── tensorboard/sim_08/
```

**Benchmarks:**
```
benchmarks/
├── sim_08_gnn_latency.csv          # Per-graph latency (bs=1)
├── sim_08_gnn_throughput_bs4.csv   # Throughput at batch_size=4
├── sim_08_gnn_throughput_bs8.csv   # Throughput at batch_size=8
└── sim_08_benchmark_summary.csv    # Consolidated results
```

**Comparisons:**
```
comparison/
├── sim_08_summary.csv              # Performance comparison table
└── sim_08_performance_comparison.png  # Visualization plots
```

### Advanced Benchmarking

#### Multi-Batch-Size Benchmarking

Test both latency (batch_size=1) and throughput (larger batches):

```bash
python3 benchmark_gnn.py \
    --model-path logs/model/sim_08_best_model.pt \
    --data-path cplantbox/data/sim_08/phloem_simulation.h5 \
    --batch-sizes 1 2 4 8 16 \
    --output-dir benchmarks
```

**Output:**
- Latency metrics for batch_size=1
- Throughput metrics for batch_size=2, 4, 8, 16
- Speedup analysis relative to batch_size=1

#### Custom Data Prefix

Override auto-detection of data prefix:

```bash
python3 compare_performance.py \
    --cplantbox-timing cplantbox/data/sim_08/phloem_timing.csv \
    --gnn-timing benchmarks/sim_08_gnn_latency.csv \
    --data-prefix my_experiment \
    --output-dir comparison
```

### Benchmark Options

#### GNN Benchmark (`benchmark_gnn.py`)

```bash
python3 benchmark_gnn.py \
    --model-path <path_to_model.pt> \
    --data-path <path_to_h5_file_or_dir> \
    --output-dir benchmarks \
    --batch-sizes 1 4 8 16 \     # Multiple batch sizes for latency + throughput
    --warmup 5 \                  # Warmup iterations (default: 5)
    --cpu                         # Force CPU (for CPU vs GPU comparison)
```

**Key Features:**
- **Latency measurement**: Uses batch_size=1 for per-graph timing
- **Throughput measurement**: Uses larger batch sizes for batched processing efficiency
- **GPU synchronization**: Automatic CUDA synchronization for accurate GPU timing
- **Same code path**: Uses `benchmark_model_inference()` from `train.py` (same as evaluation)
- **Auto-prefix**: Extracts data prefix from `--data-path` (e.g., `sim_08` from `data/sim_08/`)

#### Performance Comparison (`compare_performance.py`)

```bash
python3 compare_performance.py \
    --cplantbox-timing <path_to_csv> \
    --gnn-timing <path_to_csv> \
    --output-dir comparison \
    --data-prefix <prefix> \      # Optional, auto-detects from paths
    --plot                        # Generate visualization plots
```

**Key Features:**
- **Graph size normalization**: Time per node/edge for fair comparison
- **Auto-prefix detection**: Extracts prefix from parent directory name
- **CSV-only workflow**: No HDF5 dependency (graph sizes in timing CSV)
- **Speedup analysis**: Absolute and normalized speedup metrics

### Output Files

**CPlantBox Timing** (`phloem_timing.csv`):
```csv
step,time_s,nodes,edges
0,0.156,245,312
1,0.154,245,312
...
```

**GNN Latency** (`sim_08_gnn_latency.csv`):
```csv
batch_idx,time_ms,nodes,edges
0,5.123,245.3,312.7
1,5.087,243.1,310.2
...
```

**GNN Throughput** (`sim_08_gnn_throughput_bs4.csv`):
```csv
batch_idx,time_ms,batch_size,graphs_in_batch,nodes,edges
0,18.234,4,4,245.3,312.7
1,17.987,4,4,243.1,310.2
...
```

**Benchmark Summary** (`sim_08_benchmark_summary.csv`):
```csv
batch_size,latency_mean_ms,latency_median_ms,throughput_graphs_per_sec,...
1,5.12,5.08,195.3,...
4,4.51,4.48,221.7,...
8,4.23,4.21,237.1,...
```

**Comparison Summary** (`sim_08_summary.csv`):
```csv
Method,Mean_ms,Median_ms,Std_ms,Min_ms,Max_ms
CPlantBox,156.23,154.87,3.21,152.45,162.11
GNN,5.12,5.08,0.15,4.95,5.42
```

### Example Output

```
================================================================================
PERFORMANCE COMPARISON: CPlantBox vs GNN
================================================================================

CPlantBox Phloem Solver:
----------------------------------------
  Total steps:              12
  Mean time per step:       156.23 ms
  Median time per step:     154.87 ms

  Mean nodes:               245.3
  Mean edges:               312.7
  Time per node:            0.637 ms
  Time per edge:            0.500 ms

GNN Model:
----------------------------------------
  Total graphs:             12
  Mean time per graph:      5.12 ms
  Median time per graph:    5.08 ms

  Mean nodes:               245.3
  Mean edges:               312.7
  Time per node:            0.021 ms
  Time per edge:            0.016 ms

Speedup Analysis:
----------------------------------------
  GNN is 30.5x faster (mean time)
  GNN is 30.2x faster (median time)
  GNN is 30.2x faster (per node)
  GNN is 31.1x faster (per edge)

================================================================================
```

### Multi-Batch-Size Benchmark Output

```
======================================================================
Testing with batch_size = 1
======================================================================
Mode: LATENCY (per-graph timing)
Benchmarking: 100%|████████████| 100/100 [00:30<00:00,  3.33it/s]

Latency Results:
  Mean:    5.123 ms
  Median:  5.087 ms
  Std:     0.152 ms

======================================================================
Testing with batch_size = 4
======================================================================
Mode: THROUGHPUT (batched processing)
Benchmarking: 100%|████████████| 25/25 [00:10<00:00,  2.50it/s]

Throughput Results:
  Graphs/sec:        221.7
  Time per graph:    4.51 ms (speedup: 1.14x vs BS=1)

...

BENCHMARK SUMMARY
======================================================================

Throughput (batched processing):
  Batch Size   Graphs/sec      Time/graph (ms)    Speedup vs BS=1
----------------------------------------------------------------------
  1            195.30          5.123              1.00x
  4            221.73          4.510              1.14x
  8            237.12          4.217              1.21x
  16           243.89          4.100              1.25x

Output directory: benchmarks
  - sim_08_gnn_latency.csv           (per-graph latency, batch_size=1)
  - sim_08_gnn_throughput_bs*.csv    (throughput for each batch size)
  - sim_08_benchmark_summary.csv     (consolidated results)
======================================================================
```

### Why Timing in train.py?

The GNN benchmark uses `benchmark_model_inference()` from `train.py`, which:

1. **Measures the same code path** used during training/evaluation
2. **Reuses existing infrastructure** (no code duplication)
3. **Automatically stays in sync** with code changes
4. **More accurate** - captures real-world performance including data loading

This is superior to creating a separate benchmark script because any changes to the model automatically flow through to the benchmark.

### Understanding Results

**Key Metrics:**

1. **Latency** (batch_size=1): Time to process a single graph
   - Best for real-time/interactive applications
   - Shows per-graph overhead

2. **Throughput** (batch_size>1): Graphs processed per second
   - Best for batch processing applications
   - Shows efficiency gains from batching

3. **Normalized metrics**: Time per node/edge
   - Fair comparison regardless of graph size
   - Accounts for structural complexity

4. **Speedup factor**: CPlantBox_time / GNN_time
   - Both absolute and normalized
   - Quantifies performance improvement

### Tips for Fair Comparison

1. **Use same simulation data**: Run CPlantBox first, then benchmark GNN on saved HDF5
2. **Consider graph size**: Always compare normalized metrics (per-node/per-edge)
3. **Multiple runs**: Run benchmarks several times to account for variance
4. **Warmup included**: Scripts include warmup iterations to avoid cold-start bias
5. **Device matters**: Note whether GNN runs on CPU or GPU in comparisons
6. **Batch size**: Use batch_size=1 for latency, larger for throughput comparisons
7. **Data prefix**: Consistent naming makes organizing multi-simulation results easy

### Metrics Logging

Training and evaluation automatically log detailed metrics:

**Epoch-Level Metrics** (`sim_08_metrics.csv`):
- Loss components (MSE, physics residual, IC, BC)
- Physics metrics (S_ST RMSE, J_ax MSE, divJ correlation, etc.)
- Learning rate, epoch time

**Batch-Level Metrics** (`sim_08_batch_metrics.csv`):
- Per-batch physics residuals
- Includes epoch and batch_idx for traceability

Example epoch metrics:
```csv
epoch,learning_rate,train_loss,train_mse,train_S_ST_rmse,train_J_ax_mse,val_divJ_correlation,...
1,0.003,0.042,0.0015,0.0123,0.0089,0.987,...
2,0.003,0.038,0.0012,0.0098,0.0067,0.991,...
```

### Troubleshooting

**Missing graph size data:**
- `sim_phloem_flow.py` automatically saves nodes/edges in `phloem_timing.csv`
- No need for separate HDF5 file in comparison

**Out of memory (GPU):**
- Use `--cpu` flag for GNN benchmark
- Or reduce batch size with `--batch-sizes 1 2 4`

**Model loading errors:**
- Ensure model checkpoint matches the current `ModelConfig`
- Check model was trained with compatible hyperparameters
- Verify scaler objects are saved in checkpoint

**Inconsistent file naming:**
- All files should have data prefix (e.g., `sim_08_*`)
- If missing, check `--data-path` argument format
- Prefix extracted from parent directory name automatically

**Batch throughput slower than expected:**
- Check GPU utilization (`nvidia-smi`)
- Try different batch sizes to find optimal
- Ensure CUDA is available (`torch.cuda.is_available()`)
