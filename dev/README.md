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
