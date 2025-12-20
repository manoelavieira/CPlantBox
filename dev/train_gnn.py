"""
Training script for the phloem GNN model

Supports two training modes:
1. Traditional: Single train/val/test split (may mix files)
2. K-Fold: Cross-validation keeping simulation files separate
"""
import numpy as np

import torch
import torch_geometric
import argparse
import sys

from model.config import ModelConfig
from model import physics

import training.cli as cli
import training.logging as logging
import training.setup as setup
import training.utils as utils
import training.train as train

from training.config import TrainingConfig, LossType
from pathlib import Path


def train_single_fold(config, fold_idx=None, total_folds=None):
    """Train model on a single fold or traditional split.

    Args:
        config: Training configuration
        fold_idx: Fold index for logging (None for traditional mode)
        total_folds: Total number of folds (only used for display)

    Returns:
        Dictionary with training results
    """
    # Configure physics logging
    physics.set_physics_logging(
        enable=config.enable_physics_logging,
        log_path=config.physics_save_path
    )

    # Setup environment
    device = setup.setup_environment(config)

    if fold_idx is None:
        print(f"\n{'='*70}")
        print("Training in Traditional Mode")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        if total_folds is not None:
            print(f"Training Fold {fold_idx + 1}/{total_folds}")
        else:
            print(f"Training Fold {fold_idx}")
        print(f"{'='*70}")

    print(f"Using torch {torch.__version__}, torch_geometric {torch_geometric.__version__}")

    # Create TensorBoard writer
    writer = logging.create_tensorboard_writer(config)

    # Get data loaders
    train_loader, val_loader, test_loader = utils.get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, "
          f"Validation batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    # Setup model
    model_setup = setup.setup_model_and_scalers(train_loader, device, config.model_type)
    model_setup.model = model_setup.model.double()  # Convert to float64

    # Create model config for logging
    model_cfg = ModelConfig()

    # Log hyperparameters to TensorBoard
    logging.log_hyperparameters(writer, config, model_cfg)

    # Setup training components
    optimizer, scheduler = setup.setup_training_components(model_setup.model, config)

    # Run training loop
    training_state = train.train_model(
        model_setup, train_loader, val_loader, optimizer, scheduler,
        writer, config, model_cfg, fold_idx
    )

    # Run final evaluation and reporting
    test_metrics = train.test_model(model_setup, test_loader, writer, training_state, config)

    # Close TensorBoard writer
    writer.close()

    if fold_idx is None:
        print(f"\nTensorBoard logs saved. To view: tensorboard --logdir={config.tensorboard_log_dir}")
    else:
        print(f"\nFold {fold_idx} complete. TensorBoard logs: {config.tensorboard_log_dir}")

    return {
        'fold_idx': fold_idx,
        'test_metrics': test_metrics,
        'val_loss': training_state.best_val_loss if hasattr(training_state, 'best_val_loss') else 0.0,
        'best_epoch': training_state.best_epoch if hasattr(training_state, 'best_epoch') else 0,
        'model_path': config.model_save_path
    }


def train_all_folds(base_config):
    """Train models on all k-folds and aggregate results.

    The number of folds is automatically determined by the number of .h5 files
    in the data directory.

    Args:
        base_config: Base training configuration

    Returns:
        List of results from each fold
    """
    # Determine number of folds from number of files
    data_path = Path(base_config.data_path)
    h5_files = list(data_path.glob('**/*.h5'))
    n_folds = len(h5_files)

    if n_folds < 3:
        raise ValueError(f"Need at least 3 .h5 files for k-fold CV (found {n_folds} in {base_config.data_path})")

    print(f"\n{'='*70}")
    print(f"K-Fold Cross-Validation Training ({n_folds} folds)")
    print(f"Auto-determined from {n_folds} .h5 files in {base_config.data_path}")
    print(f"{'='*70}")
    print(f"Data path: {base_config.data_path}")
    print(f"Epochs per fold: {base_config.epochs}")
    print(f"{'='*70}\n")

    all_results = []

    # Get prefix from data path
    data_prefix = base_config.get_data_prefix()

    for fold_idx in range(n_folds):
        # Create config for this fold
        fold_config = TrainingConfig(
            data_path=base_config.data_path,
            batch_size=base_config.batch_size,
            use_kfold=True,
            current_fold=fold_idx,
            split_method=base_config.split_method,
            model_type=base_config.model_type,
            use_analytical_residual=base_config.use_analytical_residual,
            lr=base_config.lr,
            weight_decay=base_config.weight_decay,
            epochs=base_config.epochs,
            patience=base_config.patience,
            loss_type=base_config.loss_type,
            lambda_data=base_config.lambda_data,
            lambda_phys=base_config.lambda_phys,
            lambda_ic=base_config.lambda_ic,
            lambda_bc=base_config.lambda_bc,
            use_adaptive_physics_weighting=base_config.use_adaptive_physics_weighting,
            target_physics_ratio=base_config.target_physics_ratio,
            seed=base_config.seed,
            scheduler_factor=base_config.scheduler_factor,
            scheduler_patience=base_config.scheduler_patience,
            clip_grad_norm=base_config.clip_grad_norm,
            model_save_dir=base_config.model_save_dir,
            model_filename=f"{data_prefix}_best_model_fold{fold_idx}.pt",
            physics_save_dir=base_config.physics_save_dir,
            physics_save_filename=f"{data_prefix}_debugs_fold{fold_idx}.txt",
            metrics_save_dir=base_config.metrics_save_dir,
            metrics_save_filename=f"{data_prefix}_metrics.csv",  # All folds write to same CSV
            tensorboard_log_dir=f"{base_config.tensorboard_log_dir}/{data_prefix}_fold_{fold_idx}",
            enable_physics_logging=base_config.enable_physics_logging,
            enable_metrics_logging=base_config.enable_metrics_logging
        )

        # Train this fold
        fold_results = train_single_fold(fold_config, fold_idx, total_folds=n_folds)
        all_results.append(fold_results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"K-Fold Cross-Validation Results Summary")
    print(f"{'='*70}")
    print(f"\n{'Fold':<6} {'Val Loss':<12} {'Model Path'}")
    print('-'*70)

    for result in all_results:
        fold_idx = result['fold_idx']
        val_loss = result['val_loss']
        model_path = result['model_path']
        print(f"{fold_idx:<6} {val_loss:<12.4e} {model_path}")

    print('-'*70)

    # Compute statistics
    val_losses = [r['val_loss'] for r in all_results]
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    print(f"\nAggregated Validation Results:")
    print(f"  Mean Val Loss: {mean_val_loss:.4e} ± {std_val_loss:.4e}")
    print(f"{'='*70}\n")

    print(f"All fold TensorBoard logs: {base_config.tensorboard_log_dir}/{data_prefix}_fold_*")
    print(f"To view: tensorboard --logdir={base_config.tensorboard_log_dir}")

    return all_results


def main():
    """Main training function."""
    # Parse arguments and create configuration
    config = cli.parse_arguments()

    if config.use_kfold:
        # K-fold cross-validation mode - train all folds
        all_results = train_all_folds(config)
    else:
        # Traditional mode (single split)
        results = train_single_fold(config, fold_idx=None)


if __name__ == '__main__':
    main()
