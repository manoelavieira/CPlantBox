"""
Training script for the phloem GNN model
"""
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

from enum import Enum
from typing import Tuple, Optional
from pathlib import Path
from datetime import datetime

from dataset_loader import load_phloem_data
from dataset_dummy import DummyTemporalDataset
from gnn import PhloemNNConv, ModelConfig, Standardizer, physics_residual


class DatasetType(Enum):
    DUMMY = 'dummy'
    SIMULATED = 'simulated'

def collate_graphs(batch):
    return Batch.from_data_list(batch)

def create_tensorboard_writer(args: argparse.Namespace) -> SummaryWriter:
    """Create TensorBoard writer with organized logging directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dataset_name = args.dataset

    # Create descriptive experiment name
    # exp_name = (f"{dataset_name}_lr{args.lr}_bs{args.batch_size}_"
    #             f"wd{args.weight_decay}_seed{args.seed}_{timestamp}")
    exp_name = timestamp

    log_dir = Path("tensorboard_logs") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={log_dir.parent}")

    return writer

def log_hyperparameters(writer: SummaryWriter, args: argparse.Namespace, model_cfg: ModelConfig):
    """Log hyperparameters to TensorBoard."""
    hparams = {
        # Training hyperparameters
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'patience': args.patience,
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,

        # Model architecture
        'hidden_size': model_cfg.hidden_size,
        'num_layers': model_cfg.num_layers,
        'edge_feat_dim': model_cfg.edge_feat_dim,
        'node_feat_dim': model_cfg.node_feat_dim,
        'dropout': model_cfg.dropout,
    }

    if hasattr(args, 'data_path') and args.data_path:
        hparams['data_path'] = args.data_path

    metrics = {}

    writer.add_hparams(hparams, metrics)




def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        clip_grad_norm: float = 1.0
    ) -> Tuple[float, float, float, float]:
    """Train model for one epoch.

    Args:
        model: The neural network model
        loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        clip_grad_norm: Maximum norm for gradient clipping

    Returns:
        Tuple of (average_loss, average_mae, average_mse, average_physics)

    Raises:
        RuntimeError: If no training samples are processed
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_physics = 0.0
    n_batches = 0

    for batch_idx, data in enumerate(loader):
        data = data.to(next(model.parameters()).device)
        optimizer.zero_grad(set_to_none=True)

        # Keep original features for physics computation
        x_orig = data.node_feat.clone()

        # Standardize features for the model
        data.node_feat = model.feature_scaler.transform(data.node_feat)

        # Forward pass returns predictions in standardized space
        pred = model(data) # [N,1]

        y = data.y # [N,1]
        y_t = model.target_scaler.transform(y)  # Transform targets for loss computation

        # MSE in standardized space (mean over nodes in batch)
        mse = F.mse_loss(pred, y_t, reduction='mean')

        # Physics computation in original space
        pred_orig = model.target_scaler.inv_transform(pred)

        # Temporarily restore original features for physics computation
        data.node_feat = x_orig
        phys = physics_residual(pred_orig, data)  # already a mean over nodes/graphs
        phys_scalar = float(phys if phys.dim() == 0 else phys.mean())

        # Restore standardized features for next iteration
        data.node_feat = model.feature_scaler.transform(x_orig)

        # Combine with explicit physics weight
        loss = mse + getattr(model, "lambda_phys", 1.0) * phys_scalar
        loss.backward()

        # Log gradient norms before clipping
        if writer is not None and batch_idx == 0:  # Log only first batch to avoid clutter
            total_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
                    writer.add_scalar(f'Gradients/{name}', param_grad_norm,
                                    epoch * len(loader) + batch_idx)

            total_grad_norm = total_grad_norm ** 0.5
            writer.add_scalar('Gradients/total_norm', total_grad_norm,
                            epoch * len(loader) + batch_idx)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        with torch.no_grad():
            # Report MAE in original units using model's scaler
            pred_un = model.target_scaler.inv_transform(pred) if hasattr(model, 'target_scaler') and model.target_scaler is not None else pred

            # Track means per batch; we'll average by number of batches
            mae = (pred_un - y).abs().mean()
            total_mae += mae.item()
            total_mse += mse.item()
            total_physics += phys_scalar
            total_loss += (mse + getattr(model, "lambda_phys", 1.0) * phys_scalar).item()
            n_batches += 1

            # Log batch-level metrics to TensorBoard (every 10 batches to avoid clutter)
            if writer is not None and batch_idx % 10 == 0:
                step = epoch * len(loader) + batch_idx
                writer.add_scalar('Training/Batch_Loss', (mse + getattr(model, "lambda_phys", 1.0)*phys_scalar).item(), step)
                writer.add_scalar('Training/Batch_MSE', mse.item(), step)
                writer.add_scalar('Training/Batch_MAE', mae.item(), step)
                writer.add_scalar('Training/Batch_Physics', phys_scalar, step)

    if n_batches == 0:
        raise RuntimeError("No training samples this epoch.")

    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    avg_mse = total_mse / n_batches
    avg_physics = total_physics / n_batches

    return avg_loss, avg_mae, avg_mse, avg_physics

def evaluate(
        model: nn.Module,
        loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        phase: str = 'val'
    ) -> Tuple[float, float, float, float]:
    """Evaluate model on a dataset.

    Args:
        model: The neural network model
        loader: DataLoader containing evaluation data
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        phase: Phase name ('val' or 'test')

    Returns:
        Tuple of (average_loss, average_mse, average_mae, average_physics)
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_physics = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(next(model.parameters()).device)

            # Keep original features
            x_orig = data.node_feat.clone()

            # Standardize features for the model
            data.node_feat = model.feature_scaler.transform(data.node_feat)

            # Forward pass returns predictions in standardized space
            pred = model(data)

            y = data.y
            y_t = model.target_scaler.transform(y)
            mse = F.mse_loss(pred, y_t, reduction='mean')

            # Transform predictions back for MAE in original space
            pred_un = model.target_scaler.inv_transform(pred)

            # Compute physics residual for validation/test
            phys_val_scalar = 0.0
            try:
                if hasattr(data, 'time_node') and data.time_node is not None:
                    with torch.enable_grad():
                        time_node_grad = data.time_node.clone().requires_grad_(True)
                        data_with_grad = data.clone() if hasattr(data, 'clone') else data
                        data_with_grad.time_node = time_node_grad
                        data_with_grad.node_feat = model.feature_scaler.transform(x_orig)
                        pred_for_physics = model(data_with_grad)
                        pred_orig_for_physics = model.target_scaler.inv_transform(pred_for_physics)
                        data_with_grad.node_feat = x_orig
                        phys_val = physics_residual(pred_orig_for_physics, data_with_grad)
                        phys_val_scalar = float(phys_val if phys_val.dim() == 0 else phys_val.mean())
            except Exception:
                phys_val_scalar = 0.0

            mae = (pred_un - y).abs().mean()
            loss = mse + getattr(model, "lambda_phys", 1.0) * phys_val_scalar

            total_loss += float(loss)
            total_mse += float(mse)
            total_mae += float(mae)
            total_physics += phys_val_scalar
            n_batches += 1

            # Log distribution of predictions and targets (first batch only)
            if writer is not None and batch_idx == 0 and epoch % 5 == 0:
                writer.add_histogram(f'{phase}/Predictions', pred_un.cpu(), epoch)
                writer.add_histogram(f'{phase}/Targets', y.cpu(), epoch)
                writer.add_histogram(f'{phase}/Residuals', (pred_un - y).cpu(), epoch)

                # Log loss values for debugging
                writer.add_scalar(f'{phase}/MSE', float(mse), epoch)
                writer.add_scalar(f'{phase}/Physics', phys_val_scalar, epoch)
                writer.add_scalar(f'{phase}/Loss', float(loss), epoch)

    # Compute averages per batch
    denom = n_batches if n_batches > 0 else 1
    avg_loss = total_loss / denom
    avg_mse = total_mse / denom
    avg_mae = total_mae / denom
    avg_physics = total_physics / denom

    # Clear GPU memory after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, avg_mse, avg_mae, avg_physics

def validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    """Validate that dataset split ratios are valid.

    Args:
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation

    Raises:
        ValueError: If ratios are invalid
    """
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError(
            f"Sum of train_ratio ({train_ratio}) and val_ratio ({val_ratio}) "
            f"must be less than 1 to leave data for testing"
        )

def get_dataloaders(dataset_type: DatasetType, args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, validation, and test dataloaders based on dataset type.

    Args:
        dataset_type: Type of dataset to load
        args: Command line arguments containing dataset parameters

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        ValueError: If dataset parameters are invalid
    """
    # Validate split ratios
    validate_split_ratios(args.train_ratio, args.val_ratio)
    if dataset_type == DatasetType.DUMMY:
        # Create dummy dataset
        # ds = DummyTemporalDataset(n_graphs=args.n_graphs)
        ds = DummyTemporalDataset()
        print(f"\nCreated dummy dataset with {len(ds)} graphs")

        # Split dataset
        n_train = int(args.train_ratio * len(ds))
        n_val = int(args.val_ratio * len(ds))
        n_test = len(ds) - n_train - n_val

        generator = torch.Generator().manual_seed(args.seed)
        train_set, val_set, test_set = torch.utils.data.random_split(
            ds, [n_train, n_val, n_test], generator=generator)

        # Create dataloaders
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size,
            shuffle=True, collate_fn=collate_graphs
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_graphs
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_graphs
        )

    elif dataset_type == DatasetType.SIMULATED:
        # Load simulation data
        train_loader, val_loader, test_loader = load_phloem_data(
            h5_path=args.data_path,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_seed=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return train_loader, val_loader, test_loader

def print_model_summary(model: nn.Module, writer: Optional[SummaryWriter] = None):
    """Print model architecture summary and log to TensorBoard."""
    print("\nModel Architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("\nLayer Overview:")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")

    # Log model parameters to TensorBoard
    if writer is not None:
        writer.add_text('Model/Architecture',
                       f"Total: {total_params:,}, Trainable: {trainable_params:,}")
        writer.add_scalar('Model/Total_Parameters', total_params, 0)
        writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)

def log_experiment_config(args: argparse.Namespace):
    """Log experiment configuration."""
    print("\nExperiment Configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train phloem GNN model")

    parser.add_argument('--dataset', type=str, choices=['dummy', 'simulated'],
                       default='dummy', help='Dataset type to use')
    parser.add_argument('--data-path', type=str,
                       help='Path to H5 file for simulated data')
    parser.add_argument('--n-graphs', type=int, default=80,
                       help='Number of graphs for dummy dataset')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Ratio of data to use for validation')
    parser.add_argument('--lr', type=float, default=3e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs to train')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--lambda-phys', type=float, default=1.0,
                        help='Weight for physics loss term (L = MSE + lambda_phys * Physics)')
    args = parser.parse_args()

    # Validate arguments
    if args.dataset == 'simulated' and args.data_path is None:
        parser.error("--data-path is required when using --dataset simulated")

    # Set random seeds for full reproducibility
    random.seed(args.seed) # Python's random
    np.random.seed(args.seed) # NumPy
    torch.manual_seed(args.seed) # PyTorch on CPU
    torch.cuda.manual_seed(args.seed) # PyTorch on Current GPU
    torch.cuda.manual_seed_all(args.seed) # PyTorch on All GPUs

    print(f"Using torch {torch.__version__}, torch_geometric {torch_geometric.__version__}")

    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Log experiment configuration
    log_experiment_config(args)

    # Create TensorBoard writer
    writer = create_tensorboard_writer(args)

    model_cfg = ModelConfig()
    model = PhloemNNConv(model_cfg).to(device)
    # Expose physics weight on the model for easy access in training/eval
    model.lambda_phys = args.lambda_phys

    # Log hyperparameters to TensorBoard
    log_hyperparameters(writer, args, model_cfg)

    # Print detailed model summary
    print_model_summary(model, writer)

    # Get data loaders
    dataset_type = DatasetType[args.dataset.upper()]
    train_loader, val_loader, test_loader = get_dataloaders(dataset_type, args)
    print(f"Train batches: {len(train_loader)}, "
          f"Validation batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    # Setup standardization on training data
    feature_scaler = Standardizer() # for input node features (psi, vol)
    target_scaler = Standardizer() # for targets (y)
    time_scaler = Standardizer() # for graph-level time (scalar)

    # Fit scalers on training data
    with torch.no_grad():
        x_list, y_list, t_list = [], [], []

        for batch in train_loader:
            x_list.append(batch.node_feat[:, :model_cfg.node_feat_dim])
            y_list.append(batch.y)
            if hasattr(batch, 'time'):
                t_list.append(batch.time.view(-1, 1)) # collect per-graph scalars
            else:
                raise ValueError("Each Data must carry a graph-level `time` tensor.")

        Xs = torch.cat(x_list, dim=0) # [sum_N, 2]
        Ys = torch.cat(y_list, dim=0) # [sum_N, 1]
        Ts = torch.cat(t_list, dim=0) # [sum_B, 1], one per graph

        feature_scaler.fit(Xs)
        target_scaler.fit(Ys)
        time_scaler.fit(Ts)

    # Add scalers to the model
    model.feature_scaler = feature_scaler
    model.target_scaler = target_scaler
    model.time_scaler = time_scaler

    # Ensure scalers live on the same device as model (explicit)
    model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop with early stopping
    best_val = float('inf')
    patience_counter = 0
    best_epoch = 0

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Training
        tr_loss, tr_mae, tr_mse, tr_physics = train_one_epoch(
            model, train_loader, optimizer, writer, epoch)

        # Validation
        val_loss, val_mse, val_mae, val_physics = evaluate(
            model, val_loader, writer, epoch, phase='val')

        # Learning rate scheduling (use combined validation loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train_Total', tr_loss, epoch)
        writer.add_scalar('Loss/Train_MSE', tr_mse, epoch)
        writer.add_scalar('Loss/Train_Physics', tr_physics, epoch)

        writer.add_scalar('Metrics/Val_Total', val_loss, epoch)
        writer.add_scalar('Metrics/Val_MSE', val_mse, epoch)
        writer.add_scalar('Metrics/Val_Physics', val_physics, epoch)

        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Log combined metrics for easy comparison
        writer.add_scalars('MAE_Comparison', {
            'Train': tr_mae,
            'Validation': val_mae
        }, epoch)

        writer.add_scalars('Loss_Comparison', {
            'Train_Total': tr_loss,
            'Val_Total': val_loss
        }, epoch)

        writer.add_scalars('Loss_Components', {
            'MSE': tr_mse,
            'Physics': tr_physics,
            'Total': tr_loss
        }, epoch)

        # Logging
        print(f"Epoch {epoch:03d} | "
              f"train_loss={tr_loss:.4f} train_MSE={tr_mse:.4f} train_physics={tr_physics:.4f} | "
              f"val_loss={val_loss:.4f} val_MSE={val_mse:.4f} val_physics={val_physics:.4f} | "
              f"lr={current_lr:.2e}")

        # Model saving and early stopping (use combined validation loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Log best model achievement
            writer.add_scalar('Best_Model/Epoch', epoch, epoch)
            writer.add_scalar('Best_Model/Val_Loss', val_loss, epoch)
            writer.add_scalar('Best_Model/Val_MSE', val_mse, epoch)

            # Save model and scalers
            feature_scaler_state = {
                'mean': feature_scaler.mean,
                'std': feature_scaler.std,
                'device': str(feature_scaler.device)
            }
            target_scaler_state = {
                'mean': target_scaler.mean,
                'std': target_scaler.std,
                'device': str(target_scaler.device)
            }
            time_scaler_state = {
                'mean': time_scaler.mean,
                'std': time_scaler.std,
                'device': str(time_scaler.device)
            }

            torch.save({
                'epoch': epoch,
                'cfg': model_cfg.__dict__,
                'state_dict': model.state_dict(),
                'device': device.type,  # Save source device info
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_mse': val_mse,
                'feature_scaler': feature_scaler_state,
                'target_scaler': target_scaler_state,
                'time_scaler': time_scaler_state,
            }, 'model/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best validation loss: {best_val:.4f} "
                      f"at epoch {best_epoch}")

                # Log early stopping
                writer.add_text('Training/Early_Stopping',
                                f"Stopped at epoch {epoch}, best at {best_epoch}")
                break

    print("\nTraining completed!")

    # Log final training summary
    writer.add_text('Training/Summary',
                    f"Training completed. Best validation loss: {best_val:.4f} at epoch {best_epoch}")

    # Load the best model for testing
    try:
        best_checkpoint = torch.load('model/best_model.pt', map_location=device)

        # Load model state
        model.load_state_dict(best_checkpoint['state_dict'])

        # Reconstruct scalers from saved state
        feature_scaler = Standardizer()
        feature_scaler.mean = best_checkpoint['feature_scaler']['mean']
        feature_scaler.std = best_checkpoint['feature_scaler']['std']
        feature_scaler.device = device

        target_scaler = Standardizer()
        target_scaler.mean = best_checkpoint['target_scaler']['mean']
        target_scaler.std = best_checkpoint['target_scaler']['std']
        target_scaler.device = device

        time_scaler = Standardizer()
        time_scaler.mean = best_checkpoint['time_scaler']['mean']
        time_scaler.std  = best_checkpoint['time_scaler']['std']
        time_scaler.device = device

        # Assign scalers to model
        model.feature_scaler = feature_scaler
        model.target_scaler = target_scaler
        model.time_scaler   = time_scaler

        print(f"Loaded best model from epoch {best_checkpoint['epoch']} "
              f"with validation loss {best_checkpoint['val_loss']:.4f} "
              f"(MSE: {best_checkpoint['val_mse']:.4f})")
    except Exception as e:
        print(f"Error loading best model: {str(e)}")
        print("Using current model state for evaluation")

    # Final evaluation on test set
    test_loss, test_mse, test_mae, test_physics = evaluate(model, test_loader, writer,
                                                           best_epoch, phase='test')

    # Log final test metrics
    writer.add_scalar('Final/Test_Loss', test_loss, best_epoch)
    writer.add_scalar('Final/Test_MSE', test_mse, best_epoch)
    writer.add_scalar('Final/Test_MAE', test_mae, best_epoch)
    writer.add_scalar('Final/Test_Physics', test_physics, best_epoch)

    # Create final summary
    final_summary = (f"Final Results:\n"
                    f"Test Loss: {test_loss:.4f}\n"
                    f"Test MSE: {test_mse:.4f}\n"
                    f"Test MAE: {test_mae:.4f}\n"
                    f"Test Physics: {test_physics:.4f}\n"
                    f"Best epoch: {best_epoch}")

    writer.add_text('Final/Results', final_summary)

    print(f"\nFinal test metrics - Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, Physics: {test_physics:.4f}")

    # Close TensorBoard writer
    writer.close()
    print(f"\nTensorBoard logs saved. To view: tensorboard --logdir=tensorboard_logs")


if __name__ == '__main__':
    main()
