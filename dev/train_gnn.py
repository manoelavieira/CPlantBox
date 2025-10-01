"""
Training script for the phloem GNN model
"""
import argparse
from enum import Enum
from typing import Tuple, Optional
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.data import Batch

from dataset_loader import load_phloem_data
from dataset_dummy import DummyTemporalDataset
from gnn import PhloemNNConv, ModelConfig, Standardizer, physics_residual


class DatasetType(Enum):
    DUMMY = 'dummy'
    SIMULATED = 'simulated'

def collate_graphs(batch):
    return Batch.from_data_list(batch)

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, target_scaler: Optional[Standardizer] = None, clip_grad_norm: float = 1.0) -> Tuple[float, float]:
    """Train model for one epoch.

    Args:
        model: The neural network model
        loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        target_scaler: Optional scaler for target values
        clip_grad_norm: Maximum norm for gradient clipping

    Returns:
        Tuple of (average_loss, average_mae)

    Raises:
        RuntimeError: If no training samples are processed
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_nodes = 0
    for data in loader:
        data = data.to(next(model.parameters()).device)
        optimizer.zero_grad(set_to_none=True)

        # Keep original features for physics computation
        x_orig = data.x_cont.clone()

        # Standardize features for the model
        data.x_cont = model.feature_scaler.transform(data.x_cont)

        # Forward pass returns predictions in standardized space
        pred = model(data) # [N,1]

        y = data.y # [N,1]
        y_t = model.target_scaler.transform(y)  # Transform targets for loss computation

        # MSE in standardized space
        mse = F.mse_loss(pred, y_t, reduction='sum')

        # Physics computation in original space
        pred_orig = model.target_scaler.inv_transform(pred)
        # Temporarily restore original features for physics computation
        data.x_cont = x_orig
        phys = physics_residual(pred_orig, data)
        if phys.dim() > 0:
            phys = phys.sum()
        # Restore standardized features for next iteration
        data.x_cont = model.feature_scaler.transform(x_orig)

        loss = mse + phys
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        with torch.no_grad():
            # Report MAE in original units
            if target_scaler is not None:
                pred_un = target_scaler.inv_transform(pred)
            else:
                pred_un = pred

            mae = (pred_un - y).abs().sum() # (pred_un - y): per-node errors, shape [N, 1]
            total_mae += mae.item() # accumulates the sum of absolute errors across batches
            total_loss += loss.item()
            n_nodes += y.size(0)

    if n_nodes == 0:
        raise RuntimeError("No training samples this epoch.")

    avg_loss = total_loss / n_nodes
    avg_mae = total_mae / n_nodes

    return avg_loss, avg_mae

def evaluate(model: nn.Module, loader: DataLoader, target_scaler: Optional[Standardizer] = None) -> Tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: The neural network model
        loader: DataLoader containing evaluation data
        target_scaler: Optional scaler for target values

    Returns:
        Tuple of (average_mse, average_mae)
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_nodes = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(next(model.parameters()).device)

            # Keep original features
            x_orig = data.x_cont.clone()

            # Standardize features for the model
            data.x_cont = model.feature_scaler.transform(data.x_cont)

            # Forward pass returns predictions in standardized space
            pred = model(data)

            y = data.y
            y_t = model.target_scaler.transform(y)
            mse = F.mse_loss(pred, y_t, reduction='sum')

            # Transform predictions back for MAE in original space
            pred_un = model.target_scaler.inv_transform(pred)

            # Restore original features for consistent state
            data.x_cont = x_orig
            mae = (pred_un - y).abs().sum()
            total_mse += mse.item()
            total_mae += mae.item()
            n_nodes += y.size(0)

    return total_mse / max(n_nodes, 1), total_mae / max(n_nodes, 1)

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
        ds = DummyTemporalDataset(n_graphs=args.n_graphs)
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

    model_cfg = ModelConfig()
    model = PhloemNNConv(model_cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get data loaders
    dataset_type = DatasetType[args.dataset.upper()]
    train_loader, val_loader, test_loader = get_dataloaders(dataset_type, args)
    print(f"Train batches: {len(train_loader)}, "
          f"Validation batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    # Setup standardization on training data
    feature_scaler = Standardizer() # for input features (x_cont)
    target_scaler = Standardizer() # for targets (y)

    # Fit scalers on training data
    with torch.no_grad():
        x_list, y_list = [], []
        for batch in train_loader:
            x_list.append(batch.x_cont) # node features [N, 3]
            y_list.append(batch.y) # targets [N, 1]

        Xs = torch.cat(x_list, dim=0)
        Ys = torch.cat(y_list, dim=0)
        feature_scaler.fit(Xs)
        target_scaler.fit(Ys)

    # Add scalers to the model
    model.feature_scaler = feature_scaler
    model.target_scaler = target_scaler

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
        tr_loss, tr_mae = train_one_epoch(
            model, train_loader, optimizer,
            target_scaler=target_scaler
        )

        # Validation
        val_mse, val_mae = evaluate(
            model, val_loader,
            target_scaler=target_scaler
        )

        # Learning rate scheduling
        scheduler.step(val_mse)

        # Logging
        print(f"Epoch {epoch:03d} | "
              f"train_loss={tr_loss:.4f} train_MAE={tr_mae:.4f} | "
              f"val_MSE={val_mse:.4f} val_MAE={val_mae:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Model saving and early stopping
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            patience_counter = 0

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

            torch.save({
                'epoch': epoch,
                'cfg': model_cfg.__dict__,
                'state_dict': model.state_dict(),
                'device': device.type,  # Save source device info
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_mse': val_mse,
                'feature_scaler': feature_scaler_state,
                'target_scaler': target_scaler_state,
            }, 'model/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best validation MSE: {best_val:.4f} "
                      f"at epoch {best_epoch}")
                break

    print("\nTraining completed!")

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

        # Assign scalers to model
        model.feature_scaler = feature_scaler
        model.target_scaler = target_scaler

        print(f"Loaded best model from epoch {best_checkpoint['epoch']} "
              f"with validation MSE {best_checkpoint['val_mse']:.4f}")
    except Exception as e:
        print(f"Error loading best model: {str(e)}")
        print("Using current model state for evaluation")

    test_mse, test_mae = evaluate(
        model, test_loader,
        target_scaler=target_scaler
    )

    print(f"\nFinal test metrics - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")


if __name__ == '__main__':
    main()
