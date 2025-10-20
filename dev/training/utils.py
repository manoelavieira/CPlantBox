import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from typing import Tuple

from data.dataset_loader import load_phloem_data
from model.config import ModelConfig
from model.utils import Standardizer
from .config import TrainingConfig, ModelSetup


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


def get_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, validation, and test dataloaders.

    Args:
        config: Training configuration containing dataset parameters

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        ValueError: If dataset parameters are invalid
    """
    # Validate split ratios
    validate_split_ratios(config.train_ratio, config.val_ratio)

    # Load simulation data
    train_loader, val_loader, test_loader = load_phloem_data(
        h5_path=config.data_path,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        random_seed=config.seed
    )

    return train_loader, val_loader, test_loader


def save_checkpoint(
    model_setup: ModelSetup,
    model_cfg: ModelConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_loss: float,
    val_mse: float,
    filepath: str
) -> None:
    """Save model checkpoint with all necessary state.

    Args:
        model_setup: Model setup containing model and scalers
        model_cfg: Model configuration
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch number
        val_loss: Validation loss
        val_mse: Validation MSE
        filepath: Path to save checkpoint
    """
    # Prepare scaler states
    feature_scaler_state = {
        'mean': model_setup.feature_scaler.mean,
        'std': model_setup.feature_scaler.std,
        'device': str(model_setup.feature_scaler.device)
    }
    target_scaler_state = {
        'mean': model_setup.target_scaler.mean,
        'std': model_setup.target_scaler.std,
        'device': str(model_setup.target_scaler.device)
    }
    time_scaler_state = {
        'mean': model_setup.time_scaler.mean,
        'std': model_setup.time_scaler.std,
        'device': str(model_setup.time_scaler.device)
    }
    edge_scaler_state = {
        'mean': model_setup.edge_scaler.mean,
        'std': model_setup.edge_scaler.std,
        'device': str(model_setup.edge_scaler.device)
    }

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'cfg': model_cfg.__dict__,
        'state_dict': model_setup.model.state_dict(),
        'device': model_setup.device.type,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_mse': val_mse,
        'feature_scaler': feature_scaler_state,
        'target_scaler': target_scaler_state,
        'time_scaler': time_scaler_state,
        'edge_scaler': edge_scaler_state,
    }, filepath)


def load_best_model(
    model_setup: ModelSetup,
    filepath: str,
    device: torch.device
) -> bool:
    """Load the best model from checkpoint.

    Args:
        model_setup: Model setup to update with loaded state
        filepath: Path to checkpoint file
        device: Device to load model on

    Returns:
        bool: True if loading was successful, False otherwise
    """
    try:
        best_checkpoint = torch.load(filepath, map_location=device)

        # Load model state
        model_setup.model.load_state_dict(best_checkpoint['state_dict'])

        # Reconstruct scalers from saved state
        model_setup.feature_scaler = Standardizer()
        model_setup.feature_scaler.mean = best_checkpoint['feature_scaler']['mean']
        model_setup.feature_scaler.std = best_checkpoint['feature_scaler']['std']
        model_setup.feature_scaler.device = device

        model_setup.target_scaler = Standardizer()
        model_setup.target_scaler.mean = best_checkpoint['target_scaler']['mean']
        model_setup.target_scaler.std = best_checkpoint['target_scaler']['std']
        model_setup.target_scaler.device = device

        model_setup.time_scaler = Standardizer()
        model_setup.time_scaler.mean = best_checkpoint['time_scaler']['mean']
        model_setup.time_scaler.std = best_checkpoint['time_scaler']['std']
        model_setup.time_scaler.device = device

        model_setup.edge_scaler = Standardizer()
        model_setup.edge_scaler.mean = best_checkpoint['edge_scaler']['mean']
        model_setup.edge_scaler.std = best_checkpoint['edge_scaler']['std']
        model_setup.edge_scaler.device = device

        # Assign scalers to model for backward compatibility
        model_setup.model.feature_scaler = model_setup.feature_scaler
        model_setup.model.target_scaler = model_setup.target_scaler
        model_setup.model.time_scaler = model_setup.time_scaler
        model_setup.model.edge_scaler = model_setup.edge_scaler

        print(f"Loaded best model from epoch {best_checkpoint['epoch']} "
              f"with validation loss {best_checkpoint['val_loss']:.4f} "
              f"(MSE: {best_checkpoint['val_mse']:.4f})")
        return True

    except Exception as e:
        print(f"Error loading best model: {str(e)}")
        print("Using current model state for evaluation")
        return False


def prepare_model_inputs(
    data,
    model: nn.Module,
    is_training: bool = False,
    time_jitter_std: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for forward pass by handling time and feature standardization.

    Args:
        data: Input batch data
        model: The neural network model with fitted scalers
        is_training: Whether this is for training (enables time jitter)
        time_jitter_std: Standard deviation for time jitter during training

    Returns:
        Tuple of (original_features, standardized_data_ready_for_model)
    """
    # Ensure data is on correct device
    data = data.to(next(model.parameters()).device)

    # Validate required components
    if not hasattr(model, "time_scaler") or model.time_scaler is None:
        raise RuntimeError("Missing model.time_scaler (fit during setup).")
    if not hasattr(data, "time") or data.time is None:
        raise ValueError("Each Data must carry a graph-level `time` tensor.")

    # Standardize graph-level time and broadcast to nodes
    time_scaled = model.time_scaler.transform(data.time.view(-1, 1)).view(-1)

    if hasattr(data, "batch") and data.batch is not None:
        time_node = time_scaled[data.batch]
    else:
        N = data.num_nodes
        time_node = time_scaled.expand(N)
    time_node = time_node.view(-1, 1).to(next(model.parameters()).device)

    # σ_t per node (for d/dτ -> d/dt)
    time_std_scalar = model.time_scaler.std.view(-1)[0].to(time_node.device)
    time_std_node = time_std_scalar.expand_as(time_node).clone()

    # Add small random jitter to time during training
    if is_training and time_jitter_std > 0:
        jitter = torch.randn_like(time_node) * time_jitter_std
        time_node = time_node + jitter

    # Physics autograd needs time to require grad during training
    if is_training and not time_node.requires_grad:
        time_node.requires_grad_(True)

    # Store original features before standardization
    x_orig = data.node_feat.clone()

    # Attach time information to data
    data.time_node = time_node if is_training else time_node.detach()
    data.time_std_node = time_std_node if is_training else time_std_node.detach()

    # Standardize features for the model
    data.node_feat = model.feature_scaler.transform(data.node_feat)

    return x_orig, data