import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from typing import Tuple

from data.dataset_loader import load_phloem_data
from model.config import ModelConfig
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
        model_setup: Model setup containing model
        model_cfg: Model configuration
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch number
        val_loss: Validation loss
        val_mse: Validation MSE
        filepath: Path to save checkpoint
    """
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
    is_training: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for forward pass.

    This function prepares time tensors for the model without any standardization.

    Args:
        data: Input batch data
        model: The neural network model
        is_training: Whether this is for training

    Returns:
        Tuple of (original_features, data_ready_for_model)
    """
    # Move to model device
    device = next(model.parameters()).device
    data = data.to(device)

    # Validate required components
    if not hasattr(data, "time") or data.time is None:
        raise ValueError("Each Data must carry a graph-level `time` tensor.")
    if data.time.dim() != 1:
        raise ValueError(f"`data.time` must be 1D [num_graphs]; got {tuple(data.time.shape)}.")

    data.time = data.time.detach().to(device).requires_grad_(is_training)

    if hasattr(data, "batch") and data.batch is not None:
        time_per_node = data.time[data.batch].unsqueeze(-1)
    else:
        time_per_node = data.time.repeat(data.num_nodes).unsqueeze(-1)

    # Attach and (optionally) detach in eval
    data.time_per_node = time_per_node if is_training else time_per_node.detach()

    return data
