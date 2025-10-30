import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn

from pathlib import Path

from model.config import ModelConfig
from model.gnn import PhloemNNConv
from .config import TrainingConfig, ModelSetup


def setup_environment(config: TrainingConfig) -> torch.device:
    """Setup training environment including seeding and device configuration.

    Args:
        config: Training configuration

    Returns:
        torch.device: Configured device for training
    """
    # Set random seeds for full reproducibility
    random.seed(config.seed)  # Python's random
    np.random.seed(config.seed)  # NumPy
    torch.manual_seed(config.seed)  # PyTorch on CPU
    torch.cuda.manual_seed(config.seed)  # PyTorch on Current GPU
    torch.cuda.manual_seed_all(config.seed)  # PyTorch on All GPUs

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model save directory
    Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)

    return device


def setup_model(device: torch.device) -> ModelSetup:
    """Setup model.

    Args:
        device: Device to place model on

    Returns:
        ModelSetup: Configured model without scalers
    """
    # Create model
    model_cfg = ModelConfig()
    model = PhloemNNConv(model_cfg).to(device)

    # Create model setup without scalers
    model_setup = ModelSetup(
        device=device,
        model=model
    )

    # Ensure everything is on the correct device
    model_setup.to_device()

    return model_setup


def setup_training_components(
    model: nn.Module,
    config: TrainingConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Setup optimizer and learning rate scheduler.

    Args:
        model: The neural network model
        config: Training configuration

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience
    )

    return optimizer, scheduler