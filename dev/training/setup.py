import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from pathlib import Path

from model.config import ModelConfig
from model.gnn import PhloemNNConv
from model.utils import Standardizer, IdentityScaler
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


def setup_model_and_scalers(
    config: TrainingConfig,
    train_loader: DataLoader,
    device: torch.device
) -> ModelSetup:
    """Setup model and scalers.

    Args:
        config: Training configuration
        train_loader: Training data loader for fitting scalers
        device: Device to place model and scalers on

    Returns:
        ModelSetup: Configured model with fitted scalers
    """
    # Create model
    model_cfg = ModelConfig()
    model = PhloemNNConv(model_cfg).to(device)

    # Setup standardization on training data
    if config.no_standardization:
        # Use identity scalers that perform no transformation
        # This enables training in original space without standardization
        # Both MSE and physics residuals will be computed in original space
        feature_scaler = IdentityScaler()  # for input node features (psi, vol, len_leaf...)
        target_scaler = IdentityScaler()   # for targets (y)
        time_scaler = IdentityScaler()     # for graph-level time (scalar)
        edge_scaler = IdentityScaler()     # for continuous edge features (e.g., r_ST)
    else:
        # Use standard scalers that normalize to mean=0, std=1
        # Physics residuals will be converted from original to standardized space
        feature_scaler = Standardizer()  # for input node features (psi, vol, len_leaf...)
        target_scaler = Standardizer()   # for targets (y)
        time_scaler = Standardizer()     # for graph-level time (scalar)
        edge_scaler = Standardizer()     # for continuous edge features (e.g., r_ST)

    # Fit scalers on training data
    with torch.no_grad():
        x_list, y_list, t_list, e_list = [], [], [], []

        for batch in train_loader:
            x_list.append(batch.node_feat[:, :model_cfg.node_feat_dim])
            y_list.append(batch.y)
            e_list.append(batch.edge_feat[:, :model_cfg.edge_feat_dim])  # [E, D], typically D=1
            t_list.append(batch.time.view(-1, 1))  # collect per-graph scalars

        Xs = torch.cat(x_list, dim=0)  # [sum_N, 2]
        Ys = torch.cat(y_list, dim=0)  # [sum_N, 1]
        Es = torch.cat(e_list, dim=0)  # [sum_E, edge_feat_dim]
        Ts = torch.cat(t_list, dim=0)  # [sum_B, 1], one per graph

        feature_scaler.fit(Xs)
        target_scaler.fit(Ys)
        time_scaler.fit(Ts)
        edge_scaler.fit(Es)

    # Create model setup
    model_setup = ModelSetup(
        device=device,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        time_scaler=time_scaler,
        edge_scaler=edge_scaler
    )

    # Add scalers to the model for backward compatibility
    model.feature_scaler = feature_scaler
    model.target_scaler = target_scaler
    model.time_scaler = time_scaler
    model.edge_scaler = edge_scaler

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