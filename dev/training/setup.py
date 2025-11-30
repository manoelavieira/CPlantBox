import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn

from pathlib import Path

from model.config import ModelConfig
from model.gnn import PhloemNNConv
from model.gnn_operator import PhloemOperatorGNN
from .config import TrainingConfig, ModelSetup

from torch.utils.data import DataLoader
from model.utils import Standardizer


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
    train_loader: DataLoader,
    device: torch.device,
    model_type: str = "nnconv"
) -> ModelSetup:
    """Setup model and scalers.

    Args:
        train_loader: Training data loader for fitting scalers
        device: Device to place model and scalers on
        model_type: Type of model ('nnconv' or 'operator')

    Returns:
        ModelSetup: Configured model with fitted scalers
    """
    # Create model configuration
    model_cfg = ModelConfig(model_type=model_type)

    # Instantiate the appropriate model
    if model_type == "nnconv":
        model = PhloemNNConv(model_cfg).to(device)
    elif model_type == "operator":
        model = PhloemOperatorGNN(model_cfg).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'nnconv' or 'operator'")

    print(f"Created model: {model.__class__.__name__} (type={model_type})")

    # Setup standardization on training data
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

    # Create model setup with scalers
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