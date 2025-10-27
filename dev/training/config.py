"""
Configuration classes for GNN training
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from enum import Enum
import torch


class LossType(Enum):
    """Enumeration for different loss configurations."""
    DATA_ONLY = "data"         # MSE only
    PHYSICS_ONLY = "physics"   # Physics term only
    COMBINED = "combined"      # Both MSE and physics terms


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Data parameters
    data_path: str
    batch_size: int = 2
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # Training parameters
    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10
    lambda_phys: float = 1.0

    # Standard deviation of time jitter noise
    time_jitter_std: float = 0

    # Loss configuration
    loss_type: LossType = LossType.PHYSICS_ONLY

    # Reproducibility
    seed: int = 42

    # Scheduler parameters
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Gradient clipping
    clip_grad_norm: float = 1.0

    # Paths
    model_save_dir: str = "results/models"
    model_filename: str = "best_model.pt"
    tensorboard_log_dir: str = "results/tensorboard_logs"

    @property
    def model_save_path(self) -> str:
        """Get the full path for saving the model."""
        return str(Path(self.model_save_dir) / self.model_filename)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not (0 < self.train_ratio < 1):
            raise ValueError(f"train_ratio must be between 0 and 1, got {self.train_ratio}")
        if not (0 < self.val_ratio < 1):
            raise ValueError(f"val_ratio must be between 0 and 1, got {self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError(
                f"Sum of train_ratio ({self.train_ratio}) and val_ratio ({self.val_ratio}) "
                f"must be less than 1 to leave data for testing"
            )
        if self.data_path is None:
            raise ValueError("data_path is required")


@dataclass
class TrainingState:
    """State tracking for training loop."""
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    patience_counter: int = 0
    current_epoch: int = 0

    def update_best(self, val_loss: float, epoch: int) -> bool:
        """Update best validation loss and reset patience counter.

        Returns:
            True if this is a new best, False otherwise
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def should_stop(self, patience: int) -> bool:
        """Check if training should stop due to early stopping."""
        return self.patience_counter >= patience


@dataclass
class PhysicsMetrics:
    """Container for detailed physics indicators."""
    J_ax: float = 0.0               # Axial flux magnitude
    F_in: float = 0.0               # Phloem loading rate
    F_out: float = 0.0              # Sucrose outflow rate
    ds_dt: float = 0.0              # Model-predicted time derivative magnitude
    dS_dt_from_flux: float = 0.0    # Flux divergence magnitude
    dS_dt_from_physics: float = 0.0 # Total physics-based rate of change

    def __str__(self) -> str:
        return (f"J_ax={self.J_ax:.3e} F_in={self.F_in:.3e} F_out={self.F_out:.3e} "
                f"ds_dt={self.ds_dt:.3e} flux_div={self.dS_dt_from_flux:.3e}")


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    mse: float
    mae: float
    physics: float
    physics_details: Optional['PhysicsMetrics'] = None

    def __str__(self) -> str:
        base_str = f"loss={self.loss:.4f} MSE={self.mse:.4f} physics={self.physics:.4f}"
        if self.physics_details is not None:
            base_str += f" | {self.physics_details}"
        return base_str


@dataclass
class ModelSetup:
    """Configuration for model setup."""
    device: torch.device
    model: torch.nn.Module
    feature_scaler: Optional[object] = None
    target_scaler: Optional[object] = None
    time_scaler: Optional[object] = None
    edge_scaler: Optional[object] = None  # scaler for continuous edge features (e.g., r_ST)

    def to_device(self) -> None:
        """Move model and scalers to the configured device."""
        self.model.to(self.device)
        if self.feature_scaler is not None:
            self.feature_scaler.device = self.device
        if self.target_scaler is not None:
            self.target_scaler.device = self.device
        if self.time_scaler is not None:
            self.time_scaler.device = self.device
        if self.edge_scaler is not None:
            self.edge_scaler.device = self.device
