"""
Configuration classes for GNN training
"""
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from enum import Enum
import torch


class LossType(Enum):
    """Enumeration for different loss configurations."""
    DATA_ONLY = "data"              # MSE only
    PHYSICS_WITH_IC_BC = "physics"  # Physics term + initial condition supervision + boundary condition
    COMBINED = "combined"           # Both MSE and physics terms


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Data parameters
    data_path: str
    batch_size: int = 2
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # Model architecture
    model_type: str = "nnconv"  # 'nnconv' or 'operator'

    # Training parameters
    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10

    # Loss configuration
    loss_type: LossType = LossType.PHYSICS_WITH_IC_BC
    lambda_data: float = 1.0
    lambda_phys: float = 1.0
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0

    # Adaptive loss balancing for physics mode
    use_adaptive_physics_weighting: bool = False    # Balance physics vs IC/BC dynamically
    target_physics_ratio: float = 1                 # Target ratio of physics loss to supervision loss

    # Reproducibility
    seed: int = 42

    # Scheduler parameters
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # Gradient clipping
    clip_grad_norm: float = 1.0

    # Paths
    model_save_dir: str = "logs/model"
    model_filename: str = "best_model.pt"
    physics_save_dir: str = "logs/physics"
    physics_save_filename: str = "debugs.txt"
    tensorboard_log_dir: str = "logs/tensorboard"

    # Physics logging
    enable_physics_logging: bool = False

    @property
    def model_save_path(self) -> str:
        """Get the full path for saving the model."""
        return str(Path(self.model_save_dir) / self.model_filename)

    @property
    def physics_save_path(self) -> str:
        """Get the full path for saving physics debug logs."""
        return str(Path(self.physics_save_dir) / self.physics_save_filename)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.model_type not in ["nnconv", "operator"]:
            raise ValueError(f"model_type must be 'nnconv' or 'operator', got {self.model_type}")
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
    dS_dt_from_flux: float = 0.0    # Flux divergence magnitude
    dS_dt_tot: float = 0.0          # Total physics-based rate of change

    def __str__(self) -> str:
        return (f"J_ax={self.J_ax:.3e} F_in={self.F_in:.3e} F_out={self.F_out:.3e} "
                f"flux_div={self.dS_dt_from_flux:.3e} dS_dt_tot={self.dS_dt_tot:.3e}")


@dataclass
class PhysicsErrorMetrics:
    """Container for physics error metrics (MSE, RMSE, Relative Error)."""
    # J_ax errors
    J_ax_mse: float = 0.0
    J_ax_rmse: float = 0.0
    J_ax_rel_error: float = 0.0

    # divJ (divergence) errors - for mass conservation evaluation
    divJ_mse: float = 0.0
    divJ_rmse: float = 0.0
    divJ_rel_error: float = 0.0

    # dS_dt_tot (total residual) errors
    dS_dt_tot_mse: float = 0.0
    dS_dt_tot_rmse: float = 0.0
    dS_dt_tot_rel_error: float = 0.0

    # J_ax antisymmetry error (operator model only)
    J_ax_antisym_error: float = 0.0

    # Flux direction consistency metrics (physical credibility)
    J_ax_sign_accuracy: float = 0.0      # Fraction of edges with correct flux direction
    J_ax_reversal_rate: float = 0.0      # Fraction of edges with wrong direction (1 - sign_accuracy)
    delta_C_sign_accuracy: float = 0.0   # Fraction of edges with correct ΔC sign (osmotic effects)

    # Physics score metrics (dimensionless residual-based consistency)
    physics_rel_error: float = 0.0          # Normalized residual: E[|r|] / (E[|F_in|] + E[|F_out|] + eps)
    physics_satisfaction_rate: float = 0.0  # Fraction of nodes satisfying conservation within tolerance

    # Temporal consistency metrics (time-series mode)
    temporal_rel_error_pred: float = 0.0    # Normalized temporal residual for predictions
    temporal_consistency_pred: float = 0.0  # Fraction of predicted nodes satisfying temporal tolerance
    temporal_rel_error_true: float = 0.0    # Normalized temporal residual for ground truth
    temporal_consistency_true: float = 0.0  # Fraction of ground-truth nodes satisfying temporal tolerance

    def __str__(self) -> str:
        base = (f"J_ax: MSE={self.J_ax_mse:.3e} RMSE={self.J_ax_rmse:.3e} RelErr={self.J_ax_rel_error:.3e} SignAcc={self.J_ax_sign_accuracy:.3f} | "
                f"divJ: MSE={self.divJ_mse:.3e} RMSE={self.divJ_rmse:.3e} RelErr={self.divJ_rel_error:.3e} | "
                f"dS_dt_tot: MSE={self.dS_dt_tot_mse:.3e} RMSE={self.dS_dt_tot_rmse:.3e} RelErr={self.dS_dt_tot_rel_error:.3e}")
        # Always include antisymmetry error and direction metrics
        base += f" | J_ax_antisym={self.J_ax_antisym_error:.3e}"
        if self.J_ax_reversal_rate > 0 or self.delta_C_sign_accuracy > 0:
            base += f" | RevRate={self.J_ax_reversal_rate:.3f} deltaC_SignAcc={self.delta_C_sign_accuracy:.3f}"
        # Add physics score metrics
        if self.physics_rel_error > 0 or self.physics_satisfaction_rate > 0:
            base += f" | PhysRelErr={self.physics_rel_error:.4f} PhysSatisf={self.physics_satisfaction_rate:.3f}"
        return base


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    mse: float
    mae: float
    rmse: float
    rel_error: float
    physics: float
    ic_loss: float = 0.0  # Initial condition loss
    bc_loss: float = 0.0  # Boundary condition loss
    bc_nodes: float = 0.0  # Average number of boundary nodes
    bc_pct: float = 0.0  # Average percentage of boundary nodes
    physics_details: Optional['PhysicsMetrics'] = None
    physics_errors: Optional['PhysicsErrorMetrics'] = None

    def __str__(self) -> str:
        base_str = f"loss={self.loss:.3e} MSE={self.mse:.3e} RMSE={self.rmse:.3e} MAE={self.mae:.3e} RelErr={self.rel_error:.3e} physics={self.physics:.3e}"
        if self.ic_loss > 0:
            base_str += f" IC={self.ic_loss:.4f}"
        if self.bc_loss > 0:
            base_str += f" BC={self.bc_loss:.4f}"
        if self.bc_nodes > 0:
            base_str += f" BC_nodes={self.bc_nodes:.1f}({self.bc_pct:.1f}%)"
        if self.physics_details is not None:
            base_str += f" | {self.physics_details}"
        return base_str


@dataclass
class LossConfig:
    """Configuration for loss computation."""
    loss_type: LossType = LossType.PHYSICS_WITH_IC_BC
    lambda_data: float = 1.0
    lambda_phys: float = 1.0
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    use_adaptive_physics_weighting: bool = True
    target_physics_ratio: float = 1.0

    @classmethod
    def from_training_config(cls, config: 'TrainingConfig') -> 'LossConfig':
        """Create LossConfig from TrainingConfig."""
        return cls(
            loss_type=config.loss_type,
            lambda_data=config.lambda_data,
            lambda_phys=config.lambda_phys,
            lambda_ic=config.lambda_ic,
            lambda_bc=config.lambda_bc,
            use_adaptive_physics_weighting=config.use_adaptive_physics_weighting,
            target_physics_ratio=config.target_physics_ratio
        )


@dataclass
class LossResult:
    """Results from loss computation."""
    total_loss: float
    mse: float
    mae: float
    rmse: float
    rel_error: float
    phys: float
    ic: float
    bc: float
    phys_weight: float = 0.0
    bc_nodes: int = 0
    bc_pct: float = 0.0
    phys_contrib_pct: float = 0.0
    sup_or_data_contrib_pct: float = 0.0
    physics_metrics: Optional[PhysicsMetrics] = None
    physics_errors: Optional[PhysicsErrorMetrics] = None


@dataclass
class EpochResult:
    """Results from a training or evaluation epoch."""
    loss: float
    mse: float
    mae: float
    rmse: float
    rel_error: float
    phys: float
    ic: float
    bc: float
    physics_metrics: Optional[PhysicsMetrics] = None
    physics_errors: Optional[PhysicsErrorMetrics] = None
    phys_weight: float = 0.0
    supervision_weight: float = 0.0
    bc_nodes: float = 0.0
    bc_pct: float = 0.0
    phys_contrib_pct: float = 0.0
    sup_contrib_pct: float = 0.0
    weighted_supervision: float = 0.0
    weighted_physics: float = 0.0

    @classmethod
    def from_totals(cls, totals: dict) -> 'EpochResult':
        """Create EpochResult from accumulated totals dictionary."""
        n_batches = max(1, totals["n_batches"])
        return cls(
            loss=totals["loss"] / n_batches,
            mse=totals["mse"] / n_batches,
            mae=totals["mae"] / n_batches,
            rmse=totals["rmse"] / n_batches,
            rel_error=totals["rel_error"] / n_batches,
            phys=totals["phys"] / n_batches,
            ic=totals["ic"] / n_batches,
            bc=totals["bc"] / n_batches,
            physics_metrics=totals["last_phys_metrics"],
            physics_errors=totals.get("last_phys_errors", None),
            phys_weight=totals["phys_weight"] / n_batches,
            supervision_weight=totals["supervision_weight"] / n_batches,
            bc_nodes=totals["bc_nodes"] / n_batches,
            bc_pct=totals["bc_pct"] / n_batches,
            phys_contrib_pct=totals["phys_contrib_pct"] / n_batches,
            sup_contrib_pct=totals["sup_contrib_pct"] / n_batches,
            weighted_supervision=totals["weighted_supervision"] / n_batches,
            weighted_physics=totals["weighted_physics"] / n_batches,
        )


@dataclass
class ModelSetup:
    """Configuration for model setup."""
    device: torch.device
    model: torch.nn.Module
    feature_scaler: Optional[object] = None
    target_scaler: Optional[object] = None
    time_scaler: Optional[object] = None
    edge_scaler: Optional[object] = None

    def to_device(self) -> None:
        """Move model and scalers to the configured device."""
        self.model.to(self.device)
        if self.feature_scaler is not None:
            self.feature_scaler.to(self.device)
        if self.target_scaler is not None:
            self.target_scaler.to(self.device)
        if self.time_scaler is not None:
            self.time_scaler.to(self.device)
        if self.edge_scaler is not None:
            self.edge_scaler.to(self.device)