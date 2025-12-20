"""
Configuration classes for GNN training
"""
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from enum import Enum
import torch
import os


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
    split_method: str = "random"  # 'random' or 'time'

    # K-fold cross-validation parameters
    use_kfold: bool = True  # Use k-fold CV (keeps files separate) by default
    current_fold: int = 0   # Internal: which fold is being trained (set by train_all_folds)

    # Model architecture
    model_type: str = "operator"  # 'nnconv' or 'operator'
    use_analytical_residual: bool = False

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
    metrics_save_dir: str = "logs/metrics"
    metrics_save_filename: str = "metrics.csv"
    tensorboard_log_dir: str = "logs/tensorboard"

    # Physics logging
    enable_physics_logging: bool = True
    enable_metrics_logging: bool = True

    def get_data_prefix(self) -> str:
        """Extract a prefix from the data path (directory name)."""
        path = Path(self.data_path)
        # If it's a file, get the parent directory name
        if path.is_file():
            return path.parent.name
        # If it's a directory, get the directory name
        elif path.is_dir():
            return path.name
        # Fallback: try to extract from string if path doesn't exist yet
        else:
            # Remove trailing slashes and get the last component
            path_str = str(path).rstrip(os.sep)
            return os.path.basename(path_str)

    @property
    def model_save_path(self) -> str:
        """Get the full path for saving the model."""
        return str(Path(self.model_save_dir) / self.model_filename)

    @property
    def physics_save_path(self) -> str:
        """Get the full path for saving physics debug logs."""
        return str(Path(self.physics_save_dir) / self.physics_save_filename)

    @property
    def metrics_save_path(self) -> str:
        """Get the full path for saving metrics logs."""
        return str(Path(self.metrics_save_dir) / self.metrics_save_filename)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.model_type not in ["nnconv", "operator"]:
            raise ValueError(f"model_type must be 'nnconv' or 'operator', got {self.model_type}")
        if self.split_method not in ["random", "time"]:
            raise ValueError(f"split_method must be 'random' or 'time', got {self.split_method}")

        # Only validate split ratios if not using k-fold
        if not self.use_kfold:
            if not (0 < self.train_ratio < 1):
                raise ValueError(f"train_ratio must be between 0 and 1, got {self.train_ratio}")
            if not (0 < self.val_ratio < 1):
                raise ValueError(f"val_ratio must be between 0 and 1, got {self.val_ratio}")
            if self.train_ratio + self.val_ratio >= 1:
                raise ValueError(
                    f"Sum of train_ratio ({self.train_ratio}) and val_ratio ({self.val_ratio}) "
                    f"must be less than 1 to leave data for testing"
                )
        else:
            # Validate k-fold parameters
            # current_fold validation will happen at runtime when we know n_folds
            if self.current_fold < 0:
                raise ValueError(f"current_fold must be >= 0, got {self.current_fold}")

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
    """Container for comprehensive physics error metrics."""

    # ========== SUCROSE CONTENT (S_ST) METRICS ==========
    S_ST_rmse: float = 0.0
    S_ST_mae: float = 0.0
    S_ST_nmae: float = 0.0          # Normalized MAE: MAE / mean(|true|)
    S_ST_correlation: float = 0.0    # Pearson correlation coefficient

    # ========== FLUX (J_ax) METRICS ==========
    J_ax_mse: float = 0.0
    J_ax_rmse: float = 0.0
    J_ax_mae: float = 0.0
    J_ax_nmae: float = 0.0          # Normalized MAE
    J_ax_sign_accuracy: float = 0.0      # Fraction of edges with correct flux direction
    J_ax_reversal_rate: float = 0.0      # Fraction of edges with wrong direction
    J_ax_antisym_error: float = 0.0      # Antisymmetry error (operator model)
    J_ax_magnitude_ratio: float = 0.0    # mean(|pred|) / mean(|true|)
    J_ax_correlation: float = 0.0        # Pearson correlation

    # ========== DIVERGENCE (divJ) METRICS ==========
    divJ_mse: float = 0.0
    divJ_rmse: float = 0.0
    divJ_mae: float = 0.0
    divJ_nmae: float = 0.0          # Normalized MAE
    divJ_std_true: float = 0.0      # Standard deviation of true divergence
    divJ_std_pred: float = 0.0      # Standard deviation of predicted divergence
    divJ_std_ratio: float = 0.0     # std_pred / std_true
    divJ_overlap: float = 0.0       # Distribution overlap metric (Bhattacharyya coefficient)
    divJ_correlation: float = 0.0   # Pearson correlation

    # ========== TOTAL RESIDUAL (dS_dt_tot) METRICS ==========
    dS_dt_tot_mse: float = 0.0
    dS_dt_tot_rmse: float = 0.0
    dS_dt_tot_mae: float = 0.0
    dS_dt_tot_nmae: float = 0.0     # Normalized MAE
    dS_dt_tot_mean_true: float = 0.0  # Mean of true residual
    dS_dt_tot_mean_pred: float = 0.0  # Mean of predicted residual
    dS_dt_tot_std_true: float = 0.0   # Std of true residual
    dS_dt_tot_std_pred: float = 0.0   # Std of predicted residual
    dS_dt_tot_skew_true: float = 0.0  # Skewness of true residual
    dS_dt_tot_skew_pred: float = 0.0  # Skewness of predicted residual

    def __str__(self) -> str:
        base = (f"S_ST: RMSE={self.S_ST_rmse:.3e} MAE={self.S_ST_mae:.3e} NMAE={self.S_ST_nmae:.3e} Corr={self.S_ST_correlation:.3f} | "
                f"J_ax: RMSE={self.J_ax_rmse:.3e} MAE={self.J_ax_mae:.3e} SignAcc={self.J_ax_sign_accuracy:.3f} Corr={self.J_ax_correlation:.3f} | "
                f"divJ: RMSE={self.divJ_rmse:.3e} MAE={self.divJ_mae:.3e} Corr={self.divJ_correlation:.3f} | "
                f"dS_dt_tot: RMSE={self.dS_dt_tot_rmse:.3e} MAE={self.dS_dt_tot_mae:.3e}")
        return base

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for logging."""
        return {
            # Sucrose content
            'S_ST_rmse': self.S_ST_rmse,
            'S_ST_mae': self.S_ST_mae,
            'S_ST_nmae': self.S_ST_nmae,
            'S_ST_correlation': self.S_ST_correlation,
            # Flux
            'J_ax_mse': self.J_ax_mse,
            'J_ax_rmse': self.J_ax_rmse,
            'J_ax_mae': self.J_ax_mae,
            'J_ax_nmae': self.J_ax_nmae,
            'J_ax_sign_accuracy': self.J_ax_sign_accuracy,
            'J_ax_reversal_rate': self.J_ax_reversal_rate,
            'J_ax_antisym_error': self.J_ax_antisym_error,
            'J_ax_magnitude_ratio': self.J_ax_magnitude_ratio,
            'J_ax_correlation': self.J_ax_correlation,
            # Divergence
            'divJ_mse': self.divJ_mse,
            'divJ_rmse': self.divJ_rmse,
            'divJ_mae': self.divJ_mae,
            'divJ_nmae': self.divJ_nmae,
            'divJ_std_true': self.divJ_std_true,
            'divJ_std_pred': self.divJ_std_pred,
            'divJ_std_ratio': self.divJ_std_ratio,
            'divJ_overlap': self.divJ_overlap,
            'divJ_correlation': self.divJ_correlation,
            # Total residual
            'dS_dt_tot_mse': self.dS_dt_tot_mse,
            'dS_dt_tot_rmse': self.dS_dt_tot_rmse,
            'dS_dt_tot_mae': self.dS_dt_tot_mae,
            'dS_dt_tot_nmae': self.dS_dt_tot_nmae,
            'dS_dt_tot_mean_true': self.dS_dt_tot_mean_true,
            'dS_dt_tot_mean_pred': self.dS_dt_tot_mean_pred,
            'dS_dt_tot_std_true': self.dS_dt_tot_std_true,
            'dS_dt_tot_std_pred': self.dS_dt_tot_std_pred,
            'dS_dt_tot_skew_true': self.dS_dt_tot_skew_true,
            'dS_dt_tot_skew_pred': self.dS_dt_tot_skew_pred,
        }


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
    use_analytical_residual: bool = False

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
            target_physics_ratio=config.target_physics_ratio,
            use_analytical_residual=getattr(config, 'use_analytical_residual', False)
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