from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
import csv
from datetime import datetime

from torch_scatter import scatter_mean
from torch_geometric.data import Data
from training.config import PhysicsMetrics, PhysicsErrorMetrics, TrainingConfig
from . import utils
from . import config

PENALTY_WEIGHT = 1000.0     # Weight for penalizing non-positive concentrations
EPSILON = 1e-12             # Small constant to prevent division by zero
SMOOTH_WIDTH = 0.001        # Default smooth width for smooth ReLU approximation

# Module-level physics logging configuration
# Default values are imported from TrainingConfig to ensure consistency
_ENABLE_PHYSICS_LOGGING = TrainingConfig.enable_physics_logging
_PHYSICS_LOG_PATH = str(Path(TrainingConfig.physics_save_dir) / TrainingConfig.physics_save_filename)

# Module-level metrics logging configuration
_ENABLE_METRICS_LOGGING = TrainingConfig.enable_metrics_logging
_METRICS_LOG_PATH = str(Path(TrainingConfig.metrics_save_dir) / TrainingConfig.metrics_save_filename)


def smooth_relu(x: torch.Tensor, delta: float) -> torch.Tensor:
    # Smooth approximation of max(x, 0)
    # For |x| >> delta, behaves like ReLU
    return delta * F.softplus(x / delta)


def set_physics_logging(enable: bool, log_path: str = None):
    """Configure physics logging settings.

    Args:
        enable: Whether to enable physics logging
        log_path: Path to the physics log file (optional, defaults to logs/physics/debug.txt)
    """
    global _ENABLE_PHYSICS_LOGGING, _PHYSICS_LOG_PATH

    _ENABLE_PHYSICS_LOGGING = enable
    if log_path is not None:
        _PHYSICS_LOG_PATH = log_path

    # Print configuration for debugging
    print(f"\n{'='*60}")
    print(f"Physics Logging Configuration:")
    print(f"  Enabled: {_ENABLE_PHYSICS_LOGGING}")
    print(f"  Log Path: {_PHYSICS_LOG_PATH}")

    # Create directory if logging is enabled
    if _ENABLE_PHYSICS_LOGGING:
        log_dir = Path(_PHYSICS_LOG_PATH).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing log file
        with open(_PHYSICS_LOG_PATH, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Physics Debug Log - Session Started\n")
            f.write(f"{'='*60}\n")
        print(f"  Log file initialized: {_PHYSICS_LOG_PATH}")

    print(f"{'='*60}\n")


def set_metrics_logging(enable: bool, log_path: str = None):
    """Configure metrics logging settings.

    Args:
        enable: Whether to enable metrics logging
        log_path: Path to the metrics log file (optional, defaults to logs/metrics/metrics.csv)
    """
    global _ENABLE_METRICS_LOGGING, _METRICS_LOG_PATH

    _ENABLE_METRICS_LOGGING = enable
    if log_path is not None:
        _METRICS_LOG_PATH = log_path

    # Print configuration for debugging
    print(f"\n{'='*60}")
    print(f"Metrics Logging Configuration:")
    print(f"  Enabled: {_ENABLE_METRICS_LOGGING}")
    print(f"  Log Path: {_METRICS_LOG_PATH}")

    # Create directory if logging is enabled
    if _ENABLE_METRICS_LOGGING:
        log_dir = Path(_METRICS_LOG_PATH).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file with headers if it doesn't exist
        if not Path(_METRICS_LOG_PATH).exists():
            _initialize_metrics_log()
        print(f"  Log file ready: {_METRICS_LOG_PATH}")

    print(f"{'='*60}\n")


def _initialize_metrics_log():
    """Initialize the metrics CSV log file with headers."""
    headers = [
        'timestamp', 'epoch', 'phase', 'batch_idx',
        # Sucrose content
        'S_ST_rmse', 'S_ST_mae', 'S_ST_nmae', 'S_ST_correlation',
        # Flux
        'J_ax_mse', 'J_ax_rmse', 'J_ax_mae', 'J_ax_nmae',
        'J_ax_sign_accuracy', 'J_ax_reversal_rate', 'J_ax_antisym_error',
        'J_ax_magnitude_ratio', 'J_ax_correlation',
        # Divergence
        'divJ_mse', 'divJ_rmse', 'divJ_mae', 'divJ_nmae',
        'divJ_std_true', 'divJ_std_pred', 'divJ_std_ratio',
        'divJ_overlap', 'divJ_correlation',
        # Total residual
        'dS_dt_tot_mse', 'dS_dt_tot_rmse', 'dS_dt_tot_mae', 'dS_dt_tot_nmae',
        'dS_dt_tot_mean_true', 'dS_dt_tot_mean_pred',
        'dS_dt_tot_std_true', 'dS_dt_tot_std_pred',
        'dS_dt_tot_skew_true', 'dS_dt_tot_skew_pred',
    ]

    with open(_METRICS_LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def log_metrics_to_file(
    physics_errors: PhysicsErrorMetrics,
    epoch: int = None,
    phase: str = None,
    batch_idx: int = None
):
    """Log physics error metrics to CSV file.

    Args:
        physics_errors: PhysicsErrorMetrics object to log
        epoch: Current epoch number (optional)
        phase: Training phase ('train', 'val', 'test') (optional)
        batch_idx: Batch index (optional)
    """
    if not _ENABLE_METRICS_LOGGING:
        return

    # Ensure directory exists
    log_dir = Path(_METRICS_LOG_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize file if it doesn't exist
    if not Path(_METRICS_LOG_PATH).exists():
        _initialize_metrics_log()

    # Prepare row data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics_dict = physics_errors.to_dict()

    row = [
        timestamp,
        epoch if epoch is not None else '',
        phase if phase is not None else '',
        batch_idx if batch_idx is not None else '',
    ]

    # Add all metric values in the same order as headers
    row.extend([
        # Sucrose content
        metrics_dict['S_ST_rmse'], metrics_dict['S_ST_mae'],
        metrics_dict['S_ST_nmae'], metrics_dict['S_ST_correlation'],
        # Flux
        metrics_dict['J_ax_mse'], metrics_dict['J_ax_rmse'],
        metrics_dict['J_ax_mae'], metrics_dict['J_ax_nmae'],
        metrics_dict['J_ax_sign_accuracy'], metrics_dict['J_ax_reversal_rate'],
        metrics_dict['J_ax_antisym_error'], metrics_dict['J_ax_magnitude_ratio'],
        metrics_dict['J_ax_correlation'],
        # Divergence
        metrics_dict['divJ_mse'], metrics_dict['divJ_rmse'],
        metrics_dict['divJ_mae'], metrics_dict['divJ_nmae'],
        metrics_dict['divJ_std_true'], metrics_dict['divJ_std_pred'],
        metrics_dict['divJ_std_ratio'], metrics_dict['divJ_overlap'],
        metrics_dict['divJ_correlation'],
        # Total residual
        metrics_dict['dS_dt_tot_mse'], metrics_dict['dS_dt_tot_rmse'],
        metrics_dict['dS_dt_tot_mae'], metrics_dict['dS_dt_tot_nmae'],
        metrics_dict['dS_dt_tot_mean_true'], metrics_dict['dS_dt_tot_mean_pred'],
        metrics_dict['dS_dt_tot_std_true'], metrics_dict['dS_dt_tot_std_pred'],
        metrics_dict['dS_dt_tot_skew_true'], metrics_dict['dS_dt_tot_skew_pred'],
    ])

    # Append to CSV file
    with open(_METRICS_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _log_header(title: str, batch_vec: torch.Tensor = None, data: Data = None, phase: str = None):
    """Create a standardized log header.

    Args:
        title: Title for this logging section
        batch_vec: Batch vector for graph counts (optional)

    Returns:
        str: Formatted header string
    """
    num_graphs = torch.bincount(batch_vec).size(0) if batch_vec is not None else 1
    msg = f"\n{'='*60}\n{title}\n{'='*60}\n"
    msg += f"\nNumber of graphs: {num_graphs}\n"

    # Include phase if provided (train/val/test)
    if phase is not None:
        msg += f"Phase: {phase}\n"

    # Include graph-level times if available on the data object
    if data is not None and hasattr(data, 'time'):
        try:
            t = data.time
            # If tensor, convert to numpy
            if isinstance(t, torch.Tensor):
                t_arr = t.detach().cpu().numpy()
            else:
                t_arr = t

            # Format times sensibly
            if hasattr(t_arr, '__len__') and len(t_arr) > 1:
                msg += f"Graph times (hours): {t_arr}\n"
            else:
                # Scalar
                single_t = float(t_arr) if hasattr(t_arr, '__float__') else t_arr
                msg += f"Graph time (hours): {single_t}\n"
        except Exception:
            # Ignore failures to extract time
            pass

    if batch_vec is not None:
        msg += f"Nodes per graph: {torch.bincount(batch_vec).detach().cpu().numpy()}\n"

    return msg


def _log_concentrations(C_ST_true: torch.Tensor, C_ST_pred: torch.Tensor, n_samples: int = 50):
    """Log concentration values.

    Args:
        C_ST_true: True concentration values
        C_ST_pred: Predicted concentration values
        n_samples: Number of samples to display

    Returns:
        str: Formatted concentration log
    """
    msg = f"\n--- CONCENTRATION VALUES (mmol/cm³) ---\n"
    msg += f"C_ST_true: {C_ST_true[:n_samples].detach().cpu().numpy()}\n"
    msg += f"C_ST_pred: {C_ST_pred[:n_samples].detach().cpu().numpy()}\n"
    return msg


def _log_sucrose_contents(S_ST_true: torch.Tensor, S_ST_pred: torch.Tensor, n_samples: int = 50):
    """Log sucrose content values.

    Args:
        S_ST_true: True sucrose content values
        S_ST_pred: Predicted sucrose content values
        n_samples: Number of samples to display

    Returns:
        str: Formatted sucrose content log
    """
    msg = f"\n--- SUCROSE CONTENT VALUES (mmol) ---\n"
    msg += f"S_ST_true: {S_ST_true[:n_samples].detach().cpu().numpy()}\n"
    msg += f"S_ST_pred: {S_ST_pred[:n_samples].detach().cpu().numpy()}\n"
    return msg


def _log_fluxes(J_ax_true: torch.Tensor, J_ax_pred: torch.Tensor, n_samples: int = 50):
    """Log flux values.

    Args:
        J_ax_true: True flux values
        J_ax_pred: Predicted flux values
        n_samples: Number of samples to display

    Returns:
        str: Formatted flux log
    """
    msg = f"\n--- FLUX VALUES (mmol/h) ---\n"

    if J_ax_true.numel() > 0:
        msg += f"J_ax_true (mean): {J_ax_true.mean().detach().cpu().item():.6e}\n"
        msg += f"J_ax_true (std): {J_ax_true.std().detach().cpu().item():.6e}\n"
        if J_ax_true.size(0) >= n_samples:
            msg += f"J_ax_true (first {n_samples}): {J_ax_true[:n_samples].detach().cpu().numpy()}\n"

    if J_ax_pred.numel() > 0:
        msg += f"J_ax_pred (mean): {J_ax_pred.mean().detach().cpu().item():.6e}\n"
        msg += f"J_ax_pred (std): {J_ax_pred.std().detach().cpu().item():.6e}\n"
        if J_ax_pred.size(0) >= n_samples:
            msg += f"J_ax_pred (first {n_samples}): {J_ax_pred[:n_samples].detach().cpu().numpy()}\n"

    return msg


def _log_source_sink_terms(
    F_in_true: torch.Tensor,
    F_in_pred: torch.Tensor,
    F_out_true: torch.Tensor,
    F_out_pred: torch.Tensor,
    n_samples: int = 50
):
    """Log source/sink terms (F_in and F_out).

    Args:
        F_in_true: True phloem loading values
        F_in_pred: Predicted phloem loading values
        F_out_true: True sucrose outflow values
        F_out_pred: Predicted sucrose outflow values
        n_samples: Number of samples to display

    Returns:
        str: Formatted source/sink log
    """
    msg = f"\n--- SOURCE/SINK TERMS (mmol/h) ---\n"
    msg += f"F_in_true (mean): {F_in_true.mean().detach().cpu().item():.6e}\n"
    msg += f"F_in_true (first {n_samples}): {F_in_true[:n_samples].detach().cpu().numpy()}\n"
    # msg += f"F_in_pred (mean): {F_in_pred.mean().detach().cpu().item():.6e}\n"
    # msg += f"F_in_pred (first {n_samples}): {F_in_pred[:n_samples].detach().cpu().numpy()}\n"
    msg += f"\n"
    msg += f"F_out_true (mean): {F_out_true.mean().detach().cpu().item():.6e}\n"
    msg += f"F_out_true (first {n_samples}): {F_out_true[:n_samples].detach().cpu().numpy()}\n"
    # msg += f"F_out_pred (mean): {F_out_pred.mean().detach().cpu().item():.6e}\n"
    # msg += f"F_out_pred (first {n_samples}): {F_out_pred[:n_samples].detach().cpu().numpy()}\n"
    return msg


def _log_divergence(
    dS_dt_from_flux_true: torch.Tensor,
    dS_dt_from_flux_pred: torch.Tensor,
    n_samples: int = 50
):
    """Log divergence values.

    Args:
        dS_dt_from_flux_true: True divergence values
        dS_dt_from_flux_pred: Predicted divergence values
        n_samples: Number of samples to display

    Returns:
        str: Formatted divergence log
    """
    msg = f"\n--- DIVERGENCE (mmol/h) ---\n"
    msg += f"Divergence_true (mean): {dS_dt_from_flux_true.mean().detach().cpu().item():.6e}\n"
    msg += f"Divergence_true (std): {dS_dt_from_flux_true.std().detach().cpu().item():.6e}\n"
    msg += f"Divergence_true (first {n_samples}): {dS_dt_from_flux_true[:n_samples].detach().cpu().numpy()}\n"
    msg += f"\n"
    msg += f"Divergence_pred (mean): {dS_dt_from_flux_pred.mean().detach().cpu().item():.6e}\n"
    msg += f"Divergence_pred (std): {dS_dt_from_flux_pred.std().detach().cpu().item():.6e}\n"
    msg += f"Divergence_pred (first {n_samples}): {dS_dt_from_flux_pred[:n_samples].detach().cpu().numpy()}\n"
    return msg


def _log_total_residual(
    dS_dt_tot_true: torch.Tensor,
    dS_dt_tot_pred: torch.Tensor,
    n_samples: int = 50
):
    """Log total physics residual.

    Args:
        dS_dt_tot_true: True total residual values
        dS_dt_tot_pred: Predicted total residual values
        n_samples: Number of samples to display

    Returns:
        str: Formatted total residual log
    """
    msg = f"\n--- TOTAL PHYSICS RESIDUAL (mmol/h) ---\n"
    msg += f"dS_dt_tot_true (mean absolute): {dS_dt_tot_true.abs().mean().detach().cpu().item():.6e}\n"
    msg += f"dS_dt_tot_true (first {n_samples}): {dS_dt_tot_true[:n_samples].detach().cpu().numpy()}\n"
    msg += f"\n"
    msg += f"dS_dt_tot_pred (mean absolute): {dS_dt_tot_pred.abs().mean().detach().cpu().item():.6e}\n"
    msg += f"dS_dt_tot_pred (first {n_samples}): {dS_dt_tot_pred[:n_samples].detach().cpu().numpy()}\n"
    return msg


def _log_comparison_metrics(physics_errors: PhysicsErrorMetrics):
    """Log comparison metrics from PhysicsErrorMetrics.

    Args:
        physics_errors: PhysicsErrorMetrics object containing comprehensive metrics

    Returns:
        str: Formatted comparison metrics log
    """
    msg = f"\n--- PHYSICS ERROR METRICS ---"

    # Sucrose content metrics
    msg += f"\n=== SUCROSE CONTENT (S_ST) ===\n"
    msg += f"RMSE: {physics_errors.S_ST_rmse:.6e}\n"
    msg += f"MAE: {physics_errors.S_ST_mae:.6e}\n"
    msg += f"NMAE: {physics_errors.S_ST_nmae:.6e}\n"
    msg += f"Correlation: {physics_errors.S_ST_correlation:.6f}\n"

    # Flux metrics
    msg += f"\n=== FLUX (J_ax) ===\n"
    msg += f"RMSE: {physics_errors.J_ax_rmse:.6e}\n"
    msg += f"MAE: {physics_errors.J_ax_mae:.6e}\n"
    msg += f"NMAE: {physics_errors.J_ax_nmae:.6e}\n"
    msg += f"Correlation: {physics_errors.J_ax_correlation:.6f}\n"
    msg += f"Sign Accuracy: {physics_errors.J_ax_sign_accuracy:.4f} ({physics_errors.J_ax_sign_accuracy*100:.2f}%)\n"
    msg += f"Reversal Rate: {physics_errors.J_ax_reversal_rate:.4f} ({physics_errors.J_ax_reversal_rate*100:.2f}%)\n"
    msg += f"Magnitude Ratio (pred/true): {physics_errors.J_ax_magnitude_ratio:.6f}\n"
    if physics_errors.J_ax_antisym_error >= 0:
        msg += f"Antisymmetry Error: {physics_errors.J_ax_antisym_error:.6e}\n"

    # Divergence metrics
    msg += f"\n=== DIVERGENCE (divJ) ===\n"
    msg += f"RMSE: {physics_errors.divJ_rmse:.6e}\n"
    msg += f"MAE: {physics_errors.divJ_mae:.6e}\n"
    msg += f"NMAE: {physics_errors.divJ_nmae:.6e}\n"
    msg += f"Correlation: {physics_errors.divJ_correlation:.6f}\n"
    msg += f"Std (true): {physics_errors.divJ_std_true:.6e}\n"
    msg += f"Std (pred): {physics_errors.divJ_std_pred:.6e}\n"
    msg += f"Std Ratio (pred/true): {physics_errors.divJ_std_ratio:.6f}\n"
    msg += f"Distribution Overlap: {physics_errors.divJ_overlap:.6f}\n"

    # Total residual metrics
    msg += f"\n=== TOTAL RESIDUAL (dS_dt_tot) ===\n"
    msg += f"RMSE: {physics_errors.dS_dt_tot_rmse:.6e}\n"
    msg += f"MAE: {physics_errors.dS_dt_tot_mae:.6e}\n"
    msg += f"NMAE: {physics_errors.dS_dt_tot_nmae:.6e}\n"
    msg += f"Mean (true): {physics_errors.dS_dt_tot_mean_true:.6e}\n"
    msg += f"Mean (pred): {physics_errors.dS_dt_tot_mean_pred:.6e}\n"
    msg += f"Std (true): {physics_errors.dS_dt_tot_std_true:.6e}\n"
    msg += f"Std (pred): {physics_errors.dS_dt_tot_std_pred:.6e}\n"
    msg += f"Skewness (true): {physics_errors.dS_dt_tot_skew_true:.6f}\n"
    msg += f"Skewness (pred): {physics_errors.dS_dt_tot_skew_pred:.6f}\n"

    return msg


def compute_flux_direction_metrics(
    J_ax_true: torch.Tensor,
    J_ax_pred: torch.Tensor,
) -> tuple[float, float]:
    """Compute flux direction consistency metrics.

    Evaluates whether the model predicts the correct physical direction of transport,
    which is crucial for physical credibility even if magnitudes are imperfect.

    Args:
        J_ax_true: True axial fluxes [E]
        J_ax_pred: Predicted axial fluxes [E]

    Returns:
        tuple: (sign_accuracy, reversal_rate)
            - sign_accuracy: Fraction of edges where sign(J_pred) == sign(J_true)
            - reversal_rate: Fraction of edges with wrong direction (1 - sign_accuracy)
    """
    # Flux sign accuracy: Does predicted flux have correct direction?
    sign_true = torch.sign(J_ax_true)
    sign_pred = torch.sign(J_ax_pred)

    # Count edges where signs match
    sign_matches = (sign_true == sign_pred).float()
    sign_accuracy = sign_matches.mean().detach().cpu().item()
    reversal_rate = 1.0 - sign_accuracy

    return sign_accuracy, reversal_rate


def compute_correlation(
    true_vals: torch.Tensor,
    pred_vals: torch.Tensor
) -> float:
    """Compute Pearson correlation coefficient.

    Args:
        true_vals: True values [N]
        pred_vals: Predicted values [N]

    Returns:
        float: Pearson correlation coefficient [-1, 1]
    """
    if true_vals.numel() == 0:
        return 0.0

    # Center the data
    true_centered = true_vals - true_vals.mean()
    pred_centered = pred_vals - pred_vals.mean()

    # Compute correlation
    numerator = (true_centered * pred_centered).sum()
    denominator = torch.sqrt((true_centered ** 2).sum() * (pred_centered ** 2).sum())

    if denominator < EPSILON:
        return 0.0

    correlation = (numerator / denominator).detach().cpu().item()
    return correlation


def compute_distribution_overlap(
    true_vals: torch.Tensor,
    pred_vals: torch.Tensor,
    n_bins: int = 50
) -> float:
    """Compute distribution overlap using Bhattacharyya coefficient.

    The Bhattacharyya coefficient measures the similarity between two probability
    distributions. A value of 1 indicates perfect overlap, 0 indicates no overlap.

    Args:
        true_vals: True values [N]
        pred_vals: Predicted values [N]
        n_bins: Number of bins for histogram

    Returns:
        float: Bhattacharyya coefficient [0, 1]
    """
    if true_vals.numel() == 0:
        return 0.0

    # Move to CPU for numpy operations
    true_np = true_vals.detach().cpu().numpy()
    pred_np = pred_vals.detach().cpu().numpy()

    # Determine common range for histograms
    min_val = min(true_np.min(), pred_np.min())
    max_val = max(true_np.max(), pred_np.max())

    # Add small margin to avoid edge effects
    margin = (max_val - min_val) * 0.01
    bins = torch.linspace(min_val - margin, max_val + margin, n_bins + 1)

    # Compute normalized histograms
    hist_true = torch.histc(true_vals, bins=n_bins, min=min_val - margin, max=max_val + margin)
    hist_pred = torch.histc(pred_vals, bins=n_bins, min=min_val - margin, max=max_val + margin)

    # Normalize to probability distributions
    hist_true = hist_true / (hist_true.sum() + EPSILON)
    hist_pred = hist_pred / (hist_pred.sum() + EPSILON)

    # Bhattacharyya coefficient: sum of sqrt(p_i * q_i)
    overlap = torch.sqrt(hist_true * hist_pred).sum().detach().cpu().item()

    return overlap


def compute_skewness(vals: torch.Tensor) -> float:
    """Compute skewness of a distribution.

    Args:
        vals: Input values [N]

    Returns:
        float: Skewness (0 for symmetric distributions)
    """
    if vals.numel() == 0:
        return 0.0

    mean = vals.mean()
    std = vals.std()

    if std < EPSILON:
        return 0.0

    # Skewness: E[((X - mu) / sigma)^3]
    skewness = (((vals - mean) / std) ** 3).mean().detach().cpu().item()
    return skewness


def _compute_physics_error_metrics(
    S_ST_true: torch.Tensor,
    S_ST_pred: torch.Tensor,
    J_ax_true: torch.Tensor,
    J_ax_pred: torch.Tensor,
    dS_dt_from_flux_true: torch.Tensor,
    dS_dt_from_flux_pred: torch.Tensor,
    dS_dt_tot_true: torch.Tensor,
    dS_dt_tot_pred: torch.Tensor,
    edge_index: torch.Tensor = None,
    batch_vec: torch.Tensor = None
) -> PhysicsErrorMetrics:
    """Compute comprehensive physics error metrics.

    Args:
        S_ST_true: True sucrose content [N]
        S_ST_pred: Predicted sucrose content [N]
        J_ax_true: True axial fluxes [E]
        J_ax_pred: Predicted axial fluxes [E]
        dS_dt_from_flux_true: True flux divergence [N]
        dS_dt_from_flux_pred: Predicted flux divergence [N]
        dS_dt_tot_true: True total residual [N]
        dS_dt_tot_pred: Predicted total residual [N]
        edge_index: Edge connectivity [2, E] (optional)
        batch_vec: Batch indices [N] (optional)

    Returns:
        PhysicsErrorMetrics containing comprehensive metrics
    """
    # Helper function to compute basic metrics for a quantity
    def compute_basic_metrics(true_vals, pred_vals):
        mse = (true_vals - pred_vals).pow(2).mean().detach().cpu().item()
        rmse = torch.sqrt((true_vals - pred_vals).pow(2).mean()).detach().cpu().item()
        mae = torch.abs(true_vals - pred_vals).mean().detach().cpu().item()

        # Normalized MAE: MAE / mean(|true|)
        mean_true_abs = torch.abs(true_vals).mean().detach().cpu().item()
        nmae = mae / (mean_true_abs + EPSILON)

        return mse, rmse, mae, nmae

    # ========== SUCROSE CONTENT METRICS ==========
    _, S_ST_rmse, S_ST_mae, S_ST_nmae = compute_basic_metrics(S_ST_true, S_ST_pred)
    S_ST_corr = compute_correlation(S_ST_true, S_ST_pred)

    # ========== FLUX METRICS ==========
    J_ax_mse, J_ax_rmse, J_ax_mae, J_ax_nmae = compute_basic_metrics(J_ax_true, J_ax_pred)
    J_ax_corr = compute_correlation(J_ax_true, J_ax_pred)

    # Flux magnitude ratio: mean(|pred|) / mean(|true|)
    mean_J_pred = torch.abs(J_ax_pred).mean().detach().cpu().item()
    mean_J_true = torch.abs(J_ax_true).mean().detach().cpu().item()
    J_ax_mag_ratio = mean_J_pred / (mean_J_true + EPSILON)

    # Flux direction metrics
    sign_acc, rev_rate = compute_flux_direction_metrics(J_ax_true, J_ax_pred)

    # Antisymmetry error
    antisym_err = compute_flux_antisymmetry_error(J_ax_pred, edge_index)

    # ========== DIVERGENCE METRICS ==========
    divJ_mse, divJ_rmse, divJ_mae, divJ_nmae = compute_basic_metrics(
        dS_dt_from_flux_true, dS_dt_from_flux_pred
    )
    divJ_corr = compute_correlation(dS_dt_from_flux_true, dS_dt_from_flux_pred)

    # Standard deviation comparison
    divJ_std_true = dS_dt_from_flux_true.std().detach().cpu().item()
    divJ_std_pred = dS_dt_from_flux_pred.std().detach().cpu().item()
    divJ_std_ratio = divJ_std_pred / (divJ_std_true + EPSILON)

    # Distribution overlap
    divJ_overlap = compute_distribution_overlap(dS_dt_from_flux_true, dS_dt_from_flux_pred)

    # ========== TOTAL RESIDUAL METRICS ==========
    # Basic metrics (MSE, RMSE, MAE)
    residual_error = dS_dt_tot_pred - dS_dt_tot_true
    dS_dt_tot_mse = residual_error.pow(2).mean().detach().cpu().item()
    dS_dt_tot_rmse = torch.sqrt(residual_error.pow(2).mean()).detach().cpu().item()
    dS_dt_tot_mae = residual_error.abs().mean().detach().cpu().item()

    # NMAE: For steady-state, normalize by flux divergence magnitude (not by dS_dt_tot which ≈ 0)
    # This makes NMAE meaningful and prevents division by near-zero
    dS_dt_tot_nmae = dS_dt_tot_mae / (dS_dt_from_flux_true.abs().mean().detach().cpu().item() + EPSILON)

    # Distribution statistics
    dS_dt_tot_mean_true = dS_dt_tot_true.mean().detach().cpu().item()
    dS_dt_tot_mean_pred = dS_dt_tot_pred.mean().detach().cpu().item()
    dS_dt_tot_std_true = dS_dt_tot_true.std().detach().cpu().item()
    dS_dt_tot_std_pred = dS_dt_tot_pred.std().detach().cpu().item()
    dS_dt_tot_skew_true = compute_skewness(dS_dt_tot_true)
    dS_dt_tot_skew_pred = compute_skewness(dS_dt_tot_pred)

    return PhysicsErrorMetrics(
        # Sucrose content
        S_ST_rmse=S_ST_rmse,
        S_ST_mae=S_ST_mae,
        S_ST_nmae=S_ST_nmae,
        S_ST_correlation=S_ST_corr,
        # Flux
        J_ax_mse=J_ax_mse,
        J_ax_rmse=J_ax_rmse,
        J_ax_mae=J_ax_mae,
        J_ax_nmae=J_ax_nmae,
        J_ax_sign_accuracy=sign_acc,
        J_ax_reversal_rate=rev_rate,
        J_ax_antisym_error=antisym_err,
        J_ax_magnitude_ratio=J_ax_mag_ratio,
        J_ax_correlation=J_ax_corr,
        # Divergence
        divJ_mse=divJ_mse,
        divJ_rmse=divJ_rmse,
        divJ_mae=divJ_mae,
        divJ_nmae=divJ_nmae,
        divJ_std_true=divJ_std_true,
        divJ_std_pred=divJ_std_pred,
        divJ_std_ratio=divJ_std_ratio,
        divJ_overlap=divJ_overlap,
        divJ_correlation=divJ_corr,
        # Total residual
        dS_dt_tot_mse=dS_dt_tot_mse,
        dS_dt_tot_rmse=dS_dt_tot_rmse,
        dS_dt_tot_mae=dS_dt_tot_mae,
        dS_dt_tot_nmae=dS_dt_tot_nmae,
        dS_dt_tot_mean_true=dS_dt_tot_mean_true,
        dS_dt_tot_mean_pred=dS_dt_tot_mean_pred,
        dS_dt_tot_std_true=dS_dt_tot_std_true,
        dS_dt_tot_std_pred=dS_dt_tot_std_pred,
        dS_dt_tot_skew_true=dS_dt_tot_skew_true,
        dS_dt_tot_skew_pred=dS_dt_tot_skew_pred,
    )


def compute_axial_flux(
    C_ST: torch.Tensor,
    node_feat_original: torch.Tensor,
    edge_feat_original: torch.Tensor,
    edge_index: torch.Tensor,
    batch_vec: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """Compute axial sucrose flux J_ax along edges.

    This implementation follows the C++ PiafMunch algorithm (external/PiafMunch/solve.cpp):
    1. Compute osmotic pressure P_ST = C_ST * RT for each node
    2. Compute water flux JW_ST based on pressure gradients
    3. Select upstream concentration based on flow direction
    4. Compute sugar flux JS_ST = JW_ST * C_upstream

    Args:
        C_ST: Sucrose concentration per node [N] (already denormalized)
        node_feat_original: Node features in original space [N, D]
        edge_feat_original: Edge features in original space [E, D]
        edge_index: Edge connectivity [2, E]
        batch_vec: Batch assignment vector [N] (None for single graph)
        device: Target device for computations
        dtype: Data type for computations

    Returns:
        torch.Tensor: Axial flux per edge [E]
    """
    # Handle empty graph case
    if edge_index.size(1) == 0:
        return torch.zeros(0, device=device, dtype=dtype)

    src, dst = edge_index[0], edge_index[1]
    r_ST = edge_feat_original.squeeze(-1)

    # Extract node features (already in original space)
    psi = node_feat_original[:, 0]      # hydraulic potential
    Temp = node_feat_original[:, 6]     # temperature [°C]

    # ---- Step 1: Compute osmotic pressure from concentrations
    # C_ST is already denormalized and converted to concentration

    # Osmotic pressure P_ST = C_ST * RT
    # In C++, TairK_phloem is global, but in batched case we need per-graph temperature
    RT = utils.compute_RT_per_node(
        Temp=Temp,
        batch_vec=batch_vec,
        R=config.R,
        device=device,
        dtype=dtype,
    )
    P_ST_osmotic = C_ST * RT

    # Convert hydraulic potential psi to hPa and add to osmotic pressure
    psi = psi * config.cmH2O_to_hPa
    P_ST = P_ST_osmotic + psi

    # ---- Step 2: Compute water flux from pressure gradients
    P_i = P_ST[src]
    P_j = P_ST[dst]

    JW_ST = (P_i - P_j) / r_ST

    # ---- Step 3: Select upstream concentration based on flow direction
    # With dP = P_i - P_j, positive JW_ST means P_i > P_j, so flow is i -> j
    # For sugar flux, we want the upstream (source) concentration
    # If flow is i -> j, then upstream is i (src), downstream is j (dst)
    # If flow is j -> i, then upstream is j (dst), downstream is i (src)
    C_i = C_ST[src]
    C_j = C_ST[dst]

    C_upstream = torch.where(JW_ST > 0, C_i, C_j)
    C_upstream = torch.clamp(C_upstream, min=0.0)

    # ---- Step 4: Sugar flux JS_ST = JW_ST * C_upstream
    J_ax = JW_ST * C_upstream

    return J_ax


def compute_flux_divergence(J_ax: torch.Tensor, edge_index: torch.Tensor, N: int, device: torch.device) -> torch.Tensor:
    """Compute divergence of flux to get net inflow per node.

    This implements the standard divergence convention: div(J) = sum of (flux_in - flux_out)
    for each node, which represents net mass gain (positive) or loss (negative).

    **CPlantBox Sign Convention Note:**
    CPlantBox's Delta_JS_ST uses the opposite sign (-div(J)):
        Delta2[src, edge] = -1  →  source nodes accumulate -J
        Delta2[dst, edge] = +1  →  destination nodes accumulate +J
        Result: Delta_JS_ST = -div(J) in standard notation

    However, CPlantBox's conservation equation negates it:
        dS/dt = -Delta_JS_ST + F_in - F_out = +div(J) + F_in - F_out

    Our implementation computes +div(J) directly (standard convention):
        For edge src→dst with flux J: src gets -J (loss), dst gets +J (gain)
        Conservation equation: dS/dt = div(J) + F_in - F_out

    Both formulations are physically equivalent and yield the same results.

    Args:
        J_ax: Axial flux per edge [E]
        edge_index: Edge connectivity [2, E]
        N: Number of nodes
        device: Target device for computations

    Returns:
        torch.Tensor: Net flux change per node [N]
    """
    # Initialize with same dtype as J_ax
    dS_dt_from_flux = torch.zeros(N, device=device, dtype=J_ax.dtype)

    # Handle empty graph case
    if J_ax.size(0) == 0:
        return dS_dt_from_flux

    src, dst = edge_index[0], edge_index[1]

    # Standard divergence convention (physically intuitive):
    # For edge src → dst with flux J_ax:
    #   - src node (upstream) gets: -J_ax (flux leaving, mass lost → negative divergence)
    #   - dst node (downstream) gets: +J_ax (flux arriving, mass gained → positive divergence)
    # This computes: div(J) = sum of (flux_in - flux_out) for each node
    #
    # NOTE: CPlantBox uses opposite sign convention! Their Delta_JS_ST represents -div(J).
    # When loading CPlantBox data, negate Delta_JS_ST to match this convention.
    # Conservation equation: dS/dt = div(J) + F_in - F_out ≈ 0
    dS_dt_from_flux.scatter_add_(0, src, -J_ax)  # Source loses flux (negative)
    dS_dt_from_flux.scatter_add_(0, dst, +J_ax)  # Destination gains flux (positive)

    return dS_dt_from_flux


def compute_flux_antisymmetry_error(
    J_ax: torch.Tensor,
    edge_index: torch.Tensor
) -> float:
    """Compute antisymmetry error for bidirectional flux predictions.

    For each bidirectional edge pair (i,j) and (j,i), compute:
    E_antisym = 1/E * sum(|J_ij + J_ji|)

    Perfect antisymmetry (J_ji = -J_ij) yields E_antisym = 0.

    Args:
        J_ax: Edge fluxes [E]
        edge_index: Edge connectivity [2, E]

    Returns:
        float: Mean antisymmetry error across all bidirectional pairs
    """
    if J_ax.size(0) == 0:
        return 0.0

    # Create edge pair lookup: (src, dst) -> edge_index
    src, dst = edge_index[0], edge_index[1]
    edge_dict = {}

    for e_idx in range(edge_index.size(1)):
        s = src[e_idx].item()
        d = dst[e_idx].item()
        edge_dict[(s, d)] = e_idx

    # Find bidirectional pairs and compute |J_ij + J_ji|
    antisym_errors = []
    visited = set()

    for e_idx in range(edge_index.size(1)):
        s = src[e_idx].item()
        d = dst[e_idx].item()

        # Skip if we already processed this pair
        if (s, d) in visited:
            continue

        # Look for reverse edge
        if (d, s) in edge_dict:
            e_rev = edge_dict[(d, s)]
            J_ij = J_ax[e_idx].item()
            J_ji = J_ax[e_rev].item()

            # Antisymmetry error: |J_ij + J_ji|
            antisym_errors.append(abs(J_ij + J_ji))

            # Mark both directions as visited
            visited.add((s, d))
            visited.add((d, s))

    if len(antisym_errors) == 0:
        return 0.0

    # Return mean antisymmetry error
    return sum(antisym_errors) / len(antisym_errors)


def compute_phloem_loading(
    C_ST: torch.Tensor,
    node_feat_original: torch.Tensor,
    params: dict,
    node_fields: dict,
    device: torch.device
) -> torch.Tensor:
    """Compute phloem loading rate F_in per node.

    Args:
        C_ST: Sucrose concentration per node [N] (already denormalized)
        node_feat_original: Node features in original space [N, D]
        params: Simulation and step parameters
        node_fields: Node field values
        device: Target device for computations

    Returns:
        torch.Tensor: Phloem loading rate per node [N]
    """
    # Extract node features (already in original space)
    len_leaf = node_feat_original[:, 2]

    # C_ST is already denormalized and in concentration units
    CSTi_positive = torch.clamp(C_ST, min=0.0)

    # Phloem loading with feedback inhibition
    F_in = (params["Vmaxloading"] * len_leaf) * node_fields["C_meso"] / \
           (params["Mloading"] + node_fields["C_meso"]) * \
           torch.exp(-CSTi_positive * params["beta_loading"])

    return F_in


def compute_sucrose_outflow(
    C_ST: torch.Tensor,
    node_feat_original: torch.Tensor,
    params: dict,
    node_fields: dict,
    smooth_width: float = 0.0,
) -> torch.Tensor:
    """Compute sucrose outflow F_out per node.

    Args:
        C_ST: Sucrose concentration per node [N] (already denormalized)
        node_feat_original: Node features in original space [N, D]
        params: Simulation and step parameters
        node_fields: Node field values
        device: Target device for computations

    Returns:
        torch.Tensor: Sucrose outflow rate per node [N]
    """
    # Extract node features (already in original space)
    Q_Rmmax = node_feat_original[:, 3]
    Q_Grmax = node_feat_original[:, 4]
    Q_Exudmax = node_feat_original[:, 5]
    Temp = node_feat_original[:, 6]

    # C_ST is already denormalized and in concentration units
    CSTi_positive = torch.clamp(C_ST, min=0.0)

    # Apply CSTimin threshold for usage
    raw = CSTi_positive - params["CSTimin"]
    if smooth_width > 0.0:
        CSTi_effective = smooth_relu(raw, smooth_width)
    else:
        CSTi_effective = torch.clamp(raw, min=0.0)

    raw_delta = CSTi_effective - node_fields["Csoil_node"]
    if smooth_width > 0.0:
        CSTi_delta = smooth_relu(raw_delta, smooth_width)
    else:
        CSTi_delta = torch.clamp(raw_delta, min=0.0)

    # Temperature-dependent maintenance respiration
    R_mmax = (Q_Rmmax + params["krm2v"] * CSTi_effective) * \
             torch.pow(params["Q10"], (Temp - params["TrefQ10"]) / 10.0)

    # Michaelis-Menten kinetics for sucrose usage
    F_out_MM = (R_mmax + Q_Grmax) * (CSTi_effective / (CSTi_effective + params["KMfu"]))

    # Root exudation based on concentration gradient
    Exud = CSTi_delta * Q_Exudmax

    return F_out_MM + Exud


def log_physics_values(y_pred: torch.Tensor, data: Data, model_output=None, phase: str = None, epoch: int = None, batch_idx: int = None):
    """Log true and predicted physics values for analysis (no loss computation).

    This function computes and logs all physics quantities (C_ST, J_ax, divergence,
    F_in, F_out) for both true and predicted values. It's useful for evaluating
    physical consistency of models trained with DATA_ONLY loss.

    Args:
        y_pred: Predicted sucrose content [N, 1] or dict for operator model
        data: Graph data containing topology, features, and targets
        model_output: For operator models, dict containing edge_fluxes and divergences
        phase: Training phase ('train', 'val', 'test')
        epoch: Current epoch number (optional)
        batch_idx: Batch index (optional)

    Returns:
        Tuple of (PhysicsMetrics, PhysicsErrorMetrics):
            - Physics metrics for terminal display
            - Physics error metrics (MSE, RMSE, Relative Error)
            Returns (None, None) if logging disabled
    """
    if not _ENABLE_PHYSICS_LOGGING and not _ENABLE_METRICS_LOGGING:
        return None, None

    # Handle operator model case
    is_operator_model = isinstance(y_pred, dict)
    if is_operator_model:
        model_output = y_pred
        y_pred = model_output['predictions']
        edge_fluxes_pred = model_output['edge_fluxes']
        divergence_pred = model_output['divergences']

    device = y_pred.device
    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

    # Inverse-transform features
    node_feat_standardized = data.node_feat.to(device)
    node_feat_original = data.feature_scaler.inv_transform(node_feat_standardized)

    edge_feat_standardized = data.edge_feat.to(device)
    edge_feat_original = data.edge_scaler.inv_transform(edge_feat_standardized)

    edge_index = data.edge_index.to(device)
    vol_ST = node_feat_original[:, 1]

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # Compute predicted physics terms
    S_ST_pred = data.target_scaler.inv_transform(y_pred).squeeze(-1)
    C_ST_pred = S_ST_pred / vol_ST

    # Compute true physics terms
    y_true = data.y.to(device)
    S_ST_true = data.target_scaler.inv_transform(y_true).squeeze(-1)
    C_ST_true = S_ST_true / vol_ST

    if is_operator_model:
        # Operator model: use predicted fluxes/divergences directly
        J_ax_true = compute_axial_flux(
            C_ST_true, node_feat_original, edge_feat_original,
            edge_index, batch_vec, device, y_pred.dtype
        )
        dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

        F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
        F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, smooth_width=SMOOTH_WIDTH)

        F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
        F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, smooth_width=0.0)

        dS_dt_tot_true = dS_dt_from_flux_true + F_in_true - F_out_true
        dS_dt_tot_pred = divergence_pred + F_in_pred - F_out_pred

        # Compute error metrics
        physics_errors = _compute_physics_error_metrics(
            S_ST_true,
            S_ST_pred,
            J_ax_true,
            edge_fluxes_pred,
            dS_dt_from_flux_true,
            divergence_pred,
            dS_dt_tot_true,
            dS_dt_tot_pred,
            edge_index=edge_index,
            batch_vec=batch_vec
        )

        # Log metrics to CSV file
        log_metrics_to_file(physics_errors, epoch=epoch, phase=phase, batch_idx=batch_idx)

        if _ENABLE_PHYSICS_LOGGING:
            with open(_PHYSICS_LOG_PATH, "a") as f:
                msg = _log_header("DEBUG OUTPUT - OPERATOR MODEL (DATA-ONLY)", batch_vec, data=data, phase=phase)
                msg += _log_concentrations(C_ST_true, C_ST_pred)
                msg += _log_sucrose_contents(S_ST_true, S_ST_pred)
                msg += _log_fluxes(J_ax_true, edge_fluxes_pred)
                msg += _log_divergence(dS_dt_from_flux_true, divergence_pred)
                msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
                msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
                msg += _log_comparison_metrics(physics_errors)
                msg += f"{'='*60}\n"
                f.write(msg)

        # Compute averaged metrics for terminal display (operator model)
        if batch_vec is not None:
            # Batched case: compute per-graph averages
            F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
            F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
            divergence_per_graph = scatter_mean(divergence_pred.detach().abs(), batch_vec, dim=0)
            dS_dt_per_graph = scatter_mean(dS_dt_tot_pred.detach().abs(), batch_vec, dim=0)

            if edge_fluxes_pred.size(0) > 0:
                edge_batch = batch_vec[edge_index[0].to(device)]
                J_ax_per_graph = scatter_mean(edge_fluxes_pred.detach().abs(), edge_batch, dim=0)
                J_ax_avg = J_ax_per_graph.mean().item()
            else:
                J_ax_avg = 0.0

            return (
                PhysicsMetrics(
                    J_ax=J_ax_avg,
                    F_in=F_in_per_graph.mean().item(),
                    F_out=F_out_per_graph.mean().item(),
                    dS_dt_from_flux=divergence_per_graph.mean().item(),
                    dS_dt_tot=dS_dt_per_graph.mean().item()
                ),
                physics_errors
            )
        else:
            # Single graph case
            return (
                PhysicsMetrics(
                    J_ax=edge_fluxes_pred.detach().abs().mean().item() if edge_fluxes_pred.size(0) > 0 else 0.0,
                    F_in=F_in_pred.detach().mean().item(),
                    F_out=F_out_pred.detach().mean().item(),
                    dS_dt_from_flux=divergence_pred.detach().abs().mean().item(),
                    dS_dt_tot=dS_dt_tot_pred.detach().abs().mean().item()
                ),
                physics_errors
            )
    else:
        # NNConv model: reconstruct fluxes from predictions
        J_ax_pred = compute_axial_flux(C_ST_pred, node_feat_original, edge_feat_original,
                                       edge_index, batch_vec, device, y_pred.dtype)
        dS_dt_from_flux_pred = compute_flux_divergence(J_ax_pred, edge_index, N, device)

        F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
        F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, smooth_width=SMOOTH_WIDTH)
        dS_dt_tot_pred = dS_dt_from_flux_pred + F_in_pred - F_out_pred

        # Compute true values
        J_ax_true = compute_axial_flux(C_ST_true, node_feat_original, edge_feat_original,
                                       edge_index, batch_vec, device, y_pred.dtype)
        dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

        F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
        F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, smooth_width=0.0)
        dS_dt_tot_true = dS_dt_from_flux_true + F_in_true - F_out_true

        # Compute error metrics
        physics_errors = _compute_physics_error_metrics(
            S_ST_true,
            S_ST_pred,
            J_ax_true,
            J_ax_pred,
            dS_dt_from_flux_true,
            dS_dt_from_flux_pred,
            dS_dt_tot_true,
            dS_dt_tot_pred,
            edge_index=edge_index,
            batch_vec=batch_vec
        )

        # Log metrics to CSV file
        log_metrics_to_file(physics_errors, epoch=epoch, phase=phase, batch_idx=batch_idx)

        if _ENABLE_PHYSICS_LOGGING:
            with open(_PHYSICS_LOG_PATH, "a") as f:
                msg = _log_header("DEBUG OUTPUT - NNCONV MODEL (DATA-ONLY)", batch_vec, data=data, phase=phase)
                msg += _log_concentrations(C_ST_true, C_ST_pred)
                msg += _log_sucrose_contents(S_ST_true, S_ST_pred)
                msg += _log_fluxes(J_ax_true, J_ax_pred)
                msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
                msg += _log_divergence(dS_dt_from_flux_true, dS_dt_from_flux_pred)
                msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
                msg += _log_comparison_metrics(physics_errors)
                msg += f"{'='*60}\n"
                f.write(msg)

        # Compute averaged metrics for terminal display (NNConv model)
        if batch_vec is not None:
            # Batched case: compute per-graph averages
            F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
            F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
            dS_dt_from_flux_per_graph = scatter_mean(dS_dt_from_flux_pred.detach().abs(), batch_vec, dim=0)
            dS_dt_per_graph = scatter_mean(dS_dt_tot_pred.detach().abs(), batch_vec, dim=0)

            if J_ax_pred.size(0) > 0:
                edge_batch = batch_vec[edge_index[0].to(device)]
                J_ax_per_graph = scatter_mean(J_ax_pred.detach().abs(), edge_batch, dim=0)
                J_ax_avg = J_ax_per_graph.mean().item()
            else:
                J_ax_avg = 0.0

            return (
                PhysicsMetrics(
                    J_ax=J_ax_avg,
                    F_in=F_in_per_graph.mean().item(),
                    F_out=F_out_per_graph.mean().item(),
                    dS_dt_from_flux=dS_dt_from_flux_per_graph.mean().item(),
                    dS_dt_tot=dS_dt_per_graph.mean().item()
                ),
                physics_errors
            )
        else:
            # Single graph case
            return (
                PhysicsMetrics(
                    J_ax=J_ax_pred.detach().abs().mean().item() if J_ax_pred.size(0) > 0 else 0.0,
                    F_in=F_in_pred.detach().mean().item(),
                    F_out=F_out_pred.detach().mean().item(),
                    dS_dt_from_flux=dS_dt_from_flux_pred.detach().abs().mean().item(),
                    dS_dt_tot=dS_dt_tot_pred.detach().abs().mean().item()
                ),
                physics_errors
            )


def physics_residual(
    y_pred: torch.Tensor,
    data: Data,
    phase: str = None,
    epoch: int = None,
    batch_idx: int = None
):
    """Compute physics-informed residual term based on sucrose transport equations.

    Implements the governing equation for content-based sucrose transport in sieve-tubes:
    dS/dt = divJ + (F_in - F_out) ≈ 0

    For steady-state assumption, physics loss is computed by minimizing:
        residual = divJ + F_in - F_out

    Args:
        y_pred: Predicted sucrose content [N, 1] (standardized)
        data: Graph data containing topology, features, simulation parameters, and node fields
        phase: Training phase ('train', 'val', 'test') for logging
        epoch: Current epoch number (optional)
        batch_idx: Batch index (optional)

    Returns:
        tuple: (residual_loss, physics_components_dict, physics_error_metrics) where:
            - physics_components_dict contains {'J_ax', 'F_in', 'F_out', 'dS_dt_from_flux', 'dS_dt_tot'}
            - physics_error_metrics is PhysicsErrorMetrics (or None if logging disabled)
    """
    device = y_pred.device
    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

    # Inverse-transform node/edge features
    node_feat_standardized = data.node_feat.to(device)
    node_feat_original = data.feature_scaler.inv_transform(node_feat_standardized)

    edge_feat_standardized = data.edge_feat.to(device)
    edge_feat_original = data.edge_scaler.inv_transform(edge_feat_standardized)

    edge_index = data.edge_index.to(device)
    vol_ST = node_feat_original[:, 1]

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # ============================
    # Compute physics terms from PREDICTIONS
    # ============================
    S_ST_pred = data.target_scaler.inv_transform(y_pred).squeeze(-1)
    C_ST_pred = S_ST_pred / vol_ST

    # Compute axial flux and its divergence from predictions
    J_ax_pred = compute_axial_flux(C_ST_pred, node_feat_original, edge_feat_original,
                                   edge_index, batch_vec, device, y_pred.dtype)
    dS_dt_from_flux_pred = compute_flux_divergence(J_ax_pred, edge_index, N, device)

    # Always compute physics terms from true values for error metrics
    y_true = data.y.to(device)
    S_ST_true = data.target_scaler.inv_transform(y_true).squeeze(-1)
    C_ST_true = S_ST_true / vol_ST

    J_ax_true = compute_axial_flux(C_ST_true, node_feat_original, edge_feat_original,
                                   edge_index, batch_vec, device, y_pred.dtype)
    dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

    F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
    F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, smooth_width=0.0)
    dS_dt_tot_true = dS_dt_from_flux_true + F_in_true - F_out_true

    # ============================
    # Compute F_in and F_out from TRUE values (external to the system)
    # Detach to prevent gradients flowing through these terms
    # ============================
    F_in_true_detached = F_in_true.detach()
    F_out_true_detached = F_out_true.detach()

    # Also compute from predictions for logging/comparison (not used in loss)
    F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
    F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, smooth_width=SMOOTH_WIDTH)

    # Total physics-based derivative for residual calculation
    # Use TRUE values of F_in and F_out (detached)
    dS_dt_tot_pred = dS_dt_from_flux_pred + F_in_true_detached - F_out_true_detached

    # Compute error metrics
    physics_errors = _compute_physics_error_metrics(
        S_ST_true,
        S_ST_pred,
        J_ax_true,
        J_ax_pred,
        dS_dt_from_flux_true,
        dS_dt_from_flux_pred,
        dS_dt_tot_true,
        dS_dt_tot_pred,
        edge_index=edge_index,
        batch_vec=batch_vec
    )

    # Log metrics to CSV file
    log_metrics_to_file(physics_errors, epoch=epoch, phase=phase, batch_idx=batch_idx)

    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_header("DEBUG OUTPUT - NNCONV MODEL (PHYSICS RESIDUAL)", batch_vec, data=data, phase=phase)
            msg += _log_concentrations(C_ST_true, C_ST_pred)
            msg += _log_sucrose_contents(S_ST_true, S_ST_pred)
            msg += _log_fluxes(J_ax_true, J_ax_pred)
            msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
            msg += _log_divergence(dS_dt_from_flux_true, dS_dt_from_flux_pred)
            msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
            msg += _log_comparison_metrics(physics_errors)
            msg += f"{'='*60}\n"
            f.write(msg)

    # ============================
    # Compute physics residual
    # ============================
    # Minimize dS/dt_physics directly
    residual = dS_dt_tot_pred

    # Add penalty for negative concentrations (after denormalization)
    # This encourages the model to respect the physical constraint C_ST >= 0
    negative_concentration_penalty = torch.relu(-C_ST_pred).pow(2).mean()

    # Use adaptive normalization based on current residual scale
    # Use 90th percentile of absolute values as scale (robust to outliers)
    # with torch.no_grad():
    #     scale = residual.abs().quantile(0.9).clamp(min=0.1, max=10000.0)
    scale = 1.0
    residual_node = (residual / scale).pow(2)

    # Average per graph first, then across graphs
    if batch_vec is not None:
        residual_per_graph = scatter_mean(residual_node, batch_vec, dim=0)
        physics_loss = residual_per_graph.mean()
    else:
        physics_loss = residual_node.mean()

    # Combine physics residual with non-negativity penalty
    # Weight the penalty strongly to enforce physical constraint
    loss = physics_loss + PENALTY_WEIGHT * negative_concentration_penalty

    # Prepare detailed physics components for logging (use predicted values)
    if batch_vec is not None:
        # Batched case: compute per-graph averages using scatter_mean
        F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
        F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
        dS_dt_from_flux_pred_per_graph = scatter_mean(dS_dt_from_flux_pred.detach().abs(), batch_vec, dim=0)
        dS_dt_tot_pred_per_graph = scatter_mean(dS_dt_tot_pred.detach().abs(), batch_vec, dim=0)

        # Edge-level quantities: need edge-to-graph mapping
        if J_ax_pred.size(0) > 0:
            edge_batch = batch_vec[data.edge_index[0].to(device)]
            J_ax_per_graph = scatter_mean(J_ax_pred.detach().abs(), edge_batch, dim=0)
            J_ax_avg = J_ax_per_graph.mean()
        else:
            J_ax_avg = torch.tensor(0.0, device=device)

        loss_dict = {
            'J_ax': J_ax_avg,
            'F_in': F_in_per_graph.mean(),
            'F_out': F_out_per_graph.mean(),
            'dS_dt_from_flux': dS_dt_from_flux_pred_per_graph.mean(),
            'dS_dt_tot': dS_dt_tot_pred_per_graph.mean()
        }
    else:
        # Single graph case: simple mean across nodes/edges
        loss_dict = {
            'J_ax': J_ax_pred.detach().abs().mean() if J_ax_pred.size(0) > 0 else torch.tensor(0.0, device=device),
            'F_in': F_in_pred.detach().mean(),
            'F_out': F_out_pred.detach().mean(),
            'dS_dt_from_flux': dS_dt_from_flux_pred.detach().abs().mean(),
            'dS_dt_tot': dS_dt_tot_pred.detach().abs().mean()
        }

    return loss, loss_dict, physics_errors


def physics_residual_operator(
    model_output: dict,
    data: Data,
    phase: str = None,
    epoch: int = None,
    batch_idx: int = None
) -> tuple[torch.Tensor, dict]:
    """Compute physics residual for operator-based GNN using LEARNED operator.

    This function uses the edge fluxes and divergences directly predicted by the
    operator model in the physics loss. The learned operator is supervised by
    the physics residual.

    The conservation law (steady-state assumption):
        div(J) + F_in - F_out ≈ 0

    Args:
        model_output: Dict containing:
            - 'predictions': [N, 1] sucrose content predictions (standardized)
            - 'edge_fluxes': [E] predicted edge fluxes (USED in loss)
            - 'divergences': [N] divergence values (USED in loss)
        data: Graph data containing features and parameters
        phase: Training phase ('train', 'val', 'test') for logging

    Returns:
        tuple: (residual_loss, physics_components_dict, physics_error_metrics)
    """
    device = model_output['predictions'].device
    y_pred = model_output['predictions']
    edge_fluxes_pred = model_output['edge_fluxes']
    divergence_pred = model_output['divergences']

    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

    # Inverse-transform features to original space
    node_feat_standardized = data.node_feat.to(device)
    node_feat_original = data.feature_scaler.inv_transform(node_feat_standardized)

    edge_index = data.edge_index.to(device)
    vol_ST = node_feat_original[:, 1]

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # Convert predicted content to concentration
    S_ST_pred = data.target_scaler.inv_transform(y_pred).squeeze(-1)
    C_ST_pred = S_ST_pred / vol_ST

    # Always compute ground truth physics terms for error metrics
    y_true = data.y.to(device)
    S_ST_true = data.target_scaler.inv_transform(y_true).squeeze(-1)
    C_ST_true = S_ST_true / vol_ST

    # Reconstruct true fluxes for comparison
    edge_feat_standardized = data.edge_feat.to(device)
    edge_feat_original = data.edge_scaler.inv_transform(edge_feat_standardized)

    J_ax_true = compute_axial_flux(
        C_ST_true, node_feat_original, edge_feat_original,
        edge_index, batch_vec, device, y_pred.dtype
    )
    dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

    F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
    F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, smooth_width=0.0)
    dS_dt_tot_true = dS_dt_from_flux_true + F_in_true - F_out_true

    # ============================
    # Compute F_in and F_out from TRUE values (external to the system)
    # Detach to prevent gradients flowing through these terms
    # ============================
    F_in_true_detached = F_in_true.detach()
    F_out_true_detached = F_out_true.detach()

    # Also compute from predictions for logging/comparison (not used in loss)
    F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
    F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, smooth_width=SMOOTH_WIDTH)

    # Physics residual using LEARNED divergence from operator model
    # dS/dt = divergence_pred + F_in_true - F_out_true ≈ 0
    # Use TRUE values of F_in and F_out (detached)
    dS_dt_tot_pred = divergence_pred + F_in_true_detached - F_out_true_detached

    # Compute error metrics
    physics_errors = _compute_physics_error_metrics(
        S_ST_true,
        S_ST_pred,
        J_ax_true,
        edge_fluxes_pred,
        dS_dt_from_flux_true,
        divergence_pred,
        dS_dt_tot_true,
        dS_dt_tot_pred,
        edge_index=edge_index,
        batch_vec=batch_vec
    )

    # Log metrics to CSV file
    log_metrics_to_file(physics_errors, epoch=epoch, phase=phase, batch_idx=batch_idx)

    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_header("DEBUG OUTPUT - OPERATOR MODEL PHYSICS RESIDUAL (LEARNED)", batch_vec, data=data, phase=phase)
            msg += _log_concentrations(C_ST_true, C_ST_pred)
            msg += _log_sucrose_contents(S_ST_true, S_ST_pred)
            msg += _log_fluxes(J_ax_true, edge_fluxes_pred)
            msg += _log_divergence(dS_dt_from_flux_true, divergence_pred)
            msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
            msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
            msg += _log_comparison_metrics(physics_errors)
            msg += f"{'='*60}\n"
            f.write(msg)

    # ============================
    # Compute physics residual
    # ============================
    # Minimize dS/dt_physics directly
    residual = dS_dt_tot_pred

    # Add penalty for negative concentrations (after denormalization)
    # This encourages the model to respect the physical constraint C_ST >= 0
    negative_concentration_penalty = torch.relu(-C_ST_pred).pow(2).mean()

    # Adaptive normalization
    # with torch.no_grad():
    #     scale = residual.abs().quantile(0.9).clamp(min=0.1, max=10000.0)
    scale = 1.0
    residual_node = (residual / scale).pow(2)

    # Average per graph first, then across graphs
    if batch_vec is not None:
        residual_per_graph = scatter_mean(residual_node, batch_vec, dim=0)
        physics_loss = residual_per_graph.mean()
    else:
        physics_loss = residual_node.mean()

    # Combine physics residual with non-negativity penalty
    # Weight the penalty strongly to enforce physical constraint
    loss = physics_loss + PENALTY_WEIGHT * negative_concentration_penalty

    # Prepare loss dict for logging (use LEARNED fluxes/divergences)
    if batch_vec is not None:
        F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
        F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
        divergence_per_graph = scatter_mean(divergence_pred.detach().abs(), batch_vec, dim=0)
        dS_dt_per_graph = scatter_mean(dS_dt_tot_pred.detach().abs(), batch_vec, dim=0)

        # Edge-level: map edges to graphs
        if edge_fluxes_pred.size(0) > 0:
            edge_batch = batch_vec[edge_index[0].to(device)]
            J_ax_per_graph = scatter_mean(edge_fluxes_pred.detach().abs(), edge_batch, dim=0)
            J_ax_avg = J_ax_per_graph.mean()
        else:
            J_ax_avg = torch.tensor(0.0, device=device)

        loss_dict = {
            'J_ax': J_ax_avg,
            'F_in': F_in_per_graph.mean(),
            'F_out': F_out_per_graph.mean(),
            'dS_dt_from_flux': divergence_per_graph.mean(),
            'dS_dt_tot': dS_dt_per_graph.mean()
        }
    else:
        loss_dict = {
            'J_ax': edge_fluxes_pred.detach().abs().mean() if edge_fluxes_pred.size(0) > 0 else torch.tensor(0.0, device=device),
            'F_in': F_in_pred.detach().mean(),
            'F_out': F_out_pred.detach().mean(),
            'dS_dt_from_flux': divergence_pred.detach().abs().mean(),
            'dS_dt_tot': dS_dt_tot_pred.detach().abs().mean()
        }

    return loss, loss_dict, physics_errors


def physics_residual_operator_analytical(
    model_output: dict,
    data: Data,
    phase: str = None,
    epoch: int = None,
    batch_idx: int = None
) -> tuple[torch.Tensor, dict]:
    """Compute physics residual for operator-based GNN using ANALYTICAL calculations.

    This function computes fluxes and divergences analytically from predicted
    concentrations (identical to NNConv approach), while still accepting operator
    model outputs for logging/comparison purposes. The learned operator outputs
    are NOT used in the physics loss.

    This allows fair comparison between NNConv and Operator architectures with
    identical physics supervision.

    The conservation law (steady-state assumption):
        div(J) + F_in - F_out ≈ 0

    Args:
        model_output: Dict containing:
            - 'predictions': [N, 1] sucrose content predictions (standardized)
            - 'edge_fluxes': [E] predicted edge fluxes (logged but NOT used)
            - 'divergences': [N] divergence values (logged but NOT used)
        data: Graph data containing features and parameters
        phase: Training phase ('train', 'val', 'test') for logging

    Returns:
        tuple: (residual_loss, physics_components_dict, physics_error_metrics)
    """
    device = model_output['predictions'].device
    y_pred = model_output['predictions']

    # Store learned operator outputs for logging comparison (NOT used in loss)
    edge_fluxes_learned = model_output.get('edge_fluxes', None)
    divergence_learned = model_output.get('divergences', None)

    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

    # Inverse-transform features to original space
    node_feat_standardized = data.node_feat.to(device)
    node_feat_original = data.feature_scaler.inv_transform(node_feat_standardized)

    edge_index = data.edge_index.to(device)
    vol_ST = node_feat_original[:, 1]

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # Convert predicted content to concentration
    S_ST_pred = data.target_scaler.inv_transform(y_pred).squeeze(-1)
    C_ST_pred = S_ST_pred / vol_ST

    # ============================
    # ANALYTICAL CALCULATION (used in loss)
    # ============================
    edge_feat_standardized = data.edge_feat.to(device)
    edge_feat_original = data.edge_scaler.inv_transform(edge_feat_standardized)

    # Compute fluxes analytically from predicted concentrations
    J_ax_pred = compute_axial_flux(
        C_ST_pred, node_feat_original, edge_feat_original,
        edge_index, batch_vec, device, y_pred.dtype
    )
    dS_dt_from_flux_pred = compute_flux_divergence(J_ax_pred, edge_index, N, device)

    # Always compute ground truth physics terms for error metrics
    y_true = data.y.to(device)
    S_ST_true = data.target_scaler.inv_transform(y_true).squeeze(-1)
    C_ST_true = S_ST_true / vol_ST

    J_ax_true = compute_axial_flux(
        C_ST_true, node_feat_original, edge_feat_original,
        edge_index, batch_vec, device, y_pred.dtype
    )
    dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

    F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
    F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, smooth_width=0.0)
    dS_dt_tot_true = dS_dt_from_flux_true + F_in_true - F_out_true

    # ============================
    # Compute F_in and F_out from TRUE values (external to the system)
    # Detach to prevent gradients flowing through these terms
    # ============================
    F_in_true_detached = F_in_true.detach()
    F_out_true_detached = F_out_true.detach()

    # Also compute from predictions for logging/comparison (not used in loss)
    F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
    F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, smooth_width=SMOOTH_WIDTH)

    # Total rate of change from physics (analytical)
    # Use TRUE values of F_in and F_out (detached)
    dS_dt_tot_pred = dS_dt_from_flux_pred + F_in_true_detached - F_out_true_detached

    # Compute error metrics
    physics_errors = _compute_physics_error_metrics(
        S_ST_true,
        S_ST_pred,
        J_ax_true,
        J_ax_pred,
        dS_dt_from_flux_true,
        dS_dt_from_flux_pred,
        dS_dt_tot_true,
        dS_dt_tot_pred,
        edge_index=edge_index,
        batch_vec=batch_vec
    )

    # Log metrics to CSV file
    log_metrics_to_file(physics_errors, epoch=epoch, phase=phase, batch_idx=batch_idx)

    # ============================
    # LOGGING
    # ============================
    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_header("DEBUG OUTPUT - OPERATOR MODEL PHYSICS RESIDUAL (ANALYTICAL)", batch_vec, data=data, phase=phase)
            msg += _log_concentrations(C_ST_true, C_ST_pred)
            msg += _log_sucrose_contents(S_ST_true, S_ST_pred)
            msg += _log_fluxes(J_ax_true, J_ax_pred)
            msg += _log_divergence(dS_dt_from_flux_true, dS_dt_from_flux_pred)
            msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
            msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
            msg += _log_comparison_metrics(physics_errors)
            msg += f"{'='*60}\n"
            f.write(msg)

    # ============================
    # Compute physics residual
    # ============================
    # Minimize dS/dt_physics directly
    residual = dS_dt_tot_pred

    # Add penalty for negative concentrations
    negative_concentration_penalty = torch.relu(-C_ST_pred).pow(2).mean()

    # Use adaptive normalization
    scale = 1.0
    residual_node = (residual / scale).pow(2)

    # Average per graph first, then across graphs
    if batch_vec is not None:
        residual_per_graph = scatter_mean(residual_node, batch_vec, dim=0)
        physics_loss = residual_per_graph.mean()
    else:
        physics_loss = residual_node.mean()

    # Combine physics residual with non-negativity penalty
    loss = physics_loss + PENALTY_WEIGHT * negative_concentration_penalty

    # Prepare loss dict for logging (use ANALYTICAL values)
    if batch_vec is not None:
        F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
        F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
        dS_dt_from_flux_per_graph = scatter_mean(dS_dt_from_flux_pred.detach().abs(), batch_vec, dim=0)
        dS_dt_per_graph = scatter_mean(dS_dt_tot_pred.detach().abs(), batch_vec, dim=0)

        if J_ax_pred.size(0) > 0:
            edge_batch = batch_vec[edge_index[0].to(device)]
            J_ax_per_graph = scatter_mean(J_ax_pred.detach().abs(), edge_batch, dim=0)
            J_ax_avg = J_ax_per_graph.mean()
        else:
            J_ax_avg = torch.tensor(0.0, device=device)

        loss_dict = {
            'J_ax': J_ax_avg,
            'F_in': F_in_per_graph.mean(),
            'F_out': F_out_per_graph.mean(),
            'dS_dt_from_flux': dS_dt_from_flux_per_graph.mean(),
            'dS_dt_tot': dS_dt_per_graph.mean()
        }
    else:
        loss_dict = {
            'J_ax': J_ax_pred.detach().abs().mean() if J_ax_pred.size(0) > 0 else torch.tensor(0.0, device=device),
            'F_in': F_in_pred.detach().mean(),
            'F_out': F_out_pred.detach().mean(),
            'dS_dt_from_flux': dS_dt_from_flux_pred.detach().abs().mean(),
            'dS_dt_tot': dS_dt_tot_pred.detach().abs().mean()
        }

    return loss, loss_dict, physics_errors
