from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path

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
            f.write(f"{'='*60}")
            f.write(f"Physics Debug Log - Session Started\n")
            f.write(f"{'='*60}\n")
        print(f"  Log file initialized: {_PHYSICS_LOG_PATH}")

    print(f"{'='*60}\n")


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
    msg += f"F_in_pred (mean): {F_in_pred.mean().detach().cpu().item():.6e}\n"
    msg += f"F_in_pred (first {n_samples}): {F_in_pred[:n_samples].detach().cpu().numpy()}\n"
    msg += f"\n"
    msg += f"F_out_true (mean): {F_out_true.mean().detach().cpu().item():.6e}\n"
    msg += f"F_out_true (first {n_samples}): {F_out_true[:n_samples].detach().cpu().numpy()}\n"
    msg += f"F_out_pred (mean): {F_out_pred.mean().detach().cpu().item():.6e}\n"
    msg += f"F_out_pred (first {n_samples}): {F_out_pred[:n_samples].detach().cpu().numpy()}\n"
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


def _log_comparison_metrics(physics_errors: PhysicsErrorMetrics, temporal_tensors: dict = None):
    """Log comparison metrics from PhysicsErrorMetrics.

    Args:
        physics_errors: PhysicsErrorMetrics object containing MSE, RMSE, and Relative Error
        temporal_tensors: Optional dict with temporal tensors to display. Expected keys:
            {
                'pred': {
                    'dS_dt_from_state': Tensor or None,
                    'dS_dt_from_physics': Tensor or None,   # typically dS_dt_tot_pred
                },
                'true': {
                    'dS_dt_from_state': Tensor or None,
                    'dS_dt_from_physics': Tensor or None,   # typically dS_dt_tot_true
                }
            }

    Returns:
        str: Formatted comparison metrics log
    """
    msg = f"\n--- PHYSICS ERROR METRICS ---\n"

    # J_ax errors
    msg += f"J_ax      | MSE: {physics_errors.J_ax_mse:.6e}, RMSE: {physics_errors.J_ax_rmse:.6e}, RelErr: {physics_errors.J_ax_rel_error:.6e}\n"

    # Antisymmetry error (both nnconv and operator models)
    if physics_errors.J_ax_antisym_error >= 0:
        msg += f"J_ax      | Antisymmetry Error: {physics_errors.J_ax_antisym_error:.6e}\n"
    # divJ (divergence) errors - mass conservation quality
    msg += f"divJ      | MSE: {physics_errors.divJ_mse:.6e}, RMSE: {physics_errors.divJ_rmse:.6e}, RelErr: {physics_errors.divJ_rel_error:.6e}, Corr: {physics_errors.divJ_correlation:.6f}\n"

    # F_in errors
    msg += f"F_in      | MSE: {physics_errors.F_in_mse:.6e}, RMSE: {physics_errors.F_in_rmse:.6e}, RelErr: {physics_errors.F_in_rel_error:.6e}\n"

    # F_out errors
    msg += f"F_out     | MSE: {physics_errors.F_out_mse:.6e}, RMSE: {physics_errors.F_out_rmse:.6e}, RelErr: {physics_errors.F_out_rel_error:.6e}\n"

    # dS_dt_tot (total residual) errors
    msg += f"dS_dt_tot | MSE: {physics_errors.dS_dt_tot_mse:.6e}, RMSE: {physics_errors.dS_dt_tot_rmse:.6e}, RelErr: {physics_errors.dS_dt_tot_rel_error:.6e}\n"

    # Flux direction consistency (physical credibility)
    msg += f"\n--- FLUX DIRECTION CONSISTENCY ---\n"
    msg += f"J_ax Sign Accuracy: {physics_errors.J_ax_sign_accuracy:.4f} ({physics_errors.J_ax_sign_accuracy*100:.2f}%)\n"
    msg += f"J_ax Reversal Rate: {physics_errors.J_ax_reversal_rate:.4f} ({physics_errors.J_ax_reversal_rate*100:.2f}%)\n"
    msg += f"ΔC Sign Accuracy:   {physics_errors.delta_C_sign_accuracy:.4f} ({physics_errors.delta_C_sign_accuracy*100:.2f}%)\n"

    # Physics score (dimensionless residual-based consistency)
    msg += f"\n--- PHYSICS SCORE (CONSERVATION QUALITY) ---\n"
    msg += f"Normalized Residual Error: {physics_errors.physics_rel_error:.6f}\n"
    msg += f"  (Interpretation: < 0.05 tight, ~ 1 moderate violation, >> 1 poor conservation)\n"
    msg += f"Physics Satisfaction Rate: {physics_errors.physics_satisfaction_rate:.4f} ({physics_errors.physics_satisfaction_rate*100:.2f}%)\n"
    msg += f"  (Nodes within 1% tolerance of local source/sink scale)\n"

    # Temporal consistency (time-series mode only)
    msg += f"\n--- TEMPORAL CONSISTENCY (TIME-SERIES MODE) ---\n"
    msg += f"Predictions:\n"
    msg += f"  Temporal Rel Error: {physics_errors.temporal_rel_error_pred:.6f}\n"
    msg += f"  Temporal Satisfaction Rate: {physics_errors.temporal_consistency_pred:.4f} ({physics_errors.temporal_consistency_pred*100:.2f}%)\n"
    msg += f"Ground Truth:\n"
    msg += f"  Temporal Rel Error: {physics_errors.temporal_rel_error_true:.6f}\n"
    msg += f"  Temporal Satisfaction Rate: {physics_errors.temporal_consistency_true:.4f} ({physics_errors.temporal_consistency_true*100:.2f}%)\n"
    msg += f"  (Validates that ground truth data has temporal consistency)\n"

    # If caller provided tensors for temporal debugging, print first 10 node values
    if temporal_tensors is not None:
        try:
            pred = temporal_tensors.get('pred', {})
            true = temporal_tensors.get('true', {})

            if pred:
                dstate = pred.get('dS_dt_from_state', None)
                dphys = pred.get('dS_dt_from_physics', None)
                if dstate is not None:
                    msg += f"\n--- TEMPORAL VARIABLES (PREDICTIONS) ---\n"
                    msg += f"dS_dt_from_state (first 10): {dstate[:10].detach().cpu().numpy()}\n"
                if dphys is not None:
                    msg += f"dS_dt_from_physics (first 10): {dphys[:10].detach().cpu().numpy()}\n"
                if dstate is not None and dphys is not None:
                    resid = dstate - dphys
                    msg += f"residual = dS_dt_state - dS_dt_physics (first 10): {resid[:10].detach().cpu().numpy()}\n"

            if true:
                dstate_t = true.get('dS_dt_from_state', None)
                dphys_t = true.get('dS_dt_from_physics', None)
                if dstate_t is not None:
                    msg += f"\n--- TEMPORAL VARIABLES (GROUND TRUTH) ---\n"
                    msg += f"dS_dt_from_state (first 10): {dstate_t[:10].detach().cpu().numpy()}\n"
                if dphys_t is not None:
                    msg += f"dS_dt_from_physics (first 10): {dphys_t[:10].detach().cpu().numpy()}\n"
                if dstate_t is not None and dphys_t is not None:
                    resid_t = dstate_t - dphys_t
                    msg += f"residual = dS_dt_state - dS_dt_physics (first 10): {resid_t[:10].detach().cpu().numpy()}\n"
        except Exception:
            # Defensive: don't let logging fail
            msg += "\n(Note: failed to dump temporal tensors for debugging)\n"

    return msg


def compute_flux_direction_metrics(
    J_ax_true: torch.Tensor,
    J_ax_pred: torch.Tensor,
    C_ST_true: torch.Tensor,
    C_ST_pred: torch.Tensor,
    edge_index: torch.Tensor
) -> tuple[float, float, float]:
    """Compute flux direction consistency metrics.

    Evaluates whether the model predicts the correct physical direction of transport,
    which is crucial for physical credibility even if magnitudes are imperfect.

    Args:
        J_ax_true: True axial fluxes [E]
        J_ax_pred: Predicted axial fluxes [E]
        C_ST_true: True concentrations [N]
        C_ST_pred: Predicted concentrations [N]
        edge_index: Edge connectivity [2, E]

    Returns:
        tuple: (sign_accuracy, reversal_rate, delta_C_sign_accuracy)
            - sign_accuracy: Fraction of edges where sign(J_pred) == sign(J_true)
            - reversal_rate: Fraction of edges with wrong direction (1 - sign_accuracy)
            - delta_C_sign_accuracy: Fraction of edges where sign(ΔC_pred) == sign(ΔC_true)
    """
    if J_ax_true.numel() == 0 or J_ax_pred.numel() == 0:
        return 0.0, 0.0, 0.0

    # (1) Flux sign accuracy: Does predicted flux have correct direction?
    sign_true = torch.sign(J_ax_true)
    sign_pred = torch.sign(J_ax_pred)

    # Count edges where signs match
    sign_matches = (sign_true == sign_pred).float()
    sign_accuracy = sign_matches.mean().detach().cpu().item()
    reversal_rate = 1.0 - sign_accuracy

    # (2) Concentration gradient sign accuracy: Does model capture osmotic effects correctly?
    if edge_index is not None and C_ST_true.numel() > 0 and C_ST_pred.numel() > 0:
        src, dst = edge_index[0], edge_index[1]

        # Compute concentration differences
        delta_C_true = C_ST_true[src] - C_ST_true[dst]
        delta_C_pred = C_ST_pred[src] - C_ST_pred[dst]

        # Check if signs match
        delta_C_sign_true = torch.sign(delta_C_true)
        delta_C_sign_pred = torch.sign(delta_C_pred)

        delta_C_sign_matches = (delta_C_sign_true == delta_C_sign_pred).float()
        delta_C_sign_accuracy = delta_C_sign_matches.mean().detach().cpu().item()
    else:
        delta_C_sign_accuracy = 0.0

    return sign_accuracy, reversal_rate, delta_C_sign_accuracy


def compute_physics_score(
    residual: torch.Tensor,
    dS_dt_from_physics: torch.Tensor,
    batch_vec: torch.Tensor = None,
    tolerance_factor: float = 0.01
) -> tuple[float, float]:
    """Compute dimensionless physics consistency score for time-series data.

    Evaluates temporal consistency: r_i = dS/dt_state - dS/dt_physics
    Checks if state change matches expected physics-based rate of change.
    Metrics based on residual vs physics-based derivative magnitude.

    Two metrics are computed:
    (1) Normalized Relative Error (PhysRelError):
        E[|r|] / (E[|dS/dt_physics|] + eps)

        Interpretation:
        - PhysRelErr << 1 (e.g., 0.01-0.05): Physics is tight, good consistency
        - PhysRelErr ~ 1: Physics moderately violated
        - PhysRelErr >> 1: Physics badly violated

    (2) Physics Satisfaction Rate:
        Fraction satisfying |r_i| <= tolerance * |dS/dt_physics_i|

        Interpretation:
        - Rate > 0.95: Excellent physics satisfaction
        - Rate 0.8-0.95: Good physics satisfaction
        - Rate < 0.8: Poor physics satisfaction

    Args:
        residual: Physics residual [N] (mmol/h)
            Time-series: r = dS/dt_state - dS/dt_physics
        dS_dt_from_physics: Physics-based derivative [N] (mmol/h)
            Expected rate of change: divJ + F_in - F_out
        batch_vec: Batch indices for multiple graphs [N] (optional)
        tolerance_factor: Relative tolerance for satisfaction check (default: 0.01 = 1%)

    Returns:
        tuple: (physics_rel_error, physics_satisfaction_rate)
            - physics_rel_error: Normalized residual magnitude
            - physics_satisfaction_rate: Fraction of nodes within tolerance
    """
    EPSILON = 1e-10

    # Compute absolute values
    abs_residual = residual.abs()
    abs_dS_dt = dS_dt_from_physics.abs()

    # Local scale for each node: expected physics-based rate of change
    local_scale = abs_dS_dt + EPSILON

    if batch_vec is not None:
        from torch_scatter import scatter_mean

        # (1) Normalized Relative Error per graph
        mean_abs_residual_per_graph = scatter_mean(abs_residual, batch_vec, dim=0)
        mean_abs_dS_dt_per_graph = scatter_mean(abs_dS_dt, batch_vec, dim=0)

        scale_per_graph = mean_abs_dS_dt_per_graph + EPSILON
        rel_error_per_graph = mean_abs_residual_per_graph / scale_per_graph
        physics_rel_error = rel_error_per_graph.mean().detach().cpu().item()

        # (2) Satisfaction rate: fraction of nodes within tolerance
        tolerance = tolerance_factor * local_scale
        is_satisfied = (abs_residual <= tolerance).float()
        satisfaction_per_graph = scatter_mean(is_satisfied, batch_vec, dim=0)
        physics_satisfaction_rate = satisfaction_per_graph.mean().detach().cpu().item()
    else:
        # Single graph case
        # (1) Normalized Relative Error
        mean_abs_residual = abs_residual.mean()
        mean_abs_dS_dt = abs_dS_dt.mean()

        scale = mean_abs_dS_dt + EPSILON
        physics_rel_error = (mean_abs_residual / scale).detach().cpu().item()

        # (2) Satisfaction rate
        tolerance = tolerance_factor * local_scale
        is_satisfied = (abs_residual <= tolerance).float()
        physics_satisfaction_rate = is_satisfied.mean().detach().cpu().item()

    return physics_rel_error, physics_satisfaction_rate


def _compute_physics_error_metrics(
    J_ax_true: torch.Tensor,
    J_ax_pred: torch.Tensor,
    dS_dt_from_flux_true: torch.Tensor,
    dS_dt_from_flux_pred: torch.Tensor,
    F_in_true: torch.Tensor,
    F_in_pred: torch.Tensor,
    F_out_true: torch.Tensor,
    F_out_pred: torch.Tensor,
    dS_dt_tot_true: torch.Tensor,
    dS_dt_tot_pred: torch.Tensor,
    J_ax_antisym_error: float = 0.0,
    edge_index: torch.Tensor = None,
    C_ST_true: torch.Tensor = None,
    C_ST_pred: torch.Tensor = None,
    batch_vec: torch.Tensor = None,
    dS_dt_from_state_true: torch.Tensor = None,
    dS_dt_from_state_pred: torch.Tensor = None
) -> PhysicsErrorMetrics:
    """Compute MSE, RMSE, Relative Error, direction consistency, and physics score.

    Args:
        J_ax_true: True axial fluxes
        J_ax_pred: Predicted axial fluxes
        dS_dt_from_flux_true: True flux divergence
        dS_dt_from_flux_pred: Predicted flux divergence
        F_in_true: True phloem loading
        F_in_pred: Predicted phloem loading
        F_out_true: True sucrose outflow
        F_out_pred: Predicted sucrose outflow
        dS_dt_tot_true: True total residual
        dS_dt_tot_pred: Predicted total residual
        J_ax_antisym_error: Antisymmetry error for operator model (optional)
        edge_index: Edge connectivity for antisymmetry and direction calculations (optional)
        C_ST_true: True concentrations for ΔC sign accuracy (optional)
        C_ST_pred: Predicted concentrations for ΔC sign accuracy (optional)
        batch_vec: Batch indices for multiple graphs (optional)

    Returns:
        PhysicsErrorMetrics containing MSE, RMSE, relative errors, direction metrics, and physics scores
    """
    # Helper function to compute metrics for a quantity
    def compute_metrics(true_vals, pred_vals):
        mse = (true_vals - pred_vals).pow(2).mean().detach().cpu().item()
        rmse = torch.sqrt((true_vals - pred_vals).pow(2).mean()).detach().cpu().item()
        # Relative error: mean absolute error / mean absolute true value
        mae = torch.abs(true_vals - pred_vals).mean().detach().cpu().item()
        mean_true = torch.abs(true_vals).mean().detach().cpu().item()
        rel_error = mae / (mean_true + EPSILON)
        return mse, rmse, rel_error

    # Helper function to compute Pearson correlation
    def compute_correlation(true_vals, pred_vals):
        """Compute Pearson correlation coefficient between true and predicted values."""
        if true_vals.numel() == 0 or pred_vals.numel() == 0:
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

    # Compute J_ax metrics (only if we have edge-level data)
    if J_ax_true.numel() > 0 and J_ax_pred.numel() > 0:
        J_ax_mse, J_ax_rmse, J_ax_rel = compute_metrics(J_ax_true, J_ax_pred)
    else:
        J_ax_mse, J_ax_rmse, J_ax_rel = 0.0, 0.0, 0.0

    # Compute divergence metrics (including correlation for mass conservation evaluation)
    divJ_mse, divJ_rmse, divJ_rel = compute_metrics(dS_dt_from_flux_true, dS_dt_from_flux_pred)
    divJ_correlation = compute_correlation(dS_dt_from_flux_true, dS_dt_from_flux_pred)

    # Compute F_in metrics
    F_in_mse, F_in_rmse, F_in_rel = compute_metrics(F_in_true, F_in_pred)

    # Compute F_out metrics
    F_out_mse, F_out_rmse, F_out_rel = compute_metrics(F_out_true, F_out_pred)

    # Compute dS_dt_tot metrics
    dS_dt_tot_mse, dS_dt_tot_rmse, dS_dt_tot_rel = compute_metrics(dS_dt_tot_true, dS_dt_tot_pred)

    # Compute antisymmetry error if edge_index is provided and we have predicted fluxes
    antisym_err = J_ax_antisym_error
    if edge_index is not None and J_ax_pred.numel() > 0 and antisym_err == 0.0:
        antisym_err = compute_flux_antisymmetry_error(J_ax_pred, edge_index)

    # Compute flux direction consistency metrics if concentrations are provided
    if C_ST_true is not None and C_ST_pred is not None and edge_index is not None and J_ax_true.numel() > 0:
        sign_acc, rev_rate, delta_C_acc = compute_flux_direction_metrics(
            J_ax_true, J_ax_pred, C_ST_true, C_ST_pred, edge_index
        )
    else:
        sign_acc, rev_rate, delta_C_acc = 0.0, 0.0, 0.0

    # Compute physics score from residual (dimensionless consistency metric)
    phys_rel_err, phys_sat_rate = compute_physics_score(
        dS_dt_tot_pred, dS_dt_tot_pred, batch_vec, tolerance_factor=0.01
    )

    # Compute temporal consistency metrics (time-series mode)
    temporal_rel_error_pred = 0.0
    temporal_consistency_pred = 0.0
    temporal_rel_error_true = 0.0
    temporal_consistency_true = 0.0

    if dS_dt_from_state_pred is not None:
        # Compute temporal residual for predictions: r = dS/dt_state - dS/dt_physics
        residual_pred = dS_dt_from_state_pred - dS_dt_tot_pred
        temporal_rel_error_pred, temporal_consistency_pred = compute_physics_score(
            residual_pred, dS_dt_tot_pred, batch_vec, tolerance_factor=0.01
        )

    if dS_dt_from_state_true is not None:
        # Compute temporal residual for ground truth: r = dS/dt_state - dS/dt_physics
        residual_true = dS_dt_from_state_true - dS_dt_tot_true
        temporal_rel_error_true, temporal_consistency_true = compute_physics_score(
            residual_true, dS_dt_tot_true, batch_vec, tolerance_factor=0.01
        )

    return PhysicsErrorMetrics(
        J_ax_mse=J_ax_mse,
        J_ax_rmse=J_ax_rmse,
        J_ax_rel_error=J_ax_rel,
        divJ_mse=divJ_mse,
        divJ_rmse=divJ_rmse,
        divJ_rel_error=divJ_rel,
        divJ_correlation=divJ_correlation,
        F_in_mse=F_in_mse,
        F_in_rmse=F_in_rmse,
        F_in_rel_error=F_in_rel,
        F_out_mse=F_out_mse,
        F_out_rmse=F_out_rmse,
        F_out_rel_error=F_out_rel,
        dS_dt_tot_mse=dS_dt_tot_mse,
        dS_dt_tot_rmse=dS_dt_tot_rmse,
        dS_dt_tot_rel_error=dS_dt_tot_rel,
        J_ax_antisym_error=antisym_err,
        J_ax_sign_accuracy=sign_acc,
        J_ax_reversal_rate=rev_rate,
        delta_C_sign_accuracy=delta_C_acc,
        physics_rel_error=phys_rel_err,
        physics_satisfaction_rate=phys_sat_rate,
        temporal_rel_error_pred=temporal_rel_error_pred,
        temporal_consistency_pred=temporal_consistency_pred,
        temporal_rel_error_true=temporal_rel_error_true,
        temporal_consistency_true=temporal_consistency_true
    )


def _log_penalty_monitoring(
    physics_loss: torch.Tensor,
    negative_concentration_penalty: torch.Tensor,
    penalty_weight: float,
    total_loss: torch.Tensor,
    model_type: str = "NNConv"
):
    """Log penalty monitoring information.

    Args:
        physics_loss: Physics residual loss component
        negative_concentration_penalty: Penalty for negative concentrations
        penalty_weight: Weight applied to penalty
        total_loss: Combined total loss
        model_type: Type of model (for labeling)

    Returns:
        str: Formatted penalty monitoring log
    """
    ratio = (penalty_weight * negative_concentration_penalty) / (physics_loss + 1e-10)
    msg = f"\n--- PENALTY MONITORING ({model_type}) ---\n"
    msg += f"Physics loss: {physics_loss.item():.6e}\n"
    msg += f"Penalty (unweighted): {negative_concentration_penalty.item():.6e}\n"
    msg += f"Penalty (weighted): {(penalty_weight * negative_concentration_penalty).item():.6e}\n"
    msg += f"Penalty weight: {penalty_weight:.1f}\n"
    msg += f"Total loss: {total_loss.item():.6e}\n"
    msg += f"Penalty/Physics ratio: {ratio.item():.2f}\n"
    msg += f"{'='*60}\n"
    return msg


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

    # Divergence of flux -> net inflow per node
    # This computes the sum of incoming/outgoing fluxes for each node
    # dst node accumulates -J_ax
    # src node accumulates +J_ax
    dS_dt_from_flux.scatter_add_(0, dst, -J_ax)
    dS_dt_from_flux.scatter_add_(0, src, +J_ax)

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


def log_physics_values(y_pred: torch.Tensor, data: Data, model_output=None, phase: str = None):
    """Log true and predicted physics values for analysis (no loss computation).

    This function computes and logs all physics quantities (C_ST, J_ax, divergence,
    F_in, F_out) for both true and predicted values. It's useful for evaluating
    physical consistency of models trained with DATA_ONLY loss.

    Args:
        y_pred: Predicted sucrose content [N, 1] or dict for operator model
        data: Graph data containing topology, features, and targets
        model_output: For operator models, dict containing edge_fluxes and divergences

    Returns:
        Tuple of (PhysicsMetrics, PhysicsErrorMetrics):
            - Physics metrics for terminal display
            - Physics error metrics (MSE, RMSE, Relative Error)
            Returns (None, None) if logging disabled
    """
    if not _ENABLE_PHYSICS_LOGGING:
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
            J_ax_true, edge_fluxes_pred,
            dS_dt_from_flux_true, divergence_pred,
            F_in_true, F_in_pred,
            F_out_true, F_out_pred,
            dS_dt_tot_true, dS_dt_tot_pred,
            edge_index=edge_index,
            C_ST_true=C_ST_true,
            C_ST_pred=C_ST_pred,
            batch_vec=batch_vec
        )

        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_header("DEBUG OUTPUT - OPERATOR MODEL (DATA-ONLY MODE)", batch_vec, data=data, phase=phase)
            msg += _log_concentrations(C_ST_true, C_ST_pred)
            msg += _log_fluxes(J_ax_true, edge_fluxes_pred)
            msg += _log_divergence(dS_dt_from_flux_true, divergence_pred)
            msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
            msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
            msg += _log_comparison_metrics(physics_errors)
            msg += f"{'='*60}\n"
            f.write(msg)

        # Compute averaged metrics for terminal display (operator model)
        from torch_scatter import scatter_mean
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
            J_ax_true, J_ax_pred,
            dS_dt_from_flux_true, dS_dt_from_flux_pred,
            F_in_true, F_in_pred,
            F_out_true, F_out_pred,
            dS_dt_tot_true, dS_dt_tot_pred,
            edge_index=edge_index,
            C_ST_true=C_ST_true,
            C_ST_pred=C_ST_pred,
            batch_vec=batch_vec
        )

        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_header("DEBUG OUTPUT - NNCONV MODEL (DATA-ONLY MODE)", batch_vec, data=data, phase=phase)
            msg += _log_concentrations(C_ST_true, C_ST_pred)
            msg += _log_fluxes(J_ax_true, J_ax_pred)
            msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
            msg += _log_divergence(dS_dt_from_flux_true, dS_dt_from_flux_pred)
            msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
            msg += _log_comparison_metrics(physics_errors)
            msg += f"{'='*60}\n"
            f.write(msg)

        # Compute averaged metrics for terminal display (NNConv model)
        from torch_scatter import scatter_mean
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
    prev_sucrose: torch.Tensor = None,
    is_first_timestep: bool = False,
    prev_time: float = None,
    phase: str = None
):
    """Compute physics-informed residual term based on sucrose transport equations.

    Implements the governing equation for content-based sucrose transport in sieve-tubes:
    dS/dt = divJ + (F_in - F_out) ≈ 0

    where:
    - dS/dt is the discrete time derivative (S(t) - S(t-1)) / Δt
    - divJ is the divergence of axial sucrose flux
    - F_in is the phloem loading rate
    - F_out is the sucrose outflow

    For time-series mode, physics loss is computed by minimizing:
        residual = dS/dt - (divJ + F_in - F_out)

    where dS/dt is computed from the change in sucrose content between timesteps.

    Args:
        y_pred: Predicted sucrose content [N, 1] (standardized)
        data: Graph data containing topology, features, simulation parameters, and node fields
        prev_sucrose: Previous timestep sucrose content [N, 1] (standardized). None for first timestep.
        is_first_timestep: Whether this is the first timestep (skip dS/dt residual if True)
        prev_time: Time at previous timestep (hours). If None and prev_sucrose provided, will try to extract from data.

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

    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_header("DEBUG OUTPUT - PHYSICS RESIDUAL (MINIMIZE TO ZERO)", batch_vec, data=data, phase=phase)
            msg += _log_concentrations(C_ST_true, C_ST_pred)
            msg += _log_fluxes(J_ax_true, J_ax_pred)
            msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
            msg += _log_divergence(dS_dt_from_flux_true, dS_dt_from_flux_pred)
            msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
            msg += f"\n--- PHYSICS RESIDUAL (should approach zero) ---\n"
            msg += f"dS_dt_tot_pred (first 10 nodes): {dS_dt_tot_pred[:10].detach().cpu().numpy()}\n"
            msg += f"Mean absolute residual: {dS_dt_tot_pred.abs().mean().detach().cpu().item():.6e}\n"
            f.write(msg)

    # ============================
    # Compute physics residual: enforce dS/dt = divJ + F_in - F_out
    # ============================
    # Initialize temporal derivatives (will be set if not first timestep)
    dS_dt_from_state_pred = None
    dS_dt_from_state_true = None

    # For time-series mode, compute discrete time derivative dS/dt from state change
    if not is_first_timestep and prev_sucrose is not None:
        # Denormalize current and previous sucrose content
        S_ST_curr = data.target_scaler.inv_transform(y_pred).squeeze(-1)
        S_ST_prev = data.target_scaler.inv_transform(prev_sucrose).squeeze(-1)

        # Get current and previous time (assuming data.time is a scalar tensor)
        # If batched, we'd need to handle per-graph times, but we assume batch_size=1 for time-series
        if hasattr(data, 'time'):
            curr_time = data.time.item() if isinstance(data.time, torch.Tensor) else data.time
        else:
            # Fallback: extract from node features (column 7 is time)
            curr_time = node_feat_original[0, 7].item()

        # Compute delta_t from actual time difference
        if prev_time is not None:
            delta_t = curr_time - prev_time
        else:
            # Fallback: try to extract from node features if available
            # This assumes node features have time information and prev_sucrose was provided
            # but prev_time was not explicitly passed
            # Use a typical value as last resort
            delta_t = 1.0  # hours (default fallback)
            if _ENABLE_PHYSICS_LOGGING:
                with open(_PHYSICS_LOG_PATH, "a") as f:
                    f.write(f"WARNING: prev_time not provided, using default delta_t = {delta_t} hours\n")

        if delta_t <= 0:
            raise ValueError(f"Invalid delta_t: {delta_t}. Current time: {curr_time}, Previous time: {prev_time}")

        # Discrete time derivative from predictions: (S(t) - S(t-1)) / Δt  [mmol/h]
        dS_dt_from_state_pred = (S_ST_curr - S_ST_prev) / delta_t

        # Also compute temporal derivative from ground truth for metrics
        S_ST_true_curr = data.target_scaler.inv_transform(y_true).squeeze(-1)
        # For ground truth temporal derivative, we need previous ground truth
        # This requires access to data.y from the previous timestep
        # For now, we'll skip this - it would require passing prev_y_true as well
        # dS_dt_from_state_true = (S_ST_true_curr - S_ST_true_prev) / delta_t

        # Physics-based derivative from fluxes and sources/sinks
        dS_dt_from_physics = dS_dt_from_flux_pred + F_in_true_detached - F_out_true_detached

        # Residual: difference between state-based and physics-based derivatives
        # R = dS/dt_state - dS/dt_physics
        # We want this to be close to zero
        residual = dS_dt_from_state_pred - dS_dt_from_physics

        if _ENABLE_PHYSICS_LOGGING:
            with open(_PHYSICS_LOG_PATH, "a") as f:
                msg = f"\n--- TIME-SERIES MODE: Discrete Time Derivative ---\n"
                msg += f"delta_t: {delta_t} hours\n"
                msg += f"dS_dt_from_state (mean): {dS_dt_from_state_pred.mean().item():.6e}\n"
                msg += f"dS_dt_from_state (first 10): {dS_dt_from_state_pred[:10].detach().cpu().numpy()}\n"
                msg += f"dS_dt_from_physics (mean): {dS_dt_from_physics.mean().item():.6e}\n"
                msg += f"dS_dt_from_physics (first 10): {dS_dt_from_physics[:10].detach().cpu().numpy()}\n"
                msg += f"residual (mean abs): {residual.abs().mean().item():.6e}\n"
                msg += f"residual (first 10): {residual[:10].detach().cpu().numpy()}\n"
                f.write(msg)
    else:
        # First timestep or no previous state: fall back to steady-state assumption
        # Minimize dS/dt_physics directly (assume dS/dt ≈ 0)
        residual = dS_dt_tot_pred

        if _ENABLE_PHYSICS_LOGGING:
            with open(_PHYSICS_LOG_PATH, "a") as f:
                msg = f"\n--- STEADY-STATE MODE (First Timestep or No Prev State) ---\n"
                msg += f"is_first_timestep: {is_first_timestep}\n"
                msg += f"prev_sucrose is None: {prev_sucrose is None}\n"
                msg += f"Using steady-state assumption: residual = dS/dt_physics\n"
                f.write(msg)

    # Compute physics error metrics (include edge_index for antisymmetry calculation)
    # Pass temporal derivatives if available (non-first timesteps)
    physics_errors = _compute_physics_error_metrics(
        J_ax_true, J_ax_pred,
        dS_dt_from_flux_true, dS_dt_from_flux_pred,
        F_in_true, F_in_pred,
        F_out_true, F_out_pred,
        dS_dt_tot_true, dS_dt_tot_pred,
        edge_index=edge_index,
        C_ST_true=C_ST_true,
        C_ST_pred=C_ST_pred,
        batch_vec=batch_vec,
        dS_dt_from_state_true=dS_dt_from_state_true,
        dS_dt_from_state_pred=dS_dt_from_state_pred
    )

    # After computing temporal metrics, optionally dump temporal tensors for debugging
    if _ENABLE_PHYSICS_LOGGING:
        # Prepare temporal tensors (may be None)
        pred_tensors = {
            'dS_dt_from_state': dS_dt_from_state_pred,
            'dS_dt_from_physics': locals().get('dS_dt_from_physics', None) if 'dS_dt_from_physics' in locals() else dS_dt_tot_pred
        }
        true_tensors = {
            'dS_dt_from_state': dS_dt_from_state_true,
            'dS_dt_from_physics': dS_dt_tot_true
        }

        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_comparison_metrics(physics_errors, temporal_tensors={'pred': pred_tensors, 'true': true_tensors})
            f.write(msg)

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

    # Log penalty monitoring information
    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_penalty_monitoring(
                physics_loss, negative_concentration_penalty,
                PENALTY_WEIGHT, loss, "NNConv"
            )
            f.write(msg)

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
    prev_sucrose: torch.Tensor = None,
    is_first_timestep: bool = False,
    prev_time: float = None,
    phase: str = None
) -> tuple[torch.Tensor, dict]:
    """Compute physics residual for operator-based GNN using LEARNED operator.

    This function uses the edge fluxes and divergences directly predicted by the
    operator model in the physics loss. The learned operator is supervised by
    the physics residual.

    The conservation law is:
        dS/dt = div(J) + F_in - F_out ≈ 0

    For time-series mode, dS/dt is computed from the discrete change in sucrose content:
        dS/dt ≈ (S(t) - S(t-1)) / Δt

    Args:
        model_output: Dict containing:
            - 'predictions': [N, 1] sucrose content predictions (standardized)
            - 'edge_fluxes': [E] predicted edge fluxes (USED in loss)
            - 'divergences': [N] divergence values (USED in loss)
        data: Graph data containing features and parameters
        prev_sucrose: Previous timestep sucrose content [N, 1] (standardized). None for first timestep.
        is_first_timestep: Whether this is the first timestep (skip dS/dt residual if True)
        prev_time: Time at previous timestep (hours). If None and prev_sucrose provided, will try to extract from data.

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

    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_header("DEBUG OUTPUT - OPERATOR MODEL PHYSICS RESIDUAL (LEARNED)", batch_vec, data=data, phase=phase)
            msg += _log_concentrations(C_ST_true, C_ST_pred)
            msg += _log_fluxes(J_ax_true, edge_fluxes_pred)
            msg += _log_divergence(dS_dt_from_flux_true, divergence_pred)
            msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
            msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)
            f.write(msg)

    # ============================
    # Compute physics residual: enforce dS/dt = divJ + F_in - F_out
    # ============================
    # Initialize temporal derivatives (will be set if not first timestep)
    dS_dt_from_state_pred = None
    dS_dt_from_state_true = None

    # For time-series mode, compute discrete time derivative dS/dt from state change
    if not is_first_timestep and prev_sucrose is not None:
        # Denormalize current and previous sucrose content
        S_ST_curr = data.target_scaler.inv_transform(y_pred).squeeze(-1)
        S_ST_prev = data.target_scaler.inv_transform(prev_sucrose).squeeze(-1)

        # Get current time
        if hasattr(data, 'time'):
            curr_time = data.time.item() if isinstance(data.time, torch.Tensor) else data.time
        else:
            # Fallback: extract from node features (column 7 is time)
            curr_time = node_feat_original[0, 7].item()

        # Compute delta_t from actual time difference
        if prev_time is not None:
            delta_t = curr_time - prev_time
        else:
            # Fallback: try to extract from node features if available
            # This assumes node features have time information and prev_sucrose was provided
            # but prev_time was not explicitly passed
            # Use a typical value as last resort
            delta_t = 1.0  # hours (default fallback)
            if _ENABLE_PHYSICS_LOGGING:
                with open(_PHYSICS_LOG_PATH, "a") as f:
                    f.write(f"WARNING: prev_time not provided, using default delta_t = {delta_t} hours\n")

        if delta_t <= 0:
            raise ValueError(f"Invalid delta_t: {delta_t}. Current time: {curr_time}, Previous time: {prev_time}")

        # Discrete time derivative from predictions: (S(t) - S(t-1)) / Δt  [mmol/h]
        dS_dt_from_state_pred = (S_ST_curr - S_ST_prev) / delta_t

        # Physics-based derivative from fluxes and sources/sinks
        # Using LEARNED divergence from operator model
        dS_dt_from_physics = divergence_pred + F_in_true_detached - F_out_true_detached

        # Residual: difference between state-based and physics-based derivatives
        residual = dS_dt_from_state_pred - dS_dt_from_physics

        if _ENABLE_PHYSICS_LOGGING:
            with open(_PHYSICS_LOG_PATH, "a") as f:
                msg = f"\n--- TIME-SERIES MODE: Discrete Time Derivative (Operator) ---\n"
                msg += f"delta_t: {delta_t} hours\n"
                msg += f"dS_dt_from_state (mean): {dS_dt_from_state_pred.mean().item():.6e}\n"
                msg += f"dS_dt_from_state (first 10): {dS_dt_from_state_pred[:10].detach().cpu().numpy()}\n"
                msg += f"dS_dt_from_physics (mean): {dS_dt_from_physics.mean().item():.6e}\n"
                msg += f"dS_dt_from_physics (first 10): {dS_dt_from_physics[:10].detach().cpu().numpy()}\n"
                msg += f"residual (mean abs): {residual.abs().mean().item():.6e}\n"
                msg += f"residual (first 10): {residual[:10].detach().cpu().numpy()}\n"
                f.write(msg)
    else:
        # First timestep or no previous state: fall back to steady-state assumption
        residual = dS_dt_tot_pred

        if _ENABLE_PHYSICS_LOGGING:
            with open(_PHYSICS_LOG_PATH, "a") as f:
                msg = f"\n--- STEADY-STATE MODE (Operator - First Timestep or No Prev State) ---\n"
                msg += f"is_first_timestep: {is_first_timestep}\n"
                msg += f"prev_sucrose is None: {prev_sucrose is None}\n"
                msg += f"Using steady-state assumption: residual = dS/dt_physics\n"
                f.write(msg)

    # Compute physics error metrics (include edge_index for antisymmetry calculation)
    # Pass temporal derivatives if available (non-first timesteps)
    # Use LEARNED fluxes/divergences for error metrics
    physics_errors = _compute_physics_error_metrics(
        J_ax_true, edge_fluxes_pred,
        dS_dt_from_flux_true, divergence_pred,
        F_in_true, F_in_pred,
        F_out_true, F_out_pred,
        dS_dt_tot_true, dS_dt_tot_pred,
        edge_index=edge_index,
        C_ST_true=C_ST_true,
        C_ST_pred=C_ST_pred,
        batch_vec=batch_vec,
        dS_dt_from_state_true=dS_dt_from_state_true,
        dS_dt_from_state_pred=dS_dt_from_state_pred
    )

    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            pred_tensors = {
                'dS_dt_from_state': dS_dt_from_state_pred,
                'dS_dt_from_physics': locals().get('dS_dt_from_physics', None) if 'dS_dt_from_physics' in locals() else dS_dt_tot_pred
            }
            true_tensors = {
                'dS_dt_from_state': dS_dt_from_state_true,
                'dS_dt_from_physics': dS_dt_tot_true
            }
            msg = _log_comparison_metrics(physics_errors, temporal_tensors={'pred': pred_tensors, 'true': true_tensors})
            f.write(msg)

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

    # Log penalty monitoring information
    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_penalty_monitoring(
                physics_loss, negative_concentration_penalty,
                PENALTY_WEIGHT, loss, "Operator"
            )
            f.write(msg)

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
    prev_sucrose: torch.Tensor = None,
    is_first_timestep: bool = False,
    prev_time: float = None,
    phase: str = None
) -> tuple[torch.Tensor, dict]:
    """Compute physics residual for operator-based GNN using ANALYTICAL calculations.

    This function computes fluxes and divergences analytically from predicted
    concentrations (identical to NNConv approach), while still accepting operator
    model outputs for logging/comparison purposes. The learned operator outputs
    are NOT used in the physics loss.

    This allows fair comparison between NNConv and Operator architectures with
    identical physics supervision.

    The conservation law is:
        dS/dt = div(J) + F_in - F_out ≈ 0

    For time-series mode, dS/dt is computed from the discrete change in sucrose content:
        dS/dt ≈ (S(t) - S(t-1)) / Δt

    Args:
        model_output: Dict containing:
            - 'predictions': [N, 1] sucrose content predictions (standardized)
            - 'edge_fluxes': [E] predicted edge fluxes (logged but NOT used)
            - 'divergences': [N] divergence values (logged but NOT used)
        data: Graph data containing features and parameters
        prev_sucrose: Previous timestep sucrose content [N, 1] (standardized). None for first timestep.
        is_first_timestep: Whether this is the first timestep (skip dS/dt residual if True)
        prev_time: Time at previous timestep (hours). If None and prev_sucrose provided, will try to extract from data.
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

    # ============================
    # LOGGING
    # ============================
    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_header("DEBUG OUTPUT - OPERATOR MODEL PHYSICS RESIDUAL (ANALYTICAL)", batch_vec, data=data, phase=phase)
            msg += _log_concentrations(C_ST_true, C_ST_pred)
            msg += _log_fluxes(J_ax_true, J_ax_pred)
            msg += _log_divergence(dS_dt_from_flux_true, dS_dt_from_flux_pred)
            msg += _log_source_sink_terms(F_in_true, F_in_pred, F_out_true, F_out_pred)
            msg += _log_total_residual(dS_dt_tot_true, dS_dt_tot_pred)

            # Log comparison between learned and analytical operator outputs
            if edge_fluxes_learned is not None and divergence_learned is not None:
                msg += "\n" + "="*80 + "\n"
                msg += "LEARNED OPERATOR COMPARISON (not used in loss)\n"
                msg += "="*80 + "\n"
                msg += f"Edge Fluxes (Learned):\n"
                msg += f"  Mean: {edge_fluxes_learned.mean().item():.6e}\n"
                msg += f"  Std:  {edge_fluxes_learned.std().item():.6e}\n"
                msg += f"  Range: [{edge_fluxes_learned.min().item():.6e}, {edge_fluxes_learned.max().item():.6e}]\n"
                msg += f"  First 10: {edge_fluxes_learned[:10].detach().cpu().numpy()}\n"
                msg += f"\nEdge Fluxes (Analytical - USED):\n"
                msg += f"  Mean: {J_ax_pred.mean().item():.6e}\n"
                msg += f"  Std:  {J_ax_pred.std().item():.6e}\n"
                msg += f"  Range: [{J_ax_pred.min().item():.6e}, {J_ax_pred.max().item():.6e}]\n"
                msg += f"  First 10: {J_ax_pred[:10].detach().cpu().numpy()}\n"

                msg += f"\nDivergence (Learned):\n"
                msg += f"  Mean: {divergence_learned.mean().item():.6e}\n"
                msg += f"  Std:  {divergence_learned.std().item():.6e}\n"
                msg += f"  Range: [{divergence_learned.min().item():.6e}, {divergence_learned.max().item():.6e}]\n"
                msg += f"  First 10: {divergence_learned[:10].detach().cpu().numpy()}\n"
                msg += f"\nDivergence (Analytical - USED):\n"
                msg += f"  Mean: {dS_dt_from_flux_pred.mean().item():.6e}\n"
                msg += f"  Std:  {dS_dt_from_flux_pred.std().item():.6e}\n"
                msg += f"  Range: [{dS_dt_from_flux_pred.min().item():.6e}, {dS_dt_from_flux_pred.max().item():.6e}]\n"
                msg += f"  First 10: {dS_dt_from_flux_pred[:10].detach().cpu().numpy()}\n"

                # Compute difference metrics
                flux_diff = (edge_fluxes_learned - J_ax_pred).abs()
                div_diff = (divergence_learned - dS_dt_from_flux_pred).abs()
                msg += f"\nDifference (Learned - Analytical):\n"
                msg += f"  Flux MAE: {flux_diff.mean().item():.6e}\n"
                msg += f"  Divergence MAE: {div_diff.mean().item():.6e}\n"

            f.write(msg)

    # ============================
    # Compute physics residual: enforce dS/dt = divJ + F_in - F_out
    # ============================
    dS_dt_from_state_pred = None
    dS_dt_from_state_true = None

    # For time-series mode, compute discrete time derivative dS/dt from state change
    if not is_first_timestep and prev_sucrose is not None:
        # Denormalize current and previous sucrose content
        S_ST_curr = data.target_scaler.inv_transform(y_pred).squeeze(-1)
        S_ST_prev = data.target_scaler.inv_transform(prev_sucrose).squeeze(-1)

        # Get current and previous time
        if hasattr(data, 'time'):
            curr_time = data.time.item() if isinstance(data.time, torch.Tensor) else data.time
        else:
            curr_time = node_feat_original[0, 7].item()

        if prev_time is not None:
            delta_t = curr_time - prev_time
        else:
            delta_t = 1.0  # hours (default fallback)
            if _ENABLE_PHYSICS_LOGGING:
                with open(_PHYSICS_LOG_PATH, "a") as f:
                    f.write(f"WARNING: prev_time not provided, using default delta_t = {delta_t} hours\n")

        if delta_t <= 0:
            raise ValueError(f"Invalid delta_t: {delta_t}. Current time: {curr_time}, Previous time: {prev_time}")

        # Discrete time derivative from predictions
        dS_dt_from_state_pred = (S_ST_curr - S_ST_prev) / delta_t

        # Physics-based derivative from analytical fluxes
        dS_dt_from_physics = dS_dt_from_flux_pred + F_in_true_detached - F_out_true_detached

        # Residual: difference between state-based and physics-based derivatives
        residual = dS_dt_from_state_pred - dS_dt_from_physics

        if _ENABLE_PHYSICS_LOGGING:
            with open(_PHYSICS_LOG_PATH, "a") as f:
                msg = f"\n--- TIME-SERIES MODE: Discrete Time Derivative ---\n"
                msg += f"delta_t: {delta_t} hours\n"
                msg += f"dS_dt_from_state (mean): {dS_dt_from_state_pred.mean().item():.6e}\n"
                msg += f"dS_dt_from_state (first 10): {dS_dt_from_state_pred[:10].detach().cpu().numpy()}\n"
                msg += f"dS_dt_from_physics (mean): {dS_dt_from_physics.mean().item():.6e}\n"
                msg += f"dS_dt_from_physics (first 10): {dS_dt_from_physics[:10].detach().cpu().numpy()}\n"
                msg += f"residual (mean abs): {residual.abs().mean().item():.6e}\n"
                msg += f"residual (first 10): {residual[:10].detach().cpu().numpy()}\n"
                f.write(msg)
    else:
        # First timestep or no previous state: steady-state assumption
        residual = dS_dt_tot_pred

        if _ENABLE_PHYSICS_LOGGING:
            with open(_PHYSICS_LOG_PATH, "a") as f:
                msg = f"\n--- STEADY-STATE MODE (First Timestep or No Prev State) ---\n"
                msg += f"is_first_timestep: {is_first_timestep}\n"
                msg += f"prev_sucrose is None: {prev_sucrose is None}\n"
                msg += f"Using steady-state assumption: residual = dS/dt_physics\n"
                f.write(msg)

    # Compute physics error metrics
    physics_errors = _compute_physics_error_metrics(
        J_ax_true, J_ax_pred,
        dS_dt_from_flux_true, dS_dt_from_flux_pred,
        F_in_true, F_in_pred,
        F_out_true, F_out_pred,
        dS_dt_tot_true, dS_dt_tot_pred,
        edge_index=edge_index,
        C_ST_true=C_ST_true,
        C_ST_pred=C_ST_pred,
        batch_vec=batch_vec,
        dS_dt_from_state_true=dS_dt_from_state_true,
        dS_dt_from_state_pred=dS_dt_from_state_pred
    )

    # Temporal tensor logging
    if _ENABLE_PHYSICS_LOGGING:
        pred_tensors = {
            'dS_dt_from_state': dS_dt_from_state_pred,
            'dS_dt_from_physics': locals().get('dS_dt_from_physics', None) if 'dS_dt_from_physics' in locals() else dS_dt_tot_pred
        }
        true_tensors = {
            'dS_dt_from_state': dS_dt_from_state_true,
            'dS_dt_from_physics': dS_dt_tot_true
        }

        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_comparison_metrics(physics_errors, temporal_tensors={'pred': pred_tensors, 'true': true_tensors})
            f.write(msg)

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

    # Log penalty monitoring information
    if _ENABLE_PHYSICS_LOGGING:
        with open(_PHYSICS_LOG_PATH, "a") as f:
            msg = _log_penalty_monitoring(
                physics_loss, negative_concentration_penalty,
                PENALTY_WEIGHT, loss, "Operator-Analytical"
            )
            f.write(msg)

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
