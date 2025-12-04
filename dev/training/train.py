import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_mean

from typing import Tuple, Optional

from model.physics import log_physics_values
from model.config import ModelConfig
from model.physics import physics_residual, physics_residual_operator, physics_residual_operator_analytical
from .config import (
    TrainingConfig, TrainingState, TrainingMetrics, ModelSetup,
    LossType, PhysicsMetrics, PhysicsErrorMetrics, LossConfig, LossResult, EpochResult
)

import training.utils as utils
import training.logging as logging

EPSILON = 1e-12


def _compute_adaptive_weight(
    reference_loss: torch.Tensor,
    physics_loss: torch.Tensor,
    target_ratio: float,
) -> torch.Tensor:
    """Compute adaptive weight to balance physics loss with reference loss.

    Args:
        reference_loss: Reference loss to balance against (supervision or data loss)
        physics_loss: Physics residual loss
        target_ratio: Target ratio of physics to reference loss

    Returns:
        Adaptive weight
    """
    adaptive_weight = (reference_loss / physics_loss) * target_ratio
    return adaptive_weight


def _compute_loss_percentages(
    component_losses: dict,
    total_loss: torch.Tensor
) -> dict:
    """Compute percentage contribution of each loss component.

    Args:
        component_losses: Dict mapping component names to loss values
        total_loss: Total combined loss

    Returns:
        Dict mapping component names to percentage contributions
    """
    total_float = utils.to_float(total_loss)
    if total_float == 0.0:
        return {name: 0.0 for name in component_losses}

    return {
        name: 100.0 * utils.to_float(loss) / total_float
        for name, loss in component_losses.items()
    }


def accumulate_epoch_stats(
    totals: dict,
    result: LossResult,
    weighted_supervision: float = 0.0,
    weighted_physics: float = 0.0
):
    """Accumulate statistics from a LossResult into totals dictionary.

    Args:
        totals: Dictionary to accumulate statistics
        result: LossResult containing all metrics from current batch
        weighted_supervision: Actual weighted supervision component
        weighted_physics: Actual weighted physics component
    """
    totals["loss"] += result.total_loss
    totals["mse"] += result.mse
    totals["mae"] += result.mae
    totals["rmse"] += result.rmse
    totals["rel_error"] += result.rel_error
    totals["phys"] += result.phys
    totals["ic"] += result.ic
    totals["bc"] += result.bc
    totals["n_batches"] += 1
    totals["last_phys_metrics"] = result.physics_metrics
    totals["last_phys_errors"] = result.physics_errors  # Store last physics errors

    # Accumulate statistics
    totals["phys_weight"] += result.phys_weight
    totals["supervision_weight"] += 0.0  # Deprecated, kept for compatibility
    totals["bc_nodes"] += result.bc_nodes
    totals["bc_pct"] += result.bc_pct
    totals["phys_contrib_pct"] += result.phys_contrib_pct
    totals["sup_contrib_pct"] += result.sup_or_data_contrib_pct

    # Track actual weighted components
    totals["weighted_supervision"] += weighted_supervision
    totals["weighted_physics"] += weighted_physics


def run_forward(model: nn.Module, data):
    """Forward pass on inputs.

    Args:
        model: The neural network model
        data: Graph data

    Returns:
        For NNConv model: Tensor of predictions [N, 1]
        For Operator model: Dict with 'predictions', 'edge_fluxes', 'divergences'
    """
    output = model(data)
    return output


def _extract_predictions(model_output):
    """Extract prediction tensor from model output.

    Args:
        model_output: Either a tensor (NNConv) or dict (Operator)

    Returns:
        torch.Tensor: Predictions [N, 1]
    """
    if isinstance(model_output, dict):
        return model_output['predictions']
    else:
        return model_output


def _to_physics_metrics(physics_res) -> Optional[PhysicsMetrics]:
    """Convert dict of physics components to PhysicsMetrics."""
    if physics_res is not None:
        return PhysicsMetrics(
            J_ax=float(physics_res['J_ax']),
            F_in=float(physics_res['F_in']),
            F_out=float(physics_res['F_out']),
            dS_dt_from_flux=float(physics_res['dS_dt_from_flux']),
            dS_dt_tot=float(physics_res['dS_dt_tot'])
        )
    else:
        return None


def compute_physics_residual_step(
    model: nn.Module,
    data,
    model_output = None,
    loss_type: LossType = LossType.COMBINED,
    phase: str = None,
) -> Tuple[torch.Tensor, Optional[PhysicsMetrics], Optional[PhysicsErrorMetrics]]:
    """
    Unified physics residual computation for train/eval.

    Handles both NNConv and Operator model types:
    - NNConv: Reconstructs fluxes from predicted concentrations
    - Operator: Uses directly predicted edge fluxes and divergences

    Source/sink term handling:
    - COMBINED mode: Uses TRUE F_in/F_out for ALL nodes (isolates flux divergence quality)
    - PHYSICS_WITH_IC_BC mode: Uses TRUE F_in/F_out only for IC/BC nodes
    - DATA_ONLY mode: No physics residual (uses predicted F_in/F_out for logging only)

    Args:
        model: The neural network model
        data: Graph data containing features and targets
        model_output: Model output (tensor or dict). If None, will run forward pass.
        loss_type: Type of loss being used (determines source/sink term substitution strategy)
        phase: Training phase ('train', 'val', 'test') for logging

    Returns:
        Tuple of (phys_res_scalar, PhysicsMetrics|None, PhysicsErrorMetrics|None)
    """
    # If output not provided, run forward pass
    if model_output is None:
        raise ValueError("model_output must be provided (forward pass should be done before calling this function)")

    # Determine model type based on output format
    is_operator_model = isinstance(model_output, dict)

    # Compute physics residual using appropriate function
    if is_operator_model:
        # phys_res, phys_res_dict, phys_errors = physics_residual_operator(
        #     model_output, data, phase=phase
        # )
        phys_res, phys_res_dict, phys_errors = physics_residual_operator_analytical(
            model_output, data, phase=phase
        )
    else:
        # NNConv model: model_output is a tensor
        phys_res, phys_res_dict, phys_errors = physics_residual(
            model_output, data, phase=phase
        )

    # Reduce to scalar if needed
    phys_res_scalar = phys_res if phys_res.dim() == 0 else phys_res.mean()

    return phys_res_scalar, _to_physics_metrics(phys_res_dict), phys_errors


def compute_initial_condition_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    is_initial_node: torch.Tensor,
    data,
) -> torch.Tensor:
    """Compute initial condition supervision loss.

    Compares predicted vs true sucrose concentration values only for nodes that were
    present at the initial timestep (t=0) AND only when the current data represents t=0.

    This ensures initial conditions are only enforced at the actual initial time,
    not for the same nodes at later times.

    Args:
        pred: Model predictions [N, 1]
        y: Target values [N, 1]
        is_initial_node: Boolean mask indicating initial nodes [N]
        data: Graph data containing time information

    Returns:
        torch.Tensor: Initial condition loss (MSE over initial nodes only if t=0)
    """
    # Check if this is actually the first timestep
    # Since we use physical time (plant_age) instead of timestep indices, we need to compare
    # with min_time (the time at the first timestep) instead of checking time == 0
    # For batched data, check if any graph in the batch is at the initial time
    if hasattr(data, 'time') and data.time is not None and hasattr(data, 'min_time') and data.min_time is not None:
        # Use a small tolerance for floating point comparison
        tolerance = 1e-6
        is_t0_batch = (torch.abs(data.time - data.min_time) < tolerance).any()
    else:
        # Fallback: assume this is not t=0 if no time information
        is_t0_batch = False

    # Only apply initial condition loss for the first timestep
    if not is_t0_batch or not is_initial_node.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # For batched data, we need to identify which nodes belong to t=0 graphs
    if hasattr(data, 'batch') and data.batch is not None:
        # Find which graphs are at the initial time
        tolerance = 1e-6
        t0_graph_mask = (torch.abs(data.time - data.min_time) < tolerance)
        t0_graph_indices = torch.where(t0_graph_mask)[0]

        if len(t0_graph_indices) == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Create mask for nodes that belong to t=0 graphs AND are initial nodes
        t0_node_mask = torch.zeros_like(is_initial_node, dtype=torch.bool)
        for graph_idx in t0_graph_indices:
            graph_node_mask = (data.batch == graph_idx)
            t0_node_mask |= (graph_node_mask & is_initial_node)

        if not t0_node_mask.any():
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Extract predictions and targets for t=0 initial nodes only
        pred_initial = pred[t0_node_mask]
        y_initial = y[t0_node_mask]
    else:
        # Single graph case: check if it is t=0
        if not is_t0_batch:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Extract predictions and targets for initial nodes only
        pred_initial = pred[is_initial_node]
        y_initial = y[is_initial_node]

    # Compute MSE loss over initial nodes at t=0
    ic_loss = F.mse_loss(pred_initial, y_initial, reduction='mean')

    return ic_loss


def compute_boundary_condition_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    is_boundary_node: torch.Tensor,
    data,
) -> torch.Tensor:
    """Compute boundary condition supervision loss.

    Compares predicted vs true sucrose values only for boundary nodes (degree=1).
    Boundary nodes are leaves and root tips - they represent sources and sinks
    and are critical for constraining physics-based solutions.

    Unlike IC loss which only applies at t=0, BC loss applies at ALL timesteps,
    providing spatiotemporal constraints throughout the simulation.

    NOTE: Boundary nodes are detected PER-TIMESTEP during data loading, so each
    graph in the batch has its own boundary nodes based on its topology at that timestep.
    This correctly handles dynamic graph growth over time.

    Args:
        pred: Model predictions [N, 1]
        y: Target values [N, 1]
        is_boundary_node: Boolean mask indicating boundary nodes [N] (concatenated across batch)
        data: Graph data (for potential batched handling)

    Returns:
        torch.Tensor: Boundary condition loss (MSE over boundary nodes)
    """
    if not is_boundary_node.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # For batched data, apply BC loss to all boundary nodes across all graphs
    # Note: is_boundary_node is already correctly per-timestep since it was
    # computed during data loading for each graph's topology
    if hasattr(data, 'batch') and data.batch is not None:
        # Extract predictions and targets for boundary nodes
        pred_boundary = pred[is_boundary_node]
        y_boundary = y[is_boundary_node]
    else:
        # Single graph case
        pred_boundary = pred[is_boundary_node]
        y_boundary = y[is_boundary_node]

    # Compute MSE loss over boundary nodes
    bc_loss = F.mse_loss(pred_boundary, y_boundary, reduction='mean')

    return bc_loss


def compute_loss(
    loss_mse: torch.Tensor,
    loss_phys: torch.Tensor,
    loss_type: LossType,
    loss_bc: torch.Tensor = None,
    loss_ic: torch.Tensor = None,
    lambda_data: float = 1.0,
    lambda_phys: float = 1.0,
    lambda_bc: float = 1.0,
    lambda_ic: float = 1.0,
    use_adaptive_weighting: bool = True,
    target_physics_ratio: float = 0.5
) -> tuple:
    """Compute loss based on the specified loss type configuration.

    Args:
        loss_mse: Mean squared error term
        loss_phys: Physics residual term
        loss_type: Type of loss to compute
        loss_bc: Boundary condition loss term (added to all physics-based losses)
        loss_ic: Initial condition loss term (only used for PHYSICS_WITH_IC_BC loss)
        lambda_data: Data loss weight (only used for COMBINED loss, weights the MSE term)
        lambda_phys: Residual loss weight (only used with "combined" and "physics" loss)
        lambda_bc: Boundary condition term weight
        lambda_ic: Initial condition term weight (only used for PHYSICS_WITH_IC_BC loss)
        use_adaptive_weighting: If True, adaptively balance losses
        target_physics_ratio: Target ratio of physics loss to supervision loss (for physics mode)

    Returns:
        Tuple of (total_loss, physics_weight, phys_contrib_pct, ref_contrib_pct)
        where:
            - total_loss: Combined loss value
            - physics_weight: Weight applied to physics term
            - phys_contrib_pct: Percentage contribution of physics to total loss
            - ref_contrib_pct: Percentage contribution of reference loss to total loss
    """
    # Training mode: DATA ONLY
    if loss_type == LossType.DATA_ONLY:
        return loss_mse, 0.0, 0.0, 0.0

    # Training mode: PHYSICS + IC + BC
    elif loss_type == LossType.PHYSICS_WITH_IC_BC:
        if loss_ic is None or loss_bc is None:
            raise ValueError("loss_ic and loss_bc must be provided for PHYSICS_WITH_IC_BC loss type")

        # Compute supervision loss (IC + BC)
        supervision_loss = (lambda_ic * loss_ic) + (lambda_bc * loss_bc)

        # Compute adaptive weight and effective physics loss
        if use_adaptive_weighting:
            physics_weight = _compute_adaptive_weight(
                supervision_loss, loss_phys, target_physics_ratio
            )
        else:
            physics_weight = lambda_phys

        weighted_physics_loss = physics_weight * loss_phys
        total_loss = supervision_loss + weighted_physics_loss

        # Compute percentage contributions
        percentages = _compute_loss_percentages(
            {'supervision': supervision_loss, 'physics': weighted_physics_loss},
            total_loss
        )

        return (
            total_loss,
            utils.to_float(physics_weight),
            percentages['physics'],
            percentages['supervision']
        )

    # Training mode: PHYSICS + DATA (COMBINED)
    elif loss_type == LossType.COMBINED:
        # Weight the data loss term
        weighted_data_loss = lambda_data * loss_mse

        # Compute adaptive weight for physics term
        if use_adaptive_weighting:
            physics_weight = _compute_adaptive_weight(
                weighted_data_loss, loss_phys, target_physics_ratio
            )
        else:
            physics_weight = lambda_phys

        weighted_physics_loss = physics_weight * loss_phys
        total_loss = weighted_data_loss + weighted_physics_loss

        # Compute percentage contributions
        percentages = _compute_loss_percentages(
            {'data': weighted_data_loss, 'physics': weighted_physics_loss},
            total_loss
        )

        return (
            total_loss,
            utils.to_float(physics_weight),
            percentages['physics'],
            percentages['data']
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_loss_and_metrics(
    pred: torch.Tensor,
    y: torch.Tensor,
    loss_phys: torch.Tensor,
    loss_config: LossConfig,
    is_initial_node: torch.Tensor = None,
    is_boundary_node: torch.Tensor = None,
    batch_vec: torch.Tensor = None,
    data = None,
    phys_metrics: Optional[PhysicsMetrics] = None,
    phys_errors: Optional['PhysicsErrorMetrics'] = None,
) -> LossResult:
    """Compute loss and metrics with per-graph aggregation (consistent with physics residual).

    Computes metrics per-graph first, then averages across graphs in the batch.
    This ensures equal weight per graph regardless of graph size, matching the
    physics residual computation strategy.

    Args:
        pred: Model predictions [N, 1]
        y: Target values [N, 1]
        loss_phys: Precomputed physics residual (already properly averaged per-graph)
        loss_config: Loss configuration containing all loss-related parameters
        is_initial_node: Boolean mask for initial nodes (required for PHYSICS_WITH_IC_BC)
        is_boundary_node: Boolean mask for boundary nodes (optional, for BC supervision)
        batch_vec: Batch assignment for each node [N] (None for single graph)
        data: Graph data object (required for PHYSICS_WITH_IC_BC to check timesteps)
        phys_metrics: Optional physics metrics for detailed tracking
        phys_errors: Optional physics error metrics (MSE, RMSE, Relative Error)

    Returns:
        LossResult containing all computed metrics
    """
    # Squeeze predictions and targets to [N]
    pred_flat = pred.squeeze(-1)
    y_flat = y.squeeze(-1)

    # Denormalize for relative error calculation using target_scaler from data
    target_scaler = getattr(data, 'target_scaler', None)
    pred_original, y_original = _denormalize_for_metrics(pred_flat, y_flat, target_scaler)

    # Compute error metrics
    loss_mse, mae, rmse, rel_error = _compute_error_metrics(
        pred_flat, y_flat, pred_original, y_original, batch_vec
    )

    # Compute physics-specific losses (IC and BC)
    loss_ic, loss_bc, bc_node_count, bc_node_pct = _compute_physics_losses(
        pred, y, loss_config.loss_type, is_initial_node, is_boundary_node, data
    )

    # Compute total loss based on configuration
    # Returns: (total_loss, phys_weight, phys_%, ref_%)
    total_loss, phys_weight, phys_contrib_pct, ref_contrib_pct = compute_loss(
        loss_mse,
        loss_phys,
        loss_config.loss_type,
        loss_bc,
        loss_ic,
        loss_config.lambda_data,
        loss_config.lambda_phys,
        loss_config.lambda_bc,
        loss_config.lambda_ic,
        use_adaptive_weighting=loss_config.use_adaptive_physics_weighting,
        target_physics_ratio=loss_config.target_physics_ratio
    )

    return LossResult(
        total_loss=utils.to_float(total_loss),
        mse=utils.to_float(loss_mse),
        mae=utils.to_float(mae),
        rmse=utils.to_float(rmse),
        rel_error=utils.to_float(rel_error),
        phys=utils.to_float(loss_phys),
        ic=utils.to_float(loss_ic),
        bc=utils.to_float(loss_bc),
        phys_weight=phys_weight,
        bc_nodes=bc_node_count,
        bc_pct=bc_node_pct,
        phys_contrib_pct=phys_contrib_pct,
        sup_or_data_contrib_pct=ref_contrib_pct,
        physics_metrics=phys_metrics,
        physics_errors=phys_errors
    )


def _denormalize_for_metrics(
    pred_flat: torch.Tensor,
    y_flat: torch.Tensor,
    target_scaler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Denormalize predictions and targets for relative error calculation.

    Args:
        pred_flat: Flattened predictions [N] (standardized)
        y_flat: Flattened targets [N] (standardized)
        target_scaler: Standardizer instance for inverse transformation

    Returns:
        Tuple of (pred_original, y_original) in original space
    """
    if target_scaler is not None:
        # Inverse-transform from standardized space to original space
        # pred/y are [N], need to reshape to [N, 1] for scaler
        pred_original = target_scaler.inv_transform(pred_flat.unsqueeze(-1)).squeeze(-1)
        y_original = target_scaler.inv_transform(y_flat.unsqueeze(-1)).squeeze(-1)
    else:
        # No denormalization available
        pred_original = pred_flat
        y_original = y_flat

    return pred_original, y_original


def _compute_error_metrics(
    pred_flat: torch.Tensor,
    y_flat: torch.Tensor,
    pred_original: torch.Tensor,
    y_original: torch.Tensor,
    batch_vec: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute error metrics (MSE, MAE, RMSE, relative error).

    Args:
        pred_flat: Normalized predictions [N]
        y_flat: Normalized targets [N]
        pred_original: Original space predictions [N]
        y_original: Original space targets [N]
        batch_vec: Batch assignment vector [N] (None for single graph)

    Returns:
        Tuple of (mse, mae, rmse, rel_error)
    """
    # Compute per-node errors
    squared_error = (pred_flat - y_flat).pow(2)
    absolute_error = torch.abs(pred_flat - y_flat)
    absolute_error_original = torch.abs(pred_original - y_original)

    # Per-graph aggregation
    if batch_vec is not None:
        # Average errors per graph first
        mse_per_graph = scatter_mean(squared_error, batch_vec, dim=0)
        mae_per_graph = scatter_mean(absolute_error, batch_vec, dim=0)

        # Then average across graphs in batch
        loss_mse = mse_per_graph.mean()
        mae = mae_per_graph.mean()
        rmse = torch.sqrt(mse_per_graph).mean()

        # Relative error in original space: per-graph MAE / per-graph mean target
        mae_original_per_graph = scatter_mean(absolute_error_original, batch_vec, dim=0)
        y_abs_original_per_graph = scatter_mean(torch.abs(y_original), batch_vec, dim=0)
        rel_error_per_graph = mae_original_per_graph / (y_abs_original_per_graph + EPSILON)
        rel_error = rel_error_per_graph.mean()
    else:
        # Single graph case: simple average across nodes
        loss_mse = squared_error.mean()
        mae = absolute_error.mean()
        rmse = torch.sqrt(squared_error.mean())
        rel_error = absolute_error_original.mean() / (torch.abs(y_original).mean() + EPSILON)

    return loss_mse, mae, rmse, rel_error


def _compute_physics_losses(
    pred: torch.Tensor,
    y: torch.Tensor,
    loss_type: LossType,
    is_initial_node: torch.Tensor,
    is_boundary_node: torch.Tensor,
    data
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    """Compute physics-specific losses (IC and BC) and boundary node statistics.

    Args:
        pred: Model predictions [N, 1]
        y: Target values [N, 1]
        loss_type: Type of loss being computed
        is_initial_node: Boolean mask for initial nodes
        is_boundary_node: Boolean mask for boundary nodes
        data: Graph data object

    Returns:
        Tuple of (loss_ic, loss_bc, bc_node_count, bc_node_percentage)
    """
    # Initialize to zero
    loss_ic = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    loss_bc = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    bc_node_count = 0
    bc_node_percentage = 0.0

    if loss_type == LossType.PHYSICS_WITH_IC_BC:
        # Validate required inputs
        if data is None:
            raise ValueError("data must be provided for PHYSICS_WITH_IC_BC loss type to check timestep")
        if is_initial_node is None:
            raise ValueError("is_initial_node must be provided for PHYSICS_WITH_IC_BC loss type")
        if is_boundary_node is None:
            raise ValueError("is_boundary_node must be provided for PHYSICS_WITH_IC_BC loss type")

        # Compute losses
        loss_bc = compute_boundary_condition_loss(pred, y, is_boundary_node, data)
        loss_ic = compute_initial_condition_loss(pred, y, is_initial_node, data)

        # Compute statistics
        bc_node_count = int(is_boundary_node.sum().item())
        total_nodes = pred.shape[0]
        bc_node_percentage = (bc_node_count / total_nodes * 100) if total_nodes > 0 else 0.0

    return loss_ic, loss_bc, bc_node_count, bc_node_percentage


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_config: LossConfig,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        clip_grad_norm: float = 1.0,
    ) -> EpochResult:
    """Train model for one epoch.

    Args:
        model: The neural network model
        loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        loss_config: Configuration for loss computation
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        clip_grad_norm: Maximum norm for gradient clipping

    Returns:
        EpochResult containing averaged metrics from the epoch

    Raises:
        RuntimeError: If no training samples are processed
    """
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0,
              "rmse": 0.0, "rel_error": 0.0,
              "phys": 0.0, "ic": 0.0, "bc": 0.0,
              "phys_weight": 0.0, "supervision_weight": 0.0,
              "weighted_supervision": 0.0, "weighted_physics": 0.0,
              "phys_contrib_pct": 0.0, "sup_contrib_pct": 0.0,
              "bc_nodes": 0, "bc_pct": 0.0,
              "n_batches": 0, "last_phys_metrics": None, "last_phys_errors": None}

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        # Prepare data for model: add time info
        data = utils.prepare_model_inputs(
            data,
            model,
            is_training=True
        )

        # Forward pass (returns tensor or dict depending on model type)
        model_output = run_forward(model, data)

        # Extract predictions tensor for metric computation
        pred = _extract_predictions(model_output)

        # Compute physics residual (handles both model types)
        if loss_config.loss_type == LossType.DATA_ONLY:
            phys_res = torch.tensor(0.0, device=pred.device)
            # Log physics values for analysis and get metrics for terminal display
            phys_res_metrics, phys_res_errors = log_physics_values(model_output, data, phase='train')
        else:
            phys_res, phys_res_metrics, phys_res_errors = compute_physics_residual_step(
                model=model,
                data=data,
                model_output=model_output,
                loss_type=loss_config.loss_type,
                phase='train'
            )

        # Compute loss and metrics (no physics scaling needed)
        result = compute_loss_and_metrics(
            pred,
            data.y,
            phys_res,
            loss_config,
            getattr(data, 'is_initial_node', None),
            getattr(data, 'is_boundary_node', None),
            getattr(data, 'batch', None),
            data,
            phys_res_metrics,
            phys_res_errors
        )

        # Convert LossResult.total_loss back to tensor for backward
        loss_tensor = torch.tensor(result.total_loss, device=pred.device, requires_grad=True)

        # Re-compute loss tensor properly for backprop
        if loss_config.loss_type == LossType.DATA_ONLY:
            loss_tensor = F.mse_loss(pred.squeeze(-1), data.y.squeeze(-1))
        else:
            # Recompute actual loss tensor (not the float)
            pred_flat = pred.squeeze(-1)
            y_flat = data.y.squeeze(-1)
            squared_error = (pred_flat - y_flat).pow(2)
            if getattr(data, 'batch', None) is not None:
                mse_per_graph = scatter_mean(squared_error, data.batch, dim=0)
                loss_mse_tensor = mse_per_graph.mean()
            else:
                loss_mse_tensor = squared_error.mean()

            # Compute physics-specific losses as tensors
            loss_ic_tensor = compute_initial_condition_loss(pred, data.y, getattr(data, 'is_initial_node', None), data) if loss_config.loss_type == LossType.PHYSICS_WITH_IC_BC else torch.tensor(0.0, device=pred.device)
            loss_bc_tensor = compute_boundary_condition_loss(pred, data.y, getattr(data, 'is_boundary_node', None), data) if getattr(data, 'is_boundary_node', None) is not None else torch.tensor(0.0, device=pred.device)

            # Recompute total loss as tensor
            loss_tensor, _, _, _ = compute_loss(
                loss_mse_tensor,
                phys_res,
                loss_config.loss_type,
                loss_bc_tensor,
                loss_ic_tensor,
                loss_config.lambda_data,
                loss_config.lambda_phys,
                loss_config.lambda_bc,
                loss_config.lambda_ic,
                use_adaptive_weighting=loss_config.use_adaptive_physics_weighting,
                target_physics_ratio=loss_config.target_physics_ratio
            )

        loss_tensor.backward()

        # Log gradient norms (first batch only to avoid clutter)
        if writer is not None and batch_idx == 0:
            logging.log_gradient_norms(model, writer, epoch, batch_idx, len(loader))

        # Gradient clipping and optimization step
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        # Accumulate metrics for epoch averaging
        with torch.no_grad():
            # Calculate actual weighted components for this batch
            if loss_config.loss_type == LossType.PHYSICS_WITH_IC_BC:
                batch_weighted_sup = (loss_config.lambda_ic * result.ic) + (loss_config.lambda_bc * result.bc)
                batch_weighted_phys = result.phys_weight * result.phys
            elif loss_config.loss_type == LossType.COMBINED:
                batch_weighted_sup = loss_config.lambda_data * result.mse
                batch_weighted_phys = result.phys_weight * result.phys
            else:
                batch_weighted_sup = 0.0
                batch_weighted_phys = 0.0

            # Accumulate the physics residual (no normalization needed)
            accumulate_epoch_stats(totals, result, batch_weighted_sup, batch_weighted_phys)

            # Log physics metrics
            if writer is not None and batch_idx % 10 == 0:
                step = epoch * len(loader) + batch_idx
                writer.add_scalar('training/batch_physics', result.phys, step)

            # Log batch-level metrics (every 10 batches to avoid clutter)
            if writer is not None and batch_idx % 10 == 0:
                logging.log_batch_metrics(writer, epoch, batch_idx, len(loader),
                                          result.total_loss, result.mse, result.mae, result.rmse,
                                          result.rel_error, result.phys, result.ic, result.bc,
                                          result.bc_nodes, result.bc_pct)

                # Log detailed physics components if available
                if phys_res_metrics is not None:
                    logging.log_physics_components(writer, epoch, batch_idx, len(loader),
                                                   phys_res_metrics, phase='train')

    if totals["n_batches"] == 0:
        raise RuntimeError("No training samples this epoch.")

    return EpochResult.from_totals(totals)


def train_model(
    model_setup: ModelSetup,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    writer: SummaryWriter,
    config: TrainingConfig,
    model_cfg: ModelConfig
) -> TrainingState:
    """Run the main training loop with early stopping.

    Args:
        model_setup: Model setup containing model and scalers
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        writer: TensorBoard writer for logging
        config: Training configuration
        model_cfg: Model configuration for checkpointing

    Returns:
        TrainingState: Final training state with best metrics
    """
    # Initialize training state
    training_state = TrainingState()

    # Create loss configuration from training config
    loss_config = LossConfig.from_training_config(config)

    print(f"\nStarting training with loss type: {config.loss_type.value}")
    for epoch in range(1, config.epochs + 1):
        training_state.current_epoch = epoch

        # Training
        tr_result = train_epoch(
            model_setup.model,
            train_loader,
            optimizer,
            loss_config,
            writer,
            epoch,
            clip_grad_norm=config.clip_grad_norm
        )

        # Validation
        val_result = eval_model(
            model_setup.model,
            val_loader,
            loss_config,
            writer,
            epoch,
            phase='val'
        )

        # Learning rate scheduling (use combined validation loss)
        scheduler.step(val_result.loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create metrics objects
        tr_metrics = TrainingMetrics(
            tr_result.loss,
            tr_result.mse,
            tr_result.mae,
            tr_result.rmse,
            tr_result.rel_error,
            tr_result.phys,
            tr_result.ic,
            tr_result.bc,
            tr_result.bc_nodes,
            tr_result.bc_pct,
            tr_result.physics_metrics,
            tr_result.physics_errors
        )
        val_metrics = TrainingMetrics(
            val_result.loss,
            val_result.mse,
            val_result.mae,
            val_result.rmse,
            val_result.rel_error,
            val_result.phys,
            val_result.ic,
            val_result.bc,
            val_result.bc_nodes,
            val_result.bc_pct,
            val_result.physics_metrics,
            val_result.physics_errors
        )

        # Log metrics to TensorBoard
        logging.log_epoch_metrics(writer, epoch, tr_metrics, val_metrics, current_lr)

        # Console logging
        # Only print BC nodes info on first epoch (since it remains constant)
        if epoch == 1:
            base_log = f"BC nodes = {tr_result.bc_nodes:.1f}({tr_result.bc_pct:.1f}%)\n\n"
        else:
            base_log = ""

        base_log += (f"\n==== Epoch {epoch:03d} ==== "
                    f"train_tot={tr_result.loss:.3e} train_mse={tr_result.mse:.3e} "
                    f"train_relerr={tr_result.rel_error:.3e} |")
        base_log += f" train_phys={tr_result.phys:.3e} train_ic={tr_result.ic:.3e} train_bc={tr_result.bc:.3e}"

        # Add weighted component info if using adaptive weighting
        if config.use_adaptive_physics_weighting and config.loss_type == LossType.PHYSICS_WITH_IC_BC:
            base_log += (f" | sup_w={tr_result.weighted_supervision:.3e}({tr_result.sup_contrib_pct:.1f}%) "
                        f"phys_w={tr_result.weighted_physics:.3e}({tr_result.phys_contrib_pct:.1f}%)")
        elif config.use_adaptive_physics_weighting and config.loss_type == LossType.COMBINED:
            base_log += (f" | data_w={tr_result.weighted_supervision:.3e}({tr_result.sup_contrib_pct:.1f}%) "
                        f"phys_w={tr_result.weighted_physics:.3e}({tr_result.phys_contrib_pct:.1f}%)")

        base_log += (f" | val_tot={val_result.loss:.3e} val_mse={val_result.mse:.3e} "
                    f"val_phys={val_result.phys:.3e} val_relerr={val_result.rel_error:.3e}")
        # base_log += f" val_ic={val_result.ic:.3e} val_bc={val_result.bc:.3e}"

        # Add physics details to console output if available
        if tr_result.physics_metrics is not None:
            physics_log = f" | {tr_result.physics_metrics}"
            print(base_log + physics_log)
        else:
            print(base_log)

        # Add physics error metrics to console output if available
        if tr_result.physics_errors is not None:
            print(f"Physics Errors (Train): {tr_result.physics_errors}")
        if val_result.physics_errors is not None:
            print(f"Physics Errors (Valid): {val_result.physics_errors}")

        # Model saving and early stopping (use combined validation loss)
        if training_state.update_best(val_result.loss, epoch):
            # Log best model achievement
            writer.add_scalar('best_model/epoch', epoch, epoch)
            writer.add_scalar('best_model/val_loss', val_result.loss, epoch)
            writer.add_scalar('best_model/val_mse', val_result.mse, epoch)

            # Save model checkpoint
            utils.save_checkpoint(
                model_setup, model_cfg, optimizer, scheduler,
                epoch, val_result.loss, val_result.mse, val_result.phys,
                val_result.rel_error, config.model_save_path
            )
        else:
            if training_state.should_stop(config.patience):
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best validation loss: {training_state.best_val_loss:.4e} "
                      f"at epoch {training_state.best_epoch}")

                # Log early stopping
                writer.add_text('training/early_stopping',
                                f"Stopped at epoch {epoch}, best at {training_state.best_epoch}")
                break

    print("\nTraining completed!")

    # Log final training summary
    writer.add_text('training/summary',
                    f"Training completed. Best validation loss: {training_state.best_val_loss:.4f} at epoch {training_state.best_epoch}")

    return training_state


def test_model(
    model_setup: ModelSetup,
    test_loader: DataLoader,
    writer: SummaryWriter,
    training_state: TrainingState,
    config: TrainingConfig
) -> None:
    """Run final evaluation and log results.

    Args:
        model_setup: Model setup with trained model
        test_loader: Test data loader
        writer: TensorBoard writer
        training_state: Training state with best epoch info
        config: Training configuration
    """
    # Load the best model for testing
    utils.load_best_model(model_setup, config.model_save_path, model_setup.device)

    # Create loss configuration from training config
    loss_config = LossConfig.from_training_config(config)

    # Final evaluation on test set
    test_result = eval_model(
        model_setup.model,
        test_loader,
        loss_config,
        writer,
        training_state.best_epoch,
        phase='test'
    )

    # Log final test metrics
    writer.add_scalar('final/test_loss', test_result.loss, training_state.best_epoch)
    writer.add_scalar('final/test_mse', test_result.mse, training_state.best_epoch)
    writer.add_scalar('final/test_mae', test_result.mae, training_state.best_epoch)
    writer.add_scalar('final/test_rmse', test_result.rmse, training_state.best_epoch)
    writer.add_scalar('final/test_rel_error', test_result.rel_error, training_state.best_epoch)
    writer.add_scalar('final/test_physics', test_result.phys, training_state.best_epoch)
    writer.add_scalar('final/test_ic_loss', test_result.ic, training_state.best_epoch)
    writer.add_scalar('final/test_bc_loss', test_result.bc, training_state.best_epoch)

    # Log adaptive weighting metrics if available
    if config.use_adaptive_physics_weighting and config.loss_type == LossType.PHYSICS_WITH_IC_BC:
        writer.add_scalar('final/test_phys_weight', test_result.phys_weight, training_state.best_epoch)
        writer.add_scalar('final/test_supervision_weight', test_result.supervision_weight, training_state.best_epoch)
        writer.add_scalar('final/test_bc_nodes', test_result.bc_nodes, training_state.best_epoch)
        writer.add_scalar('final/test_bc_pct', test_result.bc_pct, training_state.best_epoch)

    # Create final summary
    final_summary = (f"Final Results:\n"
                    f"Test Loss: {test_result.loss:.3e}\n"
                    f"Test MSE: {test_result.mse:.3e}\n"
                    f"Test MAE: {test_result.mae:.3e}\n"
                    f"Test RMSE: {test_result.rmse:.3e}\n"
                    f"Test Rel Error: {test_result.rel_error:.3e}\n"
                    f"Test Physics: {test_result.phys:.3e}\n"
                    f"Test IC Loss: {test_result.ic:.3e}\n"
                    f"Test BC Loss: {test_result.bc:.3e}\n"
                    f"Best epoch: {training_state.best_epoch}")

    # Add adaptive weighting info if available
    if config.use_adaptive_physics_weighting and config.loss_type == LossType.PHYSICS_WITH_IC_BC:
        final_summary += (f"\nAdaptive Weight: {test_result.phys_weight:.3f}\n"
                         f"Supervision Weight: {test_result.supervision_weight:.3f}\n"
                         f"Physics Contribution: {test_result.phys_contrib_pct:.1f}%\n"
                         f"Supervision Contribution: {test_result.sup_contrib_pct:.1f}%\n"
                         f"BC Nodes: {test_result.bc_nodes:.1f} ({test_result.bc_pct:.2f}%)")

    if test_result.physics_metrics is not None:
        final_summary += f"\nTest Physics Details: {test_result.physics_metrics}"

    writer.add_text('final/results', final_summary)

    # Console output with adaptive weighting info
    console_msg = (f"\nFinal test metrics!\nLoss: {test_result.loss:.3e}, Physics Loss: {test_result.phys:.3e}, "
                   f"IC Loss: {test_result.ic:.3e}, BC Loss: {test_result.bc:.3e} | "
                   f"MSE: {test_result.mse:.3e}, RMSE: {test_result.rmse:.3e}, "
                   f"MAE: {test_result.mae:.3e}, RelErr: {test_result.rel_error:.3e}")

    if config.use_adaptive_physics_weighting and config.loss_type == LossType.PHYSICS_WITH_IC_BC:
        console_msg += (f"\n  Adaptive Weight: {test_result.phys_weight:.3f}, "
                       f"Supervision Weight: {test_result.supervision_weight:.3f}\n"
                       f"  Physics Contribution: {test_result.phys_contrib_pct:.1f}%, "
                       f"Supervision Contribution: {test_result.sup_contrib_pct:.1f}%\n"
                       f"  BC Nodes: {test_result.bc_nodes:.1f} ({test_result.bc_pct:.2f}%)")

    print(console_msg)
    if test_result.physics_metrics is not None:
        print(f"Test Physics Details: {test_result.physics_metrics}")


def eval_model(
        model: nn.Module,
        loader: DataLoader,
        loss_config: LossConfig,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        phase: str = 'val',
    ) -> EpochResult:
    """Evaluate model on a dataset.

    Args:
        model: The neural network model
        loader: DataLoader containing evaluation data
        loss_config: Configuration for loss computation
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        phase: Phase name ('val' or 'test')

    Returns:
        EpochResult containing averaged metrics from evaluation
    """
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0,
              "rmse": 0.0, "rel_error": 0.0,
              "phys": 0.0, "ic": 0.0, "bc": 0.0,
              "phys_weight": 0.0, "supervision_weight": 0.0,
              "weighted_supervision": 0.0, "weighted_physics": 0.0,
              "phys_contrib_pct": 0.0, "sup_contrib_pct": 0.0,
              "bc_nodes": 0, "bc_pct": 0.0,
              "n_batches": 0, "last_phys_metrics": None, "last_phys_errors": None}

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # Prepare data for model: add time info
            data = utils.prepare_model_inputs(
                data,
                model,
                is_training=False
            )

            # Forward pass (returns tensor or dict depending on model type)
            model_output = run_forward(model, data)

            # Extract predictions tensor for metric computation
            pred = _extract_predictions(model_output)

            # Compute physics residual (handles both model types)
            if loss_config.loss_type == LossType.DATA_ONLY:
                phys_res = torch.tensor(0.0, device=pred.device)
                # Log physics values for analysis and get metrics for terminal display
                phys_res_metrics, phys_res_errors = log_physics_values(model_output, data, phase=phase)
            else:
                phys_res, phys_res_metrics, phys_res_errors = compute_physics_residual_step(
                    model=model,
                    data=data,
                    model_output=model_output,
                    loss_type=loss_config.loss_type,
                    phase=phase
                )

            # Compute loss and metrics (no physics scaling needed)
            result = compute_loss_and_metrics(
                pred,
                data.y,
                phys_res,
                loss_config,
                getattr(data, 'is_initial_node', None),
                getattr(data, 'is_boundary_node', None),
                getattr(data, 'batch', None),
                data,
                phys_res_metrics,
                phys_res_errors
            )

            # Calculate actual weighted components for this batch (for eval)
            if loss_config.loss_type == LossType.PHYSICS_WITH_IC_BC:
                batch_weighted_sup = (loss_config.lambda_ic * result.ic) + (loss_config.lambda_bc * result.bc)
                batch_weighted_phys = result.phys_weight * result.phys
            elif loss_config.loss_type == LossType.COMBINED:
                batch_weighted_sup = loss_config.lambda_data * result.mse
                batch_weighted_phys = result.phys_weight * result.phys
            else:
                batch_weighted_sup = 0.0
                batch_weighted_phys = 0.0

            # Accumulate metrics
            accumulate_epoch_stats(totals, result, batch_weighted_sup, batch_weighted_phys)

            # Log distribution of predictions and targets (first batch only, every 5 epochs)
            if writer is not None and batch_idx == 0 and epoch % 5 == 0:
                # pred is already in original space (no standardization)
                logging.log_evaluation_histograms(writer, phase, epoch, pred, data.y)

                # Log individual loss components for debugging
                writer.add_scalar(f'{phase}/mse', result.mse, epoch)
                writer.add_scalar(f'{phase}/physics', result.phys, epoch)
                writer.add_scalar(f'{phase}/ic_loss', result.ic, epoch)
                writer.add_scalar(f'{phase}/loss', result.total_loss, epoch)

                # Log detailed physics components if available
                if phys_res_metrics is not None:
                    logging.log_physics_components(writer, epoch, 0, 1,
                                                   phys_res_metrics, phase=phase)

    # Clear GPU memory after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return EpochResult.from_totals(totals)
