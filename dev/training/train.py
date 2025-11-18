import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_mean

from typing import Tuple, Optional

from model.config import ModelConfig
from model.physics import physics_residual
from .config import TrainingConfig, TrainingState, TrainingMetrics, ModelSetup, LossType, PhysicsMetrics

import training.utils as utils
import training.logging as logging

def accumulate_epoch_stats(
    totals,
    loss: float,
    mse: float,
    mae: float,
    rmse: float,
    rel_error: float,
    phys: float,
    ic: float,
    bc: float,
    last_phys: Optional[PhysicsMetrics],
    adaptive_weight: float = 0.0,
    supervision_weight: float = 0.0,
    bc_nodes: int = 0,
    bc_pct: float = 0.0,
    phys_contrib_pct: float = 0.0,
    sup_contrib_pct: float = 0.0
):
    totals["loss"] += float(loss)
    totals["mse"] += float(mse)
    totals["mae"] += float(mae)
    totals["rmse"] += float(rmse)
    totals["rel_error"] += float(rel_error)
    totals["phys"] += float(phys)
    totals["ic"] += float(ic)
    totals["bc"] += float(bc)
    totals["n_batches"] += 1
    totals["last_phys_metrics"] = last_phys

    # Accumulate new statistics
    totals["adaptive_weight"] += adaptive_weight
    totals["supervision_weight"] += supervision_weight
    totals["bc_nodes"] += bc_nodes
    totals["bc_pct"] += bc_pct
    totals["phys_contrib_pct"] += phys_contrib_pct
    totals["sup_contrib_pct"] += sup_contrib_pct


def run_forward(model: nn.Module, data) -> torch.Tensor:
    """Forward pass on inputs.

    Returns:
        Content predictions
    """
    return model(data)


def _to_physics_metrics(physics_res) -> Optional[PhysicsMetrics]:
    """Convert dict of physics components to PhysicsMetrics."""
    if physics_res is not None:
        return PhysicsMetrics(
            J_ax=float(physics_res['J_ax']),
            F_in=float(physics_res['F_in']),
            F_out=float(physics_res['F_out']),
            dC_dt=float(physics_res['dC_dt']),
            dS_dt_from_flux=float(physics_res['dS_dt_from_flux']),
            dC_dt_from_physics=float(physics_res['dC_dt_from_physics'])
        )
    else:
        return None


def compute_physics_residual_step(
    model: nn.Module,
    data,
    pred: Optional[torch.Tensor],
    require_time_grad: bool,
) -> Tuple[torch.Tensor, Optional[PhysicsMetrics]]:
    """
    Unified physics residual computation for train/eval without standardization.

    - In training, use the already-built graph (pred computed under grad,
      and data.time_per_node requires_grad=True from prepare_model_inputs).
    - In eval, temporarily enable grad, rebuild a leaf for time_per_node, re-forward,
      then compute and detach the scalar physics residual.

    Returns (phys_res_scalar_detached, PhysicsMetrics|None).
    """
    if require_time_grad:
        # Training path: pred must already be computed under grad; time_per_node should require grad.
        phys_res, phys_res_dict = physics_residual(pred, data)
    else:
        # Eval path: re-enable grad for a one-off forward, then detach result.
        original_time = data.time_per_node
        try:
            with torch.enable_grad():
                time_leaf = original_time.detach().clone().requires_grad_(True)
                data.time_per_node = time_leaf
                pred_eval = model(data)
                phys_res, phys_res_dict = physics_residual(pred_eval, data)
        finally:
            # Restore original attribute to avoid side effects
            data.time_per_node = original_time

    # Reduce to scalar; detach ONLY in eval (no grad) path
    phys_res_scalar = phys_res if phys_res.dim() == 0 else phys_res.mean()

    # Detach only in eval path
    if not require_time_grad:
        phys_res_scalar = phys_res_scalar.detach()

    return phys_res_scalar, _to_physics_metrics(phys_res_dict)


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


def compute_loss(loss_mse: torch.Tensor, loss_phys: torch.Tensor, loss_type: LossType,
                 lambda_phys: float = 1.0, loss_ic: torch.Tensor = None, lambda_ic: float = 1.0,
                 loss_bc: torch.Tensor = None, lambda_bc: float = 1.0,
                 use_adaptive_weighting: bool = True, target_physics_ratio: float = 0.5) -> tuple:
    """Compute loss based on the specified loss type configuration.

    Args:
        loss_mse: Mean squared error term
        loss_phys: Physics residual term
        loss_type: Type of loss to compute
        lambda_phys: Physics term weight (only used for COMBINED loss)
        loss_ic: Initial condition loss term (only used for PHYSICS_WITH_IC_BC loss)
        lambda_ic: Initial condition term weight (only used for PHYSICS_WITH_IC_BC loss)
        loss_bc: Boundary condition loss term (added to all physics-based losses)
        lambda_bc: Boundary condition term weight
        use_adaptive_weighting: If True, adaptively balance losses
        target_physics_ratio: Target ratio of physics loss to supervision loss (for physics mode)

    Returns:
        Tuple of (total_loss, effective_physics_weight, supervision_weight, phys_contrib_pct, sup_contrib_pct)
    """
    if loss_type == LossType.DATA_ONLY:
        return loss_mse, 0.0, 0.0, 0.0, 0.0
    elif loss_type == LossType.PHYSICS_WITH_IC_BC:
        if loss_ic is None:
            raise ValueError("loss_ic must be provided for PHYSICS_WITH_IC_BC loss type")
        if loss_bc is None:
            raise ValueError("loss_bc must be provided for PHYSICS_WITH_IC_BC loss type")

        # Compute supervision loss (IC + BC)
        supervision_loss = lambda_ic * loss_ic + lambda_bc * loss_bc

        # === ADAPTIVE WEIGHTING FOR PHYSICS MODE ===
        # Balance physics residual with supervision losses dynamically
        adaptive_weight = 1.0  # Default
        if use_adaptive_weighting and loss_phys > 1e-8 and supervision_loss > 1e-8:
            # Scale physics loss to be a target ratio of supervision loss
            adaptive_weight = (supervision_loss / loss_phys) * target_physics_ratio
            # Clamp for stability
            adaptive_weight = torch.clamp(adaptive_weight, min=0.01, max=10.0)
            effective_physics = adaptive_weight * loss_phys
        else:
            effective_physics = loss_phys

        total_loss = effective_physics + supervision_loss

        # Return adaptive weight as float for logging
        adaptive_weight_float = float(adaptive_weight.item() if torch.is_tensor(adaptive_weight) else adaptive_weight)
        supervision_weight_float = float(supervision_loss.item() if torch.is_tensor(supervision_loss) else supervision_loss)

        # Compute percentage contributions
        total_loss_val = float(total_loss.item() if torch.is_tensor(total_loss) else total_loss)
        if total_loss_val > 1e-8:
            phys_contrib_pct = 100.0 * float(effective_physics.item() if torch.is_tensor(effective_physics) else effective_physics) / total_loss_val
            sup_contrib_pct = 100.0 * float(supervision_loss.item() if torch.is_tensor(supervision_loss) else supervision_loss) / total_loss_val
        else:
            phys_contrib_pct = 0.0
            sup_contrib_pct = 0.0

        return total_loss, adaptive_weight_float, supervision_weight_float, phys_contrib_pct, sup_contrib_pct
    elif loss_type == LossType.COMBINED:
        # === LOSS REFINEMENT 3: Adaptive weighting to auto-balance data vs physics losses ===
        # Problem: Physics loss can be 100-300x larger than data loss, making manual lambda_phys tuning fragile
        # Solution: Compute adaptive weight to make physics contribute ~50% of data loss magnitude
        adaptive_weight = lambda_phys
        if use_adaptive_weighting and loss_phys > 1e-8:  # Avoid division by near-zero
            # Compute ratio: how much to scale physics so it's ~50% of data loss
            adaptive_weight = (loss_mse / loss_phys) * 0.5
            # Clamp for stability: don't let it go too small or too large
            adaptive_weight = torch.clamp(adaptive_weight, min=0.001, max=1.0)
            # Effective lambda = base lambda * adaptive weight
            effective_lambda = lambda_phys * adaptive_weight
        else:
            effective_lambda = lambda_phys
        total_loss = loss_mse + effective_lambda * loss_phys

        # Return weights for logging
        adaptive_weight_float = float(adaptive_weight.item() if torch.is_tensor(adaptive_weight) else adaptive_weight)
        data_weight_float = float(loss_mse.item() if torch.is_tensor(loss_mse) else loss_mse)

        # Compute percentage contributions for COMBINED mode
        total_loss_val = float(total_loss.item() if torch.is_tensor(total_loss) else total_loss)
        effective_phys_val = float((effective_lambda * loss_phys).item() if torch.is_tensor(effective_lambda * loss_phys) else (effective_lambda * loss_phys))
        if total_loss_val > 1e-8:
            data_contrib_pct = 100.0 * float(loss_mse.item() if torch.is_tensor(loss_mse) else loss_mse) / total_loss_val
            phys_contrib_pct = 100.0 * effective_phys_val / total_loss_val
        else:
            data_contrib_pct = 0.0
            phys_contrib_pct = 0.0

        return total_loss, adaptive_weight_float, data_weight_float, phys_contrib_pct, data_contrib_pct
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_loss_and_metrics(
    pred: torch.Tensor,
    y: torch.Tensor,
    loss_phys: torch.Tensor,
    loss_type: LossType,
    lambda_phys: float = 1.0,
    is_initial_node: torch.Tensor = None,
    lambda_ic: float = 1.0,
    is_boundary_node: torch.Tensor = None,
    lambda_bc: float = 1.0,
    batch_vec: torch.Tensor = None,
    data = None,
    use_adaptive_physics_weighting: bool = True,
    target_physics_ratio: float = 0.5,
):
    """Compute loss and metrics with per-graph aggregation (consistent with physics residual).

    Computes metrics per-graph first, then averages across graphs in the batch.
    This ensures equal weight per graph regardless of graph size, matching the
    physics residual computation strategy.

    Args:
        pred: Model predictions [N, 1]
        y: Target values [N, 1]
        loss_phys: Precomputed physics residual (already properly averaged per-graph)
        loss_type: Type of loss to compute
        lambda_phys: Physics term weight
        is_initial_node: Boolean mask for initial nodes (required for PHYSICS_WITH_IC_BC)
        lambda_ic: Initial condition term weight
        is_boundary_node: Boolean mask for boundary nodes (optional, for BC supervision)
        lambda_bc: Boundary condition term weight
        batch_vec: Batch assignment for each node [N] (None for single graph)
        data: Graph data object (required for PHYSICS_WITH_IC_BC to check timesteps)
        use_adaptive_physics_weighting: If True, balance physics loss adaptively
        target_physics_ratio: Target ratio of physics to supervision loss

    Returns:
        Tuple of (total_loss, mse, mae, rmse, rel_error, ic_loss, bc_loss)
    """
    # Squeeze predictions and targets to [N]
    pred_flat = pred.squeeze(-1)
    y_flat = y.squeeze(-1)

    # IMPORTANT: Denormalize predictions and targets for relative error calculation
    # pred and y are normalized [0,1], so relative error
    # computed in normalized space is misleading (tiny values -> huge percentages)
    # We need to compute relative error in PHYSICAL space!
    if data is not None and hasattr(data, 'target_scale'):
        target_scale = data.target_scale.to(pred.device)
        # Handle batched case
        if batch_vec is not None and target_scale.numel() > 1:
            target_scale_per_node = target_scale[batch_vec]
        else:
            target_scale_per_node = target_scale

        # Denormalize to physical space for relative error computation
        pred_physical = pred_flat * target_scale_per_node
        y_physical = y_flat * target_scale_per_node

    # Compute per-node errors (MSE/MAE in normalized space for loss, physical for rel_error)
    squared_errors = (pred_flat - y_flat).pow(2)
    absolute_errors = torch.abs(pred_flat - y_flat)
    absolute_errors_physical = torch.abs(pred_physical - y_physical)

    # Compute per-graph metrics using scatter_mean (same as physics residual)
    if batch_vec is not None:
        # Average errors per graph first
        mse_per_graph = scatter_mean(squared_errors, batch_vec, dim=0)
        mae_per_graph = scatter_mean(absolute_errors, batch_vec, dim=0)

        # Then average across graphs in batch
        loss_mse = mse_per_graph.mean()
        mae = mae_per_graph.mean()
        rmse = torch.sqrt(mse_per_graph).mean()  # RMSE per graph, then average (for logging)

        # Relative error in PHYSICAL space: per-graph MAE / per-graph mean target
        mae_physical_per_graph = scatter_mean(absolute_errors_physical, batch_vec, dim=0)
        y_abs_physical_per_graph = scatter_mean(torch.abs(y_physical), batch_vec, dim=0)
        epsilon = 1e-12
        rel_error_per_graph = mae_physical_per_graph / (y_abs_physical_per_graph + epsilon)
        rel_error = rel_error_per_graph.mean()
    else:
        # Single graph case: simple average across nodes
        loss_mse = squared_errors.mean()
        mae = absolute_errors.mean()
        rmse = torch.sqrt(squared_errors.mean())

        # Relative error in PHYSICAL space
        epsilon = 1e-12
        rel_error = absolute_errors_physical.mean() / (torch.abs(y_physical).mean() + epsilon)

    # Compute initial condition and boundary condition loss if needed
    loss_ic = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    loss_bc = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    if loss_type == LossType.PHYSICS_WITH_IC_BC:
        if is_initial_node is None:
            raise ValueError("is_initial_node must be provided for PHYSICS_WITH_IC_BC loss type")
        if data is None:
            raise ValueError("data must be provided for PHYSICS_WITH_IC_BC loss type to check timestep")
        loss_ic = compute_initial_condition_loss(pred, y, is_initial_node, data)
        if is_boundary_node is not None and is_boundary_node.any():
            loss_bc = compute_boundary_condition_loss(pred, y, is_boundary_node, data)

    # Compute total loss based on configuration
    base_loss, adaptive_weight, supervision_or_data_weight, phys_contrib_pct, sup_or_data_contrib_pct = compute_loss(
        loss_mse, loss_phys, loss_type, lambda_phys, loss_ic, lambda_ic, loss_bc, lambda_bc,
        use_adaptive_weighting=use_adaptive_physics_weighting,
        target_physics_ratio=target_physics_ratio
    )

    # Use base_loss as total_loss (no constraint penalties)
    total_loss = base_loss

    # Compute BC node statistics
    bc_node_count = 0
    bc_node_percentage = 0.0
    total_nodes = pred.shape[0]

    if is_boundary_node is not None:
        bc_node_count = int(is_boundary_node.sum().item())
        bc_node_percentage = (bc_node_count / total_nodes * 100) if total_nodes > 0 else 0.0

    return total_loss, loss_mse, mae, rmse, rel_error, loss_ic, loss_bc, adaptive_weight, supervision_or_data_weight, bc_node_count, bc_node_percentage, phys_contrib_pct, sup_or_data_contrib_pct


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        clip_grad_norm: float = 1.0,
        loss_type: LossType = LossType.COMBINED,
        lambda_phys: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_bc: float = 1.0,
        use_adaptive_physics_weighting: bool = True,
        target_physics_ratio: float = 0.5
    ) -> Tuple[float, float, float, float, float, float, float, float, Optional[PhysicsMetrics], float, float, float, float, float, float]:
    """Train model for one epoch.

    Args:
        model: The neural network model
        loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        clip_grad_norm: Maximum norm for gradient clipping
        loss_type: Type of loss to compute (data_only, physics, or combined)
        lambda_phys: Weight for physics term (only used with combined loss)
        lambda_ic: Weight for initial condition term (only used with physics loss)
        lambda_bc: Weight for boundary condition term (optional supervision)
        use_adaptive_physics_weighting: If True, balance physics loss adaptively
        target_physics_ratio: Target ratio of physics to supervision loss

    Returns:
        Tuple of (average_loss, average_mae, average_mse, average_rmse, average_rel_error,
                  average_physics, average_ic_loss, average_bc_loss, last_physics_metrics,
                  average_adaptive_weight, average_supervision_weight, average_bc_nodes, average_bc_pct,
                  average_phys_contrib_pct, average_sup_contrib_pct)

    Raises:
        RuntimeError: If no training samples are processed
    """
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "rmse": 0.0, "rel_error": 0.0, "phys": 0.0, "ic": 0.0, "bc": 0.0,
              "adaptive_weight": 0.0, "supervision_weight": 0.0, "bc_nodes": 0, "bc_pct": 0.0,
              "phys_contrib_pct": 0.0, "sup_contrib_pct": 0.0,
              "n_batches": 0, "last_phys_metrics": None}

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        # Prepare data for model: add time info
        data = utils.prepare_model_inputs(
            data,
            model,
            is_training=True
        )

        # Forward pass (no standardization, already in original space)
        pred = run_forward(model, data)

        # Compute physics residual
        if loss_type == LossType.DATA_ONLY:
            phys_res, phys_res_metrics = torch.tensor(0.0, device=pred.device), None
        else:
            phys_res, phys_res_metrics = compute_physics_residual_step(
                model=model,
                data=data,
                pred=pred,
                require_time_grad=True
            )

        # Compute loss and metrics (no physics scaling needed)
        loss, mse, mae, rmse, rel_error, ic_loss, bc_loss, adaptive_weight, supervision_weight, bc_nodes, bc_pct, phys_contrib_pct, sup_contrib_pct = compute_loss_and_metrics(
            pred,
            data.y,
            phys_res,
            loss_type,
            lambda_phys,
            getattr(data, 'is_initial_node', None),
            lambda_ic,
            getattr(data, 'is_boundary_node', None),
            lambda_bc,
            getattr(data, 'batch', None),
            data,
            use_adaptive_physics_weighting,
            target_physics_ratio
        )

        if loss_type == LossType.DATA_ONLY:
            mean_y = data.y.mean().item()
            mean_pred = pred.mean().item()

            log_path = "results/debug_output.txt"
            with open(log_path, "a") as f:
                if batch_idx == 0 or batch_idx == len(loader) - 1:
                    msg = (
                        f"\nEpoch {epoch:03d} | Batch {batch_idx} | Number of nodes: {data.y.shape[0]}\n"
                        f"S_ST true:\n{data.y.detach().cpu().numpy()[:10]}\n"
                        f"S_ST pred:\n{pred.detach().cpu().numpy()[:10]}\n"
                        f"Mean S_ST true: {mean_y:.6e}, Mean S_ST pred: {mean_pred:.6e}\n"
                    )
                    # print(msg)
                    f.write(msg + "\n")

        loss.backward()

        # Log gradient norms (first batch only to avoid clutter)
        if writer is not None and batch_idx == 0:
            logging.log_gradient_norms(model, writer, epoch, batch_idx, len(loader))

        # Gradient clipping and optimization step
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        # Accumulate metrics for epoch averaging
        with torch.no_grad():
            # Accumulate the physics residual (no normalization needed)
            accumulate_epoch_stats(totals, loss, mse, mae, rmse, rel_error, phys_res, ic_loss, bc_loss, phys_res_metrics,
                                   adaptive_weight, supervision_weight, bc_nodes, bc_pct, phys_contrib_pct, sup_contrib_pct)

            # Log physics metrics
            if writer is not None and batch_idx % 10 == 0:
                step = epoch * len(loader) + batch_idx
                writer.add_scalar('training/batch_physics', float(phys_res), step)

            # Log batch-level metrics (every 10 batches to avoid clutter)
            if writer is not None and batch_idx % 10 == 0:
                logging.log_batch_metrics(writer, epoch, batch_idx, len(loader),
                                          loss, mse, mae, rmse, rel_error, phys_res, ic_loss)

                # Log detailed physics components if available
                if phys_res_metrics is not None:
                    logging.log_physics_components(writer, epoch, batch_idx, len(loader),
                                                   phys_res_metrics, phase='train')

    if totals["n_batches"] == 0:
        raise RuntimeError("No training samples this epoch.")

    return (totals["loss"] / totals["n_batches"],
            totals["mae"] / totals["n_batches"],
            totals["mse"] / totals["n_batches"],
            totals["rmse"] / totals["n_batches"],
            totals["rel_error"] / totals["n_batches"],
            totals["phys"] / totals["n_batches"],
            totals["ic"] / totals["n_batches"],
            totals["bc"] / totals["n_batches"],
            totals["last_phys_metrics"],
            totals["adaptive_weight"] / totals["n_batches"],
            totals["supervision_weight"] / totals["n_batches"],
            totals["bc_nodes"] / totals["n_batches"],
            totals["bc_pct"] / totals["n_batches"],
            totals["phys_contrib_pct"] / totals["n_batches"],
            totals["sup_contrib_pct"] / totals["n_batches"])


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

    print(f"\nStarting training with loss type: {config.loss_type.value}")
    for epoch in range(1, config.epochs + 1):
        training_state.current_epoch = epoch

        # Use configured loss type directly
        current_loss_type = config.loss_type

        # Training
        tr_loss, tr_mae, tr_mse, tr_rmse, tr_rel_error, tr_phys, tr_ic, tr_bc, tr_phys_metrics, tr_adap_w, tr_sup_w, tr_bc_nodes, tr_bc_pct, tr_phys_pct, tr_sup_pct = train_epoch(
            model_setup.model,
            train_loader,
            optimizer,
            writer,
            epoch,
            clip_grad_norm=config.clip_grad_norm,
            loss_type=current_loss_type,
            lambda_phys=config.lambda_phys,
            lambda_ic=config.lambda_ic,
            lambda_bc=config.lambda_bc,
            use_adaptive_physics_weighting=config.use_adaptive_physics_weighting,
            target_physics_ratio=config.target_physics_ratio
        )

        # Validation
        val_loss, val_mse, val_mae, val_rmse, val_rel_error, val_phys, val_ic, val_bc, val_phys_metrics, val_adap_w, val_sup_w, val_bc_nodes, val_bc_pct, val_phys_pct, val_sup_pct = eval_model(
            model_setup.model,
            val_loader,
            writer,
            epoch,
            phase='val',
            loss_type=current_loss_type,
            lambda_phys=config.lambda_phys,
            lambda_ic=config.lambda_ic,
            lambda_bc=config.lambda_bc,
            use_adaptive_physics_weighting=config.use_adaptive_physics_weighting,
            target_physics_ratio=config.target_physics_ratio
        )

        # Learning rate scheduling (use combined validation loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create metrics objects
        tr_metrics = TrainingMetrics(tr_loss, tr_mse, tr_mae, tr_rmse, tr_rel_error, tr_phys, tr_ic, tr_bc, tr_phys_metrics)
        val_metrics = TrainingMetrics(val_loss, val_mse, val_mae, val_rmse, val_rel_error, val_phys, val_ic, val_bc, val_phys_metrics)

        # Log metrics to TensorBoard
        logging.log_epoch_metrics(writer, epoch, tr_metrics, val_metrics, current_lr)

        # Console logging
        base_log = (f"Epoch {epoch:03d} | "
                    f"train_tot={tr_loss:.3e} train_mse={tr_mse:.3e} train_phys={tr_phys:.3e} train_rmse={tr_rmse:.3e} train_relerr={tr_rel_error:.3e}")
        base_log += f" train_ic={tr_ic:.3e} train_bc={tr_bc:.3e}"

        # Add adaptive weighting info if using adaptive weighting
        if config.use_adaptive_physics_weighting and current_loss_type == LossType.PHYSICS_WITH_IC_BC:
            base_log += f" | train_phys_w={tr_adap_w:.3f} train_sup_w={tr_sup_w:.3f} [phys={tr_phys_pct:.1f}% sup={tr_sup_pct:.1f}%] train_bc_nodes={tr_bc_nodes:.1f}({tr_bc_pct:.1f}%)"

        base_log += (f" | val_tot={val_loss:.3e} val_mse={val_mse:.3e} val_phys={val_phys:.3e} val_rmse={val_rmse:.3e} val_relerr={val_rel_error:.3e}")

        base_log += f" val_ic={val_ic:.3e} val_bc={val_bc:.3e}"

        # Add adaptive weighting info for validation
        if config.use_adaptive_physics_weighting and current_loss_type == LossType.PHYSICS_WITH_IC_BC:
            base_log += f" | val_phys_w={val_adap_w:.3f} val_sup_w={val_sup_w:.3f} [phys={val_phys_pct:.1f}% sup={val_sup_pct:.1f}%] val_bc_nodes={val_bc_nodes:.1f}({val_bc_pct:.1f}%)"        # Add physics details to console output if available
        if tr_phys_metrics is not None:
            physics_log = f" | {tr_phys_metrics}"
            print(base_log + physics_log)
        else:
            print(base_log)

        # Model saving and early stopping (use combined validation loss)
        if training_state.update_best(val_loss, epoch):
            # Log best model achievement
            writer.add_scalar('best_model/epoch', epoch, epoch)
            writer.add_scalar('best_model/val_loss', val_loss, epoch)
            writer.add_scalar('best_model/val_mse', val_mse, epoch)

            # Save model checkpoint
            utils.save_checkpoint(
                model_setup, model_cfg, optimizer, scheduler,
                epoch, val_loss, val_mse, config.model_save_path
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

    # Final evaluation on test set
    test_loss, test_mse, test_mae, test_rmse, test_rel_error, test_phys, test_ic, test_bc, test_phys_metrics, test_adap_w, test_sup_w, test_bc_nodes, test_bc_pct, test_phys_pct, test_sup_pct = eval_model(
        model_setup.model,
        test_loader,
        writer,
        training_state.best_epoch,
        phase='test',
        loss_type=config.loss_type,
        lambda_phys=config.lambda_phys,
        lambda_ic=config.lambda_ic,
        lambda_bc=config.lambda_bc,
        use_adaptive_physics_weighting=config.use_adaptive_physics_weighting,
        target_physics_ratio=config.target_physics_ratio
    )

    # Log final test metrics
    writer.add_scalar('final/test_loss', test_loss, training_state.best_epoch)
    writer.add_scalar('final/test_mse', test_mse, training_state.best_epoch)
    writer.add_scalar('final/test_mae', test_mae, training_state.best_epoch)
    writer.add_scalar('final/test_rmse', test_rmse, training_state.best_epoch)
    writer.add_scalar('final/test_rel_error', test_rel_error, training_state.best_epoch)
    writer.add_scalar('final/test_physics', test_phys, training_state.best_epoch)
    writer.add_scalar('final/test_ic_loss', test_ic, training_state.best_epoch)
    writer.add_scalar('final/test_bc_loss', test_bc, training_state.best_epoch)

    # Log adaptive weighting metrics if available
    if config.use_adaptive_physics_weighting and config.loss_type == LossType.PHYSICS_WITH_IC_BC:
        writer.add_scalar('final/test_adaptive_weight', test_adap_w, training_state.best_epoch)
        writer.add_scalar('final/test_supervision_weight', test_sup_w, training_state.best_epoch)
        writer.add_scalar('final/test_bc_nodes', test_bc_nodes, training_state.best_epoch)
        writer.add_scalar('final/test_bc_pct', test_bc_pct, training_state.best_epoch)

    # Create final summary
    final_summary = (f"Final Results:\n"
                    f"Test Loss: {test_loss:.3e}\n"
                    f"Test MSE: {test_mse:.3e}\n"
                    f"Test MAE: {test_mae:.3e}\n"
                    f"Test RMSE: {test_rmse:.3e}\n"
                    f"Test Rel Error: {test_rel_error:.3e}\n"
                    f"Test Physics: {test_phys:.3e}\n"
                    f"Test IC Loss: {test_ic:.3e}\n"
                    f"Test BC Loss: {test_bc:.3e}\n"
                    f"Best epoch: {training_state.best_epoch}")

    # Add adaptive weighting info if available
    if config.use_adaptive_physics_weighting and config.loss_type == LossType.PHYSICS_WITH_IC_BC:
        final_summary += (f"\nAdaptive Weight: {test_adap_w:.3f}\n"
                         f"Supervision Weight: {test_sup_w:.3f}\n"
                         f"Physics Contribution: {test_phys_pct:.1f}%\n"
                         f"Supervision Contribution: {test_sup_pct:.1f}%\n"
                         f"BC Nodes: {test_bc_nodes:.1f} ({test_bc_pct:.2f}%)")

    if test_phys_metrics is not None:
        final_summary += f"\nTest Physics Details: {test_phys_metrics}"

    writer.add_text('final/results', final_summary)

    # Console output with adaptive weighting info
    console_msg = (f"\nFinal test metrics - Loss: {test_loss:.3e}, MSE: {test_mse:.3e}, RMSE: {test_rmse:.3e}, "
                   f"MAE: {test_mae:.3e}, RelErr: {test_rel_error:.3e}, Physics: {test_phys:.3e}, "
                   f"IC Loss: {test_ic:.3e}, BC Loss: {test_bc:.3e}")

    if config.use_adaptive_physics_weighting and config.loss_type == LossType.PHYSICS_WITH_IC_BC:
        console_msg += (f"\n  Adaptive Weight: {test_adap_w:.3f}, Supervision Weight: {test_sup_w:.3f}\n"
                       f"  Physics Contribution: {test_phys_pct:.1f}%, Supervision Contribution: {test_sup_pct:.1f}%\n"
                       f"  BC Nodes: {test_bc_nodes:.1f} ({test_bc_pct:.2f}%)")

    print(console_msg)
    if test_phys_metrics is not None:
        print(f"Test Physics Details: {test_phys_metrics}")


def eval_model(
        model: nn.Module,
        loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        phase: str = 'val',
        loss_type: LossType = LossType.COMBINED,
        lambda_phys: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_bc: float = 1.0,
        use_adaptive_physics_weighting: bool = True,
        target_physics_ratio: float = 0.5
    ) -> Tuple[float, float, float, float, float, float, float, float, Optional[PhysicsMetrics], float, float, float, float, float, float]:
    """Evaluate model on a dataset.

    Args:
        model: The neural network model
        loader: DataLoader containing evaluation data
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        phase: Phase name ('val' or 'test')
        loss_type: Type of loss to compute (data, physics, or combined)
        lambda_phys: Weight for physics term (only used with combined loss)
        lambda_ic: Weight for initial condition term (only used with physics loss)
        lambda_bc: Weight for boundary condition term (used with physics loss)
        use_adaptive_physics_weighting: If True, balance physics loss adaptively
        target_physics_ratio: Target ratio of physics to supervision loss

    Returns:
        Tuple of (average_loss, average_mse, average_mae, average_rmse, average_rel_error,
                  average_physics, average_ic_loss, average_bc_loss, last_physics_metrics,
                  average_adaptive_weight, average_supervision_weight, average_bc_nodes, average_bc_pct,
                  average_phys_contrib_pct, average_sup_contrib_pct)
    """
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "rmse": 0.0, "rel_error": 0.0, "phys": 0.0, "ic": 0.0, "bc": 0.0,
              "adaptive_weight": 0.0, "supervision_weight": 0.0, "bc_nodes": 0, "bc_pct": 0.0,
              "phys_contrib_pct": 0.0, "sup_contrib_pct": 0.0,
              "n_batches": 0, "last_phys_metrics": None}

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # Prepare data for model: add time info
            data = utils.prepare_model_inputs(
                data,
                model,
                is_training=False
            )

            # Forward pass with prepared data (no standardization)
            pred = run_forward(model, data)

            # Compute physics residual (eval path re-enables grad internally)
            if loss_type == LossType.DATA_ONLY:
                phys_res, phys_res_metrics = torch.tensor(0.0, device=pred.device), None
            else:
                phys_res, phys_res_metrics = compute_physics_residual_step(
                    model=model,
                    data=data,
                    pred=None,
                    require_time_grad=False
                )

            # Compute loss and metrics (no physics scaling needed)
            loss, mse, mae, rmse, rel_error, ic_loss, bc_loss, adaptive_weight, supervision_weight, bc_nodes, bc_pct, phys_contrib_pct, sup_contrib_pct = compute_loss_and_metrics(
                pred,
                data.y,
                phys_res,
                loss_type,
                lambda_phys,
                getattr(data, 'is_initial_node', None),
                lambda_ic,
                getattr(data, 'is_boundary_node', None),
                lambda_bc,
                getattr(data, 'batch', None),
                data,
                use_adaptive_physics_weighting,
                target_physics_ratio
            )

            # Accumulate metrics
            accumulate_epoch_stats(totals, loss, mse, mae, rmse, rel_error, phys_res, ic_loss, bc_loss, phys_res_metrics,
                                   adaptive_weight, supervision_weight, bc_nodes, bc_pct, phys_contrib_pct, sup_contrib_pct)

            # Log distribution of predictions and targets (first batch only, every 5 epochs)
            if writer is not None and batch_idx == 0 and epoch % 5 == 0:
                # pred is already in original space (no standardization)
                logging.log_evaluation_histograms(writer, phase, epoch, pred, data.y)

                # Log individual loss components for debugging
                writer.add_scalar(f'{phase}/mse', float(mse), epoch)
                writer.add_scalar(f'{phase}/physics', float(phys_res), epoch)
                writer.add_scalar(f'{phase}/ic_loss', float(ic_loss), epoch)
                writer.add_scalar(f'{phase}/loss', float(loss), epoch)

                # Log detailed physics components if available
                if phys_res_metrics is not None:
                    logging.log_physics_components(writer, epoch, 0, 1,
                                                   phys_res_metrics, phase=phase)

    # Clear GPU memory after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute averages per batch
    denom = max(1, totals["n_batches"])
    return (totals["loss"] / denom,
            totals["mse"] / denom,
            totals["mae"] / denom,
            totals["rmse"] / denom,
            totals["rel_error"] / denom,
            totals["phys"] / denom,
            totals["ic"] / denom,
            totals["bc"] / denom,
            totals["last_phys_metrics"],
            totals["adaptive_weight"] / denom,
            totals["supervision_weight"] / denom,
            totals["bc_nodes"] / denom,
            totals["bc_pct"] / denom,
            totals["phys_contrib_pct"] / denom,
            totals["sup_contrib_pct"] / denom)
