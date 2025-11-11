import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    last_phys: Optional[PhysicsMetrics]
):
    totals["loss"] += float(loss)
    totals["mse"] += float(mse)
    totals["mae"] += float(mae)
    totals["rmse"] += float(rmse)
    totals["rel_error"] += float(rel_error)
    totals["phys"] += float(phys)
    totals["ic"] += float(ic)
    totals["n_batches"] += 1
    totals["last_phys_metrics"] = last_phys


def run_forward(model: nn.Module, data) -> torch.Tensor:
    """Forward pass on inputs (tiny wrapper for symmetry)."""
    return model(data)


def _to_physics_metrics(physics_res) -> Optional[PhysicsMetrics]:
    """Convert dict of physics components to PhysicsMetrics."""
    if physics_res is not None:
        return PhysicsMetrics(
            J_ax=float(physics_res['J_ax']),
            F_in=float(physics_res['F_in']),
            F_out=float(physics_res['F_out']),
            ds_dt=float(physics_res['ds_dt']),
            dS_dt_from_flux=float(physics_res['dS_dt_from_flux']),
            dS_dt_from_physics=float(physics_res['dS_dt_from_physics'])
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
) -> torch.Tensor:
    """Compute initial condition supervision loss.

    Compares predicted vs true sucrose values only for nodes that were
    present at the initial timestep (t=0).

    Args:
        pred: Model predictions [N, 1]
        y: Target values [N, 1]
        is_initial_node: Boolean mask indicating initial nodes [N]

    Returns:
        torch.Tensor: Initial condition loss (MSE over initial nodes only)
    """
    if not is_initial_node.any():
        # No initial nodes in this batch, return zero loss
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Extract predictions and targets for initial nodes only
    pred_initial = pred[is_initial_node]
    y_initial = y[is_initial_node]

    # Compute MSE loss over initial nodes
    ic_loss = F.mse_loss(pred_initial, y_initial, reduction='mean')

    return ic_loss


def compute_loss(loss_mse: torch.Tensor, loss_phys: torch.Tensor, loss_type: LossType,
                 lambda_phys: float = 1.0, loss_ic: torch.Tensor = None, lambda_ic: float = 1.0) -> torch.Tensor:
    """Compute loss based on the specified loss type configuration.

    Args:
        loss_mse: Mean squared error term
        loss_phys: Physics residual term
        loss_type: Type of loss to compute
        lambda_phys: Physics term weight (only used for COMBINED loss)
        loss_ic: Initial condition loss term (only used for PHYSICS_WITH_IC loss)
        lambda_ic: Initial condition term weight (only used for PHYSICS_WITH_IC loss)

    Returns:
        Computed loss tensor
    """
    if loss_type == LossType.DATA_ONLY:
        return loss_mse
    elif loss_type == LossType.PHYSICS_ONLY:
        return loss_phys
    elif loss_type == LossType.PHYSICS_WITH_IC:
        if loss_ic is None:
            raise ValueError("loss_ic must be provided for PHYSICS_WITH_IC loss type")
        return loss_phys + lambda_ic * loss_ic
    elif loss_type == LossType.COMBINED:
        return loss_mse + lambda_phys * loss_phys
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
    batch_vec: torch.Tensor = None,
):
    """Compute loss and metrics with per-graph aggregation (consistent with physics residual).

    Computes metrics per-graph first, then averages across graphs in the batch.
    This ensures equal weight per graph regardless of graph size, matching the
    physics residual computation strategy.

    Args:
        pred: Model predictions (already in original space) [N, 1]
        y: Target values (in original space) [N, 1]
        loss_phys: Precomputed physics residual (already properly averaged per-graph)
        loss_type: Type of loss to compute
        lambda_phys: Physics term weight
        is_initial_node: Boolean mask for initial nodes (required for PHYSICS_WITH_IC)
        lambda_ic: Initial condition term weight
        batch_vec: Batch assignment for each node [N] (None for single graph)

    Returns:
        Tuple of (total_loss, mse, mae, rmse, rel_error, ic_loss)
    """
    from torch_scatter import scatter_mean

    # Squeeze predictions and targets to [N]
    pred_flat = pred.squeeze(-1)
    y_flat = y.squeeze(-1)

    # Compute per-node errors
    squared_errors = (pred_flat - y_flat).pow(2)
    absolute_errors = torch.abs(pred_flat - y_flat)

    # Compute per-graph metrics using scatter_mean (same as physics residual)
    if batch_vec is not None:
        # Average errors per graph first
        mse_per_graph = scatter_mean(squared_errors, batch_vec, dim=0)
        mae_per_graph = scatter_mean(absolute_errors, batch_vec, dim=0)

        # Then average across graphs in batch
        loss_mse = mse_per_graph.mean()
        mae = mae_per_graph.mean()
        rmse = torch.sqrt(mse_per_graph).mean()  # RMSE per graph, then average

        # Relative error: per-graph MAE / per-graph mean target
        y_abs_per_graph = scatter_mean(torch.abs(y_flat), batch_vec, dim=0)
        epsilon = 1e-12
        rel_error_per_graph = mae_per_graph / (y_abs_per_graph + epsilon)
        rel_error = rel_error_per_graph.mean()
    else:
        # Single graph case: simple average across nodes
        loss_mse = squared_errors.mean()
        mae = absolute_errors.mean()
        rmse = torch.sqrt(loss_mse)

        epsilon = 1e-12
        rel_error = mae / (torch.abs(y_flat).mean() + epsilon)

    # Compute initial condition loss if needed
    loss_ic = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    if loss_type == LossType.PHYSICS_WITH_IC:
        if is_initial_node is None:
            raise ValueError("is_initial_node must be provided for PHYSICS_WITH_IC loss type")
        loss_ic = compute_initial_condition_loss(pred, y, is_initial_node)

    # Compute total loss based on configuration
    total_loss = compute_loss(loss_mse, loss_phys, loss_type, lambda_phys, loss_ic, lambda_ic)

    return total_loss, loss_mse, mae, rmse, rel_error, loss_ic


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        clip_grad_norm: float = 1.0,
        loss_type: LossType = LossType.COMBINED,
        lambda_phys: float = 1.0,
        lambda_ic: float = 1.0
    ) -> Tuple[float, float, float, float, float, float, float, Optional[PhysicsMetrics]]:
    """Train model for one epoch.

    Args:
        model: The neural network model
        loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        clip_grad_norm: Maximum norm for gradient clipping
        loss_type: Type of loss to compute (data_only, physics_only, physics_ic, or combined)
        lambda_phys: Weight for physics term (only used with combined loss)
        lambda_ic: Weight for initial condition term (only used with physics_ic loss)

    Returns:
        Tuple of (average_loss, average_mae, average_mse, average_rmse, average_rel_error,
                  average_physics, average_ic_loss, last_physics_metrics)

    Raises:
        RuntimeError: If no training samples are processed
    """
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "rmse": 0.0, "rel_error": 0.0, "phys": 0.0, "ic": 0.0, "n_batches": 0, "last_phys_metrics": None}

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
        loss, mse, mae, rmse, rel_error, ic_loss = compute_loss_and_metrics(
            pred,
            data.y,
            phys_res,
            loss_type,
            lambda_phys,
            getattr(data, 'is_initial_node', None),
            lambda_ic,
            getattr(data, 'batch', None)
        )

        if loss_type == LossType.DATA_ONLY:
            mean_y = data.y.mean().item()
            mean_pred = pred.mean().item()

            log_path = "results/debug_output.txt"
            with open(log_path, "a") as f:
                if batch_idx == 0 or batch_idx == len(loader) - 1:
                    msg = (
                        f"\nEpoch {epoch:03d} | Batch {batch_idx} | Number of nodes: {data.y.shape[0]}\n"
                        f"Q_ST true:\n{data.y.detach().cpu().numpy()[:10]}\n"
                        f"Q_ST pred:\n{pred.detach().cpu().numpy()[:10]}\n"
                        f"Mean Q_ST true: {mean_y:.6e}, Mean Q_ST pred: {mean_pred:.6e}"
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
            accumulate_epoch_stats(totals, loss, mse, mae, rmse, rel_error, phys_res, ic_loss, phys_res_metrics)

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
            totals["last_phys_metrics"])


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

        # Training
        tr_loss, tr_mae, tr_mse, tr_rmse, tr_rel_error, tr_phys, tr_ic, tr_phys_metrics = train_epoch(
            model_setup.model,
            train_loader,
            optimizer,
            writer,
            epoch,
            clip_grad_norm=config.clip_grad_norm,
            loss_type=config.loss_type,
            lambda_phys=config.lambda_phys,
            lambda_ic=config.lambda_ic
        )

        # Validation
        val_loss, val_mse, val_mae, val_rmse, val_rel_error, val_phys, val_ic, val_phys_metrics = eval_model(
            model_setup.model,
            val_loader,
            writer,
            epoch,
            phase='val',
            loss_type=config.loss_type,
            lambda_phys=config.lambda_phys,
            lambda_ic=config.lambda_ic
        )

        # Learning rate scheduling (use combined validation loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create metrics objects
        tr_metrics = TrainingMetrics(tr_loss, tr_mse, tr_mae, tr_rmse, tr_rel_error, tr_phys, tr_ic, tr_phys_metrics)
        val_metrics = TrainingMetrics(val_loss, val_mse, val_mae, val_rmse, val_rel_error, val_phys, val_ic, val_phys_metrics)

        # Log metrics to TensorBoard
        logging.log_epoch_metrics(writer, epoch, tr_metrics, val_metrics, current_lr)

        # Console logging
        base_log = (f"Epoch {epoch:03d} | "
                    f"train_tot={tr_loss:.3e} train_mse={tr_mse:.3e} train_phys={tr_phys:.3e} train_rmse={tr_rmse:.3e} train_relerr={tr_rel_error:.3e}")

        # Add IC loss if it's meaningful (> 0)
        if tr_ic > 0:
            base_log += f" train_ic={tr_ic:.3e}"

        base_log += (f" | val_tot={val_loss:.3e} val_mse={val_mse:.3e} val_phys={val_phys:.3e} val_rmse={val_rmse:.3e} val_relerr={val_rel_error:.3e}")

        if val_ic > 0:
            base_log += f" val_ic={val_ic:.3e}"

        # Add physics details to console output if available
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
    test_loss, test_mse, test_mae, test_rmse, test_rel_error, test_phys, test_ic, test_phys_metrics = eval_model(
        model_setup.model,
        test_loader,
        writer,
        training_state.best_epoch,
        phase='test',
        loss_type=config.loss_type,
        lambda_phys=config.lambda_phys,
        lambda_ic=config.lambda_ic
    )

    # Log final test metrics
    writer.add_scalar('final/test_loss', test_loss, training_state.best_epoch)
    writer.add_scalar('final/test_mse', test_mse, training_state.best_epoch)
    writer.add_scalar('final/test_mae', test_mae, training_state.best_epoch)
    writer.add_scalar('final/test_rmse', test_rmse, training_state.best_epoch)
    writer.add_scalar('final/test_rel_error', test_rel_error, training_state.best_epoch)
    writer.add_scalar('final/test_physics', test_phys, training_state.best_epoch)
    writer.add_scalar('final/test_ic_loss', test_ic, training_state.best_epoch)

    # Create final summary
    final_summary = (f"Final Results:\n"
                    f"Test Loss: {test_loss:.3e}\n"
                    f"Test MSE: {test_mse:.3e}\n"
                    f"Test MAE: {test_mae:.3e}\n"
                    f"Test RMSE: {test_rmse:.3e}\n"
                    f"Test Rel Error: {test_rel_error:.3e}\n"
                    f"Test Physics: {test_phys:.3e}\n"
                    f"Test IC Loss: {test_ic:.3e}\n"
                    f"Best epoch: {training_state.best_epoch}")

    if test_phys_metrics is not None:
        final_summary += f"\nTest Physics Details: {test_phys_metrics}"

    writer.add_text('final/results', final_summary)

    print(f"\nFinal test metrics - Loss: {test_loss:.3e}, MSE: {test_mse:.3e}, RMSE: {test_rmse:.3e}, MAE: {test_mae:.3e}, RelErr: {test_rel_error:.3e}, Physics: {test_phys:.3e}, IC Loss: {test_ic:.3e}")
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
        lambda_ic: float = 1.0
    ) -> Tuple[float, float, float, float, float, float, float, Optional[PhysicsMetrics]]:
    """Evaluate model on a dataset.

    Args:
        model: The neural network model
        loader: DataLoader containing evaluation data
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        phase: Phase name ('val' or 'test')
        loss_type: Type of loss to compute (data, physics, physics_ic, or combined)
        lambda_phys: Weight for physics term (only used with combined loss)
        lambda_ic: Weight for initial condition term (only used with physics_ic loss)

    Returns:
        Tuple of (average_loss, average_mse, average_mae, average_rmse, average_rel_error,
                  average_physics, average_ic_loss, last_physics_metrics)
    """
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "rmse": 0.0, "rel_error": 0.0, "phys": 0.0, "ic": 0.0, "n_batches": 0, "last_phys_metrics": None}

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
            loss, mse, mae, rmse, rel_error, ic_loss = compute_loss_and_metrics(
                pred,
                data.y,
                phys_res,
                loss_type,
                lambda_phys,
                getattr(data, 'is_initial_node', None),
                lambda_ic,
                getattr(data, 'batch', None)
            )

            # Accumulate metrics
            accumulate_epoch_stats(totals, loss, mse, mae, rmse, rel_error, phys_res, ic_loss, phys_res_metrics)

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
            totals["last_phys_metrics"])