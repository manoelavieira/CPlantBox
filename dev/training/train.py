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

def compute_loss(loss_mse: torch.Tensor, loss_phys: torch.Tensor, loss_type: LossType, lambda_phys: float = 1.0) -> torch.Tensor:
    """Compute loss based on the specified loss type configuration.

    Args:
        loss_mse: Mean squared error term
        loss_phys: Physics residual term
        loss_type: Type of loss to compute
        lambda_phys: Physics term weight (only used for COMBINED loss)

    Returns:
        Computed loss tensor
    """
    if loss_type == LossType.DATA_ONLY:
        return loss_mse
    elif loss_type == LossType.PHYSICS_ONLY:
        return loss_phys
    elif loss_type == LossType.COMBINED:
        return loss_mse + lambda_phys * loss_phys
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_physics_residual(
    pred_norm: torch.Tensor,
    data,
    model: nn.Module,
    node_feat_orig: torch.Tensor
):
    """Compute physics residual with proper feature handling.

    Args:
        pred_norm: Model predictions in standardized space
        data: Batch data with time information
        model: Neural network model with scalers
        node_feat_orig: Original (unstandardized) node features

    Returns:
        tuple: (physics_residual, PhysicsMetrics)
    """
    try:
        # Transform features and predictions back to original space for physics
        pred_orig = model.target_scaler.inv_transform(pred_norm)
        node_feat_std_backup = data.node_feat.clone()
        data.node_feat = node_feat_orig

        # Compute physics residual with optional detailed components
        phys_res, phys_res_dict = physics_residual(pred_orig, data)
        phys_res = phys_res if phys_res.dim() == 0 else phys_res.mean()

        # Convert tensor components to PhysicsMetrics
        phys_res_metrics = PhysicsMetrics(
            J_ax=float(phys_res_dict['J_ax']),
            F_in=float(phys_res_dict['F_in']),
            F_out=float(phys_res_dict['F_out']),
            ds_dt=float(phys_res_dict['ds_dt']),
            dS_dt_from_flux=float(phys_res_dict['dS_dt_from_flux']),
            dS_dt_from_physics=float(phys_res_dict['dS_dt_from_physics'])
        )

        # Restore standardized features
        data.node_feat = node_feat_std_backup

        return phys_res, phys_res_metrics

    except Exception as e:
        raise RuntimeError(f"Physics computation failed: {e}")


def compute_loss_and_metrics(
    pred_norm: torch.Tensor,
    y: torch.Tensor,
    loss_phys: torch.Tensor,
    model: nn.Module,
    loss_type: LossType,
    lambda_phys: float = 1.0,
):
    """Compute loss and metrics in a unified way.

    Args:
        pred_norm: Model predictions in standardized space
        targets: Target values in original space
        loss_phys: Precomputed physics residual
        model: Neural network model with scalers
        loss_type: Type of loss to compute
        lambda_phys: Physics term weight

    Returns:
        Tuple of (total_loss, mse, mae)
    """
    # Transform targets to standardized space for MSE computation
    y_std = model.target_scaler.transform(y)

    # MSE in standardized space
    loss_mse = F.mse_loss(pred_norm, y_std, reduction='mean')

    # MAE in original space for interpretability
    pred_orig = model.target_scaler.inv_transform(pred_norm)
    mae = (pred_orig - y).abs().mean()

    # Compute total loss based on configuration
    total_loss = compute_loss(loss_mse, loss_phys, loss_type, lambda_phys)

    return total_loss, loss_mse, mae


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        clip_grad_norm: float = 1.0,
        loss_type: LossType = LossType.COMBINED,
        lambda_phys: float = 1.0,
        time_jitter_std : float = 0.01
    ) -> Tuple[float, float, float, float, Optional[PhysicsMetrics]]:
    """Train model for one epoch.

    Args:
        model: The neural network model
        loader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        clip_grad_norm: Maximum norm for gradient clipping
        loss_type: Type of loss to compute (data_only, physics_only, or combined)
        lambda_phys: Weight for physics term (only used with combined loss)
        time_jitter_std: Standard deviation for time jitter

    Returns:
        Tuple of (average_loss, average_mae, average_mse, average_physics, last_physics_metrics)

    Raises:
        RuntimeError: If no training samples are processed
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_phys = 0.0
    n_batches = 0
    last_phys_metrics = None  # Store physics metrics from last batch for logging

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        # Prepare data for model: add time info, jitter (if training), and normalize features
        node_feat_orig, data_norm = utils.prepare_model_inputs(
            data,
            model,
            is_training=True,
            time_jitter_std=time_jitter_std
        )

        # Forward pass with standardized data
        pred_norm = model(data_norm)

        # Compute physics residual safely
        phys_res, phys_res_metrics = compute_physics_residual(
            pred_norm,
            data_norm,
            model,
            node_feat_orig
        )

        # Compute loss and metrics
        loss, mse, mae = compute_loss_and_metrics(
            pred_norm,
            data.y,
            phys_res,
            model,
            loss_type,
            lambda_phys,
        )

        # Backward pass
        loss.backward()

        # Log gradient norms (first batch only to avoid clutter)
        if writer is not None and batch_idx == 0:
            logging.log_gradient_norms(model, writer, epoch, batch_idx, len(loader))

        # Gradient clipping and optimization step
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        # Accumulate metrics for epoch averaging
        with torch.no_grad():
            total_loss += float(loss)
            total_mse += float(mse)
            total_mae += float(mae)
            total_phys += float(phys_res)
            n_batches += 1
            last_phys_metrics = phys_res_metrics  # keep last batch's physics for representative logging

            # Log batch-level metrics (every 10 batches to avoid clutter)
            if writer is not None and batch_idx % 10 == 0:
                logging.log_batch_metrics(writer, epoch, batch_idx, len(loader),
                                          loss, mse, mae, phys_res)

                # Log detailed physics components if available
                if phys_res_metrics is not None:
                    logging.log_physics_components(writer, epoch, batch_idx, len(loader),
                                                   phys_res_metrics, phase='train')

    if n_batches == 0:
        raise RuntimeError("No training samples this epoch.")

    # Return epoch averages
    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    avg_mse = total_mse / n_batches
    avg_phys = total_phys / n_batches

    return (avg_loss, avg_mae, avg_mse, avg_phys, last_phys_metrics)


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
        tr_loss, tr_mae, tr_mse, tr_physics, tr_physics_details = train_epoch(
            model_setup.model,
            train_loader,
            optimizer,
            writer,
            epoch,
            clip_grad_norm=config.clip_grad_norm,
            loss_type=config.loss_type,
            lambda_phys=config.lambda_phys,
            time_jitter_std=config.time_jitter_std
        )

        # Validation
        val_loss, val_mse, val_mae, val_physics, val_physics_details = eval_model(
            model_setup.model,
            val_loader,
            writer,
            epoch,
            phase='val',
            loss_type=config.loss_type,
            lambda_phys=config.lambda_phys
        )

        # Learning rate scheduling (use combined validation loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create metrics objects
        train_metrics = TrainingMetrics(tr_loss, tr_mse, tr_mae, tr_physics, tr_physics_details)
        val_metrics = TrainingMetrics(val_loss, val_mse, val_mae, val_physics, val_physics_details)

        # Log metrics to TensorBoard
        logging.log_epoch_metrics(writer, epoch, train_metrics, val_metrics, current_lr)

        # Console logging
        base_log = (f"Epoch {epoch:03d} | "
                   f"train_tot={tr_loss:.3e} train_mse={tr_mse:.4f} train_phys={tr_physics:.3e} | "
                   f"val_tot={val_loss:.3e} val_mse={val_mse:.4f} val_phys={val_physics:.3e}")

        # Add physics details to console output if available
        if tr_physics_details is not None:
            physics_log = f" | {tr_physics_details}"
            print(base_log + physics_log)
        else:
            print(base_log)

        # Model saving and early stopping (use combined validation loss)
        if training_state.update_best(val_loss, epoch):
            # Log best model achievement
            writer.add_scalar('Best_Model/Epoch', epoch, epoch)
            writer.add_scalar('Best_Model/Val_Loss', val_loss, epoch)
            writer.add_scalar('Best_Model/Val_MSE', val_mse, epoch)

            # Save model checkpoint
            utils.save_checkpoint(
                model_setup, model_cfg, optimizer, scheduler,
                epoch, val_loss, val_mse, config.model_save_path
            )
        else:
            if training_state.should_stop(config.patience):
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best validation loss: {training_state.best_val_loss:.4f} "
                      f"at epoch {training_state.best_epoch}")

                # Log early stopping
                writer.add_text('Training/Early_Stopping',
                                f"Stopped at epoch {epoch}, best at {training_state.best_epoch}")
                break

    print("\nTraining completed!")

    # Log final training summary
    writer.add_text('Training/Summary',
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
    test_loss, test_mse, test_mae, test_physics, test_physics_details = eval_model(
        model_setup.model,
        test_loader,
        writer,
        training_state.best_epoch,
        phase='test',
        loss_type=config.loss_type,
        lambda_phys=config.lambda_phys
    )

    # Log final test metrics
    writer.add_scalar('Final/Test_Loss', test_loss, training_state.best_epoch)
    writer.add_scalar('Final/Test_MSE', test_mse, training_state.best_epoch)
    writer.add_scalar('Final/Test_MAE', test_mae, training_state.best_epoch)
    writer.add_scalar('Final/Test_Physics', test_physics, training_state.best_epoch)

    # Create final summary
    final_summary = (f"Final Results:\n"
                    f"Test Loss: {test_loss:.4f}\n"
                    f"Test MSE: {test_mse:.4f}\n"
                    f"Test MAE: {test_mae:.4f}\n"
                    f"Test Physics: {test_physics:.4f}\n"
                    f"Best epoch: {training_state.best_epoch}")

    if test_physics_details is not None:
        final_summary += f"\nTest Physics Details: {test_physics_details}"

    writer.add_text('Final/Results', final_summary)

    print(f"\nFinal test metrics - Loss: {test_loss:.3e}, MSE: {test_mse:.4f}, Physics: {test_physics:.3e}")
    if test_physics_details is not None:
        print(f"Physics details: {test_physics_details}")


def eval_model(
        model: nn.Module,
        loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        phase: str = 'val',
        loss_type: LossType = LossType.COMBINED,
        lambda_phys: float = 1.0
    ) -> Tuple[float, float, float, float, Optional[PhysicsMetrics]]:
    """Evaluate model on a dataset.

    Args:
        model: The neural network model
        loader: DataLoader containing evaluation data
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        phase: Phase name ('val' or 'test')
        loss_type: Type of loss to compute (data_only, physics_only, or combined)
        lambda_phys: Weight for physics term (only used with combined loss)

    Returns:
        Tuple of (average_loss, average_mse, average_mae, average_physics, last_physics_metrics)
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_phys = 0.0
    n_batches = 0
    last_phys_metrics = None  # Store physics metrics from last batch for logging

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # Prepare data for forward pass (no training mode, no jitter)
            node_feat_orig, data_norm = utils.prepare_model_inputs(
                data,
                model,
                is_training=False,
                time_jitter_std=0.0
            )

            # Forward pass with prepared data
            pred_norm = model(data_norm)

            # Compute physics residual safely with gradient context for evaluation
            phys_res = torch.tensor(0.0, device=pred_norm.device)
            phys_res_metrics = None
            with torch.enable_grad():
                time_norm_grad = data_norm.time_norm.clone().requires_grad_(True)
                data_norm_with_grad = data_norm
                data_norm_with_grad.time_norm = time_norm_grad
                pred_for_phys = model(data_norm_with_grad)
                pred_orig_for_physics = model.target_scaler.inv_transform(pred_for_phys)
                data_norm_with_grad.node_feat = node_feat_orig
                phys_res, phys_res_metrics = physics_residual(pred_orig_for_physics, data_norm_with_grad)
                phys_res = phys_res if phys_res.dim() == 0 else phys_res.mean()
                phys_res = phys_res.detach()

                # Convert to PhysicsMetrics for logging
                phys_res_metrics = PhysicsMetrics(
                    J_ax=float(phys_res_metrics['J_ax']),
                    F_in=float(phys_res_metrics['F_in']),
                    F_out=float(phys_res_metrics['F_out']),
                    ds_dt=float(phys_res_metrics['ds_dt']),
                    dS_dt_from_flux=float(phys_res_metrics['dS_dt_from_flux']),
                    dS_dt_from_physics=float(phys_res_metrics['dS_dt_from_physics'])
                )

            # Compute loss and metrics using helper function
            loss, mse, mae = compute_loss_and_metrics(
                pred_norm,
                data.y,
                phys_res,
                model,
                loss_type,
                lambda_phys,
            )

            # Accumulate metrics
            total_loss += float(loss)
            total_mse += float(mse)
            total_mae += float(mae)
            total_phys += float(phys_res)
            n_batches += 1
            last_phys_metrics = phys_res_metrics  # Keep last batch's physics for representative logging

            # Log distribution of predictions and targets (first batch only, every 5 epochs)
            if writer is not None and batch_idx == 0 and epoch % 5 == 0:
                pred_original = model.target_scaler.inv_transform(pred_norm)
                logging.log_evaluation_histograms(writer, phase, epoch, pred_original, data.y)

                # Log individual loss components for debugging
                writer.add_scalar(f'{phase}/MSE', float(mse), epoch)
                writer.add_scalar(f'{phase}/Physics', float(phys_res), epoch)
                writer.add_scalar(f'{phase}/Loss', float(loss), epoch)

                # Log detailed physics components if available
                if phys_res_metrics is not None:
                    logging.log_physics_components(writer, epoch, 0, 1,
                                                 phys_res_metrics, phase=phase)

    # Compute averages per batch
    denom = n_batches if n_batches > 0 else 1
    avg_loss = total_loss / denom
    avg_mse = total_mse / denom
    avg_mae = total_mae / denom
    avg_phys = total_phys / denom

    # Clear GPU memory after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, avg_mse, avg_mae, avg_phys, last_phys_metrics