import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from typing import Tuple, Optional

from models.gnn import ModelConfig, physics_residual
from config import TrainingConfig, TrainingState, TrainingMetrics, ModelSetup, LossType

import training.utils as utils
import training.logging as logging

def compute_loss(mse: torch.Tensor, physics: torch.Tensor, loss_type: LossType, lambda_phys: float = 1.0) -> torch.Tensor:
    """Compute loss based on the specified loss type configuration.

    Args:
        mse: Mean squared error term
        physics: Physics residual term
        loss_type: Type of loss to compute
        lambda_phys: Physics term weight (only used for COMBINED loss)

    Returns:
        Computed loss tensor
    """
    if loss_type == LossType.DATA_ONLY:
        return mse
    elif loss_type == LossType.PHYSICS_ONLY:
        return physics
    elif loss_type == LossType.COMBINED:
        return mse + lambda_phys * physics
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_physics_residual(
    pred_standardized: torch.Tensor,
    data,
    model: nn.Module,
    original_features: torch.Tensor
) -> torch.Tensor:
    """Safely compute physics residual with proper feature handling.

    Args:
        pred_standardized: Model predictions in standardized space
        data: Batch data with time information
        model: Neural network model with scalers
        original_features: Original (unstandardized) node features

    Returns:
        Physics residual tensor (scalar or mean reduced)
    """
    try:
        # Transform predictions back to original space for physics
        pred_orig = model.target_scaler.inv_transform(pred_standardized)

        # Temporarily restore original features for physics computation
        data_feat_backup = data.node_feat.clone()
        data.node_feat = original_features

        # Compute physics residual
        phys = physics_residual(pred_orig, data)
        phys_tensor = phys if phys.dim() == 0 else phys.mean()

        # Restore standardized features
        data.node_feat = data_feat_backup

        return phys_tensor

    except Exception as e:
        # Fallback to zero if physics computation fails
        return torch.tensor(0.0, device=pred_standardized.device)


def compute_loss_and_metrics(
    pred_standardized: torch.Tensor,
    targets: torch.Tensor,
    physics_residual_tensor: torch.Tensor,
    model: nn.Module,
    loss_type: LossType,
    lambda_phys: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss and metrics in a unified way.

    Args:
        pred_standardized: Model predictions in standardized space
        targets: Target values in original space
        physics_residual_tensor: Precomputed physics residual
        model: Neural network model with scalers
        loss_type: Type of loss to compute
        lambda_phys: Physics term weight

    Returns:
        Tuple of (total_loss, mse, mae, physics_residual)
    """
    # Transform targets to standardized space for MSE computation
    targets_standardized = model.target_scaler.transform(targets)

    # MSE in standardized space
    mse = F.mse_loss(pred_standardized, targets_standardized, reduction='mean')

    # MAE in original space for interpretability
    pred_original = model.target_scaler.inv_transform(pred_standardized)
    mae = (pred_original - targets).abs().mean()

    # Compute total loss based on configuration
    total_loss = compute_loss(mse, physics_residual_tensor, loss_type, lambda_phys)

    return total_loss, mse, mae, physics_residual_tensor


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
    ) -> Tuple[float, float, float, float]:
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
        Tuple of (average_loss, average_mae, average_mse, average_physics)

    Raises:
        RuntimeError: If no training samples are processed
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_physics = 0.0
    n_batches = 0

    for batch_idx, data in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        # Prepare data for forward pass (handles time, jitter, standardization)
        original_features, prepared_data = utils.prepare_model_inputs(
            data, model, is_training=True, time_jitter_std=time_jitter_std
        )

        # Forward pass with prepared data
        pred_standardized = model(prepared_data)

        # Compute physics residual safely
        physics_residual_tensor = compute_physics_residual(
            pred_standardized, prepared_data, model, original_features
        )

        # Compute loss and metrics
        loss, mse, mae, physics_tensor = compute_loss_and_metrics(
            pred_standardized, data.y, physics_residual_tensor,
            model, loss_type, lambda_phys
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
            total_physics += float(physics_tensor)
            n_batches += 1

            # Log batch-level metrics (every 10 batches to avoid clutter)
            if writer is not None and batch_idx % 10 == 0:
                logging.log_batch_metrics(writer, epoch, batch_idx, len(loader),
                                loss, mse, mae, physics_tensor)

    if n_batches == 0:
        raise RuntimeError("No training samples this epoch.")

    # Return epoch averages
    return (total_loss / n_batches, total_mae / n_batches,
            total_mse / n_batches, total_physics / n_batches)


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
        tr_loss, tr_mae, tr_mse, tr_physics = train_epoch(
            model_setup.model, train_loader, optimizer, writer, epoch,
            clip_grad_norm=config.clip_grad_norm,
            loss_type=config.loss_type,
            lambda_phys=config.lambda_phys,
            time_jitter_std=config.time_jitter_std)

        # Validation
        val_loss, val_mse, val_mae, val_physics = eval_model(
            model_setup.model, val_loader, writer, epoch, phase='val',
            loss_type=config.loss_type, lambda_phys=config.lambda_phys)

        # Learning rate scheduling (use combined validation loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Create metrics objects
        train_metrics = TrainingMetrics(tr_loss, tr_mse, tr_mae, tr_physics)
        val_metrics = TrainingMetrics(val_loss, val_mse, val_mae, val_physics)

        # Log metrics to TensorBoard
        logging.log_epoch_metrics(writer, epoch, train_metrics, val_metrics, current_lr)

        # Console logging
        print(f"Epoch {epoch:03d} | "
              f"train_loss={tr_loss:.4f} train_MSE={tr_mse:.4f} train_physics={tr_physics:.4f} | "
              f"val_loss={val_loss:.4f} val_MSE={val_mse:.4f} val_physics={val_physics:.4f} | "
              f"lr={current_lr:.2e}")

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
    test_loss, test_mse, test_mae, test_physics = eval_model(
        model_setup.model, test_loader, writer,
        training_state.best_epoch, phase='test',
        loss_type=config.loss_type, lambda_phys=config.lambda_phys
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

    writer.add_text('Final/Results', final_summary)

    print(f"\nFinal test metrics - Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, Physics: {test_physics:.4f}")


def eval_model(
        model: nn.Module,
        loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        phase: str = 'val',
        loss_type: LossType = LossType.COMBINED,
        lambda_phys: float = 1.0
    ) -> Tuple[float, float, float, float]:
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
        Tuple of (average_loss, average_mse, average_mae, average_physics)
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_physics = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # Prepare data for forward pass (no training mode, no jitter)
            original_features, prepared_data = utils.prepare_model_inputs(
                data, model, is_training=False, time_jitter_std=0.0
            )

            # Forward pass with prepared data
            pred_standardized = model(prepared_data)

            # Compute physics residual safely with gradient context for evaluation
            physics_residual_tensor = torch.tensor(0.0, device=pred_standardized.device)
            with torch.enable_grad():
                time_node_grad = prepared_data.time_node.clone().requires_grad_(True)
                data_with_grad = prepared_data  # Use prepared data directly
                data_with_grad.time_node = time_node_grad
                pred_for_physics = model(data_with_grad)
                pred_orig_for_physics = model.target_scaler.inv_transform(pred_for_physics)
                data_with_grad.node_feat = original_features
                phys_val = physics_residual(pred_orig_for_physics, data_with_grad)
                physics_residual_tensor = phys_val if phys_val.dim() == 0 else phys_val.mean()
                physics_residual_tensor = physics_residual_tensor.detach()

            # Compute loss and metrics using helper function
            loss, mse, mae, physics_tensor = compute_loss_and_metrics(
                pred_standardized, data.y, physics_residual_tensor,
                model, loss_type, lambda_phys
            )

            # Accumulate metrics
            total_loss += float(loss)
            total_mse += float(mse)
            total_mae += float(mae)
            total_physics += float(physics_tensor)
            n_batches += 1

            # Log distribution of predictions and targets (first batch only, every 5 epochs)
            if writer is not None and batch_idx == 0 and epoch % 5 == 0:
                pred_original = model.target_scaler.inv_transform(pred_standardized)
                logging.log_evaluation_histograms(writer, phase, epoch, pred_original, data.y)

                # Log individual loss components for debugging
                writer.add_scalar(f'{phase}/MSE', float(mse), epoch)
                writer.add_scalar(f'{phase}/Physics', float(physics_tensor), epoch)
                writer.add_scalar(f'{phase}/Loss', float(loss), epoch)

    # Compute averages per batch
    denom = n_batches if n_batches > 0 else 1
    avg_loss = total_loss / denom
    avg_mse = total_mse / denom
    avg_mae = total_mae / denom
    avg_physics = total_physics / denom

    # Clear GPU memory after evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_loss, avg_mse, avg_mae, avg_physics