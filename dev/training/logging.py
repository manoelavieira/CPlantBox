import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from typing import Optional
from pathlib import Path
from datetime import datetime
import csv

from model.config import ModelConfig
from .config import TrainingConfig, TrainingMetrics, PhysicsMetrics, PhysicsErrorMetrics


def create_tensorboard_writer(config: TrainingConfig) -> SummaryWriter:
    """Create TensorBoard writer with organized logging directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = timestamp

    log_dir = Path(config.tensorboard_log_dir) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={log_dir.parent}")

    return writer


def print_model_summary(model: nn.Module, writer: Optional[SummaryWriter] = None):
    """Print model architecture summary and log to TensorBoard."""
    print("\nModel Architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("\nLayer Overview:")
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")

    # Log model parameters to TensorBoard
    if writer is not None:
        writer.add_text('model/architecture',
                       f"total: {total_params:,}, trainable: {trainable_params:,}")
        writer.add_scalar('model/total_parameters', total_params, 0)
        writer.add_scalar('model/trainable_parameters', trainable_params, 0)


def print_experiment_config(config: TrainingConfig):
    """Print experiment configuration."""
    print("\nExperiment Configuration:")
    for field_name, field_value in config.__dict__.items():
        print(f"{field_name}: {field_value}")


def log_physics_components(
    writer: SummaryWriter,
    epoch: int,
    batch_idx: int,
    loader_len: int,
    physics_metrics: 'PhysicsMetrics',
    phase: str = 'train'
) -> None:
    """Log detailed physics components to TensorBoard.

    Args:
        writer: TensorBoard writer
        epoch: Current epoch number
        batch_idx: Current batch index
        loader_len: Total number of batches in loader
        physics_metrics: Detailed physics metrics
        phase: Phase name ('train', 'val', 'test')
    """
    step = epoch * loader_len + batch_idx
    prefix = f'physics_{phase}' if phase != 'train' else 'physics'

    writer.add_scalar(f'{prefix}/J_ax', physics_metrics.J_ax, step)
    writer.add_scalar(f'{prefix}/F_in', physics_metrics.F_in, step)
    writer.add_scalar(f'{prefix}/F_out', physics_metrics.F_out, step)
    writer.add_scalar(f'{prefix}/dS_dt_from_flux', physics_metrics.dS_dt_from_flux, step)
    writer.add_scalar(f'{prefix}/dS_dt_tot', physics_metrics.dS_dt_tot, step)

    # Also log as a scalar group for easy comparison
    writer.add_scalars(f'{prefix}_components', {
        'J_ax': physics_metrics.J_ax,
        'F_in': physics_metrics.F_in,
        'F_out': physics_metrics.F_out,
        'flux_div': physics_metrics.dS_dt_from_flux,
        'dS_dt_tot': physics_metrics.dS_dt_tot
    }, step)


def log_epoch_metrics(
    writer: SummaryWriter,
    epoch: int,
    train_metrics: TrainingMetrics,
    val_metrics: TrainingMetrics,
    current_lr: float
) -> None:
    """Log epoch metrics to TensorBoard.

    Args:
        writer: TensorBoard writer
        epoch: Current epoch number
        train_metrics: Training metrics for this epoch
        val_metrics: Validation metrics for this epoch
        current_lr: Current learning rate
    """
    # Log individual metrics
    writer.add_scalar('loss/train_total', train_metrics.loss, epoch)
    writer.add_scalar('loss/train_mse', train_metrics.mse, epoch)
    writer.add_scalar('loss/train_physics', train_metrics.physics, epoch)
    writer.add_scalar('loss/train_ic', train_metrics.ic_loss, epoch)
    writer.add_scalar('loss/train_bc', train_metrics.bc_loss, epoch)

    writer.add_scalar('metrics/train_mae', train_metrics.mae, epoch)
    writer.add_scalar('metrics/train_rmse', train_metrics.rmse, epoch)
    writer.add_scalar('metrics/train_rel_error', train_metrics.rel_error, epoch)

    # Log boundary condition metrics
    if train_metrics.bc_nodes > 0:
        writer.add_scalar('metrics/train_bc_nodes', train_metrics.bc_nodes, epoch)
        writer.add_scalar('metrics/train_bc_pct', train_metrics.bc_pct, epoch)

    writer.add_scalar('metrics/val_total', val_metrics.loss, epoch)
    writer.add_scalar('metrics/val_mse', val_metrics.mse, epoch)
    writer.add_scalar('metrics/val_mae', val_metrics.mae, epoch)
    writer.add_scalar('metrics/val_rmse', val_metrics.rmse, epoch)
    writer.add_scalar('metrics/val_rel_error', val_metrics.rel_error, epoch)
    writer.add_scalar('metrics/val_physics', val_metrics.physics, epoch)
    writer.add_scalar('metrics/val_ic', val_metrics.ic_loss, epoch)
    writer.add_scalar('metrics/val_bc', val_metrics.bc_loss, epoch)

    # Log validation boundary condition metrics
    if val_metrics.bc_nodes > 0:
        writer.add_scalar('metrics/val_bc_nodes', val_metrics.bc_nodes, epoch)
        writer.add_scalar('metrics/val_bc_pct', val_metrics.bc_pct, epoch)

    writer.add_scalar('learning_rate', current_lr, epoch)

    writer.add_scalars('loss/comparison', {
        'train_total': train_metrics.loss,
        'val_total': val_metrics.loss
    }, epoch)

    writer.add_scalars('metrics/mse_comparison', {
        'train': train_metrics.mse,
        'val': val_metrics.mse
    }, epoch)

    writer.add_scalars('metrics/rmse_comparison', {
        'train': train_metrics.rmse,
        'val': val_metrics.rmse
    }, epoch)

    writer.add_scalars('metrics/mae_comparison', {
        'train': train_metrics.mae,
        'val': val_metrics.mae
    }, epoch)

    writer.add_scalars('metrics/rel_error_comparison', {
        'train': train_metrics.rel_error,
        'val': val_metrics.rel_error
    }, epoch)

    writer.add_scalars('loss/components', {
        'mse': train_metrics.mse,
        'physics': train_metrics.physics,
        'total': train_metrics.loss
    }, epoch)

    # Log detailed physics components if available
    if train_metrics.physics_details is not None:
        log_physics_components(writer, epoch, 0, 1, train_metrics.physics_details, phase='train_epoch')

    if val_metrics.physics_details is not None:
        log_physics_components(writer, epoch, 0, 1, val_metrics.physics_details, phase='val_epoch')

    # Log physics error metrics if available
    if train_metrics.physics_errors is not None:
        log_physics_error_metrics(writer, epoch, train_metrics.physics_errors, phase='train')

    if val_metrics.physics_errors is not None:
        log_physics_error_metrics(writer, epoch, val_metrics.physics_errors, phase='val')


def log_physics_error_metrics(
    writer: SummaryWriter,
    epoch: int,
    physics_errors: 'PhysicsErrorMetrics',
    phase: str = 'train'
) -> None:
    """Log all physics error metrics to TensorBoard.

    Args:
        writer: TensorBoard writer
        epoch: Current epoch number
        physics_errors: Physics error metrics (comprehensive)
        phase: Phase name ('train', 'val', 'test')
    """
    prefix = f'physics_errors_{phase}'

    # ========== SUCROSE CONTENT (S_ST) METRICS ==========
    writer.add_scalar(f'{prefix}/S_ST_rmse', physics_errors.S_ST_rmse, epoch)
    writer.add_scalar(f'{prefix}/S_ST_mae', physics_errors.S_ST_mae, epoch)
    writer.add_scalar(f'{prefix}/S_ST_nmae', physics_errors.S_ST_nmae, epoch)
    writer.add_scalar(f'{prefix}/S_ST_correlation', physics_errors.S_ST_correlation, epoch)

    # ========== FLUX (J_ax) METRICS ==========
    writer.add_scalar(f'{prefix}/J_ax_mse', physics_errors.J_ax_mse, epoch)
    writer.add_scalar(f'{prefix}/J_ax_rmse', physics_errors.J_ax_rmse, epoch)
    writer.add_scalar(f'{prefix}/J_ax_mae', physics_errors.J_ax_mae, epoch)
    writer.add_scalar(f'{prefix}/J_ax_nmae', physics_errors.J_ax_nmae, epoch)
    writer.add_scalar(f'{prefix}/J_ax_sign_accuracy', physics_errors.J_ax_sign_accuracy, epoch)
    writer.add_scalar(f'{prefix}/J_ax_reversal_rate', physics_errors.J_ax_reversal_rate, epoch)
    writer.add_scalar(f'{prefix}/J_ax_antisym_error', physics_errors.J_ax_antisym_error, epoch)
    writer.add_scalar(f'{prefix}/J_ax_magnitude_ratio', physics_errors.J_ax_magnitude_ratio, epoch)
    writer.add_scalar(f'{prefix}/J_ax_correlation', physics_errors.J_ax_correlation, epoch)

    # ========== DIVERGENCE (divJ) METRICS ==========
    writer.add_scalar(f'{prefix}/divJ_mse', physics_errors.divJ_mse, epoch)
    writer.add_scalar(f'{prefix}/divJ_rmse', physics_errors.divJ_rmse, epoch)
    writer.add_scalar(f'{prefix}/divJ_mae', physics_errors.divJ_mae, epoch)
    writer.add_scalar(f'{prefix}/divJ_nmae', physics_errors.divJ_nmae, epoch)
    writer.add_scalar(f'{prefix}/divJ_std_true', physics_errors.divJ_std_true, epoch)
    writer.add_scalar(f'{prefix}/divJ_std_pred', physics_errors.divJ_std_pred, epoch)
    writer.add_scalar(f'{prefix}/divJ_std_ratio', physics_errors.divJ_std_ratio, epoch)
    writer.add_scalar(f'{prefix}/divJ_overlap', physics_errors.divJ_overlap, epoch)
    writer.add_scalar(f'{prefix}/divJ_correlation', physics_errors.divJ_correlation, epoch)

    # ========== TOTAL RESIDUAL (dS_dt_tot) METRICS ==========
    writer.add_scalar(f'{prefix}/dS_dt_tot_mse', physics_errors.dS_dt_tot_mse, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_rmse', physics_errors.dS_dt_tot_rmse, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_mae', physics_errors.dS_dt_tot_mae, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_nmae', physics_errors.dS_dt_tot_nmae, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_mean_true', physics_errors.dS_dt_tot_mean_true, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_mean_pred', physics_errors.dS_dt_tot_mean_pred, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_std_true', physics_errors.dS_dt_tot_std_true, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_std_pred', physics_errors.dS_dt_tot_std_pred, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_skew_true', physics_errors.dS_dt_tot_skew_true, epoch)
    writer.add_scalar(f'{prefix}/dS_dt_tot_skew_pred', physics_errors.dS_dt_tot_skew_pred, epoch)

    # ========== GROUPED METRICS FOR COMPARISON ==========
    # MSE comparison
    writer.add_scalars(f'{prefix}_mse_all', {
        'S_ST': physics_errors.S_ST_rmse ** 2,  # Approximate from RMSE
        'J_ax': physics_errors.J_ax_mse,
        'divJ': physics_errors.divJ_mse,
        'dS_dt_tot': physics_errors.dS_dt_tot_mse
    }, epoch)

    # RMSE comparison
    writer.add_scalars(f'{prefix}_rmse_all', {
        'S_ST': physics_errors.S_ST_rmse,
        'J_ax': physics_errors.J_ax_rmse,
        'divJ': physics_errors.divJ_rmse,
        'dS_dt_tot': physics_errors.dS_dt_tot_rmse
    }, epoch)

    # MAE comparison
    writer.add_scalars(f'{prefix}_mae_all', {
        'S_ST': physics_errors.S_ST_mae,
        'J_ax': physics_errors.J_ax_mae,
        'divJ': physics_errors.divJ_mae,
        'dS_dt_tot': physics_errors.dS_dt_tot_mae
    }, epoch)

    # NMAE comparison (normalized errors)
    writer.add_scalars(f'{prefix}_nmae_all', {
        'S_ST': physics_errors.S_ST_nmae,
        'J_ax': physics_errors.J_ax_nmae,
        'divJ': physics_errors.divJ_nmae,
        'dS_dt_tot': physics_errors.dS_dt_tot_nmae
    }, epoch)

    # Correlation comparison
    writer.add_scalars(f'{prefix}_correlation_all', {
        'S_ST': physics_errors.S_ST_correlation,
        'J_ax': physics_errors.J_ax_correlation,
        'divJ': physics_errors.divJ_correlation,
    }, epoch)

    # Standard deviation comparison
    writer.add_scalars(f'{prefix}_std_comparison', {
        'divJ_true': physics_errors.divJ_std_true,
        'divJ_pred': physics_errors.divJ_std_pred,
        'dS_dt_tot_true': physics_errors.dS_dt_tot_std_true,
        'dS_dt_tot_pred': physics_errors.dS_dt_tot_std_pred,
    }, epoch)


def log_gradient_norms(
    model: nn.Module,
    writer: SummaryWriter,
    epoch: int,
    batch_idx: int,
    loader_len: int
) -> None:
    """Log gradient norms to TensorBoard."""
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += param_grad_norm ** 2
            writer.add_scalar(f'gradients/{name}', param_grad_norm,
                            epoch * loader_len + batch_idx)

    total_grad_norm = total_grad_norm ** 0.5
    writer.add_scalar('gradients/total_norm', total_grad_norm,
                    epoch * loader_len + batch_idx)


def log_batch_metrics(
    writer: SummaryWriter,
    epoch: int,
    batch_idx: int,
    loader_len: int,
    loss: torch.Tensor,
    mse: torch.Tensor,
    mae: torch.Tensor,
    rmse: torch.Tensor,
    rel_error: torch.Tensor,
    physics: torch.Tensor,
    ic_loss: torch.Tensor = None,
    bc_loss: torch.Tensor = None,
    bc_nodes: int = 0,
    bc_pct: float = 0.0
) -> None:
    """Log batch-level metrics to TensorBoard."""
    step = epoch * loader_len + batch_idx
    writer.add_scalar('training/batch_loss', float(loss), step)
    writer.add_scalar('training/batch_mse', float(mse), step)
    writer.add_scalar('training/batch_mae', float(mae), step)
    writer.add_scalar('training/batch_rmse', float(rmse), step)
    writer.add_scalar('training/batch_rel_error', float(rel_error), step)
    writer.add_scalar('training/batch_physics', float(physics), step)
    if ic_loss is not None:
        writer.add_scalar('training/batch_ic_loss', float(ic_loss), step)
    if bc_loss is not None:
        writer.add_scalar('training/batch_bc_loss', float(bc_loss), step)
    if bc_nodes > 0:
        writer.add_scalar('training/batch_bc_nodes', bc_nodes, step)
        writer.add_scalar('training/batch_bc_pct', bc_pct, step)


def log_evaluation_histograms(
    writer: SummaryWriter,
    phase: str,
    epoch: int,
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> None:
    """Log distribution histograms for evaluation metrics."""
    writer.add_histogram(f'{phase}/predictions', predictions.cpu(), epoch)
    writer.add_histogram(f'{phase}/targets', targets.cpu(), epoch)
    writer.add_histogram(f'{phase}/residuals', (predictions - targets).cpu(), epoch)


def log_hyperparameters(writer: SummaryWriter, config: TrainingConfig, model_cfg: ModelConfig):
    """Log hyperparameters to TensorBoard."""
    hparams = {
        # Training hyperparameters
        'learning_rate': config.lr,
        'batch_size': config.batch_size,
        'weight_decay': config.weight_decay,
        'epochs': config.epochs,
        'patience': config.patience,
        'seed': config.seed,
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'lambda_data': config.lambda_data,
        'loss_type': config.loss_type.value,

        # Model architecture
        'hidden_size': model_cfg.hidden_size,
        'num_layers': model_cfg.num_layers,
        'edge_feat_dim': model_cfg.edge_feat_dim,
        'node_feat_dim': model_cfg.node_feat_dim,
        'dropout': model_cfg.dropout,
    }

    if config.data_path:
        hparams['data_path'] = config.data_path

    metrics = {}

    writer.add_hparams(hparams, metrics)

def save_metrics_to_csv(
    config: TrainingConfig,
    epoch: int,
    fold_idx: Optional[int],
    train_metrics: TrainingMetrics,
    val_metrics: TrainingMetrics,
    current_lr: float
) -> None:
    """Save epoch metrics to CSV file.

    Args:
        config: Training configuration
        epoch: Current epoch number
        fold_idx: Current fold index (None for traditional mode)
        train_metrics: Training metrics for this epoch
        val_metrics: Validation metrics for this epoch
        current_lr: Current learning rate
    """
    if not config.enable_metrics_logging:
        return

    metrics_path = Path(config.metrics_save_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to determine if we need to write headers
    file_exists = metrics_path.exists()

    # Open file in append mode
    with open(metrics_path, 'a', newline='') as f:
        fieldnames = [
            'fold', 'epoch', 'learning_rate',
            'train_loss', 'train_mse', 'train_mae', 'train_rmse', 'train_rel_error',
            'train_phys', 'train_ic', 'train_bc',
            'val_loss', 'val_mse', 'val_mae', 'val_rmse', 'val_rel_error',
            'val_phys', 'val_ic', 'val_bc'
        ]

        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            csv_writer.writeheader()

        # Write metrics row
        csv_writer.writerow({
            'fold': fold_idx if fold_idx is not None else 'N/A',
            'epoch': epoch,
            'learning_rate': current_lr,
            'train_loss': train_metrics.loss,
            'train_mse': train_metrics.mse,
            'train_mae': train_metrics.mae,
            'train_rmse': train_metrics.rmse,
            'train_rel_error': train_metrics.rel_error,
            'train_phys': train_metrics.physics,
            'train_ic': train_metrics.ic_loss,
            'train_bc': train_metrics.bc_loss,
            'val_loss': val_metrics.loss,
            'val_mse': val_metrics.mse,
            'val_mae': val_metrics.mae,
            'val_rmse': val_metrics.rmse,
            'val_rel_error': val_metrics.rel_error,
            'val_phys': val_metrics.physics,
            'val_ic': val_metrics.ic_loss,
            'val_bc': val_metrics.bc_loss,
        })
