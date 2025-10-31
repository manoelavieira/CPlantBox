import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from typing import Optional
from pathlib import Path
from datetime import datetime

from model.config import ModelConfig
from .config import TrainingConfig, TrainingMetrics, PhysicsMetrics


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
    writer.add_scalar(f'{prefix}/ds_dt', physics_metrics.ds_dt, step)
    writer.add_scalar(f'{prefix}/dS_dt_from_flux', physics_metrics.dS_dt_from_flux, step)
    writer.add_scalar(f'{prefix}/dS_dt_from_physics', physics_metrics.dS_dt_from_physics, step)

    # Also log as a scalar group for easy comparison
    writer.add_scalars(f'{prefix}_components', {
        'J_ax': physics_metrics.J_ax,
        'F_in': physics_metrics.F_in,
        'F_out': physics_metrics.F_out,
        'ds_dt': physics_metrics.ds_dt,
        'flux_div': physics_metrics.dS_dt_from_flux
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

    writer.add_scalar('metrics/val_total', val_metrics.loss, epoch)
    writer.add_scalar('metrics/val_mse', val_metrics.mse, epoch)
    writer.add_scalar('metrics/val_physics', val_metrics.physics, epoch)
    writer.add_scalar('metrics/val_ic', val_metrics.ic_loss, epoch)

    writer.add_scalar('learning_rate', current_lr, epoch)

    writer.add_scalars('loss/comparison', {
        'train_total': train_metrics.loss,
        'val_total': val_metrics.loss
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
    physics: torch.Tensor,
    ic_loss: torch.Tensor = None
) -> None:
    """Log batch-level metrics to TensorBoard."""
    step = epoch * loader_len + batch_idx
    writer.add_scalar('training/batch_loss', float(loss), step)
    writer.add_scalar('training/batch_mse', float(mse), step)
    writer.add_scalar('training/batch_mae', float(mae), step)
    writer.add_scalar('training/batch_physics', float(physics), step)
    if ic_loss is not None:
        writer.add_scalar('training/batch_ic_loss', float(ic_loss), step)


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
        'lambda_phys': config.lambda_phys,
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