import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from typing import Optional
from pathlib import Path
from datetime import datetime

from model.config import ModelConfig
from .config import TrainingConfig, TrainingMetrics


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
        writer.add_text('Model/Architecture',
                       f"Total: {total_params:,}, Trainable: {trainable_params:,}")
        writer.add_scalar('Model/Total_Parameters', total_params, 0)
        writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)


def print_experiment_config(config: TrainingConfig):
    """Print experiment configuration."""
    print("\nExperiment Configuration:")
    for field_name, field_value in config.__dict__.items():
        print(f"{field_name}: {field_value}")


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
    writer.add_scalar('Loss/Train_Total', train_metrics.loss, epoch)
    writer.add_scalar('Loss/Train_MSE', train_metrics.mse, epoch)
    writer.add_scalar('Loss/Train_Physics', train_metrics.physics, epoch)

    writer.add_scalar('Metrics/Val_Total', val_metrics.loss, epoch)
    writer.add_scalar('Metrics/Val_MSE', val_metrics.mse, epoch)
    writer.add_scalar('Metrics/Val_Physics', val_metrics.physics, epoch)

    writer.add_scalar('Learning_Rate', current_lr, epoch)

    writer.add_scalars('Loss_Comparison', {
        'Train_Total': train_metrics.loss,
        'Val_Total': val_metrics.loss
    }, epoch)

    writer.add_scalars('Loss_Components', {
        'MSE': train_metrics.mse,
        'Physics': train_metrics.physics,
        'Total': train_metrics.loss
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
            writer.add_scalar(f'Gradients/{name}', param_grad_norm,
                            epoch * loader_len + batch_idx)

    total_grad_norm = total_grad_norm ** 0.5
    writer.add_scalar('Gradients/total_norm', total_grad_norm,
                    epoch * loader_len + batch_idx)


def log_batch_metrics(
    writer: SummaryWriter,
    epoch: int,
    batch_idx: int,
    loader_len: int,
    loss: torch.Tensor,
    mse: torch.Tensor,
    mae: torch.Tensor,
    physics: torch.Tensor
) -> None:
    """Log batch-level metrics to TensorBoard."""
    step = epoch * loader_len + batch_idx
    writer.add_scalar('Training/Batch_Loss', float(loss), step)
    writer.add_scalar('Training/Batch_MSE', float(mse), step)
    writer.add_scalar('Training/Batch_MAE', float(mae), step)
    writer.add_scalar('Training/Batch_Physics', float(physics), step)


def log_evaluation_histograms(
    writer: SummaryWriter,
    phase: str,
    epoch: int,
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> None:
    """Log distribution histograms for evaluation metrics."""
    writer.add_histogram(f'{phase}/Predictions', predictions.cpu(), epoch)
    writer.add_histogram(f'{phase}/Targets', targets.cpu(), epoch)
    writer.add_histogram(f'{phase}/Residuals', (predictions - targets).cpu(), epoch)


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