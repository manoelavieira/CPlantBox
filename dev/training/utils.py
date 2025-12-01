import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from typing import Tuple

from data.dataset_loader import load_phloem_data
from model.config import ModelConfig
from .config import TrainingConfig, ModelSetup
from model.utils import Standardizer


def to_float(x):
    if torch.is_tensor(x):
        return x.item()
    return float(x)


def validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    """Validate that dataset split ratios are valid.

    Args:
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation

    Raises:
        ValueError: If ratios are invalid
    """
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError(
            f"Sum of train_ratio ({train_ratio}) and val_ratio ({val_ratio}) "
            f"must be less than 1 to leave data for testing"
        )


def get_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, validation, and test dataloaders.

    Args:
        config: Training configuration containing dataset parameters

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        ValueError: If dataset parameters are invalid
    """
    # Validate split ratios
    validate_split_ratios(config.train_ratio, config.val_ratio)

    # Load simulation data
    train_loader, val_loader, test_loader = load_phloem_data(
        h5_path=config.data_path,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        random_seed=config.seed
    )

    return train_loader, val_loader, test_loader


def get_dataloaders_with_train_graphs(config: TrainingConfig):
    """Get train graphs list and validation/test loaders for curriculum learning.

    Returns train graphs as a list instead of DataLoader to enable curriculum filtering.

    Args:
        config: Training configuration containing dataset parameters

    Returns:
        Tuple of (train_graphs_list, val_loader, test_loader, collate_fn)
    """
    from data.dataset_loader import load_phloem_data, load_graphs_from_file, train_test_split, collate_graphs
    from pathlib import Path

    # Validate split ratios
    validate_split_ratios(config.train_ratio, config.val_ratio)

    # Load all graphs from file
    path = Path(config.data_path)
    if path.is_file():
        print(f"Loading data from single file: {config.data_path}")
        graphs = load_graphs_from_file(str(config.data_path), None)
    else:
        raise RuntimeError(f"Curriculum learning currently only supports single file input")

    # Split into train/val/test
    import torch
    from torch.utils.data import DataLoader

    n_samples = len(graphs)
    n_train = int(config.train_ratio * n_samples)
    n_val = int(config.val_ratio * n_samples)

    # Sort graphs by their time attribute for time-series splitting
    times = [graph.time.item() for graph in graphs]
    sorted_indices = sorted(range(n_samples), key=lambda i: times[i])

    # Create contiguous chunks: train (earliest), val (middle), test (latest)
    train_idx = sorted_indices[:n_train]
    val_idx = sorted_indices[n_train:n_train + n_val]
    test_idx = sorted_indices[n_train + n_val:]

    print(f"\nTime-series split for curriculum learning:")
    print(f"  Train: {len(train_idx)} graphs, time range [{times[train_idx[0]]:.2f}, {times[train_idx[-1]]:.2f}]")
    print(f"  Val:   {len(val_idx)} graphs, time range [{times[val_idx[0]]:.2f}, {times[val_idx[-1]]:.2f}]")
    print(f"  Test:  {len(test_idx)} graphs, time range [{times[test_idx[0]]:.2f}, {times[test_idx[-1]]:.2f}]")

    # Get train graphs as list (for curriculum filtering)
    train_graphs = [graphs[i] for i in train_idx]

    # Create val/test loaders normally
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    val_loader = DataLoader(val_graphs, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(test_graphs, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate_graphs)

    return train_graphs, val_loader, test_loader, collate_graphs


def save_checkpoint(
    model_setup: ModelSetup,
    model_cfg: ModelConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_loss: float,
    val_mse: float,
    val_phys: float,
    val_rel_error: float,
    filepath: str
) -> None:
    """Save model checkpoint with all necessary state.

    Args:
        model_setup: Model setup containing model and scalers
        model_cfg: Model configuration
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch number
        val_loss: Validation loss
        val_mse: Validation MSE
        filepath: Path to save checkpoint
    """
    # Prepare scaler states
    feature_scaler_state = {
        'mean': model_setup.feature_scaler.mean,
        'std': model_setup.feature_scaler.std,
        'device': str(model_setup.feature_scaler.device)
    }
    target_scaler_state = {
        'mean': model_setup.target_scaler.mean,
        'std': model_setup.target_scaler.std,
        'device': str(model_setup.target_scaler.device)
    }
    time_scaler_state = {
        'mean': model_setup.time_scaler.mean,
        'std': model_setup.time_scaler.std,
        'device': str(model_setup.time_scaler.device)
    }
    edge_scaler_state = {
        'mean': model_setup.edge_scaler.mean,
        'std': model_setup.edge_scaler.std,
        'device': str(model_setup.edge_scaler.device)
    }

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'cfg': model_cfg.__dict__,
        'state_dict': model_setup.model.state_dict(),
        'device': model_setup.device.type,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_mse': val_mse,
        'val_phys': val_phys,
        'val_rel_error': val_rel_error,
        'feature_scaler': feature_scaler_state,
        'target_scaler': target_scaler_state,
        'time_scaler': time_scaler_state,
        'edge_scaler': edge_scaler_state,
    }, filepath)


def load_best_model(
    model_setup: ModelSetup,
    filepath: str,
    device: torch.device
) -> bool:
    """Load the best model from checkpoint.

    Args:
        model_setup: Model setup to update with loaded state
        filepath: Path to checkpoint file
        device: Device to load model on

    Returns:
        bool: True if loading was successful, False otherwise
    """
    try:
        best_checkpoint = torch.load(filepath, map_location=device)

        # Load model state
        model_setup.model.load_state_dict(best_checkpoint['state_dict'])

        # Reconstruct scalers from saved state
        model_setup.feature_scaler = Standardizer()
        model_setup.feature_scaler.mean = best_checkpoint['feature_scaler']['mean']
        model_setup.feature_scaler.std = best_checkpoint['feature_scaler']['std']
        model_setup.feature_scaler.device = device

        model_setup.target_scaler = Standardizer()
        model_setup.target_scaler.mean = best_checkpoint['target_scaler']['mean']
        model_setup.target_scaler.std = best_checkpoint['target_scaler']['std']
        model_setup.target_scaler.device = device

        model_setup.time_scaler = Standardizer()
        model_setup.time_scaler.mean = best_checkpoint['time_scaler']['mean']
        model_setup.time_scaler.std = best_checkpoint['time_scaler']['std']
        model_setup.time_scaler.device = device

        model_setup.edge_scaler = Standardizer()
        model_setup.edge_scaler.mean = best_checkpoint['edge_scaler']['mean']
        model_setup.edge_scaler.std = best_checkpoint['edge_scaler']['std']
        model_setup.edge_scaler.device = device

        # Assign scalers to model for backward compatibility
        model_setup.model.feature_scaler = model_setup.feature_scaler
        model_setup.model.target_scaler = model_setup.target_scaler
        model_setup.model.time_scaler = model_setup.time_scaler
        model_setup.model.edge_scaler = model_setup.edge_scaler

        print(f"Loaded best model from epoch {best_checkpoint['epoch']} "
              f"with validation loss {best_checkpoint['val_loss']:.4e} "
              f"(MSE: {best_checkpoint['val_mse']:.4e} Physics: {best_checkpoint['val_phys']:.4e} RelErr: {best_checkpoint['val_rel_error']:.4e})")
        return True

    except Exception as e:
        print(f"Error loading best model: {str(e)}")
        print("Using current model state for evaluation")
        return False


def prepare_model_inputs(
    data,
    model: nn.Module,
    is_training: bool = False
):
    """Prepare data for forward pass by handling standardization of features, targets, and time.

    This function standardizes:
    - Node features using feature_scaler
    - Edge features using edge_scaler
    - Graph-level time using time_scaler
    - Targets using target_scaler

    Args:
        data: Input batch data
        model: The neural network model with fitted scalers
        is_training: Whether this is for training

    Returns:
        Modified data object ready for model forward pass
    """
    # Move to model device
    device = next(model.parameters()).device
    data = data.to(device)

    # Validate required components
    if not hasattr(data, "time") or data.time is None:
        raise ValueError("Each Data must carry a graph-level `time` tensor.")
    if data.time.dim() != 1:
        raise ValueError(f"`data.time` must be 1D [num_graphs]; got {tuple(data.time.shape)}.")
    if not hasattr(model, "time_scaler") or model.time_scaler is None:
        raise RuntimeError("Missing model.time_scaler (fit during setup).")

    # Standardize graph-level time and broadcast to nodes
    time_standardized_value = model.time_scaler.transform(data.time.view(-1, 1)).view(-1)

    if hasattr(data, "batch") and data.batch is not None:
        time_standardized = time_standardized_value[data.batch]
    else:
        time_standardized = time_standardized_value.expand(data.num_nodes)

    # time_standardized: the standardized input used by the GNN
    # time_sigma: the conversion factor σ_t used to scale d/dτ -> d/dt
    # time_standardized and time_sigma are tensors of shape [N, 1]
    std_deviation_value = model.time_scaler.std.view(-1)[0].to(time_standardized.device)
    time_standardized = time_standardized.view(-1, 1).to(next(model.parameters()).device)
    time_sigma = std_deviation_value.expand_as(time_standardized).clone()


    # Attach time information to data
    if is_training:
        data.time_per_node = time_standardized
        data.time_sigma = time_sigma
    else:
        data.time_per_node = time_standardized.detach()
        data.time_sigma = time_sigma.detach()

    # Standardize node/edge features and targets (will be used by model)
    data.node_feat = model.feature_scaler.transform(data.node_feat)
    data.edge_feat = model.edge_scaler.transform(data.edge_feat)
    data.y = model.target_scaler.transform(data.y)

    # Attach scalers to data for physics residual computation
    # This allows physics functions to denormalize predictions back to original space
    data.target_scaler = model.target_scaler
    data.feature_scaler = model.feature_scaler
    data.edge_scaler = model.edge_scaler
    data.time_scaler = model.time_scaler

    return data