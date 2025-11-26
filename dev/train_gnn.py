"""
Training script for the phloem GNN model
"""
import numpy as np

import torch
import torch_geometric

from model.config import ModelConfig

import training.cli as cli
import training.logging as logging
import training.setup as setup
import training.utils as utils
import training.train as train


def main():
    """Main training function."""
    # Parse arguments and create configuration
    config = cli.parse_arguments()

    # Setup environment
    device = setup.setup_environment(config)
    print(f"Using torch {torch.__version__}, torch_geometric {torch_geometric.__version__}")

    # Print experiment configuration
    # print_experiment_config(config)

    # Create TensorBoard writer
    writer = logging.create_tensorboard_writer(config)

    # Get data loaders
    train_loader, val_loader, test_loader = utils.get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, "
          f"Validation batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    # Setup model
    model_setup = setup.setup_model_and_scalers(train_loader, device, config.model_type)
    model_setup.model = model_setup.model.double()  # Convert to float64

    # Create model config for logging
    model_cfg = ModelConfig()

    # Log hyperparameters to TensorBoard
    logging.log_hyperparameters(writer, config, model_cfg)

    # Print detailed model summary
    # print_model_summary(model_setup.model, writer)

    # Setup training components
    optimizer, scheduler = setup.setup_training_components(model_setup.model, config)

    # Run training loop
    training_state = train.train_model(
        model_setup, train_loader, val_loader, optimizer, scheduler,
        writer, config, model_cfg
    )

    # Run final evaluation and reporting
    train.test_model(model_setup, test_loader, writer, training_state, config)

    # Close TensorBoard writer
    writer.close()
    print(f"\nTensorBoard logs saved. To view: tensorboard --logdir={config.tensorboard_log_dir}")


if __name__ == '__main__':
    main()
