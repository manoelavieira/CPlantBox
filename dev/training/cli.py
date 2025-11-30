import argparse

from .config import TrainingConfig, LossType


def parse_arguments() -> TrainingConfig:
    """Parse command line arguments and create training configuration.

    Note: Data standardization has been removed - all training happens in original space.

    Returns:
        TrainingConfig: Validated training configuration
    """
    parser = argparse.ArgumentParser(description="Train phloem GNN model")

    parser.add_argument('--data-path', type=str,
                       help='Path to H5 file for simulated data')
    parser.add_argument('--model-type', type=str, default='nnconv',
                       choices=['nnconv', 'operator'],
                       help='Model architecture: nnconv (baseline) or operator (flux-based)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Ratio of data to use for validation')
    parser.add_argument('--lr', type=float, default=3e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs to train')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--lambda-data', type=float, default=1.0,
                        help='Weight for data MSE loss term (only used with "combined" loss)')
    parser.add_argument('--lambda-phys', type=float, default=1.0,
                        help='Weight for residual loss term (only used with "combined" and "physics" loss)')
    parser.add_argument('--lambda-ic', type=float, default=1.0,
                        help='Weight for initial condition loss term (only used with "physics" loss)')
    parser.add_argument('--lambda-bc', type=float, default=1.0,
                        help='Weight for boundary condition loss term (used with "physics" loss)')
    parser.add_argument('--loss-type', type=str, default='physics',
                        choices=['data', 'physics', 'combined'],
                        help='Type of loss to use: data (MSE), physics, or combined (data + physics)')
    parser.add_argument('--enable-physics-logging', action='store_true',
                        help='Enable detailed physics debug logging to file')
    parser.add_argument('--tensorboard-log-dir', type=str, default='results/tensorboard_logs',
                        help='Directory for TensorBoard logs')

    args = parser.parse_args()

    # Create training configuration
    config = TrainingConfig(
        data_path=args.data_path,
        model_type=args.model_type,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        epochs=args.epochs,
        seed=args.seed,
        lambda_data=args.lambda_data,
        lambda_phys=args.lambda_phys,
        lambda_ic=args.lambda_ic,
        lambda_bc=args.lambda_bc,
        loss_type=LossType(args.loss_type),
        enable_physics_logging=args.enable_physics_logging,
    )

    # Validate configuration
    config.validate()

    return config