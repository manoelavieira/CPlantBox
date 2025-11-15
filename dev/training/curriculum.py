"""
Curriculum learning utilities for progressive training on increasingly complex graphs.

Implements difficulty-based curriculum where simpler (smaller) graphs are learned first,
then progressively more complex graphs are added to the training set.
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple
from torch_geometric.data import Data


def get_graph_difficulty(graph: Data) -> float:
    """Compute difficulty score for a graph.

    Difficulty is based on graph size (number of nodes) as a proxy for complexity.
    Smaller graphs (early timesteps) are simpler, larger graphs (late timesteps) are harder.

    Args:
        graph: PyG Data object

    Returns:
        Difficulty score (higher = harder)
    """
    return float(graph.num_nodes)


def create_curriculum_subsets(
    graphs: List[Data],
    difficulty_thresholds: List[float],
) -> List[List[int]]:
    """Create progressive subsets of graphs based on difficulty thresholds.

    Args:
        graphs: List of graph data objects
        difficulty_thresholds: List of difficulty cutoffs for each curriculum stage
                               e.g., [40, 60, 100] means:
                               - Stage 1: graphs with <= 40 nodes
                               - Stage 2: graphs with <= 60 nodes
                               - Stage 3: graphs with <= 100 nodes (all)

    Returns:
        List of index lists, one per curriculum stage
    """
    # Compute difficulty for each graph
    difficulties = [get_graph_difficulty(g) for g in graphs]

    # Create cumulative subsets for each threshold
    curriculum_indices = []
    for threshold in difficulty_thresholds:
        indices = [i for i, diff in enumerate(difficulties) if diff <= threshold]
        curriculum_indices.append(indices)
        print(f"Curriculum stage with threshold {threshold}: {len(indices)} graphs")

    return curriculum_indices


def create_curriculum_loaders(
    train_graphs: List[Data],
    val_loader: DataLoader,
    test_loader: DataLoader,
    difficulty_thresholds: List[float],
    batch_size: int,
    collate_fn
) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """Create curriculum-based training data loaders.

    Creates multiple training loaders with progressively more difficult graphs,
    while keeping validation and test sets unchanged.

    Args:
        train_graphs: List of training graph data objects
        val_loader: Validation data loader (unchanged)
        test_loader: Test data loader (unchanged)
        difficulty_thresholds: Difficulty cutoffs for curriculum stages
        batch_size: Batch size for training loaders
        collate_fn: Collate function for batching graphs

    Returns:
        Tuple of (curriculum_train_loaders, val_loader, test_loader)
    """
    # Get indices for each curriculum stage
    curriculum_indices = create_curriculum_subsets(train_graphs, difficulty_thresholds)

    # Create data loaders for each stage (skip empty stages)
    curriculum_loaders = []
    for stage_idx, indices in enumerate(curriculum_indices):
        if not indices:
            print(f"Warning: Stage {stage_idx} has no graphs, skipping!")
            continue

        # Create subset of training data
        stage_graphs = [train_graphs[i] for i in indices]

        # Create loader for this stage
        stage_loader = DataLoader(
            stage_graphs,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        curriculum_loaders.append(stage_loader)

        # Report statistics
        num_nodes = [g.num_nodes for g in stage_graphs]
        print(f"Stage {len(curriculum_loaders)}: {len(stage_graphs)} graphs, "
              f"nodes: min={min(num_nodes)}, max={max(num_nodes)}, mean={sum(num_nodes)/len(num_nodes):.1f}")

    if not curriculum_loaders:
        raise ValueError("No valid curriculum stages created! Check difficulty thresholds.")

    return curriculum_loaders, val_loader, test_loader


def get_default_curriculum_stages(total_graphs: int, num_stages: int = 3) -> List[float]:
    """Get default curriculum thresholds based on data statistics.

    Args:
        total_graphs: Total number of training graphs
        num_stages: Number of curriculum stages (default: 3)

    Returns:
        List of difficulty thresholds
    """
    # For phloem data, typical node counts range from ~25 (early) to ~100 (late)
    # Use empirical thresholds based on typical growth patterns
    if num_stages == 3:
        return [70.0, 85.0, 150.0]  # Easy (early), Medium (mid), All (late)
    elif num_stages == 4:
        return [60.0, 75.0, 90.0, 150.0]  # Very Easy, Easy, Medium, All
    elif num_stages == 5:
        return [55.0, 70.0, 80.0, 90.0, 150.0]  # Very Easy, Easy, Medium, Hard, All
    else:
        # Linear spacing for custom number of stages
        return list(range(60, 151, (150 - 60) // (num_stages - 1)))


class CurriculumScheduler:
    """Scheduler for curriculum learning that tracks progress through stages."""

    def __init__(
        self,
        curriculum_loaders: List[DataLoader],
        epochs_per_stage: List[int],
        val_loader: DataLoader,
        test_loader: DataLoader
    ):
        """
        Args:
            curriculum_loaders: List of data loaders, one per curriculum stage
            epochs_per_stage: Number of epochs to train on each stage
            val_loader: Validation data loader (unchanged across stages)
            test_loader: Test data loader (unchanged across stages)
        """
        self.curriculum_loaders = curriculum_loaders
        self.epochs_per_stage = epochs_per_stage
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.num_stages = len(curriculum_loaders)
        self.current_stage = 0
        self.current_epoch_in_stage = 0
        self.total_epochs_completed = 0

        # Validate inputs
        assert len(curriculum_loaders) == len(epochs_per_stage), \
            "Number of loaders must match number of epoch specifications"

        print(f"\n{'='*60}")
        print(f"Curriculum Learning Scheduler Initialized")
        print(f"{'='*60}")
        print(f"Total stages: {self.num_stages}")
        for i, (loader, epochs) in enumerate(zip(curriculum_loaders, epochs_per_stage)):
            print(f"  Stage {i+1}: {len(loader.dataset)} graphs, {epochs} epochs")
        print(f"{'='*60}\n")

    def get_current_loader(self) -> DataLoader:
        """Get the current stage's training loader."""
        return self.curriculum_loaders[self.current_stage]

    def get_current_stage_info(self) -> dict:
        """Get information about current curriculum stage."""
        return {
            'stage': self.current_stage + 1,
            'total_stages': self.num_stages,
            'epoch_in_stage': self.current_epoch_in_stage + 1,
            'epochs_per_stage': self.epochs_per_stage[self.current_stage],
            'total_epochs': self.total_epochs_completed,
            'num_graphs': len(self.curriculum_loaders[self.current_stage].dataset)
        }

    def step(self) -> bool:
        """Advance to next epoch, potentially moving to next curriculum stage.

        Returns:
            True if training should continue, False if curriculum is complete
        """
        self.current_epoch_in_stage += 1
        self.total_epochs_completed += 1

        # Check if current stage is complete
        if self.current_epoch_in_stage >= self.epochs_per_stage[self.current_stage]:
            # Move to next stage
            self.current_stage += 1
            self.current_epoch_in_stage = 0

            if self.current_stage >= self.num_stages:
                # Curriculum complete
                return False
            else:
                # Starting new stage
                print(f"\n{'='*60}")
                print(f"ADVANCING TO CURRICULUM STAGE {self.current_stage + 1}/{self.num_stages}")
                print(f"{'='*60}")
                stage_info = self.get_current_stage_info()
                print(f"Training graphs: {stage_info['num_graphs']}")
                print(f"Epochs for this stage: {stage_info['epochs_per_stage']}")
                print(f"{'='*60}\n")

        return True

    def is_stage_boundary(self) -> bool:
        """Check if we just started a new curriculum stage."""
        return self.current_epoch_in_stage == 0
