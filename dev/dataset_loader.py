"""
Dataset loader for phloem flow simulation data from HDF5 files.

This module provides utilities to load and preprocess phloem flow simulation
data for training the PhloemNNConv GNN model. Supports loading from both
single HDF5 files and batch loading from directories containing multiple
HDF5 files.
"""

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import Tuple, List, Sequence, Dict
from torch.utils.data import DataLoader
import warnings
import os
from pathlib import Path

def collate_graphs(batch):
    return Batch.from_data_list(batch)

def get_edge_index_from_topology(I_Upflow: np.ndarray, I_Downflow: np.ndarray, connectivity: np.ndarray = None) -> torch.Tensor:
    """
    Convert topology arrays (edge->node mapping) into a PyG edge_index.

    Args:
        I_Upflow: array of upstream node IDs for each edge
        I_Downflow: array of downstream node IDs for each edge
        connectivity: optional array [E, 2] with [upstream, downstream] node IDs per edge

    Returns:
        edge_index: LongTensor [2, E] with source->target node IDs
    """
    edges = []

    if connectivity is not None:
        # print("> Using connectivity array for edge construction")
        n_edges = len(connectivity)
        for e in range(n_edges):
            up, down = connectivity[e]
            edges.append([up, down])  # add directed edge
    else:
        # print("> Using I_Upflow/I_Downflow arrays for edge construction")
        n_edges = len(I_Upflow)
        for e in range(n_edges):
            up, down = I_Upflow[e], I_Downflow[e]
            edges.append([up, down])  # add directed edge

    if not edges:
        raise ValueError("No valid edges found")

    # print(f"Number of edges: {len(edges)}")

    # Convert list of [src, dst] edges (shape [E, 2]) into PyG format [2, E]:
    #   1. torch.tensor(..., long) -> create integer tensor of edges
    #   2. .t() -> transpose to [2, E], i.e. [sources; targets]
    #   3. .contiguous() -> ensure memory layout is contiguous for safe GPU ops
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def load_graph_data(h5_file: h5py.File, timestep: int) -> Data:
    """Load graph data for a specific timestep.

    Args:
        h5_file: Open HDF5 file containing simulation data
        timestep: Index of timestep to load

    Returns:
        data: PyG Data object containing graph structure and features
    """
    step_key = f'step_{timestep:03d}'

    step_params = h5_file[f'{step_key}'].attrs
    step_params_names: Sequence[str] = (
        "PAR", "RH", "Tair", "co2",
        "iteration", "plant_age"
    )

    step_params_vals = []
    step_params_missing: Dict[str, bool] = {}
    for k in step_params_names:
        if k in step_params:
            val = np.array(step_params[k]).item()
            step_params_vals.append(float(val))
        else:
            step_params_missing[k] = True
            step_params_vals.append(float("nan"))

    if step_params_missing:
        missing_keys = ", ".join(step_params_missing.keys())
        print(f"[WARN] Missing step attrs at {step_key}: {missing_keys}")
    step_params = torch.tensor(step_params_vals, dtype=torch.float32).view(1, -1)

    # Load node features
    psi = torch.tensor(h5_file[f'{step_key}/nodes/psiXyl4Phloem'][:], dtype=torch.float32)
    vol = torch.tensor(h5_file[f'{step_key}/nodes/vol_ST'][:], dtype=torch.float32)
    len_leaf = torch.tensor(h5_file[f'{step_key}/nodes/len_leaf'][:], dtype=torch.float32)

    node_feat = torch.stack([psi, vol, len_leaf], dim=1)  # [N, 3]
    num_nodes = psi.shape[0]  # get actual number of nodes

    # Load additional features needed physics residual calculation
    node_fields_names = [
        "C_ST_np", "C_meso", "Csoil_node", "Q_Exud", "Q_Exudmax",
        "Q_Gr", "Q_Grmax", "Q_Rm", "Q_Rmmax", "Q_meso",
        "vol_Meso", "vol_ST"
    ]

    node_fields = torch.empty((num_nodes, len(node_fields_names)), dtype=torch.float32)
    for j, n in enumerate(node_fields_names):
        node_fields[:, j] = torch.from_numpy(h5_file[f"{step_key}/nodes/{n}"][:]).to(torch.float32)

    node_pos_np = h5_file[f"{step_key}/nodes/positions"][:]  # [N, 3]
    node_pos = torch.from_numpy(node_pos_np.astype(np.float32, copy=False))

    print(f"\nStep {timestep}: Number of nodes = {num_nodes}")
    # print(f"node_feat_shape: {node_feat.shape}, node_feat_dtype: {node_feat.dtype}")

    # Get graph structure from both methods for comparison
    I_Up = h5_file[f'{step_key}/arrays/I_Upflow'][:]
    I_Down = h5_file[f'{step_key}/arrays/I_Downflow'][:]
    connectivity = h5_file[f'{step_key}/segments/connectivity'][:]

    # Try both methods and compare
    edge_index_flow = get_edge_index_from_topology(I_Up, I_Down)
    edge_index_conn = get_edge_index_from_topology(None, None, connectivity)

    # Compare the two edge_index tensors
    # print("\nComparing edge construction methods:")
    # if torch.equal(edge_index_flow, edge_index_conn):
    #     print("Both methods yield the same edge_index.")

    # Use connectivity method for edge_index
    edge_index = edge_index_conn

    # Load per-edge resistance values r_ST [E], then reshape to [E, 1]
    # so each edge has an explicit single feature column (required by PyG)
    r_st = torch.tensor(h5_file[f'{step_key}/segments/r_ST'][:], dtype=torch.float32)
    edge_feat = r_st.view(-1, 1)  # [E, 1]
    # print(f"edge_feat_shape: {edge_feat.shape}, edge_feat_dtype: {edge_feat.dtype}")

    org_types = torch.tensor(h5_file[f'{step_key}/segments/organ_types'][:], dtype=torch.long)
    # Map organ types: 2->0 (root), 3->1 (stem), 4->2 (leaf) for embedding lookup
    org_type_map = {2: 0, 3: 1, 4: 2}
    edge_org = torch.tensor([org_type_map[int(t)] for t in org_types], dtype=torch.long)
    # print(f"edge_org_shape: {edge_org.shape}, edge_org_dtype: {edge_org.dtype}")

    # Load target values (sucrose concentration)
    y = torch.tensor(h5_file[f'{step_key}/nodes/Q_ST'][:], dtype=torch.float32).view(-1, 1)
    # print(f"y_shape: {y.shape}, y_dtype: {y.dtype}")

    # Use timestep as time feature
    time = torch.tensor(timestep, dtype=torch.float32)

    # Load physics constants from parameters
    sim_params = h5_file['parameters/sieve_tube'].attrs

    sim_params_names: Sequence[str] = (
        "CSTimin", "C_targ", "KMfu", "Mloading", "Q10", "TrefQ10",
        "beta_loading", "Vmaxloading", "krm2v"
    )

    sim_param_vals = []
    sim_param_missing: Dict[str, bool] = {}
    for k in sim_params_names:
        if k in sim_params:
            val = np.array(sim_params[k]).item()
            sim_param_vals.append(float(val))
        else:
            sim_param_missing[k] = True
            sim_param_vals.append(float("nan"))

    if sim_param_missing:
        missing_keys = ", ".join(sim_param_missing.keys())
        print(f"[WARN] Missing sieve_tube attrs at {step_key}: {missing_keys}")

    sim_params = torch.tensor(sim_param_vals, dtype=torch.float32).view(1, -1)  # [1, K]

    # Create PyG Data object with explicit num_nodes
    data = Data(
        node_feat=node_feat,    # Node features [N, 3] - [psi, vol_ST, len_leaf]
        edge_index=edge_index,  # Graph connectivity [2, E]
        edge_feat=edge_feat,    # Edge features [E, 1]
        edge_org=edge_org,      # Edge organ types [E]
        y=y,                    # Target values [N, 1]
        time=time,              # Graph-level time feature [1]
        num_nodes=num_nodes,    # Explicitly set number of nodes
        sim_params=sim_params,                      # [1,K]
        sim_params_names=list(sim_params_names),    # list[str] (not used in math; for reference)
        step_params=step_params,                    # [1,J]
        step_params_names=list(step_params_names),  # list[str] (not used in math; for reference)
        node_fields=node_fields,                    # Additional node fields for physics residual [N, len(names)]
        node_fields_names=list(node_fields_names),  # list[str] (not used in math; for reference)
        node_pos=node_pos                           # Node positions [N, 3] (optional; for visualization)
    )

    return data


def load_graphs_from_file(h5_path: str) -> List[Data]:
    """Load all graph data from a single HDF5 file.

    Args:
        h5_path: Path to HDF5 file containing simulation data

    Returns:
        graphs: List of PyG Data objects loaded from the file
    """
    graphs: List[Data] = []

    try:
        with h5py.File(h5_path, 'r') as f:
            n_steps = None
            sim_path = "parameters/simulation"
            if sim_path in f:
                sim_group = f[sim_path]
                n_steps = sim_group.attrs.get('steps', None)
            print(f"File {h5_path}: Simulation steps: {n_steps}")

            if n_steps is None or n_steps <= 0:
                raise ValueError(f"No timesteps found in HDF5 file {h5_path}")

            # Load graphs from each timestep
            for i in range(n_steps):
                try:
                    graph = load_graph_data(f, i)
                    graphs.append(graph)
                except Exception as e:
                    warnings.warn(f"Error loading timestep {i} from {h5_path}: {str(e)}")
                    continue
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {h5_path}: {str(e)}")

    if not graphs:
        warnings.warn(f"No valid graphs loaded from file {h5_path}")

    return graphs


def load_phloem_data(h5_path: str, batch_size: int = 32,
                     train_ratio: float = 0.8, val_ratio: float = 0.1,
                     random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load phloem simulation data and create train/val/test DataLoaders.

    Args:
        h5_path: Path to HDF5 file or directory containing HDF5 files with simulation data
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    print(f"train_ratio: {train_ratio}, val_ratio: {val_ratio}, test_ratio: {1 - train_ratio - val_ratio}")

    graphs: List[Data] = []

    # Check if h5_path is a file or directory
    path = Path(h5_path)

    if path.is_file():
        # Single file case (original behavior)
        print(f"Loading data from single file: {h5_path}")
        graphs = load_graphs_from_file(h5_path)

    elif path.is_dir():
        # Directory case (new batch loading functionality)
        print(f"Loading data from directory: {h5_path}")

        # Find all .h5 files in the directory
        h5_files = list(path.rglob("*.h5"))
        if not h5_files:
            raise RuntimeError(f"No .h5 files found in directory {h5_path}")

        print(f"Found {len(h5_files)} .h5 files in directory")

        # Load graphs from each file
        for h5_file in sorted(h5_files):  # Sort for consistent ordering
            print(f"\nProcessing file: {h5_file}")
            try:
                file_graphs = load_graphs_from_file(str(h5_file))
                graphs.extend(file_graphs)
                print(f"\nLoaded {len(file_graphs)} graphs from {h5_file}")
            except Exception as e:
                warnings.warn(f"Failed to load file {h5_file}: {str(e)}")
                continue

    else:
        raise RuntimeError(f"Path {h5_path} is neither a file nor a directory")

    if not graphs:
        raise RuntimeError("No valid graphs loaded")

    print(f"\nSuccessfully loaded {len(graphs)} total graphs")

    # Split graphs into train/val/test
    n_samples = len(graphs)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    n_test = n_samples - n_train - n_val

    # Shuffle with fixed seed
    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(n_samples, generator=generator)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_graphs, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(test_graphs, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_graphs)

    return train_loader, val_loader, test_loader


def main():
    # Example 1: Load from single file (original behavior)
    h5_path = 'data/sim_00/phloem_simulation.h5'
    print("====== Loading from single file ======")
    try:
        train_loader, val_loader, test_loader = load_phloem_data(h5_path)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Single file loading failed: {e}")

    # Example 2: Load from directory (new batch loading functionality)
    h5_dir = 'data/'  # Directory containing multiple .h5 files
    print("\n====== Loading from directory ======")
    try:
        train_loader, val_loader, test_loader = load_phloem_data(h5_dir)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Directory loading failed: {e}")

    print("\nNote: Use either a single .h5 file path or a directory path containing .h5 files")

if __name__ == '__main__':
    main()