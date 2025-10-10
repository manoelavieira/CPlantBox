"""
Dataset loader for phloem flow simulation data from HDF5 files.

This module provides utilities to load and preprocess phloem flow simulation
data for training the PhloemNNConv GNN model.
"""

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import Tuple, List, Sequence, Dict
from torch.utils.data import DataLoader
import warnings

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
    print(f"node_feat_shape: {node_feat.shape}, node_feat_dtype: {node_feat.dtype}")

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


def load_phloem_data(h5_path: str, batch_size: int = 32,
                     train_ratio: float = 0.8, val_ratio: float = 0.1,
                     random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load phloem simulation data and create train/val/test DataLoaders.

    Args:
        h5_path: Path to HDF5 file containing simulation data
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    print(f"train_ratio: {train_ratio}, val_ratio: {val_ratio}, test_ratio: {1 - train_ratio - val_ratio}")

    graphs: List[Data] = []

    try:
        with h5py.File(h5_path, 'r') as f:
            n_steps = None
            sim_path = "parameters/simulation"
            if sim_path in f:
                sim_group = f[sim_path]
                n_steps = sim_group.attrs.get('steps', None)
            print(f"Simulation steps: {n_steps}")

            if n_steps is None or n_steps <= 0:
                raise ValueError("No timesteps found in HDF5 file")

            # Load graphs from each timestep
            for i in range(n_steps):
                try:
                    graph = load_graph_data(f, i)
                    graphs.append(graph)
                except Exception as e:
                    warnings.warn(f"Error loading timestep {i}: {str(e)}")
                    continue
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {h5_path}: {str(e)}")

    if not graphs:
        raise RuntimeError("No valid graphs loaded from file")

    print(f"\nSuccessfully loaded {len(graphs)} graphs from {h5_path}")

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
