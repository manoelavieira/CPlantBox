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


# CPlantBox organ type constants for reference
CPLANTBOX_ORGAN_TYPES = {
    0: "organ",
    1: "seed",
    2: "root",
    3: "stem",
    4: "leaf"
}

# Remapped organ types for GNN model (only effective types: root, stem, leaf)
# Maps CPlantBox indices [2,3,4] -> [0,1,2]
GNN_ORGAN_TYPES = {
    0: "root",
    1: "stem",
    2: "leaf"
}

CPLANTBOX_TO_GNN_MAPPING = {2: 0, 3: 1, 4: 2}  # root -> 0, stem -> 1, leaf -> 2


def collate_graphs(batch):
    """Custom collate function that handles parameter names properly."""
    batched_data = Batch.from_data_list(batch)

    # Fix parameter names: they should be the same across all graphs in a batch
    # So we just take the first one instead of creating nested lists
    if hasattr(batched_data, 'sim_params_names') and isinstance(batched_data.sim_params_names, list):
        if len(batched_data.sim_params_names) > 0 and isinstance(batched_data.sim_params_names[0], list):
            batched_data.sim_params_names = batched_data.sim_params_names[0]

    if hasattr(batched_data, 'step_params_names') and isinstance(batched_data.step_params_names, list):
        if len(batched_data.step_params_names) > 0 and isinstance(batched_data.step_params_names[0], list):
            batched_data.step_params_names = batched_data.step_params_names[0]

    if hasattr(batched_data, 'node_fields_names') and isinstance(batched_data.node_fields_names, list):
        if len(batched_data.node_fields_names) > 0 and isinstance(batched_data.node_fields_names[0], list):
            batched_data.node_fields_names = batched_data.node_fields_names[0]

    return batched_data


def get_edge_index_from_topology(
        I_Upflow: np.ndarray,
        I_Downflow: np.ndarray,
        connectivity: np.ndarray = None
    ) -> torch.Tensor:
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
        n_edges = len(connectivity)
        for e in range(n_edges):
            up, down = connectivity[e]
            edges.append([up, down])  # add directed edge
    else:
        n_edges = len(I_Upflow)
        for e in range(n_edges):
            up, down = I_Upflow[e], I_Downflow[e]
            edges.append([up, down])  # add directed edge

    if not edges:
        raise ValueError("No valid edges found")

    # Convert list of [src, dst] edges (shape [E, 2]) into PyG format [2, E]:
    # 1. torch.tensor(..., long) -> create integer tensor of edges
    # 2. .t() -> transpose to [2, E], i.e. [sources; targets]
    # 3. .contiguous() -> ensure memory layout is contiguous for safe GPU ops
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


def load_graph_data(h5_file: h5py.File, timestep: int, initial_node_count: int = None) -> Data:
    """Load graph data for a specific timestep.

    Args:
        h5_file: Open HDF5 file containing simulation data
        timestep: Index of timestep to load
        initial_node_count: Number of initial nodes from timestep 0.
                            If None, will be determined from step_000.
                            Used to create initial node mask.

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
    step_params = torch.tensor(step_params_vals, dtype=torch.float64).view(1, -1)

    # Load node features
    psi = torch.tensor(h5_file[f'{step_key}/nodes/psiXyl4Phloem'][:], dtype=torch.float64)
    vol = torch.tensor(h5_file[f'{step_key}/nodes/vol_ST'][:], dtype=torch.float64)
    len_leaf = torch.tensor(h5_file[f'{step_key}/nodes/len_leaf'][:], dtype=torch.float64)
    Q_Rmmax = torch.tensor(h5_file[f'{step_key}/nodes/Q_Rmmax'][:], dtype=torch.float64)
    Q_Grmax = torch.tensor(h5_file[f'{step_key}/nodes/Q_Grmax'][:], dtype=torch.float64)
    Q_Exudmax = torch.tensor(h5_file[f'{step_key}/nodes/Q_Exudmax'][:], dtype=torch.float64)
    Temp = step_params[0, step_params_names.index("Tair")].repeat(psi.shape[0])

    node_feat = torch.stack([psi, vol, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp], dim=1)  # [N, 7]
    num_nodes = psi.shape[0]  # get actual number of nodes

    # Load additional features needed physics residual calculation
    node_fields_names = [
        "C_ST_np", "C_meso", "Csoil_node", "Q_Exud", "Q_Exudmax",
        "Q_Gr", "Q_Grmax", "Q_Rm", "Q_Rmmax", "Q_meso",
        "vol_Meso", "vol_ST"
    ]

    node_fields = torch.empty((num_nodes, len(node_fields_names)), dtype=torch.float64)
    for j, n in enumerate(node_fields_names):
        node_fields[:, j] = torch.from_numpy(h5_file[f"{step_key}/nodes/{n}"][:]).to(torch.float64)

    node_pos_np = h5_file[f"{step_key}/nodes/positions"][:]  # [N, 3]
    node_pos = torch.from_numpy(node_pos_np.astype(np.float64, copy=False))

    # Get graph structure from both methods for comparison
    I_Up = h5_file[f'{step_key}/arrays/I_Upflow'][:]
    I_Down = h5_file[f'{step_key}/arrays/I_Downflow'][:]
    connectivity = h5_file[f'{step_key}/segments/connectivity'][:]

    # Try both methods and compare
    edge_index_flow = get_edge_index_from_topology(I_Up, I_Down)
    edge_index_conn = get_edge_index_from_topology(None, None, connectivity)

    # Use connectivity method for edge_index
    edge_index = edge_index_conn

    # Load per-edge resistance values r_ST [E], then reshape to [E, 1]
    # so each edge has an explicit single feature column (required by PyG)
    r_st = torch.tensor(h5_file[f'{step_key}/segments/r_ST'][:], dtype=torch.float64)
    edge_feat = r_st.view(-1, 1)  # [E, 1]

    # Load organ types and remap to GNN indices
    # CPlantBox: ot_organ=0, ot_seed=1, ot_root=2, ot_stem=3, ot_leaf=4
    # GNN model: ot_root=0, ot_stem=1, ot_leaf=2 (only effective types)
    org_types_raw = torch.tensor(h5_file[f'{step_key}/segments/organ_types'][:], dtype=torch.long)

    # Filter out unused organ types (0=organ, 1=seed) and remap the rest
    valid_mask = (org_types_raw >= 2)  # Only keep root(2), stem(3), leaf(4)
    if not valid_mask.all():
        print(f"[WARN] Step {timestep}: Found {(~valid_mask).sum()} edges with organ types 0 or 1, filtering them out")

    # Remap organ types: 2->0, 3->1, 4->2
    edge_org = torch.tensor([CPLANTBOX_TO_GNN_MAPPING.get(t.item(), -1) for t in org_types_raw], dtype=torch.long)

    # Validate remapped organ types are within expected range [0, 2]
    if edge_org.numel() > 0:
        min_org_type, max_org_type = edge_org.min().item(), edge_org.max().item()
        if min_org_type < 0 or max_org_type > 2:
            raise ValueError(f"Step {timestep}: Remapped organ types outside range [0,2]: min={min_org_type}, max={max_org_type}")

        # Print organ type distribution for debugging (only first timestep)
        if timestep == 0:
            unique_types, counts = torch.unique(edge_org, return_counts=True)
            type_dist = {GNN_ORGAN_TYPES.get(t.item(), f"unknown_{t.item()}"): c.item()
                        for t, c in zip(unique_types, counts)}
            print(f"GNN organ type distribution: {type_dist}")

    # Load target values (sucrose concentration)
    y = torch.tensor(h5_file[f'{step_key}/nodes/C_ST_np'][:], dtype=torch.float64).view(-1, 1)

    # Use physical time (plant_age) as time feature instead of timestep index
    # This is CRITICAL for physics-informed learning: dC/dt must be computed with respect to
    # actual physical time, not arbitrary timestep indices
    plant_age = h5_file[f'{step_key}'].attrs['plant_age']
    time = torch.tensor(plant_age, dtype=torch.float64)

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

    sim_params = torch.tensor(sim_param_vals, dtype=torch.float64).view(1, -1)  # [1, K]

    # Create initial node mask based on node indices
    is_initial_node = torch.zeros(num_nodes, dtype=torch.bool)

    try:
        if initial_node_count is not None:
            t0_nodes = initial_node_count
        else:
            t0_nodes = get_initial_node_count(h5_file)

        if timestep == 0:
            # For t=0, all nodes are initial
            is_initial_node[:] = True
        else:
            # For subsequent timesteps, nodes 0 to (t0_nodes-1) are initial
            # Only mark as initial if the node index is within the original range
            initial_count = min(t0_nodes, num_nodes)
            is_initial_node[:initial_count] = True

    except Exception as e:
        warnings.warn(f"Could not determine initial nodes for timestep {timestep}: {str(e)}")

        if timestep == 0:
            is_initial_node[:] = True

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
        node_pos=node_pos,                          # Node positions [N, 3] (optional; for visualization)
        is_initial_node=is_initial_node             # Boolean mask indicating nodes present at t=0 [N]
    )

    return data


def get_initial_node_count(h5_file: h5py.File) -> int:
    """Get the number of nodes in the first timestep (t=0) of an HDF5 file.

    Args:
        h5_file: Open HDF5 file containing simulation data

    Returns:
        int: Number of nodes in timestep 0
    """
    try:
        step_key = 'step_000'  # First timestep
        node_count = h5_file[f"{step_key}/nodes/positions"].shape[0]
        return node_count
    except KeyError:
        warnings.warn(f"Could not find step_000 in HDF5 file")
        return 0


def find_common_initial_node_count(h5_files: List[Path]) -> int:
    """Find the minimum number of initial nodes across all simulation files.

    This identifies the common initial node count that exists across all
    simulation runs of the same plant. Uses the minimum count to ensure
    all files have at least this many initial nodes.

    Args:
        h5_files: List of HDF5 file paths

    Returns:
        int: Minimum number of initial nodes across all files
    """
    if not h5_files:
        return 0

    min_initial_count = float('inf')

    for h5_file_path in h5_files:
        try:
            with h5py.File(h5_file_path, 'r') as f:
                file_initial_count = get_initial_node_count(f)
                min_initial_count = min(min_initial_count, file_initial_count)

        except Exception as e:
            warnings.warn(f"Error reading initial node count from {h5_file_path}: {str(e)}")
            continue

    if min_initial_count == float('inf'):
        return 0

    print(f"Found {min_initial_count} common initial nodes across {len(h5_files)} files")

    return min_initial_count


def load_graphs_from_file(h5_path: str, initial_node_count: int = None) -> List[Data]:
    """Load all graph data from a single HDF5 file.

    Args:
        h5_path: Path to HDF5 file containing simulation data
        initial_node_count: Number of initial nodes (for index-based marking).
                            If None, will be determined from step_000.

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
                    graph = load_graph_data(f, i, initial_node_count)
                    graphs.append(graph)
                except Exception as e:
                    warnings.warn(f"Error loading timestep {i} from {h5_path}: {str(e)}")
                    continue
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {h5_path}: {str(e)}")

    if not graphs:
        warnings.warn(f"No valid graphs loaded from file {h5_path}")

    return graphs


def load_graphs_from_directory(h5_dir: str) -> List[Data]:
    """Load all graph data from all HDF5 files in a directory.

    Args:
        h5_dir: Path to directory containing HDF5 files
    Returns:
        graphs: List of PyG Data objects loaded from all files
    """
    graphs: List[Data] = []

    # Find all .h5 files in the directory
    h5_files = list(Path(h5_dir).rglob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No .h5 files found in directory {h5_dir}")

    print(f"Found {len(h5_files)} .h5 files in directory")

    # Find common initial node count across all files
    common_initial_count = find_common_initial_node_count(h5_files)

    # Load graphs from each file
    for h5_file in sorted(h5_files):  # Sort for consistent ordering
        print(f"\nProcessing file: {h5_file}")
        try:
            file_graphs = load_graphs_from_file(str(h5_file), common_initial_count)
            graphs.extend(file_graphs)
            print(f"Loaded {len(file_graphs)} graphs from {h5_file}")
        except Exception as e:
            warnings.warn(f"Failed to load file {h5_file}: {str(e)}")
            continue

    if not graphs:
        raise RuntimeError("No valid graphs loaded")

    print(f"\nSuccessfully loaded {len(graphs)} total graphs")

    return graphs


def train_test_split(
        graphs: List[Data],
        train_ratio: float,
        val_ratio: float,
        batch_size: int = 8,
        random_seed: int = 42
    ) -> Tuple[List[Data], List[Data], List[Data]]:
    """Split graphs into train/val/test sets.

    Args:
        graphs: List of PyG Data objects
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation
        batch_size: Batch size for DataLoaders
        random_seed: Random seed for reproducibility

    Returns:
        train_graphs, val_graphs, test_graphs: Lists of graphs for each split
    """
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


def load_phloem_data(
       h5_path: str,
       batch_size: int = 8,
       train_ratio: float = 0.8,
       val_ratio: float = 0.1,
       random_seed: int = 42
   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    path = Path(h5_path)

    if path.is_file():
        print(f"Loading data from single file: {h5_path}")
        graphs = load_graphs_from_file(h5_path, None)
    elif path.is_dir():
        print(f"Loading data from directory: {h5_path}")
        graphs = load_graphs_from_directory(h5_path)
    else:
        raise RuntimeError(f"Path {h5_path} is neither a file nor a directory")

    if not graphs:
        raise RuntimeError("No valid graphs loaded")

    print(f"\nSuccessfully loaded {len(graphs)} total graphs")

    train_loader, val_loader, test_loader = train_test_split(graphs, train_ratio, val_ratio, batch_size, random_seed)

    return train_loader, val_loader, test_loader


def main():
    # ==== Example 1: Load from single file (original behavior)
    h5_path = './cplantbox/data/sim_01/phloem_simulation.h5'
    print("====== Loading from single file ======")
    try:
        train_loader, val_loader, test_loader = load_phloem_data(h5_path)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

        # Test the initial node mask
        graphs = load_graphs_from_file(h5_path, None)
        print(f"Loaded {len(graphs)} graphs from file")

        # Show details for first few graphs
        for i, graph in enumerate(graphs[:3]):
            print(f"Graph {i} at timestep {graph.time.item()}: {graph.is_initial_node.sum().item()} initial nodes out of {graph.is_initial_node.size(0)} total nodes")

    except Exception as e:
        print(f"Single file loading failed: {e}")

    # ==== Example 2: Load from directory (new batch loading functionality)
    h5_dir = './cplantbox/data/'  # directory containing multiple .h5 files
    print("\n====== Loading from directory ======")
    try:
        train_loader, val_loader, test_loader = load_phloem_data(h5_dir)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"Directory loading failed: {e}")

    print("\nNote: Use either a single .h5 file path or a directory path containing .h5 files")

if __name__ == '__main__':
    main()