from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_parameters(data, device, batch_vec=None, y_pred_size=None):
    """Extract and broadcast simulation and step parameters to per-node tensors.

    Args:
        data: Data object containing sim_params and step_params
        device: Target device for tensors
        batch_vec: Batch vector for batched graphs (optional)
        y_pred_size: Size of predictions for single graph case

    Returns:
        dict: Parameter name -> per-node tensor mapping
    """
    # Use parameter names stored in the data object (from dataset_loader.py)
    sim_params_names = data.sim_params_names
    step_params_names = data.step_params_names

    sim_params = data.sim_params.to(device)
    step_params = data.step_params.to(device)

    params = {}

    if batch_vec is not None:
        # Batched case: map parameters to nodes via batch vector
        for i, name in enumerate(sim_params_names):
            params[f"{name}"] = sim_params[batch_vec, i]
        for i, name in enumerate(step_params_names):
            params[f"{name}"] = step_params[batch_vec, i]
    else:
        # Single graph case: broadcast to all nodes
        N = y_pred_size
        for i, name in enumerate(sim_params_names):
            params[f"{name}"] = sim_params[0, i].expand(N)
        for i, name in enumerate(step_params_names):
            params[f"{name}"] = step_params[0, i].expand(N)

    return params


def extract_node_fields(data, device):
    """Extract node fields into a dictionary using names from data object.

    Args:
        data: Data object containing node_fields and node_fields_names
        device: Target device for tensors

    Returns:
        dict: Field name -> tensor mapping
    """
    node_fields_names = data.node_fields_names
    node_fields = data.node_fields.to(device)  # [N, num_fields]

    fields = {}
    for i, name in enumerate(node_fields_names):
        fields[f"{name}"] = node_fields[:, i]

    return fields


def compute_RT_per_node(
    Temp: torch.Tensor,
    batch_vec: Optional[torch.Tensor],
    R: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute per-node RT given per-node temperature (°C).
    - If batched, uses the temperature of the first node of each graph,
      then broadcasts within that graph (mirrors C++ global-per-graph T behavior).
    - If single-graph, uses Temp[0] and broadcasts to all nodes.

    Returns:
        torch.Tensor: RT per node [N], on `device` and in `dtype`.
    """
    Temp = Temp.to(device=device, dtype=dtype)

    if batch_vec is not None:
        # Batched case: use temperature from first node of each graph
        batch_vec = batch_vec.to(device)
        unique_graphs = torch.unique(batch_vec)
        first_node_per_graph = torch.zeros(len(unique_graphs), dtype=torch.long, device=device)

        for i, graph_id in enumerate(unique_graphs):
            first_node_per_graph[i] = torch.where(batch_vec == graph_id)[0][0]

        # Get temperature for each graph and broadcast to all nodes in that graph
        temp_per_graph = Temp[first_node_per_graph]
        RT_per_graph = R * (temp_per_graph + 273.15)
        RT = RT_per_graph[batch_vec]
    else:
        RT_scalar = R * (Temp[0] + 273.15)
        RT = RT_scalar.expand_as(Temp)

    return RT