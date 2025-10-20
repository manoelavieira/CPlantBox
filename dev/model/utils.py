from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Standardizer:
    """Feature-wise standardization (mean, std deviation) with safe inverse-transform.

    Call fit() on a Tensor [N, D], then use transform()/inv_transform().
    """
    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.device = torch.device('cpu')

    def fit(self, X: torch.Tensor):
        self.device = X.device
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True).clamp_min(1e-8)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return X
        if X.device != self.device:
            self.to(X.device)
        return (X - self.mean) / self.std

    def inv_transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return X
        if X.device != self.device:
            self.to(X.device)
        return X * self.std + self.mean

    def to(self, device) -> 'Standardizer':
        """Move internal tensors to the specified device.

        Args:
            device: The device to move the tensors to (torch.device, str, or int)

        Returns:
            self: The Standardizer instance for method chaining
        """
        device = torch.device(device)

        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        self.device = device
        return self


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