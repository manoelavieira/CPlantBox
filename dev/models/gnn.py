"""
Phloem GNN (baseline) — NNConv with edge features
-------------------------------------------------
Predicts sucrose concentration per node at timestep t, given:
- Plant topology (edge_index)
- Node features at time t: water potential (psi), sieve-tube volume (vol_st), leaf length (len_leaf)
- Edge features at time t: sieve-tube resistance (r_st), organ type (categorical)


Expected `Data` fields per graph (per timestep)
----------------------------------------------
- data.edge_index: LongTensor  [2, E]
- data.edge_feat:  FloatTensor [E, 1]      # r_st (resistance)
- data.node_feat:  FloatTensor [N, 3]      # [psi, vol_st, len_leaf]
- data.time:       FloatTensor [1]         # time in days (graph-level)
- data.y:          FloatTensor [N, 1]      # target sucrose at t
- Optional: data.batch for mini-batching multiple graphs
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from torch_geometric.nn.norm import GraphNorm  # per-graph normalization

# Global constants
R = 83.14  # universal gas constant


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


@dataclass
class ModelConfig:
    """Configuration for PhloemNNConv model.

    Attributes:
        node_feat_dim: Dimension of continuous node features
        num_org_types: Number of organ types [LEAF, STEM, ROOT]
        org_emb_size: Dimension of organ type embeddings
        hidden_size: Dimension of hidden layers in NNConv/MLPs
        num_layers: Number of NNConv layers
        edge_feat_dim: Dimension of continuous edge features [r_st]
        aggr: NNConv aggregator type ("add", "mean", or "max")
        dropout: Dropout probability
    """
    node_feat_dim: int = 7  # [psi, vol_st, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp]
    edge_feat_dim: int = 1  # [r_st]
    num_org_types: int = 3
    org_emb_size: int = 8  # embedding dimension for categorical organ type
    hidden_size: int = 64
    num_layers: int = 3
    aggr: str = "add"
    dropout: float = 0.0

    def __post_init__(self):
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Dropout must be between 0 and 1, got {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"Number of layers must be positive, got {self.num_layers}")
        if self.aggr not in ["add", "mean", "max"]:
            raise ValueError(f"Aggregator must be one of ['add', 'mean', 'max'], got {self.aggr}")
        if any(d < 1 for d in [self.node_feat_dim, self.edge_feat_dim, self.num_org_types,
                               self.org_emb_size, self.hidden_size]):
            raise ValueError("All dimensions must be positive integers")


class EdgeNet(nn.Module):
    """Edge MLP producing weight matrices for NNConv.

    Maps edge features (continuous + categorical) -> [E, out_channels * in_channels].
    MLP that turns per-edge features into a per-edge weight matrix W_e that NNConv
    will use to transform neighbor node features.
    """
    def __init__(self, edge_feat_cont_dim: int, num_org_types: int, org_emb_size: int,
                 in_node_dim: int, out_node_dim: int, hidden_size: int = 64):
        super().__init__()

        self.in_node_dim = in_node_dim
        self.out_node_dim = out_node_dim

        self.org_emb = nn.Embedding(num_org_types, org_emb_size)
        edge_input_dim = edge_feat_cont_dim + org_emb_size

        # Combined MLP for both continuous and categorical features
        # It outputs a flattened weight matrix of size [out_channels * in_channels] per edge
        self.mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            # Final linear layer has no activation
            # We want raw learned weights, not squashed by ReLU/sigmoid
            nn.Linear(hidden_size, in_node_dim * out_node_dim)
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        # edge_features: [E, D+1] where D is edge_feat_cont_dim and last column is organ type
        edge_feat_cont = edge_features[:, :-1]  # continuous features
        edge_feat_cat = edge_features[:, -1].long()  # organ type as long tensor (int64)

        # Combine continuous edge features with organ embeddings
        edge_emb = self.org_emb(edge_feat_cat)
        edge_inputs = torch.cat([edge_feat_cont, edge_emb], dim=-1)

        return self.mlp(edge_inputs)


class PhloemNNConv(nn.Module):
    """Neural network model for phloem flow prediction using NNConv layers.

    Combines node features with edge features through multiple NNConv
    layers to predict sucrose concentration.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.feature_scaler = None      # for node features
        self.target_scaler = None       # for output values
        self.time_scaler = None         # for time normalization
        self.edge_scaler = None         # for edge features
        self._validated_input = False   # Track if input has been validated

        # Node input = continuous node features + time
        in_node_dim = cfg.node_feat_dim + 1
        current_dim = in_node_dim

        conv_layers = []
        norm_layers = []
        for _ in range(cfg.num_layers):
            edge_mlp = EdgeNet(edge_feat_cont_dim = cfg.edge_feat_dim,
                               num_org_types = cfg.num_org_types,
                               org_emb_size = cfg.org_emb_size,
                               in_node_dim = current_dim,
                               out_node_dim = cfg.hidden_size,
                               hidden_size = cfg.hidden_size)

            # EdgeNet returns [E, in_channels * hidden_size]
            # NNConv reshapes to [E, in_channels, hidden_size]
            conv = NNConv(in_channels = current_dim,
                          out_channels = cfg.hidden_size,
                          nn = edge_mlp,
                          aggr = cfg.aggr)
            conv_layers.append(conv)
            norm_layers.append(GraphNorm(cfg.hidden_size))  # per-graph stats
            current_dim = cfg.hidden_size

        self.convs = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norm_layers)

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _validate_input(self, data: Data) -> None:
        """Validate input data dimensions and types. Only runs once on first forward pass."""
        if self._validated_input:
            return

        must_have = ["node_feat", "edge_feat", "edge_index", "edge_org",
                     "time", "node_fields", "sim_params", "step_params"]

        for k in must_have:
            if not hasattr(data, k):
                raise ValueError(f"Data must have {k} attribute")

        # Shapes & dtypes
        if not (isinstance(data.edge_index, torch.Tensor) and data.edge_index.ndim == 2 and data.edge_index.size(0) == 2):
            raise ValueError(f"edge_index must be [2, E], got {getattr(data.edge_index, 'shape', None)}")
        if data.edge_index.dtype != torch.long:
            raise ValueError(f"edge_index must be torch.long, got {data.edge_index.dtype}")

        if data.node_feat.size(1) != self.cfg.node_feat_dim:
            raise ValueError(f"Expected node_feat dim {self.cfg.node_feat_dim}, got {data.node_feat.size(1)}")
        if data.edge_feat.size(1) != self.cfg.edge_feat_dim:
            raise ValueError(f"Expected edge_feat dim {self.cfg.edge_feat_dim}, got {data.edge_feat.size(1)}")

        if data.edge_org.numel() == 0:
            raise ValueError("edge_org is empty")
        if data.edge_org.dtype != torch.long:
            raise ValueError(f"edge_org must be torch.long, got {data.edge_org.dtype}")
        if data.edge_org.max() >= self.cfg.num_org_types:
            raise ValueError(f"Edge organ type index {data.edge_org.max()} >= num_org_types {self.cfg.num_org_types}")

        # time must be scalar or 1D
        if not isinstance(data.time, torch.Tensor):
            raise ValueError("time must be a Tensor")
        if data.time.ndim > 1:
            raise ValueError(f"time must be scalar or 1D, got shape {tuple(data.time.shape)}")

        self._validated_input = True
        print("Input validation successful: data format matches model configuration.")

    def to(self, device):
        """Move the model and its scalers to the specified device."""
        if hasattr(self, 'feature_scaler') and self.feature_scaler is not None:
            self.feature_scaler.to(device)
        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            self.target_scaler.to(device)
        if hasattr(self, 'time_scaler') and self.time_scaler is not None:
            self.time_scaler.to(device)
        if hasattr(self, 'edge_scaler') and self.edge_scaler is not None:
            self.edge_scaler.to(device)
        return super().to(device)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model.

        IMPORTANT: This method modifies the input data object by adding data.time_node,
        which is essential for physics-informed loss computation. Always use the same
        data object for both model forward pass and physics_residual calculation.

        Args:
            data: Graph data object containing node features, edge features, topology and time

        Returns:
            torch.Tensor: Predicted sucrose concentration for each node [N, 1]
        """
        self._validate_input(data)
        device = next(self.parameters()).device

        # Assert critical fields
        if not hasattr(data, 'time'):
            raise ValueError("Missing graph-level time. data.time is required.")

        node_feat: torch.Tensor = data.node_feat.to(device)
        edge_index: torch.Tensor = data.edge_index.to(device)
        edge_feat: torch.Tensor = data.edge_feat.to(device)
        edge_org: torch.Tensor = data.edge_org.to(device)
        time = data.time.to(device)

        if time.dim() == 0:
            time = time.unsqueeze(0)

        if self.time_scaler is None or getattr(self.time_scaler, "mean", None) is None:
            raise RuntimeError(
                "Missing or unfitted time_scaler. "
                "A fitted Standardizer is required for consistent d/dt computation."
            )
        if self.edge_scaler is None or getattr(self.edge_scaler, "mean", None) is None:
            raise RuntimeError(
                "Missing or unfitted edge_scaler. "
                "A fitted Standardizer is required for consistent edge feature scaling."
            )

        # Capture the std used for scaling so we can convert d/dτ -> d/dt later
        time_scaled = self.time_scaler.transform(time.view(-1, 1)).view(-1)
        time_std_scalar = self.time_scaler.std.view(-1)[0]

        # Create per-node time
        if hasattr(data, 'batch') and data.batch is not None:
            time_node = time_scaled[data.batch].clone()
            time_std_node = time_std_scalar.expand(time_node.size(0))
        else:
            time_node = time_scaled.expand(node_feat.size(0)).clone()
            time_std_node = time_std_scalar.expand(time_node.size(0))

        # Ensure time has gradients and reshape
        time_node = time_node.view(-1, 1)                  # [N, 1]
        time_node.requires_grad_(True)                     # make time differentiable per node
        data.time_node = time_node                         # scaled time (τ)
        data.time_std_node = time_std_node.view(-1, 1)     # σ_t per node (for ∂/∂t conversion)

        # Concatenate as extra channel
        node_feat = torch.cat([node_feat, time_node], dim=1)

        # Optionally standardize continuous edge features (e.g., r_ST) before EdgeNet
        edge_feat = self.edge_scaler.transform(edge_feat)

        # Pre-allocate tensor for edge features (continuous + categorical)
        edge_features = torch.empty(
            edge_feat.size(0), edge_feat.size(1) + 1,
            device=device, dtype=torch.float32
        )  # [E, D + 1] where D = number of continuous edge features

        edge_features[:, :-1] = edge_feat
        edge_features[:, -1] = edge_org.to(torch.long)

        batch_vec = getattr(data, "batch", None)

        # Iterate over all convolutional blocks (NNConv + GraphNorm)
        for conv, bn in zip(self.convs, self.norms):
            # Apply edge-conditioned convolution (message passing step)
            h = conv(node_feat, edge_index, edge_features)

            # Apply per-graph normalization
            if batch_vec is None:
                # Single-graph case: create a fake batch (all nodes belong to graph 0)
                fake_batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
                h = bn(h, fake_batch)
            else:
                # Batched case: use actual graph assignments
                h = bn(h, batch_vec)

            h = F.relu(h)
            h = self.dropout(h)

            # Residual connection: if input and output dims match (after first layer),
            # add skip connection
            if h.shape == node_feat.shape:
                node_feat = node_feat + h
            else:
                node_feat = h

        # Final MLP head to produce scalar output per node
        out = self.head(node_feat)

        return out


def physics_residual(y_pred: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute physics-informed residual term based on sucrose transport equations.

    Implements the governing equation:
    ds_{st}/dt = J_ax + (F_in - F_out)

    where:
    - J_ax is the axial sucrose flux
    - F_in is the phloem loading rate
    - F_out is the sucrose outflow

    IMPORTANT: y_pred MUST come from a model forward pass using the same data object,
    so that data.time_node is properly connected to y_pred in the computation graph.
    Without this connection, ds/dt cannot be computed and the physics constraint is meaningless.

    Args:
        y_pred: Predicted sucrose content [N, 1] - MUST be connected to data.time_node
        data: Graph data containing topology, features, simulation parameters, and node fields

    Returns:
        torch.Tensor: Physics residual loss term

    Raises:
        ValueError: If y_pred is not connected to data.time_node in the computation graph
    """
    device = y_pred.device

    # Get edge topology and features
    edge_index = data.edge_index.to(device)  # [2, E]

    src, dst = edge_index[0], edge_index[1]
    r_ST = data.edge_feat.to(device).squeeze(-1)  # [E, 1] -> [E]

    # Node features already in original space
    node_feat = data.node_feat.to(device)
    psi = node_feat[:, 0]
    vol_ST = node_feat[:, 1]
    len_leaf = node_feat[:, 2]
    Q_Rmmax = node_feat[:, 3]
    Q_Grmax = node_feat[:, 4]
    Q_Exudmax = node_feat[:, 5]
    Temp = node_feat[:, 6]

    batch_vec = getattr(data, "batch", None)

    # Extract sim_parms and step_params (see parameters list in dataset_loader.py)
    params = extract_parameters(data, device, batch_vec, y_pred.size(0))

    # Extract node fields (see node_fields_names in dataset_loader.py)
    node_fields = extract_node_fields(data, device)

    # Sucrose at endpoints
    s_i = y_pred[src, 0]
    s_j = y_pred[dst, 0]

    # Water potential differences
    psi_i = psi[src]
    psi_j = psi[dst]

    # Jw < 0: psi_i > psi_j (physical flow src -> dst (arrow direction))
    # Jw > 0: psi_j > psi_i (physical flow dst -> src (arrow direction))
    Jw = psi_j - psi_i

    # (Optional / debug) Incidence matrices aligned by physics; not used in residual here
    # delta2_physics = create_delta2(data, psi)

    # Use the upstream node’s sucrose for the advective term:
    # If Jw < 0 (src -> dst), upstream is src, thus pick s_i
    # If Jw > 0 (dst -> src), upstream is dst, thus pick s_j
    s_ij = torch.where(Jw < 0, s_i, s_j)

    # Sign choice below makes J_ax > 0 correspond to flow src -> dst (arrow direction)
    # Jax > 0: Flow along edge direction (src → dst)
    # Jax < 0: Flow opposite to edge direction (dst → src)
    # If psi_i > psi_j (and/or s_i > s_j), driving > 0, giving J_ax > 0 (src -> dst).
    RT_i = R * (Temp[src] + 273.15)  # Use temperature at source nodes for each edge
    J_ax = (1 / r_ST) * s_ij * (RT_i * (s_i - s_j) + (psi_i - psi_j))

    # Divergence of flux -> net inflow per node
    # This computes the sum of incoming/outgoing fluxes for each node
    # dst node accumulates +J_ax   (incoming)
    # src node accumulates -J_ax   (outgoing)
    N = y_pred.size(0)
    dS_dt_from_flux = torch.zeros(N, device=device)
    dS_dt_from_flux.scatter_add_(0, dst, J_ax)   # Add incoming fluxes
    dS_dt_from_flux.scatter_add_(0, src, -J_ax)  # Subtract outgoing fluxes

    # We need ds/dt from the model with respect to a differentiable time feature
    # This is ESSENTIAL for physics-informed learning: without it, the physics constraint is meaningless
    if not hasattr(data, 'time_node') or data.time_node is None:
        raise ValueError("data.time_node not found. data.time_node is required for physics residual computation.")
    if not data.time_node.requires_grad:
        raise ValueError("data.time_node must have requires_grad=True for physics residual computation.")
    if not hasattr(data, "time_std_node") or data.time_std_node is None:
        raise ValueError("data.time_std_node missing; ensure model.forward() sets it.")

    # Compute gradient of predictions w.r.t. **scaled** time_node τ [N,1]
    # We'll convert to real time derivative via  ∂/∂t = (1/σ_t) ∂/∂τ
    try:
        ds_dt = torch.autograd.grad(
            y_pred.sum(),        # sum to get scalar for gradient computation. trick to avoid building the full Jacobian
            data.time_node,      # [N, 1] per-node time features
            create_graph=True,   # needed for second backward pass
            retain_graph=True,   # keep graph for subsequent loss computation
            allow_unused=False   # ERROR if time_node is not connected - this is required for physics
        )[0]
        ds_dt = ds_dt.squeeze()
    except RuntimeError as e:
        if "not have been used in the graph" in str(e):
            raise ValueError(
                "Physics residual computation failed: y_pred is not connected to time_node. "
                "This indicates that the model predictions don't depend on time, which breaks the physics constraint. "
                "Ensure that y_pred comes from a model forward pass that uses the same data object, "
                "or that the model architecture properly utilizes the time feature."
            ) from e
        else:
            raise

    # Convert ∂/∂τ to ∂/∂t using stored σ_t (τ = (t - μ_t)/σ_t)
    ds_dt /= data.time_std_node.squeeze()  # now ds_dt is in real time units

    # CSTi is the sucrose concentration in sieve tube
    CSTi = y_pred.squeeze(-1) / vol_ST
    CSTi_positive = torch.clamp(CSTi, min=0.0)  # ensure non-negative concentrations

    # F_in: phloem loading term per node
    # IMPORTANT: Uses original CSTi_positive (before CSTimin threshold) as per original code
    F_in = (params["Vmaxloading"] * len_leaf) * node_fields["C_meso"] / (params["Mloading"] + node_fields["C_meso"]) * torch.exp(-CSTi_positive * params["beta_loading"])

    # F_out: sucrose outflow from sieve tubes (uses CSTi_thresholded for usage)
    # F_out = F_out_MM + Exud
    # where F_out_MM = (R_mmax + Q_Grmax) * (CSTi / (CSTi + KMfu))
    # Apply CSTimin threshold: if CSTi < CSTimin, no sucrose usage
    CSTi_effective = torch.clamp(CSTi_positive - params["CSTimin"], min=0.0)
    CSTi_delta = torch.clamp(CSTi_effective - node_fields["Csoil_node"], min=0.0)

    # R_mmax = Q_Rmmax_ (PiafMunch2.cpp)
    R_mmax = (Q_Rmmax + params["krm2v"] * CSTi_effective) * torch.pow(params["Q10"], (Temp - params["TrefQ10"]) / 10.0)

    # Sucrose usage rate for growth + maintenance
    # F_out_MM = Fu_lim (PiafMunch2.cpp)
    F_out_MM = (R_mmax + Q_Grmax) * (CSTi_effective / (CSTi_effective + params["KMfu"]))

    # Root exudation rate (passive transport based on concentration gradient)
    Exud = CSTi_delta * Q_Exudmax

    # Total outflow
    F_out = F_out_MM + Exud

    # Total rate of change from physics: axial transport + phloem loading - outflow
    dS_dt_from_physics = dS_dt_from_flux + F_in - F_out

    # Compute residual as the difference between model derivative and physics derivative
    residual_node = (ds_dt.squeeze() - dS_dt_from_physics).pow(2)

    # Average per graph first (so each graph contributes equally),
    # then average across graphs. Fall back to simple mean if no batch.
    if batch_vec is not None:
        residual_per_graph = scatter_mean(residual_node, batch_vec, dim=0)
        loss = residual_per_graph.mean()
    else:
        loss = residual_node.mean()

    return loss