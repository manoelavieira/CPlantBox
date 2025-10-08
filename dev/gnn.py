"""
Phloem GNN (baseline) — NNConv with edge features
-------------------------------------------------
Predicts sucrose concentration per node at timestep t, given:
- Plant topology (edge_index)
- Node features at time t: water potential (psi), sieve-tube volume (vol_st)
- Edge features at time t: sieve-tube resistance (r_st), organ type (categorical)


Expected `Data` fields per graph (per timestep)
----------------------------------------------
- data.edge_index: LongTensor  [2, E]
- data.edge_feat:  FloatTensor [E, 1]      # r_st (resistance)
- data.node_feat:  FloatTensor [N, 2]      # [psi, vol_st]
- data.time:       FloatTensor [1]         # time in days (graph-level)
- data.y:          FloatTensor [N, 1]      # target sucrose at t
- Optional: data.batch for mini-batching multiple graphs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import NNConv
import warnings

# -----------------------------
# Small utilities
# -----------------------------
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
        # Ensure mean/std are on the same device as X
        if X.device != self.device:
            self.to(X.device)
        return (X - self.mean) / self.std

    def inv_transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return X
        # Ensure mean/std are on the same device as X
        if X.device != self.device:
            self.to(X.device)
        return X * self.std + self.mean

    def to(self, device: torch.device) -> 'Standardizer':
        """Move internal tensors to the specified device.

        Args:
            device: The device to move the tensors to

        Returns:
            self: The Standardizer instance for method chaining
        """
        if not isinstance(device, (torch.device, str)):
            raise TypeError(f"device must be torch.device or str, got {type(device)}")

        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        self.device = torch.device(device)
        return self

# -----------------------------
# Model
# -----------------------------
@dataclass
class ModelConfig:
    """Configuration for PhloemNNConv model.

    Attributes:
        node_feat_dim: Dimension of continuous node features [psi, vol_st]
        num_org_types: Number of organ types [LEAF, STEM, ROOT]
        org_emb_size: Dimension of organ type embeddings
        hidden_size: Dimension of hidden layers in NNConv/MLPs
        num_layers: Number of NNConv layers
        edge_feat_dim: Dimension of continuous edge features [r_st]
        aggr: NNConv aggregator type ("add", "mean", or "max")
        dropout: Dropout probability
    """
    node_feat_dim: int = 2 # [psi, vol_st]
    edge_feat_dim: int = 1 # [r_st]
    num_org_types: int = 3
    org_emb_size: int = 8 # embedding dimension for categorical organ type
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
        if any(d < 1 for d in [self.node_feat_dim, self.num_org_types, self.org_emb_size,
                               self.hidden_size, self.edge_feat_dim]):
            raise ValueError("All dimensions must be positive integers")


class EdgeNet(nn.Module):
    """Edge MLP producing weight matrices for NNConv.

    Maps edge features (continuous + organ type) -> [E, out_channels * in_channels].
    MLP that turns per-edge features into a per-edge weight matrix W_e that NNConv
    will use to transform neighbor node features.
    """
    def __init__(self, edge_feat_dim: int, num_org_types: int, org_emb_size: int,
                 in_node_dim: int, out_node_dim: int, hidden_size: int = 64):
        super().__init__()

        self.in_node_dim = in_node_dim
        self.out_node_dim = out_node_dim

        self.org_emb = nn.Embedding(num_org_types, org_emb_size)
        edge_input_dim = edge_feat_dim + org_emb_size

        # Combined MLP for both continuous and embedded features
        # It outputs a flattened weight matrix of size [out_channels * in_channels] per edge
        self.mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            # Final linear layer has no activation
            # We want raw learned weights, not squashed by ReLU/sigmoid
            nn.Linear(hidden_size, out_node_dim * in_node_dim)
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        # edge_features: [E, D+1] where D is edge_feat_dim and last column is organ type
        edge_feat = edge_features[:, :-1]  # continuous features
        edge_org = edge_features[:, -1].long()  # organ type as long tensor (int64)

        # Combine continuous edge features with organ embeddings
        edge_emb = self.org_emb(edge_org)
        edge_inputs = torch.cat([edge_feat, edge_emb], dim=-1)

        return self.mlp(edge_inputs)


class PhloemNNConv(nn.Module):
    """Neural network model for phloem flow prediction using NNConv layers.

    Combines node features (continuous and organ type) with edge features
    through multiple NNConv layers to predict sucrose concentration.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Initialize all scalers as None
        self.feature_scaler = None  # for node features
        self.target_scaler = None   # for output values
        self.time_scaler = None     # for time normalization
        self._validated_input = False  # Track if input has been validated

        # Node input = continuous node features + (scaled) time
        in_node_dim = cfg.node_feat_dim + 1
        current_dim = in_node_dim

        conv_layers = []
        norm_layers = []
        for _ in range(cfg.num_layers):
            edge_mlp = EdgeNet(edge_feat_dim=cfg.edge_feat_dim,
                               num_org_types=cfg.num_org_types,
                               org_emb_size=cfg.org_emb_size,
                               in_node_dim=current_dim,
                               out_node_dim=cfg.hidden_size,
                               hidden_size=cfg.hidden_size)

            # EdgeNet returns [E, c_in * hidden_size]
            # NNConv reshapes to [E, c_in, hidden_size]
            conv = NNConv(in_channels=current_dim,
                          out_channels=cfg.hidden_size,
                          nn=edge_mlp,
                          aggr=cfg.aggr)
            conv_layers.append(conv)
            norm_layers.append(nn.BatchNorm1d(cfg.hidden_size))
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
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _validate_input(self, data: Data) -> None:
        """Validate input data dimensions and types. Only runs once on first forward pass."""
        if self._validated_input:
            return

        if not hasattr(data, 'node_feat'):
            raise ValueError("Data must have node_feat attribute")
        if data.node_feat.size(1) != self.cfg.node_feat_dim:
            raise ValueError(f"Expected node_feat dim {self.cfg.node_feat_dim}, got {data.node_feat.size(1)}")
        if data.edge_feat.size(1) != self.cfg.edge_feat_dim:
            raise ValueError(f"Expected edge_feat dim {self.cfg.edge_feat_dim}, got {data.edge_feat.size(1)}")
        if not hasattr(data, 'edge_org'):
            raise ValueError("Data must have edge_org attribute")
        if data.edge_org.max() >= self.cfg.num_org_types:
            raise ValueError(f"Edge organ type index {data.edge_org.max()} >= num_org_types {self.cfg.num_org_types}")

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
        return super().to(device)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            data: Graph data object containing node features, edge features, topology and time

        Returns:
            torch.Tensor: Predicted sucrose concentration for each node [N, 1]
        """
        self._validate_input(data)
        device = next(self.parameters()).device

        node_feat: torch.Tensor = data.node_feat.to(device) # [N, 2] [psi, vol_st]
        edge_index: torch.Tensor = data.edge_index.to(device) # [2, E] graph connectivity (sources, targets indices)
        edge_feat: torch.Tensor = data.edge_feat.to(device)  # [E, De] continuous edge features
        edge_org: torch.Tensor = data.edge_org.to(device)  # [E] categorical organ type per edge

        # Graph-level time handling
        if not hasattr(data, 'time'):
            raise ValueError("Missing graph-level time. data.time is required.")
        t = data.time.to(device)

        # Create differentiable per-node time feature
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Standardize time if scaler available
        if self.time_scaler is not None and getattr(self.time_scaler, "mean", None) is not None:
            t_scaled = self.time_scaler.transform(t.view(-1, 1)).view(-1)
        else:
            t_scaled = t

        # Create per-node time with requires_grad=True
        if hasattr(data, 'batch') and data.batch is not None:
            # For batched graphs
            t_node = t_scaled[data.batch].clone()
        else:
            # Single graph
            t_node = t_scaled.expand(node_feat.size(0)).clone()

        # Ensure time has gradients and reshape
        t_node = t_node.view(-1, 1)     # [N, 1]
        t_node.requires_grad_(True)     # make time differentiable per node
        data.time_node = t_node         # store time in data for physics computation

        # Concatenate as extra channel
        node_feat = torch.cat([node_feat, t_node], dim=1)  # [N, 3]

        # Pre-allocate tensor for edge features
        edge_features = torch.empty(edge_feat.size(0), edge_feat.size(1) + 1,
                                    device=device, dtype=torch.float32)
        edge_features[:, :-1] = edge_feat
        edge_features[:, -1] = edge_org.float()

        # stack NNConv layers with residual connections
        for conv, bn in zip(self.convs, self.norms):
            # Process combined edge features through NNConv
            h = conv(node_feat, edge_index, edge_features)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

            # residual if dimensions match (for layers after first)
            if h.shape == node_feat.shape:
                node_feat = node_feat + h
            else:
                node_feat = h

        out = self.head(node_feat) # [N, 1]
        return out


# -----------------
# Physics hook
# -----------------
def create_delta2(data: Data, psi: torch.Tensor, align_to_upstream: bool = True,
                  dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
    """Create a flow-aligned incidence matrix per-edge using node potentials `psi`.

    For each edge e (src->dst) we compute Jw = psi_j - psi_i. The sign of Jw indicates
    the flow proxy. If `align_to_upstream=True`, columns are multiplied so that the +1
    entry lies on the upstream node (the node where the fluid originates). If False,
    +1 will be placed on the downstream node.

    Args:
        data: torch_geometric.data.Data with `edge_index` and optional `batch`/`num_graphs`.
        psi: per-node potential tensor of shape [N] (or [N,] float tensor).
        align_to_upstream: if True, +1 is on upstream node (source of flow); else +1 on downstream.
        dtype, device: optional dtype/device for outputs.

    Returns:
        sparse_coo_tensor or list of such (or dense tensors if as_dense=True).
    """
    if device is None:
        device = data.edge_index.device if hasattr(data, 'edge_index') else torch.device('cpu')

    edge_index = data.edge_index.to(device)
    psi = psi.to(device)

    N_total = int(edge_index.max().item() + 1)

    src = edge_index[0]
    dst = edge_index[1]

    # Flow proxy Jw and its sign
    # If Jw > 0, flow is dst -> src (dst is upstream); If Jw < 0, flow is src -> dst (src is upstream)
    Jw = psi[dst] - psi[src]
    sign = torch.sign(Jw)

    # Replace zeros (no gradient) with +1 to keep a deterministic orientation
    sign[sign == 0.] = 1.

    # If align_to_upstream is True, multiply by sign so that +1 is at upstream node
    # Original column has [-1 at src, +1 at dst]. Multiplying by sign gives [-sign, +sign].
    # If align_to_upstream=False, flip the sign to put +1 on downstream.
    col_multiplier = sign if align_to_upstream else -sign

    # Per-graph signed incidence
    batch = data.batch.to(device)
    edge_graph = batch[src]
    num_graphs = int(batch.max().item() + 1)
    out = []
    for g in range(num_graphs):
        mask_nodes = (batch == g)
        node_idx = torch.nonzero(mask_nodes, as_tuple=False).view(-1) # torch.nonzero returns the indices where mask_nodes is True
        N_g = node_idx.numel()

        mask_edges = (edge_graph == g)
        E_g = int(mask_edges.sum().item())

        # local mapping
        local_map = torch.full((N_total,), -1, device=device, dtype=torch.long)
        local_map[node_idx] = torch.arange(N_g, device=device, dtype=torch.long) # torch.arange(N_g) generates [0, 1, 2, ..., N_g-1].

        src_g = local_map[src[mask_edges]]
        dst_g = local_map[dst[mask_edges]]

        cm_g = col_multiplier[mask_edges]

        # For an N x E incidence matrix (rows=nodes, cols=edges) we need
        # indices shaped as [row_indices(node), col_indices(edge)]. Each edge
        # contributes two entries: -1 on source node row, +1 on target node row.
        row_indices = torch.cat([src_g, dst_g], dim=0)               # node rows
        col_indices = torch.cat([torch.arange(E_g, device=device), torch.arange(E_g, device=device)], dim=0)  # edge cols

        # Optionally sort by column (edge) for nicer ordering/debugging
        perm = torch.argsort(col_indices)
        row_indices = row_indices[perm]
        col_indices = col_indices[perm]

        indices = torch.stack([row_indices, col_indices], dim=0)  # [2, nnz]

        vals = torch.cat([-cm_g, cm_g], dim=0).to(torch.float32)
        vals = vals[perm]

        # Create sparse tensor with shape (N_g, E_g): rows=nodes, cols=edges
        sparse_g = torch.sparse_coo_tensor(indices, vals, size=(N_g, E_g), dtype=torch.float32, device=device)
        # print(f"[GNN][CREATE_DELTA2] Graph {g}: N_g={N_g}, E_g={E_g}, nnz: {sparse_g._nnz()}")
        # print(f"[GNN][CREATE_DELTA2] Indices sample: {sparse_g._indices()}")
        out.append(sparse_g)

    return out


def physics_residual(y_pred: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute physics-informed residual term based on sucrose transport equations.

    Implements the simplified governing equation:
    ds_{st,q,j}/dt = J_{ax,st,j}

    Args:
        y_pred: Predicted sucrose concentrations [N, 1]
        data: Graph data containing topology and features
        model: PhloemNNConv model instance for scaling transformations

    Returns:
        torch.Tensor: Physics residual loss term
    """
    device = y_pred.device

    # Constants (should be loaded from H5 file in practice)
    R = 8.314  # Gas constant [J/(mol·K)]
    T = 298.15  # Temperature [K]
    RT = R * T

    # Get edge topology and features
    edge_index = data.edge_index.to(device)  # [2, E]

    src, dst = edge_index[0], edge_index[1]
    r_st = data.edge_feat.to(device).squeeze(-1)  # [E, 1] resistance terms (K_ax/L)

    node_graph = data.batch  # graph index per node
    edge_graph = data.batch[src]  # graph index per edge (edges never connect nodes from different graphs)
    # for g in range(data.num_graphs):
    #     mask_nodes = (node_graph == g)
    #     mask_edges = (edge_graph == g)

    # Node features (already in original space)
    node_feat = data.node_feat.to(device)
    psi = node_feat[:, 0]

    # Sucrose at endpoints (original units)
    s_i = y_pred[src, 0]
    s_j = y_pred[dst, 0]

    # Water potential differences
    psi_i = psi[src]
    psi_j = psi[dst]

    # Flow direction proxy: Jw < 0 means i -> j; Jw > 0 means j -> i
    Jw = psi_j - psi_i

    deltas_physics = create_delta2(data, psi)

    # Upwind sucrose (take upstream node’s sucrose)
    # s_ij is the sucrose value of the node where the fluid originates (the source/upstream node)
    s_ij = torch.where(Jw < 0, s_i, s_j)

    for g in range(data.num_graphs):
        mask_edges = (edge_graph == g)
        s_ij_g = s_ij[mask_edges]

        # print(f"s_ij for graph {g}: {s_ij_g}")

    # Compute axial flux J_ax for each edge
    # J_{ax} = s_ij * r_st * (RT * (s_j - s_i) + (psi_j - psi_i))
    J_ax = s_ij * r_st * (RT * (s_j - s_i) + (psi_j - psi_i))

    # Divergence of flux -> net inflow per node
    # This computes the sum of incoming/outgoing fluxes for each node
    N = y_pred.size(0)
    dS_dt_from_flux = torch.zeros(N, device=device)
    dS_dt_from_flux.scatter_add_(0, dst, J_ax)   # Add incoming fluxes
    dS_dt_from_flux.scatter_add_(0, src, -J_ax)  # Subtract outgoing fluxes

    # Get time derivatives using per-node time gradients
    if hasattr(data, 'time_node'):
        # Compute gradient of predictions w.r.t. time_node [N,1]
        ds_dt = torch.autograd.grad(
            y_pred.sum(),        # sum to get scalar for gradient computation
            data.time_node,      # [N,1] per-node time features
            create_graph=True,   # needed for second backward pass
            retain_graph=True    # keep graph for subsequent loss computation
        )[0].squeeze()
    else:
        raise ValueError("data.time_node not found. data.time_node is required for physics residual computation.")

    # Compute residual as the difference between computed derivatives
    residual = (ds_dt.squeeze() - dS_dt_from_flux)**2

    return residual.mean()
