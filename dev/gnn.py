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

    def forward(self, data: Data, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            data: Graph data object containing node features, edge features, and topology

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
        # Accept t from argument, or read from data.time (required if t is None)
        if t is None:
            if not hasattr(data, 'time'):
                raise ValueError("Missing graph-level time. Provide t or set data.time.")
            t = data.time
        t = t.to(device)

        # Ensure shape [num_graphs] for batching logic
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Standardize time if a scaler is available (keep it differentiable)
        if self.time_scaler is not None and getattr(self.time_scaler, "mean", None) is not None:
            t_scaled = self.time_scaler.transform(t.view(-1, 1)).view(-1)
        else:
            t_scaled = t

        # Broadcast per-graph time to per-node time
        if hasattr(data, 'batch') and data.batch is not None:
            t_node = t_scaled[data.batch]
        else:
            t_node = t_scaled.expand(node_feat.size(0))

        # Concatenate as an extra channel (keep gradient w.r.t. t_node)
        node_feat = torch.cat([node_feat, t_node.view(-1, 1)], dim=1) # [N, 3]

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


# -----------------------------------------
# Physics hook (non operational for now)
# -----------------------------------------
def physics_residual(y_pred: torch.Tensor, data: Data) -> torch.Tensor:
    """Placeholder for future physics-informed residuals.
    Currently returns 0.0 and issues a warning. When implemented, will compute
    physical residuals using the predicted values and data fields.

    Args:
        y_pred: Predicted values from the model
        data: Graph data object containing features and topology

    Returns:
        torch.Tensor: Physics-based residual term (currently 0.0)
    """
    warnings.warn(
        "physics_residual() is not yet implemented and returns 0.0. "
        "Physical constraints are not being enforced.",
        RuntimeWarning, stacklevel=2
    )
    return torch.tensor(0.0, device=y_pred.device)