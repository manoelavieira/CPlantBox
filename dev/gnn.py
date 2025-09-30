"""
Phloem GNN (baseline) — NNConv with edge features
-------------------------------------------------
Predicts sucrose concentration per node at timestep t, given:
- Plant topology (edge_index)
- Node features at time t: water potential (psi), sieve-tube volume (vol_st)
- Edge features at time t: sieve-tube resistance (r_st), organ type (categorical)


Expected `Data` fields per graph (per timestep)
----------------------------------------------
- data.edge_index: LongTensor [2, E]
- data.edge_attr:  FloatTensor [E, 1]      # r_st (resistance)
- data.x_cont:     FloatTensor [N, 3]      # [psi, vol_st, time]  (time in days)
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
        x_cont_dim: Dimension of continuous node features [psi, vol_st, time]
        n_org_types: Number of organ types [LEAF, STEM, ROOT]
        org_emb_dim: Dimension of organ type embeddings
        hidden_dim: Hidden dimension in neural networks
        n_layers: Number of NNConv layers
        edge_cont_dim: Dimension of continuous edge features [r_st]
        aggr: NNConv aggregator type ("add", "mean", or "max")
        dropout: Dropout probability
    """
    x_cont_dim: int = 3  # [psi, vol_st, time]
    n_org_types: int = 3
    org_emb_dim: int = 8
    hidden_dim: int = 64
    n_layers: int = 3
    edge_cont_dim: int = 1
    aggr: str = "add"
    dropout: float = 0.0

    def __post_init__(self):
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Dropout must be between 0 and 1, got {self.dropout}")
        if self.n_layers < 1:
            raise ValueError(f"Number of layers must be positive, got {self.n_layers}")
        if self.aggr not in ["add", "mean", "max"]:
            raise ValueError(f"Aggregator must be one of ['add', 'mean', 'max'], got {self.aggr}")
        if any(d < 1 for d in [self.x_cont_dim, self.n_org_types, self.org_emb_dim,
                              self.hidden_dim, self.edge_cont_dim]):
            raise ValueError("All dimensions must be positive integers")


class EdgeNet(nn.Module):
    """Edge MLP producing weight matrices for NNConv.

    Maps edge features (continuous + organ type) -> [E, out_channels * in_channels].
    MLP that turns per-edge features into a per-edge weight matrix W_e that NNConv
    will use to transform neighbor node features.
    """
    def __init__(self, edge_cont_dim: int, n_org_types: int, org_emb_dim: int,
                 in_channels: int, out_channels: int, hidden: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Organ type embedding
        self.org_emb = nn.Embedding(n_org_types, org_emb_dim)

        # Input dim is: continuous edge features + embedded organ type
        edge_feat_dim = edge_cont_dim + org_emb_dim

        # Combined MLP for both continuous and embedded features
        # It outputs a flattened weight matrix of size [out_channels * in_channels] per edge
        self.mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_channels * in_channels)
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        # edge_features: [E, D+1] where D is edge_cont_dim and last column is organ type
        edge_attr = edge_features[:, :-1]  # continuous features
        edge_org = edge_features[:, -1].long()  # organ type as long tensor

        # Combine continuous edge features with organ embeddings
        edge_emb = self.org_emb(edge_org)
        edge_feat_combined = torch.cat([edge_attr, edge_emb], dim=-1)
        return self.mlp(edge_feat_combined)


class PhloemNNConv(nn.Module):
    """Neural network model for phloem flow prediction using NNConv layers.

    Combines node features (continuous and organ type) with edge features
    through multiple NNConv layers to predict sucrose concentration.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Use continuous node features directly
        in_channels = cfg.x_cont_dim

        layers = []
        norms = []

        c_in = in_channels
        for _ in range(cfg.n_layers):
            edge_mlp = EdgeNet(edge_cont_dim=cfg.edge_cont_dim,
                               n_org_types=cfg.n_org_types,
                               org_emb_dim=cfg.org_emb_dim,
                               in_channels=c_in,
                               out_channels=cfg.hidden_dim,
                               hidden=cfg.hidden_dim)

            # EdgeNet returns [E, c_in * hidden_dim]
            # NNConv reshapes to [E, c_in, hidden_dim]
            conv = NNConv(c_in, cfg.hidden_dim, nn=edge_mlp, aggr=cfg.aggr)
            layers.append(conv)
            norms.append(nn.BatchNorm1d(cfg.hidden_dim))
            c_in = cfg.hidden_dim

        self.convs = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1)
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # Initialize weights
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
        """Validate input data dimensions and types."""
        if not hasattr(data, 'x_cont'):
            raise ValueError("Data must have x_cont attribute")
        if data.x_cont.size(1) != self.cfg.x_cont_dim:
            raise ValueError(f"Expected x_cont dim {self.cfg.x_cont_dim}, got {data.x_cont.size(1)}")
        if data.edge_attr.size(1) != self.cfg.edge_cont_dim:
            raise ValueError(f"Expected edge_attr dim {self.cfg.edge_cont_dim}, got {data.edge_attr.size(1)}")
        if not hasattr(data, 'edge_org'):
            raise ValueError("Data must have edge_org attribute")
        if data.edge_org.max() >= self.cfg.n_org_types:
            raise ValueError(f"Edge organ type index {data.edge_org.max()} >= n_org_types {self.cfg.n_org_types}")

    def to(self, device):
        """Move the model and its scalers to the specified device."""
        if hasattr(self, 'feature_scaler') and self.feature_scaler is not None:
            self.feature_scaler.to(device)
        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            self.target_scaler.to(device)
        return super().to(device)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            data: Graph data object containing node features, edge features, and topology

        Returns:
            torch.Tensor: Predicted sucrose concentration for each node [N, 1]
        """
        self._validate_input(data)
        device = next(self.parameters()).device

        x: torch.Tensor = data.x_cont.to(device) # [N, Dc] node features
        edge_index: torch.Tensor = data.edge_index.to(device) # [2, E] graph connectivity (sources, targets indices)
        edge_attr: torch.Tensor = data.edge_attr.to(device)  # [E, De] continuous edge features
        edge_org: torch.Tensor = data.edge_org.to(device)  # [E] categorical organ type per edge

        # Pre-allocate tensor for edge features
        edge_features = torch.empty(edge_attr.size(0), edge_attr.size(1) + 1,
                                  device=device, dtype=torch.float32)
        edge_features[:, :-1] = edge_attr
        edge_features[:, -1] = edge_org.float()

        # stack NNConv layers with residual connections
        for conv, bn in zip(self.convs, self.norms):
            # Process combined edge features through NNConv
            h = conv(x, edge_index, edge_features)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

            # residual if dimensions match (for layers after first)
            if h.shape == x.shape:
                x = x + h
            else:
                x = h

        out = self.head(x) # [N, 1]
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