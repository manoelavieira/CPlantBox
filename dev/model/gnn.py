"""
Phloem GNN (baseline) — NNConv with edge features
-------------------------------------------------
Predicts sucrose content per node at timestep t, given:
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from torch_geometric.nn.norm import GraphNorm  # per-graph normalization

from .config import ModelConfig
import math

class EdgeNet(nn.Module):
    """Edge MLP producing weight matrices for NNConv.

    Maps edge features (continuous + categorical) -> [E, out_channels * in_channels].
    MLP that turns per-edge features into a per-edge weight matrix W_e that NNConv
    will use to transform neighbor node features.
    """
    def __init__(self, edge_feat_cont_dim: int, num_org_types: int,
                 in_node_dim: int, out_node_dim: int, hidden_size: int = 64):
        super().__init__()

        self.in_node_dim = in_node_dim
        self.out_node_dim = out_node_dim
        self.num_org_types = num_org_types

        edge_input_dim = edge_feat_cont_dim + num_org_types

        # Combined MLP for both continuous and categorical features
        # It outputs a flattened weight matrix of size [out_channels * in_channels] per edge
        self.mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            # Final linear layer has no activation
            # We want raw learned weights, not squashed by ReLU/sigmoid
            nn.Linear(hidden_size, in_node_dim * out_node_dim)
        )

        # Ensure EdgeNet is in float64 to match data loader dtype
        self.double()
        self.mlp.double()

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        # edge_features: [E, D+1] where D is edge_feat_cont_dim and last column is organ type
        edge_feat_cont = edge_features[:, :-1]  # continuous features
        edge_feat_cat = edge_features[:, -1].long()  # organ type as long tensor (int64)

        # Convert organ type to one-hot encoding
        edge_one_hot = F.one_hot(edge_feat_cat, num_classes=self.num_org_types).to(edge_features.dtype)

        # Combine continuous edge features with one-hot organ type
        edge_inputs = torch.cat([edge_feat_cont, edge_one_hot], dim=-1)

        # Ensure dtype consistency by converting to model weights dtype
        model_dtype = next(self.mlp.parameters()).dtype
        edge_inputs = edge_inputs.to(dtype=model_dtype)

        return self.mlp(edge_inputs)


class PhloemNNConv(nn.Module):
    """Neural network model for phloem flow prediction using NNConv layers.

    Combines node features with edge features through multiple NNConv
    layers to predict sucrose content.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self._validated_input = False   # Track if input has been validated

        # Node input = continuous node features + time
        in_node_dim = cfg.node_feat_dim + 1
        current_dim = in_node_dim

        conv_layers = []
        norm_layers = []
        for _ in range(cfg.num_layers):
            edge_mlp = EdgeNet(edge_feat_cont_dim=cfg.edge_feat_dim,
                               num_org_types=cfg.num_org_types,
                               in_node_dim=current_dim,
                               out_node_dim=cfg.hidden_size,
                               hidden_size=cfg.hidden_size)

            # EdgeNet returns [E, in_channels * hidden_size]
            # NNConv reshapes to [E, in_channels, hidden_size]
            conv = NNConv(in_channels=current_dim,
                          out_channels=cfg.hidden_size,
                          nn=edge_mlp,
                          aggr=cfg.aggr)
            conv_layers.append(conv)
            norm_layers.append(GraphNorm(cfg.hidden_size))  # per-graph stats
            current_dim = cfg.hidden_size

        self.convs = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norm_layers)

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        # Learnable output gain for softplus scaling; init near target magnitude
        self.log_alpha = nn.Parameter(torch.tensor(math.log(5e-5), dtype=torch.float64))

        self.dropout = nn.Dropout(cfg.dropout)
        self._init_weights()

        # Ensure model is in float64 to match data loader dtype
        self.double()

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
                     "node_fields", "sim_params", "step_params",]

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
        if data.edge_org.max().item() >= self.cfg.num_org_types:
            raise ValueError(
                f"Edge organ type index {data.edge_org.max().item()} >= num_org_types {self.cfg.num_org_types}"
            )

        if data.time_per_node is None:
            raise ValueError("time_per_node is required for physics-informed loss computation")
        if data.time_per_node.ndim != 2 or data.time_per_node.size(1) != 1:
            raise ValueError(f"time_per_node must be [N,1], got {tuple(data.time_per_node.shape)}")

        self._validated_input = True
        print("Input validation successful: data format matches model configuration.")

    def to(self, device):
        """Move the model to the specified device."""
        return super().to(device)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model.

        IMPORTANT: This method expects data.time_per_node to be present;
        which is essential for physics-informed loss computation. Always use the same
        data object for both model forward pass and physics_residual calculation.

        Args:
            data: Graph data object containing node features, edge features, topology and time

        Returns:
            torch.Tensor: Predicted sucrose content for each node [N, 1]
        """
        self._validate_input(data)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Ensure inputs match model device and dtype (model is set to double())
        node_feat: torch.Tensor = data.node_feat.to(device=device, dtype=dtype)
        edge_index: torch.Tensor = data.edge_index.to(device)
        edge_feat: torch.Tensor = data.edge_feat.to(device=device, dtype=dtype)
        edge_org: torch.Tensor = data.edge_org.to(device)
        time_per_node: torch.Tensor = data.time_per_node.to(device=device, dtype=dtype)

        # Shape checks
        if time_per_node.dim() != 2 or time_per_node.size(1) != 1 or time_per_node.size(0) != node_feat.size(0):
            raise RuntimeError(
                f"`time_per_node` must be [N,1]; got {tuple(time_per_node.shape)} with N={node_feat.size(0)}."
            )

        # Concatenate as extra channel
        node_feat = torch.cat([node_feat, time_per_node], dim=1)

        # Pre-allocate tensor for edge features (continuous + categorical)
        # Use the same dtype as edge_feat to maintain consistency
        edge_features = torch.empty(
            edge_feat.size(0), edge_feat.size(1) + 1,
            device=device, dtype=dtype
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

        # Positive output with learnable scale
        out = torch.exp(self.log_alpha) * F.softplus(out)

        # For sharper non-negativity barrier, put the scale inside Softplus
        # out = F.softplus(torch.exp(self.log_alpha) * out)

        return out