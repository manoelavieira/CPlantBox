"""
Operator-based GNN: Message Passing as Discrete Transport Operator
-------------------------------------------------
Implements PhloemOperatorGNN where message passing layers learn edge fluxes,
compute divergences, and update node embeddings accordingly.

Expected `Data` fields per graph (per timestep)
----------------------------------------------
- data.edge_index: LongTensor  [2, E]
- data.edge_feat:  FloatTensor [E, 1]      # r_st (resistance)
- data.node_feat:  FloatTensor [N, 3]      # [psi, vol_st, len_leaf]
- data.time:       FloatTensor [1]         # time in days (graph-level)
- data.y:          FloatTensor [N, 1]      # target sucrose concentration at t
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


class FluxMessagePassing(nn.Module):
    """Operator-like message passing layer where messages are learned edge fluxes.

    This layer implements transport operator explicitly:
    - Step 1: Compute scalar flux J_ij for each edge from node pairs and edge features
    - Step 2: Compute divergence per node: div_i = sum_{j->i} J_ji - sum_{i->k} J_ik
    - Step 3: Update node embeddings: h'_i = MLP([h_i, div_i])

    This is a true discrete operator: fluxes → divergence → node update.
    We compute everything explicitly in forward() for clarity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_feat_dim: int,
        num_org_types: int,
        hidden_size: int = 64,
    ):
        """Initialize flux message passing layer.

        Args:
            in_channels: Input node feature dimension
            out_channels: Output node feature dimension
            edge_feat_dim: Continuous edge feature dimension
            num_org_types: Number of organ type categories
            hidden_size: Hidden dimension for MLPs
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_org_types = num_org_types

        # Edge input = continuous features + one-hot organ type
        edge_input_dim = edge_feat_dim + num_org_types

        # MLP to compute edge flux from node features and edge features
        # Input: [h_src, h_dst, edge_features]
        # Output: scalar flux per edge
        self.flux_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # scalar flux per edge
        )

        # MLP to update node embeddings from [h_old, divergence]
        self.node_update_mlp = nn.Sequential(
            nn.Linear(in_channels + 1, hidden_size),  # +1 for divergence
            nn.ReLU(),
            nn.Linear(hidden_size, out_channels)
        )

        self.double()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: flux → divergence → node update.

        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_features: Edge features [E, D+1] (continuous + categorical)

        Returns:
            tuple: (updated_node_features [N, out_channels], edge_fluxes [E])
        """
        N = x.size(0)
        device = x.device
        dtype = x.dtype

        # Extract source and destination nodes
        src, dst = edge_index[0], edge_index[1]

        # --- Step 1: Compute edge fluxes ---
        # Process edge features: separate continuous and categorical
        edge_feat_cont = edge_features[:, :-1]  # continuous features
        edge_feat_cat = edge_features[:, -1].long()  # organ type

        # One-hot encode organ type
        edge_one_hot = F.one_hot(edge_feat_cat, num_classes=self.num_org_types).to(dtype)

        # Combine all edge features
        edge_inputs = torch.cat([edge_feat_cont, edge_one_hot], dim=-1)

        # Get node features at edge endpoints
        x_src = x[src]  # [E, in_channels]
        x_dst = x[dst]  # [E, in_channels]

        # Compute flux from node pair and edge features
        flux_input = torch.cat([x_src, x_dst, edge_inputs], dim=-1)
        edge_fluxes = self.flux_mlp(flux_input).squeeze(-1)  # [E]

        # --- Step 2: Compute divergence per node ---
        # Divergence: net outflow - net inflow
        # For edge i->j with flux J_ij:
        #   node i: +J_ij (outgoing)
        #   node j: -J_ij (incoming)
        divergence = torch.zeros(N, device=device, dtype=dtype)
        divergence.scatter_add_(0, src, edge_fluxes)   # +J at source (outflow)
        divergence.scatter_add_(0, dst, -edge_fluxes)  # -J at destination (inflow)

        # --- Step 3: Node update using divergence ---
        x_updated = self.node_update_mlp(torch.cat([x, divergence.unsqueeze(-1)], dim=-1))

        return x_updated, edge_fluxes


class PhloemOperatorGNN(nn.Module):
    """Operator-based GNN where message passing implements discrete transport.

    Unlike PhloemNNConv which predicts concentrations and reconstructs fluxes,
    this model directly predicts edge fluxes via message passing. The messages
    are flux-like quantities, aggregation is divergence, and node updates use
    the divergence to mimic discrete conservation laws.

    Returns a dictionary with:
        - 'predictions': Node-wise sucrose content predictions [N, 1]
        - 'edge_fluxes': Edge-wise flux predictions [E]
        - 'divergences': Node-wise divergence values [N]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self._validated_input = False

        # Node input = continuous node features + time
        in_node_dim = cfg.node_feat_dim + 1
        current_dim = in_node_dim

        conv_layers = []
        norm_layers = []

        for _ in range(cfg.num_layers):
            conv = FluxMessagePassing(
                in_channels=current_dim,
                out_channels=cfg.hidden_size,
                edge_feat_dim=cfg.edge_feat_dim,
                num_org_types=cfg.num_org_types,
                hidden_size=cfg.hidden_size,
            )
            conv_layers.append(conv)
            norm_layers.append(GraphNorm(cfg.hidden_size))
            current_dim = cfg.hidden_size

        self.convs = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norm_layers)

        # Head to predict sucrose content from final embeddings
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        # Learnable output gain
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1e-1), dtype=torch.float64))

        self.dropout = nn.Dropout(cfg.dropout)
        self._init_weights()
        self.double()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _validate_input(self, data: Data) -> None:
        """Validate input data dimensions and types."""
        if self._validated_input:
            return

        must_have = ["node_feat", "edge_feat", "edge_index", "edge_org",
                     "node_fields", "sim_params", "step_params"]

        for k in must_have:
            if not hasattr(data, k):
                raise ValueError(f"Data must have {k} attribute")
        if not (isinstance(data.edge_index, torch.Tensor) and
                data.edge_index.ndim == 2 and data.edge_index.size(0) == 2):
            raise ValueError(f"edge_index must be [2, E], got {getattr(data.edge_index, 'shape', None)}")
        if data.edge_index.dtype != torch.long:
            raise ValueError(f"edge_index must be torch.long, got {data.edge_index.dtype}")
        if data.node_feat.size(1) != self.cfg.node_feat_dim:
            raise ValueError(f"Expected node_feat dim {self.cfg.node_feat_dim}, got {data.node_feat.size(1)}")
        if data.edge_feat.size(1) != self.cfg.edge_feat_dim:
            raise ValueError(f"Expected edge_feat dim {self.cfg.edge_feat_dim}, got {data.edge_feat.size(1)}")
        if data.time_per_node is None:
            raise ValueError("time_per_node is required")
        if data.time_per_node.ndim != 2 or data.time_per_node.size(1) != 1:
            raise ValueError(f"time_per_node must be [N,1], got {tuple(data.time_per_node.shape)}")

        self._validated_input = True
        print("Input validation successful (PhloemOperatorGNN)")

    def forward(self, data: Data) -> dict:
        """Forward pass returning predictions, edge fluxes, and divergences.

        Args:
            data: Graph data object

        Returns:
            dict with keys:
                - 'predictions': [N, 1] sucrose content predictions
                - 'edge_fluxes': [E] predicted edge fluxes from last layer
                - 'divergences': [N] divergence values from last layer
        """
        self._validate_input(data)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Prepare inputs
        node_feat = data.node_feat.to(device=device, dtype=dtype)
        edge_index = data.edge_index.to(device)
        edge_feat = data.edge_feat.to(device=device, dtype=dtype)
        edge_org = data.edge_org.to(device)
        time_per_node = data.time_per_node.to(device=device, dtype=dtype)

        if time_per_node.dim() != 2 or time_per_node.size(1) != 1:
            raise RuntimeError(f"time_per_node must be [N,1]; got {tuple(time_per_node.shape)}")

        # Concatenate time as extra channel
        node_feat = torch.cat([node_feat, time_per_node], dim=1)

        # Prepare edge features (continuous + categorical)
        edge_features = torch.empty(
            edge_feat.size(0), edge_feat.size(1) + 1,
            device=device, dtype=dtype
        )
        edge_features[:, :-1] = edge_feat
        edge_features[:, -1] = edge_org.to(dtype)

        batch_vec = getattr(data, "batch", None)

        # Track edge fluxes from last layer for output
        # (divergence is computed internally by FluxMessagePassing)
        final_edge_fluxes = None
        final_divergence = None

        # Message passing iterations
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Apply flux message passing (computes fluxes → divergence → node update)
            h, edge_fluxes = conv(node_feat, edge_index, edge_features)

            # Store outputs from last layer
            if i == len(self.convs) - 1:
                final_edge_fluxes = edge_fluxes
                # Recompute divergence for output (same as computed internally)
                N = node_feat.size(0)
                final_divergence = torch.zeros(N, device=device, dtype=dtype)
                if edge_fluxes.size(0) > 0:
                    src, dst = edge_index[0], edge_index[1]
                    # Divergence: +J at source (outflow), -J at destination (inflow)
                    final_divergence.scatter_add_(0, src, edge_fluxes)
                    final_divergence.scatter_add_(0, dst, -edge_fluxes)

            # Apply normalization
            if batch_vec is None:
                fake_batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
                h = norm(h, fake_batch)
            else:
                h = norm(h, batch_vec)

            h = F.relu(h)
            h = self.dropout(h)

            # Residual connection
            if h.shape == node_feat.shape:
                node_feat = node_feat + h
            else:
                node_feat = h

        # Final prediction head
        out = self.head(node_feat)

        # NOTE: No activation here - model outputs in standardized space
        # The target values are standardized (mean=0, std=1), so predictions should match
        # Non-negativity constraint is enforced after denormalization in physics functions

        # Return dictionary with all outputs
        return {
            'predictions': out,
            'edge_fluxes': final_edge_fluxes if final_edge_fluxes is not None else torch.tensor([], device=device),
            'divergences': final_divergence if final_divergence is not None else torch.zeros(node_feat.size(0), device=device)
        }
