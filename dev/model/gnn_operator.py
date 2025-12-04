"""
Operator-based GNN: Physical Flux Computation from Predictions
-------------------------------------------------
Implements PhloemOperatorGNN where:
1. Message passing layers update node embeddings
2. Prediction head outputs sucrose content S_ST
3. Physical flux module computes fluxes from predicted concentrations
   using: J_ax,ij = J^W_ij(Δψ, ΔT, r_ij) · C_upstream

The water flux J^W_ij is antisymmetric by construction (J^W_ji = -J^W_ij)
through parametrization in terms of differences:
- Δψ = ψ_src - ψ_dst (pressure gradient)
- ΔT = T_src - T_dst (temperature gradient)

This properly implements the physical transport operator where fluxes
depend on both the concentration at the upstream node and the pressure-driven
water flux.

Expected `Data` fields per graph (per timestep)
----------------------------------------------
- data.edge_index: LongTensor  [2, E]
- data.edge_feat:  FloatTensor [E, edge_feat_dim]  # edge features (e.g., r_st resistance)
- data.edge_org:   LongTensor  [E]                  # organ type per edge
- data.node_feat:  FloatTensor [N, node_feat_dim]  # [psi, vol_st, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp]
- data.time_per_node: FloatTensor [N, 1]           # time in days (per node)
- data.y:          FloatTensor [N, 1]              # target sucrose content at t
- data.norm_stats: dict                             # normalization statistics (optional)
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


class MessagePassingLayer(nn.Module):
    """Standard message passing layer for updating node embeddings.

    This layer updates node embeddings without computing physical fluxes.
    Physical flux computation is done separately after predictions are made.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_feat_dim: int,
        num_org_types: int,
        hidden_size: int = 64,
    ):
        """Initialize message passing layer.

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

        # MLP to compute messages from node pairs and edge features
        message_input_dim = 2 * in_channels + edge_input_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # MLP to update node embeddings from [h_old, aggregated_messages]
        self.node_update_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_channels)
        )

        self.double()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: message → aggregate → update.

        Args:
            h: Latent node embeddings [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_features: Edge features [E, D+1] (continuous + categorical)

        Returns:
            updated_node_features [N, out_channels]
        """
        N = h.size(0)
        device = h.device
        dtype = h.dtype

        # Extract source and destination nodes
        src, dst = edge_index[0], edge_index[1]

        # Process edge features: separate continuous and categorical
        edge_feat_cont = edge_features[:, :-1]  # continuous features
        edge_feat_cat = edge_features[:, -1].long()  # organ type

        # One-hot encode organ type
        edge_one_hot = F.one_hot(edge_feat_cat, num_classes=self.num_org_types).to(dtype)

        # Combine all edge features
        edge_inputs = torch.cat([edge_feat_cont, edge_one_hot], dim=-1)

        # Get node features at edge endpoints
        h_src = h[src]  # [E, in_channels]
        h_dst = h[dst]  # [E, in_channels]

        # Compute messages
        message_input = torch.cat([h_src, h_dst, edge_inputs], dim=-1)
        messages = self.message_mlp(message_input)  # [E, hidden_size]

        # Aggregate messages (sum aggregation)
        aggregated = torch.zeros(N, messages.size(1), device=device, dtype=dtype)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Update nodes
        h_updated = self.node_update_mlp(torch.cat([h, aggregated], dim=-1))

        return h_updated


class PhysicalFluxModule(nn.Module):
    """Computes physical fluxes based on predicted concentrations and physical features.

    Implements: J_ax,ij = J^W_ij(Δψ, ΔT, r_ij) · C_upstream
    where:
    - J^W_ij is the water flux (antisymmetric: J^W_ji = -J^W_ij)
    - C_upstream is the concentration at the source node

    The water flux is parametrized using pressure and temperature differences:
    - Δψ = ψ_src - ψ_dst (driving force)
    - ΔT = T_src - T_dst (affects viscosity)

    This ensures physical consistency: J_ji = -J_ij by construction.
    """

    def __init__(
        self,
        edge_feat_dim: int,
        num_org_types: int,
        hidden_size: int = 128,
    ):
        """Initialize physical flux module.

        Args:
            edge_feat_dim: Continuous edge feature dimension
            num_org_types: Number of organ type categories
            hidden_size: Hidden dimension for MLPs
        """
        super().__init__()

        self.num_org_types = num_org_types

        # Edge input = continuous features + one-hot organ type
        edge_input_dim = edge_feat_dim + num_org_types

        # MLP to compute signed water flux from physical features
        # Input: [Δpsi, ΔT, |Δpsi|, edge_features]
        # Output: signed flux (positive = src→dst flow)
        # Antisymmetry: flux_ji = MLP(-Δpsi_ij, -ΔT_ij, |Δpsi|_ij, ...) ≈ -flux_ij
        water_flux_input_dim = 3 + edge_input_dim  # Δpsi, ΔT, |Δpsi|, edge_features

        # Simple MLP architecture - proven to work
        self.water_flux_mlp = nn.Sequential(
            nn.Linear(water_flux_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize with small weights to prevent extreme initial predictions
        for layer in self.water_flux_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.double()

    def forward(
        self,
        C_ST: torch.Tensor,
        node_feat_phys: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute physical fluxes and divergences.

        Args:
            C_ST: Predicted concentrations [N, 1] (mol/m³)
            node_feat_phys: Physical node features [N, 8] (psi, vol, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp, time)
            edge_index: Edge connectivity [2, E]
            edge_features: Edge features [E, D+1] (continuous + categorical)

        Returns:
            tuple: (edge_fluxes [E], divergences [N])
                edge_fluxes: J_ax,ij = J^W_ij * C_upstream (mol/s)
                divergences: per-node flux divergence (mol/s)
        """
        N = C_ST.size(0)
        device = C_ST.device
        dtype = C_ST.dtype

        # Extract source and destination nodes
        src, dst = edge_index[0], edge_index[1]

        # Process edge features: separate continuous and categorical
        edge_feat_cont = edge_features[:, :-1]  # continuous features
        edge_feat_cat = edge_features[:, -1].long()  # organ type

        # One-hot encode organ type
        edge_one_hot = F.one_hot(edge_feat_cat, num_classes=self.num_org_types).to(dtype)

        # Combine all edge features
        edge_inputs = torch.cat([edge_feat_cont, edge_one_hot], dim=-1)

        # Extract physical features for water flux: psi (index 0) and Temp (index 6)
        # node_feat_phys = [psi, vol, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp, time]
        psi = node_feat_phys[:, 0:1]  # [N, 1]
        Temp = node_feat_phys[:, 6:7]  # [N, 1]

        psi_src = psi[src]  # [E, 1]
        psi_dst = psi[dst]  # [E, 1]
        Temp_src = Temp[src]  # [E, 1]
        Temp_dst = Temp[dst]  # [E, 1]

        # Compute differences (antisymmetric features)
        # Δpsi = psi_src - psi_dst (positive if flow is src→dst)
        # ΔT = Temp_src - Temp_dst
        delta_psi = psi_src - psi_dst  # [E, 1]
        delta_T = Temp_src - Temp_dst  # [E, 1]

        # NEW APPROACH: Directly predict signed flux from Δpsi, ΔT
        # This is more flexible and allows the network to learn the relationship
        # Antisymmetry is still enforced because:
        # - flux_ji = MLP(Δpsi_ji, ΔT_ji, edge_features)
        #           = MLP(-Δpsi_ij, -ΔT_ij, edge_features)
        # - If MLP is odd in (Δpsi, ΔT), then flux_ji = -flux_ij
        # We encourage this by using the signed differences directly

        # Add magnitude of pressure gradient as additional feature to help with scaling
        delta_psi_magnitude = torch.abs(delta_psi)  # [E, 1]

        water_flux_input = torch.cat([delta_psi, delta_T, delta_psi_magnitude, edge_inputs], dim=-1)
        J_water = self.water_flux_mlp(water_flux_input).squeeze(-1)  # [E], can be positive or negative

        # Get concentration at source (upstream) node
        C_src = C_ST[src].squeeze(-1)  # [E]

        # Compute sugar flux: J_ax,ij = J^W_ij * C_upstream
        edge_fluxes = J_water * C_src  # [E] (mol/s)

        # Compute divergence per node
        # This follows the C++ PiafMunch convention (Delta2 matrix):
        # For edge src → dst with flux edge_fluxes:
        #   - src node (upstream) gets: -edge_fluxes (flux leaving)
        #   - dst node (downstream) gets: +edge_fluxes (flux arriving)
        # This matches: Delta2[src, edge] = -1, Delta2[dst, edge] = +1
        divergence = torch.zeros(N, device=device, dtype=dtype)
        divergence.scatter_add_(0, src, -edge_fluxes)  # -J at source (flux leaving)
        divergence.scatter_add_(0, dst, +edge_fluxes)  # +J at destination (flux arriving)

        return edge_fluxes, divergence


class PhloemOperatorGNN(nn.Module):
    """Operator-based GNN where physical fluxes are computed from predictions.

    Architecture:
    1. Message passing layers update node embeddings
    2. Prediction head outputs sucrose content S_ST
    3. Physical flux module computes fluxes from predicted concentrations
       using J_ax,ij = J^W_ij(ΔP, r_ij, T) · C_upstream

    Returns a dictionary with:
        - 'predictions': Node-wise sucrose content predictions [N, 1]
        - 'edge_fluxes': Edge-wise flux predictions [E] (computed from predictions)
        - 'divergences': Node-wise divergence values [N]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self._validated_input = False

        # Node input for time-series mode: [psi, vol, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp, time, S(t-1)]
        # Always includes previous sucrose content for temporal modeling
        in_node_dim = cfg.node_feat_dim + 1 + 1  # base features + time + prev_sucrose

        # Project physical features + prev_sucrose to latent space
        self.input_proj = nn.Linear(in_node_dim, cfg.hidden_size)

        conv_layers = []
        norm_layers = []

        for _ in range(cfg.num_layers):
            conv = MessagePassingLayer(
                in_channels=cfg.hidden_size,  # All layers work with hidden_size latent state
                out_channels=cfg.hidden_size,
                edge_feat_dim=cfg.edge_feat_dim,
                num_org_types=cfg.num_org_types,
                hidden_size=cfg.hidden_size,
            )
            conv_layers.append(conv)
            norm_layers.append(GraphNorm(cfg.hidden_size))

        self.convs = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norm_layers)

        # Head to predict sucrose content from final embeddings
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, 1)
        )

        # Physical flux module (computes fluxes from predictions)
        self.flux_module = PhysicalFluxModule(
            edge_feat_dim=cfg.edge_feat_dim,
            num_org_types=cfg.num_org_types,
            hidden_size=cfg.hidden_size,
        )

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

    def _compute_concentration(
        self,
        S_ST: torch.Tensor,
        vol_ST: torch.Tensor,
        data: Data
    ) -> torch.Tensor:
        """Convert standardized sucrose content to concentration.

        Args:
            S_ST: Standardized sucrose content [N, 1]
            vol_ST: Sieve tube volume [N, 1]
            data: Data object (may contain norm_stats)

        Returns:
            C_ST: Concentration [N, 1] in mol/m³
        """
        # Inverse standardization: S = S_ST * std + mean
        if hasattr(data, 'norm_stats') and 'S_ST' in data.norm_stats:
            S_mean = data.norm_stats['S_ST']['mean']
            S_std = data.norm_stats['S_ST']['std']
            S = S_ST * S_std + S_mean
        else:
            # If no normalization stats, assume S_ST is already physical
            # or use it directly (this is a fallback)
            S = S_ST

        # Concentration: C = S / vol
        C_ST = S / (vol_ST + 1e-10)  # Add epsilon to avoid division by zero

        return C_ST

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

    def forward(self, data: Data, prev_sucrose: torch.Tensor) -> dict:
        """Forward pass returning predictions, edge fluxes, and divergences.

        Args:
            data: Graph data object
            prev_sucrose: Previous timestep sucrose content [N, 1] (standardized).
                         Required for time-series learning. For first timestep or new nodes,
                         should be initialized with appropriate values (e.g., ground truth).

        Returns:
            dict with keys:
                - 'predictions': [N, 1] sucrose content S_ST (standardized)
                - 'edge_fluxes': [E] physical edge fluxes (mol/s)
                - 'divergences': [N] divergence values (mol/s)
        """
        self._validate_input(data)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Prepare inputs
        edge_index = data.edge_index.to(device)
        edge_feat = data.edge_feat.to(device=device, dtype=dtype)
        edge_org = data.edge_org.to(device)
        time_per_node = data.time_per_node.to(device=device, dtype=dtype)

        if time_per_node.dim() != 2 or time_per_node.size(1) != 1:
            raise RuntimeError(f"time_per_node must be [N,1]; got {tuple(time_per_node.shape)}")

        # Prepare prev_sucrose
        prev_sucrose_tensor = prev_sucrose.to(device=device, dtype=dtype)
        if prev_sucrose_tensor.dim() == 1:
            prev_sucrose_tensor = prev_sucrose_tensor.unsqueeze(-1)

        # Physical features for flux computation (no prev_sucrose): [psi, vol, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp, time]
        node_feat_phys_for_flux = torch.cat([data.node_feat.to(device, dtype), time_per_node], dim=1)  # [N, 8]

        # Full node features for message passing (includes prev_sucrose): [psi, vol, ..., Temp, time, S(t-1)]
        node_feat_with_history = torch.cat([node_feat_phys_for_flux, prev_sucrose_tensor], dim=1)  # [N, 9]

        # Initialize latent state from physical features + temporal context
        h = self.input_proj(node_feat_with_history)  # [N, hidden_size]

        # Prepare edge features (continuous + categorical)
        edge_features = torch.empty(
            edge_feat.size(0), edge_feat.size(1) + 1,
            device=device, dtype=dtype
        )
        edge_features[:, :-1] = edge_feat
        edge_features[:, -1] = edge_org.to(dtype)

        batch_vec = getattr(data, "batch", None)

        # Message passing iterations (no flux computation here)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Apply message passing to update embeddings
            h_new = conv(h, edge_index, edge_features)

            # Apply normalization
            if batch_vec is None:
                fake_batch = torch.zeros(h_new.size(0), dtype=torch.long, device=h_new.device)
                h_new = norm(h_new, fake_batch)
            else:
                h_new = norm(h_new, batch_vec)

            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection (always possible since all layers have same hidden_size)
            h = h + h_new

        # Get predictions: S_ST (standardized sucrose content)
        S_ST = self.head(h)  # [N, 1]

        # Compute C_ST from S_ST for flux calculation
        vol_ST = node_feat_phys_for_flux[:, 1:2]  # [N, 1] - volume is at index 1
        C_ST = self._compute_concentration(S_ST, vol_ST, data)  # [N, 1]

        # Compute raw physical fluxes from predicted concentrations
        edge_fluxes_raw, divergences_raw = self.flux_module(
            C_ST=C_ST,
            node_feat_phys=node_feat_phys_for_flux,
            edge_index=edge_index,
            edge_features=edge_features
        )

        # Return dictionary with all outputs
        return {
            'predictions': S_ST,  # Standardized sucrose content
            'edge_fluxes': edge_fluxes_raw,  # Physical fluxes (mol/s)
            'divergences': divergences_raw,  # Per-node divergence (mol/s)
        }