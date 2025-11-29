"""
Operator-based GNN: Physical Flux Computation from Predictions
-------------------------------------------------
Implements PhloemOperatorGNN where:
1. Message passing layers update node embeddings
2. Prediction head outputs sucrose content S_ST
3. Physical flux module computes fluxes from predicted concentrations
   using: J_ax,ij = J^W_ij(Δψ, ΔC, T, r_ij) · C_upstream

The water flux J^W_ij is antisymmetric by construction (J^W_ji = -J^W_ij)
through parametrization in terms of differences:
- Δψ = ψ_src - ψ_dst (pressure gradient)
- ΔC = C_src - C_dst (concentration gradient)
- T = (T_src + T_dst) / 2 (average temperature, not a gradient)

This properly implements the physical transport operator where fluxes
depend on the concentration at the upstream node (based on flow direction)
and the pressure-driven water flux. The concentration gradient ΔC affects
the flux magnitude through osmotic effects, while temperature T affects
it through the RT (gas constant × temperature) term.

Expected `Data` fields per graph (per timestep)
----------------------------------------------
- data.edge_index: LongTensor  [2, E]
- data.edge_feat:  FloatTensor [E, edge_feat_dim]  # edge features (e.g., r_st resistance)
- data.edge_org:   LongTensor  [E]                 # organ type per edge
- data.node_feat:  FloatTensor [N, node_feat_dim]  # [psi, vol_st, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp]
- data.time_per_node: FloatTensor [N, 1]           # time in days (per node)
- data.y:          FloatTensor [N, 1]              # target sucrose content at t
- data.norm_stats: dict                            # normalization statistics (optional)
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

        # MLP to update node embeddings
        # Input is concatenation of [h_old, aggregated_messages]
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

    Implements: J_ax,ij = J^W_ij(Δψ, ΔC, T, r_ij) · C_upstream
    where:
    - J^W_ij is the water flux (antisymmetric: J^W_ji = -J^W_ij)
    - C_upstream is the concentration at the upstream node (source of the flow)

    The water flux is parametrized using:
    - Δψ = ψ_src - ψ_dst (pressure gradient, driving force for water movement)
    - ΔC = C_src - C_dst (concentration gradient, affects flow via osmotic effects)
    - T = average temperature (affects viscosity and RT term, NOT a gradient)

    Note: Unlike traditional Münch models, we don't use ΔT (temperature difference)
    because temperature affects the flux through the RT (gas constant × T) term in
    osmotic pressure, not through a gradient. In practice, all nodes have the same
    temperature at any given time step.

    This ensures physical consistency: J_ji = -J_ij by construction.
    """

    def __init__(
        self,
        edge_feat_dim: int,
        num_org_types: int,
        hidden_size: int = 64,
    ):
        """Initialize physical flux module.

        Args:
            edge_feat_dim: Continuous edge feature dimension
            num_org_types: Number of organ type categories
            hidden_size: Hidden dimension for MLPs
        """
        super().__init__()

        self.num_org_types = num_org_types

        # Learnable linear combination of (Δpsi, ΔC) -> antisymmetric scalar a_ij
        # a_ij = w_psi * Δpsi + w_C * ΔC
        self.antisym_linear = nn.Linear(2, 1, bias=False)

        # Edge input = continuous features + one-hot organ type
        edge_input_dim = edge_feat_dim + num_org_types

        # MLP to compute water flux magnitude from symmetric edge representation
        # Input: [|Δpsi|, |ΔC|, T_edge, edge_features_symmetric]
        # Output: scalar flux magnitude (always positive or zero)
        sym_input_dim = 3 + edge_input_dim  # |Δpsi|, |ΔC|, T_edge, edge_features
        self.sym_mlp = nn.Sequential(
            nn.Linear(sym_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # flux magnitude
            nn.Softplus()  # ensures non-negative magnitude
        )

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
        psi = node_feat_phys[:, 0:1]
        Temp = node_feat_phys[:, 6:7]

        psi_src, psi_dst = psi[src], psi[dst]
        T_src, T_dst = Temp[src], Temp[dst]

        # Sucrose concentrations at endpoints
        C_src = C_ST[src]
        C_dst = C_ST[dst]

        # Compute differences (antisymmetric features)
        # Following CPlantBox convention where JW = (P_src - P_dst) / r_ST
        # Note: src/dst are graph topology labels, NOT flow direction!
        # Positive flux: P_src > P_dst -> water flows HIGH to LOW -> src -> dst
        # We compute delta_psi = psi_src - psi_dst to match this convention
        delta_psi = psi_src - psi_dst
        delta_C = C_src - C_dst


        # Antisymmetric linear combination: a_ij = w_psi * Δpsi + w_C * ΔC
        antisym_input = torch.cat([delta_psi, delta_C], dim=-1)  # [E, 2]
        a_ij = self.antisym_linear(antisym_input).squeeze(-1)    # [E]

        # Use absolute values for symmetric magnitude computation
        # This ensures magnitude is the same regardless of edge direction
        abs_delta_psi = torch.abs(delta_psi)
        abs_delta_C = torch.abs(delta_C)

        # Use absolute temperature / edge-average, not ΔT
        T_edge = 0.5 * (T_src + T_dst)

        # Inputs for symmetric magnitude MLP: [|Δpsi|, |ΔC|, T_edge, edge_features]
        sym_input = torch.cat(
            [abs_delta_psi, abs_delta_C, T_edge, edge_inputs],
            dim=-1
        )
        scale = self.sym_mlp(sym_input).squeeze(-1)  # [E], always >= 0

        # Water flux: J_water_ij = a_ij * scale_ij
        # - a_ij = w_psi * Δpsi + w_C * ΔC (antisymmetric: flips sign if src/dst are swapped)
        # - scale_ij = SymMLP(|Δpsi|, |ΔC|, T_edge, edge_features) ≥ 0 (symmetric)
        # This guarantees J_water_ji = -J_water_ij by construction
        # Water flux: antisymmetric scalar * symmetric positive scale
        J_water = a_ij * scale  # [E]

        # Select upstream concentration based on flow direction
        # CPlantBox convention: JW_ST = (P_src - P_dst) / r_ST
        # If J_water > 0: P_src > P_dst -> flow is src -> dst -> upstream is src
        # If J_water < 0: P_dst > P_src -> flow is dst -> src -> upstream is dst
        # Upstream = where the water (and sucrose) flows FROM
        # Compute sugar flux: J_ax,ij = J^W_ij * C_upstream
        C_src = C_ST[src].squeeze(-1)
        C_dst = C_ST[dst].squeeze(-1)
        C_upstream = torch.where(J_water > 0, C_src, C_dst)
        edge_fluxes = J_water * C_upstream

        # Compute divergence per node
        divergence = torch.zeros(N, device=device, dtype=dtype)
        divergence.scatter_add_(0, src, edge_fluxes)   # +J at source (outflow)
        divergence.scatter_add_(0, dst, -edge_fluxes)  # -J at destination (inflow)

        return edge_fluxes, divergence


class PhloemOperatorGNN(nn.Module):
    """Operator-based GNN where physical fluxes are computed from predictions.

    Architecture:
    1. Message passing layers update node embeddings
    2. Prediction head outputs sucrose content S_ST
    3. Physical flux module computes fluxes from predicted concentrations
    using J_ax,ij = J^W_ij(Δψ, ΔC, T, r_ij) · C_upstream, where J^W_ij is
    parameterized as a learnable linear combination of Δψ and ΔC times a
    symmetric nonlinear scale factor.

    Returns a dictionary with:
        - 'predictions': Node-wise sucrose content predictions [N, 1]
        - 'edge_fluxes': Edge-wise flux predictions [E] (computed from predictions)
        - 'divergences': Node-wise divergence values [N]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self._validated_input = False

        # Node input = continuous node features + time
        in_node_dim = cfg.node_feat_dim + 1  # Physical features: [psi, vol, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp, time]

        # Project physical features to latent space
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

    def forward(self, data: Data) -> dict:
        """Forward pass returning predictions, edge fluxes, and divergences.

        Args:
            data: Graph data object

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

        # Physical features: [psi, vol, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp, time]
        # This tensor is kept unchanged throughout all layers
        node_feat_phys = torch.cat([data.node_feat.to(device, dtype), time_per_node], dim=1)  # [N, 8]

        # Initialize latent state from physical features
        h = self.input_proj(node_feat_phys)  # [N, hidden_size]

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
        S_ST = self.head(h)

        # Compute C_ST from S_ST for flux calculation
        vol_ST = node_feat_phys[:, 1:2]
        C_ST = self._compute_concentration(S_ST, vol_ST, data)

        # Compute raw physical fluxes from predicted concentrations
        edge_fluxes_raw, divergences_raw = self.flux_module(
            C_ST=C_ST,
            node_feat_phys=node_feat_phys,
            edge_index=edge_index,
            edge_features=edge_features
        )

        # Return dictionary with all outputs
        return {
            'predictions': S_ST,  # Standardized sucrose content
            'edge_fluxes': edge_fluxes_raw,  # Physical fluxes (mol/s)
            'divergences': divergences_raw,  # Per-node divergence (mol/s)
        }

