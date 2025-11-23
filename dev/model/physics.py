from __future__ import annotations

import torch

from torch_scatter import scatter_mean
from torch_geometric.data import Data

from . import utils
from . import config

DEBUG = True  # Debug flag: set to True to enable detailed physics loss debugging
debug_path = "results/debug_physics_logs.txt"


def compute_axial_flux(
    C_ST: torch.Tensor,
    node_feat_original: torch.Tensor,
    edge_feat_original: torch.Tensor,
    edge_index: torch.Tensor,
    batch_vec: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """Compute axial sucrose flux J_ax along edges.

    This implementation follows the C++ PiafMunch algorithm (external/PiafMunch/solve.cpp):
    1. Compute osmotic pressure P_ST = C_ST * RT for each node
    2. Compute water flux JW_ST based on pressure gradients
    3. Select upstream concentration based on flow direction
    4. Compute sugar flux JS_ST = JW_ST * C_upstream

    Args:
        C_ST: Sucrose concentration per node [N] (already denormalized)
        node_feat_original: Node features in original space [N, D]
        edge_feat_original: Edge features in original space [E, D]
        edge_index: Edge connectivity [2, E]
        batch_vec: Batch assignment vector [N] (None for single graph)
        device: Target device for computations
        dtype: Data type for computations

    Returns:
        torch.Tensor: Axial flux per edge [E]
    """
    # Handle empty graph case
    if edge_index.size(1) == 0:
        return torch.zeros(0, device=device, dtype=dtype)

    src, dst = edge_index[0], edge_index[1]
    r_ST = edge_feat_original.squeeze(-1)

    # Extract node features (already in original space)
    psi = node_feat_original[:, 0]      # hydraulic potential
    Temp = node_feat_original[:, 6]     # temperature [°C]

    # ---- Step 1: Compute osmotic pressure from concentrations
    # C_ST is already denormalized and converted to concentration

    # Osmotic pressure P_ST = C_ST * RT
    # In C++, TairK_phloem is global, but in batched case we need per-graph temperature
    RT = utils.compute_RT_per_node(
        Temp=Temp,
        batch_vec=batch_vec,
        R=config.R,
        device=device,
        dtype=dtype,
    )
    P_ST_osmotic = C_ST * RT

    # Convert hydraulic potential psi to hPa and add to osmotic pressure
    psi = psi * config.cmH2O_to_hPa
    P_ST = P_ST_osmotic + psi

    # ---- Step 2: Compute water flux from pressure gradients
    P_i = P_ST[src]
    P_j = P_ST[dst]

    JW_ST = (P_j - P_i) / r_ST

    # ---- Step 3: Select upstream concentration based on flow direction
    # With dP = P_j - P_i, positive JW_ST means P_j > P_i, so flow is j -> i
    # For sugar flux, we want the upstream (source) concentration
    # If flow is j -> i, then upstream is j (dst), downstream is i (src)
    C_i = C_ST[src]
    C_j = C_ST[dst]

    C_upstream = torch.where(JW_ST > 0, C_j, C_i)
    C_upstream = torch.clamp(C_upstream, min=0.0)

    # ---- Step 4: Sugar flux JS_ST = JW_ST * C_upstream
    J_ax = JW_ST * C_upstream

    return J_ax


def compute_flux_divergence(J_ax: torch.Tensor, edge_index: torch.Tensor, N: int, device: torch.device) -> torch.Tensor:
    """Compute divergence of flux to get net inflow per node.

    Args:
        J_ax: Axial flux per edge [E]
        edge_index: Edge connectivity [2, E]
        N: Number of nodes
        device: Target device for computations

    Returns:
        torch.Tensor: Net flux change per node [N]
    """
    # Initialize with same dtype as J_ax
    dS_dt_from_flux = torch.zeros(N, device=device, dtype=J_ax.dtype)

    # Handle empty graph case
    if J_ax.size(0) == 0:
        return dS_dt_from_flux

    src, dst = edge_index[0], edge_index[1]

    # Divergence of flux -> net inflow per node
    # This computes the sum of incoming/outgoing fluxes for each node
    # dst node accumulates -J_ax
    # src node accumulates +J_ax
    dS_dt_from_flux.scatter_add_(0, dst, -J_ax)
    dS_dt_from_flux.scatter_add_(0, src, +J_ax)

    return dS_dt_from_flux


def compute_phloem_loading(
    C_ST: torch.Tensor,
    node_feat_original: torch.Tensor,
    params: dict,
    node_fields: dict,
    device: torch.device
) -> torch.Tensor:
    """Compute phloem loading rate F_in per node.

    Args:
        C_ST: Sucrose concentration per node [N] (already denormalized)
        node_feat_original: Node features in original space [N, D]
        params: Simulation and step parameters
        node_fields: Node field values
        device: Target device for computations

    Returns:
        torch.Tensor: Phloem loading rate per node [N]
    """
    # Extract node features (already in original space)
    len_leaf = node_feat_original[:, 2]

    # C_ST is already denormalized and in concentration units
    CSTi_positive = torch.clamp(C_ST, min=0.0)

    # Phloem loading with feedback inhibition
    F_in = (params["Vmaxloading"] * len_leaf) * node_fields["C_meso"] / \
           (params["Mloading"] + node_fields["C_meso"]) * \
           torch.exp(-CSTi_positive * params["beta_loading"])

    return F_in


def compute_sucrose_outflow(
    C_ST: torch.Tensor,
    node_feat_original: torch.Tensor,
    params: dict,
    node_fields: dict,
    device: torch.device
) -> torch.Tensor:
    """Compute sucrose outflow F_out per node.

    Args:
        C_ST: Sucrose concentration per node [N] (already denormalized)
        node_feat_original: Node features in original space [N, D]
        params: Simulation and step parameters
        node_fields: Node field values
        device: Target device for computations

    Returns:
        torch.Tensor: Sucrose outflow rate per node [N]
    """
    # Extract node features (already in original space)
    Q_Rmmax = node_feat_original[:, 3]
    Q_Grmax = node_feat_original[:, 4]
    Q_Exudmax = node_feat_original[:, 5]
    Temp = node_feat_original[:, 6]

    # C_ST is already denormalized and in concentration units
    CSTi_positive = torch.clamp(C_ST, min=0.0)

    # Apply CSTimin threshold for usage
    CSTi_effective = torch.clamp(CSTi_positive - params["CSTimin"], min=0.0)
    CSTi_delta = torch.clamp(CSTi_effective - node_fields["Csoil_node"], min=0.0)

    # Temperature-dependent maintenance respiration
    R_mmax = (Q_Rmmax + params["krm2v"] * CSTi_effective) * \
             torch.pow(params["Q10"], (Temp - params["TrefQ10"]) / 10.0)

    # Michaelis-Menten kinetics for sucrose usage
    F_out_MM = (R_mmax + Q_Grmax) * (CSTi_effective / (CSTi_effective + params["KMfu"]))

    # Root exudation based on concentration gradient
    Exud = CSTi_delta * Q_Exudmax

    return F_out_MM + Exud





def physics_residual(y_pred: torch.Tensor, data: Data):
    """Compute physics-informed residual term based on sucrose transport equations.

    Implements the governing equation for content-based sucrose transport in sieve-tubes:
    dS/dt = divJ + (F_in - F_out) ≈ 0

    where:
    - divJ is the divergence of axial sucrose flux
    - F_in is the phloem loading rate
    - F_out is the sucrose outflow

    Physics loss is computed by minimizing dS_dt_from_physics_pred to enforce
    the conservation law on predicted concentrations.

    Args:
        y_pred: Predicted sucrose content [N, 1]
        data: Graph data containing topology, features, simulation parameters, and node fields

    Returns:
        tuple: (residual_loss, physics_components_dict) where physics_components_dict contains
            {'J_ax', 'F_in', 'F_out', 'dS_dt_from_flux', 'dS_dt_from_physics'}
    """
    device = y_pred.device
    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

    # Inverse-transform node/edge features
    node_feat_standardized = data.node_feat.to(device)
    node_feat_original = data.feature_scaler.inv_transform(node_feat_standardized)

    edge_feat_standardized = data.edge_feat.to(device)
    edge_feat_original = data.edge_scaler.inv_transform(edge_feat_standardized)

    edge_index = data.edge_index.to(device)
    vol_ST = node_feat_original[:, 1]

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # ============================
    # Compute physics terms from PREDICTIONS
    # ============================
    S_ST_pred = data.target_scaler.inv_transform(y_pred).squeeze(-1)
    C_ST_pred = S_ST_pred / vol_ST

    # Compute axial flux and its divergence from predictions
    J_ax_pred = compute_axial_flux(C_ST_pred, node_feat_original, edge_feat_original,
                                   edge_index, batch_vec, device, y_pred.dtype)
    dS_dt_from_flux_pred = compute_flux_divergence(J_ax_pred, edge_index, N, device)

    # Compute phloem loading and outflow from predictions
    F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
    F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, device)

    # Total physics-based derivative from predictions (in physical units: mmol/h)
    dS_dt_from_physics_pred = dS_dt_from_flux_pred + F_in_pred - F_out_pred

    if DEBUG:
        # Compute physics terms from true values for comparison/debugging
        y_true = data.y.to(device)
        S_ST_true = data.target_scaler.inv_transform(y_true).squeeze(-1)
        C_ST_true = S_ST_true / vol_ST

        J_ax_true = compute_axial_flux(C_ST_true, node_feat_original, edge_feat_original,
                                       edge_index, batch_vec, device, y_pred.dtype)
        dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

        F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
        F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, device)
        dS_dt_from_physics_true = dS_dt_from_flux_true + F_in_true - F_out_true
        with open(debug_path, "a") as f:
            msg = (
                f"\n{'='*60}\n"
                f"DEBUG OUTPUT - PHYSICS RESIDUAL (MINIMIZE TO ZERO)\n"
                f"{'='*60}\n"
                f"\nNumber of graphs in batch: {torch.bincount(batch_vec).size(0) if batch_vec is not None else 1}\n"
            )
            if batch_vec is not None:
                msg += f"Number of nodes per graph: {torch.bincount(batch_vec).detach().cpu().numpy()}\n"
            f.write(msg)

            msg = (
                f"\n--- CONCENTRATION VALUES (mmol/cm³) ---\n"
                f"C_ST_true: {C_ST_true[:10].detach().cpu().numpy()}\n"
                f"C_ST_pred: {C_ST_pred[:10].detach().cpu().numpy()}\n"

                f"\n--- FLUX VALUES (mmol/h) ---\n"
                f"J_ax_true (mean): {J_ax_true.mean().detach().cpu().item():.6e}\n"
                f"J_ax_pred (mean): {J_ax_pred.mean().detach().cpu().item():.6e}\n"

                f"\nF_in_true: {F_in_true[:10].detach().cpu().numpy()}\n"
                f"F_in_pred: {F_in_pred[:10].detach().cpu().numpy()}\n"

                f"\nF_out_true: {F_out_true[:10].detach().cpu().numpy()}\n"
                f"F_out_pred: {F_out_pred[:10].detach().cpu().numpy()}\n"

                f"\n--- TIME DERIVATIVES FROM DISCRETE FLUX LAW (mmol/h) ---\n"
                f"dS_dt_from_flux_true: {dS_dt_from_flux_true[:10].detach().cpu().numpy()}\n"
                f"dS_dt_from_flux_pred: {dS_dt_from_flux_pred[:10].detach().cpu().numpy()}\n"

                f"\ndS_dt_from_physics_true (total): {dS_dt_from_physics_true[:10].detach().cpu().numpy()}\n"
                f"dS_dt_from_physics_pred (total): {dS_dt_from_physics_pred[:10].detach().cpu().numpy()}\n"

                f"\n--- PHYSICS RESIDUAL (should approach zero) ---\n"
                f"dS_dt_from_physics_pred (first 10 nodes): {dS_dt_from_physics_pred[:10].detach().cpu().numpy()}\n"
                f"Mean absolute residual: {dS_dt_from_physics_pred.abs().mean().detach().cpu().item():.6e}\n"
                f"{'='*60}\n"
            )
            f.write(msg)

    # ============================
    # Compute physics residual: minimize dS_dt_from_physics_pred to zero
    # ============================
    # Physics loss: enforce conservation law by minimizing the residual
    # dS/dt = divJ + F_in - F_out ≈ 0
    # residual = divJ_pred + F_in_pred - F_out_pred
    residual = dS_dt_from_physics_pred

    # Use adaptive normalization based on current residual scale
    # Use 90th percentile of absolute values as scale (robust to outliers)
    with torch.no_grad():
        scale = residual.abs().quantile(0.9).clamp(min=0.1, max=10000.0)

    # Normalize residual by adaptive scale
    residual_node = (residual / scale).pow(2)

    # Average per graph first, then across graphs
    if batch_vec is not None:
        residual_per_graph = scatter_mean(residual_node, batch_vec, dim=0)
        loss = residual_per_graph.mean()
    else:
        loss = residual_node.mean()

    # Prepare detailed physics components for logging (use predicted values)
    if batch_vec is not None:
        # Batched case: compute per-graph averages using scatter_mean
        F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
        F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
        dS_dt_from_flux_pred_per_graph = scatter_mean(dS_dt_from_flux_pred.detach().abs(), batch_vec, dim=0)
        dS_dt_from_physics_pred_per_graph = scatter_mean(dS_dt_from_physics_pred.detach().abs(), batch_vec, dim=0)

        # Edge-level quantities: need edge-to-graph mapping
        if J_ax_pred.size(0) > 0:
            edge_batch = batch_vec[data.edge_index[0].to(device)]
            J_ax_per_graph = scatter_mean(J_ax_pred.detach().abs(), edge_batch, dim=0)
            J_ax_avg = J_ax_per_graph.mean()
        else:
            J_ax_avg = torch.tensor(0.0, device=device)

        loss_dict = {
            'J_ax': J_ax_avg,
            'F_in': F_in_per_graph.mean(),
            'F_out': F_out_per_graph.mean(),
            'dS_dt': dS_dt_from_physics_pred_per_graph.mean(),  # For compatibility with logging
            'dS_dt_from_flux': dS_dt_from_flux_pred_per_graph.mean(),
            'dS_dt_from_physics': dS_dt_from_physics_pred_per_graph.mean()
        }
    else:
        # Single graph case: simple mean across nodes/edges
        loss_dict = {
            'J_ax': J_ax_pred.detach().abs().mean() if J_ax_pred.size(0) > 0 else torch.tensor(0.0, device=device),
            'F_in': F_in_pred.detach().mean(),
            'F_out': F_out_pred.detach().mean(),
            'dS_dt': dS_dt_from_physics_pred.detach().abs().mean(),  # For compatibility with logging
            'dS_dt_from_flux': dS_dt_from_flux_pred.detach().abs().mean(),
            'dS_dt_from_physics': dS_dt_from_physics_pred.detach().abs().mean()
        }

    return loss, loss_dict