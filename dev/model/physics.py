from __future__ import annotations

import torch

from torch_scatter import scatter_mean
from torch_geometric.data import Data
from training.config import PhysicsMetrics
from . import utils
from . import config

DEBUG = True                # Debug flag: set to True to enable detailed physics loss debugging
PENALTY_WEIGHT = 1000.0     # Weight for penalizing non-positive concentrations
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


def log_physics_values(y_pred: torch.Tensor, data: Data, model_output=None):
    """Log true and predicted physics values for analysis (no loss computation).

    This function computes and logs all physics quantities (C_ST, J_ax, divergence,
    F_in, F_out) for both true and predicted values. It's useful for evaluating
    physical consistency of models trained with DATA_ONLY loss.

    Args:
        y_pred: Predicted sucrose content [N, 1] or dict for operator model
        data: Graph data containing topology, features, and targets
        model_output: For operator models, dict containing edge_fluxes and divergences

    Returns:
        PhysicsMetrics: Computed physics metrics for terminal display, or None if DEBUG=False
    """
    if not DEBUG:
        return None

    # Handle operator model case
    is_operator_model = isinstance(y_pred, dict)
    if is_operator_model:
        model_output = y_pred
        y_pred = model_output['predictions']
        edge_fluxes_pred = model_output['edge_fluxes']
        divergence_pred = model_output['divergences']

    device = y_pred.device
    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

    # Inverse-transform features
    node_feat_standardized = data.node_feat.to(device)
    node_feat_original = data.feature_scaler.inv_transform(node_feat_standardized)

    edge_feat_standardized = data.edge_feat.to(device)
    edge_feat_original = data.edge_scaler.inv_transform(edge_feat_standardized)

    edge_index = data.edge_index.to(device)
    vol_ST = node_feat_original[:, 1]

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # Compute predicted physics terms
    S_ST_pred = data.target_scaler.inv_transform(y_pred).squeeze(-1)
    C_ST_pred = S_ST_pred / vol_ST

    # Compute true physics terms
    y_true = data.y.to(device)
    S_ST_true = data.target_scaler.inv_transform(y_true).squeeze(-1)
    C_ST_true = S_ST_true / vol_ST

    if is_operator_model:
        # Operator model: use predicted fluxes/divergences directly
        J_ax_true = compute_axial_flux(
            C_ST_true, node_feat_original, edge_feat_original,
            edge_index, batch_vec, device, y_pred.dtype
        )
        dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

        F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
        F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, device)

        F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
        F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, device)

        dS_dt_from_physics_true = dS_dt_from_flux_true + F_in_true - F_out_true
        dS_dt_from_physics_pred = divergence_pred + F_in_pred - F_out_pred

        with open(debug_path, "a") as f:
            msg = (
                f"\n{'='*60}\n"
                f"DEBUG OUTPUT - OPERATOR MODEL (DATA-ONLY MODE)\n"
                f"{'='*60}\n"
                f"\nNumber of graphs: {torch.bincount(batch_vec).size(0) if batch_vec is not None else 1}\n"
            )
            if batch_vec is not None:
                msg += f"Nodes per graph: {torch.bincount(batch_vec).detach().cpu().numpy()}\n"

            msg += (
                f"\n--- CONCENTRATION VALUES (mmol/cm³) ---\n"
                f"C_ST_true: {C_ST_true[:10].detach().cpu().numpy()}\n"
                f"C_ST_pred: {C_ST_pred[:10].detach().cpu().numpy()}\n"

                f"\n--- EDGE FLUXES (mmol/h) ---\n"
                f"J_ax_true (reconstructed from physics):\n"
                f"  First 10: {J_ax_true[:10].detach().cpu().numpy()}\n"
                f"  Mean: {J_ax_true.mean().detach().cpu().item():.6e}\n"
                f"  Std: {J_ax_true.std().detach().cpu().item():.6e}\n"
                f"J_ax_pred (from operator model):\n"
                f"  First 10: {edge_fluxes_pred[:10].detach().cpu().numpy()}\n"
                f"  Mean: {edge_fluxes_pred.mean().detach().cpu().item():.6e}\n"
                f"  Std: {edge_fluxes_pred.std().detach().cpu().item():.6e}\n"

                f"\n--- DIVERGENCE (mmol/h) ---\n"
                f"Divergence_true:\n"
                f"  First 10: {dS_dt_from_flux_true[:10].detach().cpu().numpy()}\n"
                f"  Mean: {dS_dt_from_flux_true.mean().detach().cpu().item():.6e}\n"
                f"Divergence_pred:\n"
                f"  First 10: {divergence_pred[:10].detach().cpu().numpy()}\n"
                f"  Mean: {divergence_pred.mean().detach().cpu().item():.6e}\n"

                f"\n--- SOURCE/SINK TERMS (mmol/h) ---\n"
                f"F_in_true: {F_in_true[:10].detach().cpu().numpy()}\n"
                f"F_in_pred: {F_in_pred[:10].detach().cpu().numpy()}\n"
                f"F_out_true: {F_out_true[:10].detach().cpu().numpy()}\n"
                f"F_out_pred: {F_out_pred[:10].detach().cpu().numpy()}\n"

                f"\n--- TOTAL PHYSICS RESIDUAL (mmol/h) ---\n"
                f"dS_dt_from_physics_true: {dS_dt_from_physics_true[:10].detach().cpu().numpy()}\n"
                f"dS_dt_from_physics_pred: {dS_dt_from_physics_pred[:10].detach().cpu().numpy()}\n"

                f"\n--- COMPARISON METRICS ---\n"
                f"Flux MSE: {((J_ax_true - edge_fluxes_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"Divergence MSE: {((dS_dt_from_flux_true - divergence_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"F_in MSE: {((F_in_true - F_in_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"F_out MSE: {((F_out_true - F_out_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"Total residual MSE: {((dS_dt_from_physics_true - dS_dt_from_physics_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"{'='*60}\n"
            )
            f.write(msg)

        # Compute averaged metrics for terminal display (operator model)
        from torch_scatter import scatter_mean
        if batch_vec is not None:
            # Batched case: compute per-graph averages
            F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
            F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
            divergence_per_graph = scatter_mean(divergence_pred.detach().abs(), batch_vec, dim=0)
            dS_dt_per_graph = scatter_mean(dS_dt_from_physics_pred.detach().abs(), batch_vec, dim=0)

            if edge_fluxes_pred.size(0) > 0:
                edge_batch = batch_vec[edge_index[0].to(device)]
                J_ax_per_graph = scatter_mean(edge_fluxes_pred.detach().abs(), edge_batch, dim=0)
                J_ax_avg = J_ax_per_graph.mean().item()
            else:
                J_ax_avg = 0.0

            return PhysicsMetrics(
                J_ax=J_ax_avg,
                F_in=F_in_per_graph.mean().item(),
                F_out=F_out_per_graph.mean().item(),
                dS_dt=dS_dt_per_graph.mean().item(),
                dS_dt_from_flux=divergence_per_graph.mean().item(),
                dS_dt_from_physics=dS_dt_per_graph.mean().item()
            )
        else:
            # Single graph case
            return PhysicsMetrics(
                J_ax=edge_fluxes_pred.detach().abs().mean().item() if edge_fluxes_pred.size(0) > 0 else 0.0,
                F_in=F_in_pred.detach().mean().item(),
                F_out=F_out_pred.detach().mean().item(),
                dS_dt=dS_dt_from_physics_pred.detach().abs().mean().item(),
                dS_dt_from_flux=divergence_pred.detach().abs().mean().item(),
                dS_dt_from_physics=dS_dt_from_physics_pred.detach().abs().mean().item()
            )
    else:
        # NNConv model: reconstruct fluxes from predictions
        J_ax_pred = compute_axial_flux(C_ST_pred, node_feat_original, edge_feat_original,
                                       edge_index, batch_vec, device, y_pred.dtype)
        dS_dt_from_flux_pred = compute_flux_divergence(J_ax_pred, edge_index, N, device)

        F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
        F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, device)
        dS_dt_from_physics_pred = dS_dt_from_flux_pred + F_in_pred - F_out_pred

        # Compute true values
        J_ax_true = compute_axial_flux(C_ST_true, node_feat_original, edge_feat_original,
                                       edge_index, batch_vec, device, y_pred.dtype)
        dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

        F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
        F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, device)
        dS_dt_from_physics_true = dS_dt_from_flux_true + F_in_true - F_out_true

        with open(debug_path, "a") as f:
            msg = (
                f"\n{'='*60}\n"
                f"DEBUG OUTPUT - NNCONV MODEL (DATA-ONLY MODE)\n"
                f"{'='*60}\n"
                f"\nNumber of graphs: {torch.bincount(batch_vec).size(0) if batch_vec is not None else 1}\n"
            )
            if batch_vec is not None:
                msg += f"Nodes per graph: {torch.bincount(batch_vec).detach().cpu().numpy()}\n"

            msg += (
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

                f"\n--- TIME DERIVATIVES (mmol/h) ---\n"
                f"dS_dt_from_flux_true: {dS_dt_from_flux_true[:10].detach().cpu().numpy()}\n"
                f"dS_dt_from_flux_pred: {dS_dt_from_flux_pred[:10].detach().cpu().numpy()}\n"

                f"\ndS_dt_from_physics_true (total): {dS_dt_from_physics_true[:10].detach().cpu().numpy()}\n"
                f"dS_dt_from_physics_pred (total): {dS_dt_from_physics_pred[:10].detach().cpu().numpy()}\n"

                f"\n--- COMPARISON METRICS ---\n"
                f"Flux MSE: {((J_ax_true - J_ax_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"Divergence MSE: {((dS_dt_from_flux_true - dS_dt_from_flux_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"F_in MSE: {((F_in_true - F_in_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"F_out MSE: {((F_out_true - F_out_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"Total residual MSE: {((dS_dt_from_physics_true - dS_dt_from_physics_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"{'='*60}\n"
            )
            f.write(msg)

        # Compute averaged metrics for terminal display (NNConv model)
        from torch_scatter import scatter_mean
        if batch_vec is not None:
            # Batched case: compute per-graph averages
            F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
            F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
            dS_dt_from_flux_per_graph = scatter_mean(dS_dt_from_flux_pred.detach().abs(), batch_vec, dim=0)
            dS_dt_per_graph = scatter_mean(dS_dt_from_physics_pred.detach().abs(), batch_vec, dim=0)

            if J_ax_pred.size(0) > 0:
                edge_batch = batch_vec[edge_index[0].to(device)]
                J_ax_per_graph = scatter_mean(J_ax_pred.detach().abs(), edge_batch, dim=0)
                J_ax_avg = J_ax_per_graph.mean().item()
            else:
                J_ax_avg = 0.0

            return PhysicsMetrics(
                J_ax=J_ax_avg,
                F_in=F_in_per_graph.mean().item(),
                F_out=F_out_per_graph.mean().item(),
                dS_dt=dS_dt_per_graph.mean().item(),
                dS_dt_from_flux=dS_dt_from_flux_per_graph.mean().item(),
                dS_dt_from_physics=dS_dt_per_graph.mean().item()
            )
        else:
            # Single graph case
            return PhysicsMetrics(
                J_ax=J_ax_pred.detach().abs().mean().item() if J_ax_pred.size(0) > 0 else 0.0,
                F_in=F_in_pred.detach().mean().item(),
                F_out=F_out_pred.detach().mean().item(),
                dS_dt=dS_dt_from_physics_pred.detach().abs().mean().item(),
                dS_dt_from_flux=dS_dt_from_flux_pred.detach().abs().mean().item(),
                dS_dt_from_physics=dS_dt_from_physics_pred.detach().abs().mean().item()
            )


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

    # Add penalty for negative concentrations (after denormalization)
    # This encourages the model to respect the physical constraint C_ST >= 0
    negative_concentration_penalty = torch.relu(-C_ST_pred).pow(2).mean()

    # Use adaptive normalization based on current residual scale
    # Use 90th percentile of absolute values as scale (robust to outliers)
    with torch.no_grad():
        scale = residual.abs().quantile(0.9).clamp(min=0.1, max=10000.0)

    # Normalize residual by adaptive scale
    residual_node = (residual / scale).pow(2)

    # Average per graph first, then across graphs
    if batch_vec is not None:
        residual_per_graph = scatter_mean(residual_node, batch_vec, dim=0)
        physics_loss = residual_per_graph.mean()
    else:
        physics_loss = residual_node.mean()

    # Combine physics residual with non-negativity penalty
    # Weight the penalty strongly to enforce physical constraint
    loss = physics_loss + PENALTY_WEIGHT * negative_concentration_penalty

    # Log penalty monitoring information
    if DEBUG:
        with open(debug_path, "a") as f:
            ratio = (PENALTY_WEIGHT * negative_concentration_penalty) / (physics_loss + 1e-10)
            msg = (
                f"\n--- PENALTY MONITORING (NNConv) ---\n"
                f"Physics loss: {physics_loss.item():.6e}\n"
                f"Penalty (unweighted): {negative_concentration_penalty.item():.6e}\n"
                f"Penalty (weighted): {(PENALTY_WEIGHT * negative_concentration_penalty).item():.6e}\n"
                f"Penalty weight: {PENALTY_WEIGHT:.1f}\n"
                f"Total loss: {loss.item():.6e}\n"
                f"Penalty/Physics ratio: {ratio.item():.2f}\n"
            )
            f.write(msg)

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


def physics_residual_operator(
    model_output: dict,
    data: Data
) -> tuple[torch.Tensor, dict]:
    """Compute physics residual for operator-based GNN.

    Unlike physics_residual which reconstructs fluxes from predicted concentrations,
    this function uses the edge fluxes and divergences directly predicted by the
    operator model.

    The conservation law is:
        dS/dt = div(J) + F_in - F_out ≈ 0

    where:
        - div(J) is already computed by the model (from edge fluxes)
        - F_in and F_out are computed from predicted concentrations

    Args:
        model_output: Dict containing:
            - 'predictions': [N, 1] sucrose content predictions (standardized)
            - 'edge_fluxes': [E] predicted edge fluxes
            - 'divergences': [N] divergence values
        data: Graph data containing features and parameters

    Returns:
        tuple: (residual_loss, physics_components_dict)
    """
    device = model_output['predictions'].device
    y_pred = model_output['predictions']
    edge_fluxes_pred = model_output['edge_fluxes']
    divergence_pred = model_output['divergences']

    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

    # Inverse-transform features to original space
    node_feat_standardized = data.node_feat.to(device)
    node_feat_original = data.feature_scaler.inv_transform(node_feat_standardized)

    edge_index = data.edge_index.to(device)
    vol_ST = node_feat_original[:, 1]

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # Convert predicted content to concentration
    S_ST_pred = data.target_scaler.inv_transform(y_pred).squeeze(-1)
    C_ST_pred = S_ST_pred / vol_ST

    # Compute source/sink terms from predicted concentrations
    F_in_pred = compute_phloem_loading(C_ST_pred, node_feat_original, params, node_fields, device)
    F_out_pred = compute_sucrose_outflow(C_ST_pred, node_feat_original, params, node_fields, device)

    # Physics residual using model's divergence directly
    # dS/dt = divergence + F_in - F_out ≈ 0
    dS_dt_from_physics_pred = divergence_pred + F_in_pred - F_out_pred

    if DEBUG:
        # For debugging, compute ground truth physics terms
        y_true = data.y.to(device)
        S_ST_true = data.target_scaler.inv_transform(y_true).squeeze(-1)
        C_ST_true = S_ST_true / vol_ST

        # Reconstruct true fluxes for comparison
        edge_feat_standardized = data.edge_feat.to(device)
        edge_feat_original = data.edge_scaler.inv_transform(edge_feat_standardized)

        J_ax_true = compute_axial_flux(
            C_ST_true, node_feat_original, edge_feat_original,
            edge_index, batch_vec, device, y_pred.dtype
        )
        dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

        F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
        F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, device)
        dS_dt_from_physics_true = dS_dt_from_flux_true + F_in_true - F_out_true

        with open(debug_path, "a") as f:
            msg = (
                f"\n{'='*60}\n"
                f"DEBUG OUTPUT - OPERATOR MODEL PHYSICS RESIDUAL\n"
                f"{'='*60}\n"
                f"\nNumber of graphs: {torch.bincount(batch_vec).size(0) if batch_vec is not None else 1}\n"
            )
            if batch_vec is not None:
                msg += f"Nodes per graph: {torch.bincount(batch_vec).detach().cpu().numpy()}\n"

            msg += (
                f"\n--- CONCENTRATION VALUES (mmol/cm³) ---\n"
                f"C_ST_true: {C_ST_true[:10].detach().cpu().numpy()}\n"
                f"C_ST_pred: {C_ST_pred[:10].detach().cpu().numpy()}\n"

                f"\n--- EDGE FLUXES (mmol/h) ---\n"
                f"J_ax_true (reconstructed from physics):\n"
                f"  First 10: {J_ax_true[:10].detach().cpu().numpy()}\n"
                f"  Mean: {J_ax_true.mean().detach().cpu().item():.6e}\n"
                f"  Std: {J_ax_true.std().detach().cpu().item():.6e}\n"
                f"J_ax_pred (from operator model):\n"
                f"  First 10: {edge_fluxes_pred[:10].detach().cpu().numpy()}\n"
                f"  Mean: {edge_fluxes_pred.mean().detach().cpu().item():.6e}\n"
                f"  Std: {edge_fluxes_pred.std().detach().cpu().item():.6e}\n"

                f"\n--- DIVERGENCE (mmol/h) ---\n"
                f"Divergence_true (reconstructed):\n"
                f"  First 10: {dS_dt_from_flux_true[:10].detach().cpu().numpy()}\n"
                f"  Mean: {dS_dt_from_flux_true.mean().detach().cpu().item():.6e}\n"
                f"  Std: {dS_dt_from_flux_true.std().detach().cpu().item():.6e}\n"
                f"Divergence_pred (from operator model):\n"
                f"  First 10: {divergence_pred[:10].detach().cpu().numpy()}\n"
                f"  Mean: {divergence_pred.mean().detach().cpu().item():.6e}\n"
                f"  Std: {divergence_pred.std().detach().cpu().item():.6e}\n"

                f"\n--- SOURCE/SINK TERMS (mmol/h) ---\n"
                f"F_in_true: {F_in_true[:10].detach().cpu().numpy()}\n"
                f"F_in_pred: {F_in_pred[:10].detach().cpu().numpy()}\n"
                f"F_in mean - true: {F_in_true.mean().detach().cpu().item():.6e}, pred: {F_in_pred.mean().detach().cpu().item():.6e}\n"
                f"\n"
                f"F_out_true: {F_out_true[:10].detach().cpu().numpy()}\n"
                f"F_out_pred: {F_out_pred[:10].detach().cpu().numpy()}\n"
                f"F_out mean - true: {F_out_true.mean().detach().cpu().item():.6e}, pred: {F_out_pred.mean().detach().cpu().item():.6e}\n"

                f"\n--- TOTAL PHYSICS RESIDUAL (mmol/h) ---\n"
                f"dS_dt_from_physics_true (total):\n"
                f"  First 10: {dS_dt_from_physics_true[:10].detach().cpu().numpy()}\n"
                f"  Mean absolute: {dS_dt_from_physics_true.abs().mean().detach().cpu().item():.6e}\n"
                f"dS_dt_from_physics_pred (total):\n"
                f"  First 10: {dS_dt_from_physics_pred[:10].detach().cpu().numpy()}\n"
                f"  Mean absolute: {dS_dt_from_physics_pred.abs().mean().detach().cpu().item():.6e}\n"

                f"\n--- COMPARISON METRICS ---\n"
                f"Flux MSE: {((J_ax_true - edge_fluxes_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"Divergence MSE: {((dS_dt_from_flux_true - divergence_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"F_in MSE: {((F_in_true - F_in_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"F_out MSE: {((F_out_true - F_out_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"Total residual MSE: {((dS_dt_from_physics_true - dS_dt_from_physics_pred).pow(2).mean()).detach().cpu().item():.6e}\n"
                f"{'='*60}\n"
            )
            f.write(msg)

    # Compute loss
    residual = dS_dt_from_physics_pred

    # Add penalty for negative concentrations (after denormalization)
    # This encourages the model to respect the physical constraint C_ST >= 0
    negative_concentration_penalty = torch.relu(-C_ST_pred).pow(2).mean()

    # Adaptive normalization
    with torch.no_grad():
        scale = residual.abs().quantile(0.9).clamp(min=0.1, max=10000.0)

    residual_node = (residual / scale).pow(2)

    # Average per graph first, then across graphs
    if batch_vec is not None:
        residual_per_graph = scatter_mean(residual_node, batch_vec, dim=0)
        physics_loss = residual_per_graph.mean()
    else:
        physics_loss = residual_node.mean()

    # Combine physics residual with non-negativity penalty
    # Weight the penalty strongly to enforce physical constraint
    loss = physics_loss + PENALTY_WEIGHT * negative_concentration_penalty

    # Log penalty monitoring information
    if DEBUG:
        with open(debug_path, "a") as f:
            ratio = (PENALTY_WEIGHT * negative_concentration_penalty) / (physics_loss + 1e-10)
            msg = (
                f"\n--- PENALTY MONITORING (Operator) ---\n"
                f"Physics loss: {physics_loss.item():.6e}\n"
                f"Penalty (unweighted): {negative_concentration_penalty.item():.6e}\n"
                f"Penalty (weighted): {(PENALTY_WEIGHT * negative_concentration_penalty).item():.6e}\n"
                f"Penalty weight: {PENALTY_WEIGHT:.1f}\n"
                f"Total loss: {loss.item():.6e}\n"
                f"Penalty/Physics ratio: {ratio.item():.2f}\n"
            )
            f.write(msg)

    # Prepare loss dict for logging
    if batch_vec is not None:
        F_in_per_graph = scatter_mean(F_in_pred.detach(), batch_vec, dim=0)
        F_out_per_graph = scatter_mean(F_out_pred.detach(), batch_vec, dim=0)
        divergence_per_graph = scatter_mean(divergence_pred.detach().abs(), batch_vec, dim=0)
        dS_dt_per_graph = scatter_mean(dS_dt_from_physics_pred.detach().abs(), batch_vec, dim=0)

        # Edge-level: map edges to graphs
        if edge_fluxes_pred.size(0) > 0:
            edge_batch = batch_vec[edge_index[0].to(device)]
            J_ax_per_graph = scatter_mean(edge_fluxes_pred.detach().abs(), edge_batch, dim=0)
            J_ax_avg = J_ax_per_graph.mean()
        else:
            J_ax_avg = torch.tensor(0.0, device=device)

        loss_dict = {
            'J_ax': J_ax_avg,
            'F_in': F_in_per_graph.mean(),
            'F_out': F_out_per_graph.mean(),
            'dS_dt': dS_dt_per_graph.mean(),
            'dS_dt_from_flux': divergence_per_graph.mean(),
            'dS_dt_from_physics': dS_dt_per_graph.mean()
        }
    else:
        loss_dict = {
            'J_ax': edge_fluxes_pred.detach().abs().mean() if edge_fluxes_pred.size(0) > 0 else torch.tensor(0.0, device=device),
            'F_in': F_in_pred.detach().mean(),
            'F_out': F_out_pred.detach().mean(),
            'dS_dt': dS_dt_from_physics_pred.detach().abs().mean(),
            'dS_dt_from_flux': divergence_pred.detach().abs().mean(),
            'dS_dt_from_physics': dS_dt_from_physics_pred.detach().abs().mean()
        }

    return loss, loss_dict
