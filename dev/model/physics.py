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


def compute_time_derivative(y_pred: torch.Tensor, data: Data) -> torch.Tensor:
    """Compute time derivative of y_pred (sucrose content) from model predictions.

    Args:
        y_pred: Predicted sucrose content [N, 1] - must be connected to data.time_per_node
        data: Graph data containing time features

    Returns:
        torch.Tensor: Time derivative dy/dt per node [N]

    Raises:
        ValueError: If y_pred is not connected to data.time_per_node in computation graph
    """

    # We need dS/dt from the model with respect to a differentiable time feature
    # This is ESSENTIAL for physics-informed learning: without it, the physics constraint is meaningless
    if not hasattr(data, 'time_per_node') or data.time_per_node is None:
        raise ValueError("data.time_per_node not found. Required for physics residual computation.")
    if not data.time_per_node.requires_grad:
        raise ValueError("data.time_per_node must have requires_grad=True for physics residual computation.")

    # Compute gradient of predictions w.r.t. time_per_node [N,1]
    try:
        dy_dt = torch.autograd.grad(
            y_pred.sum(),        # sum to get scalar for gradient computation
            data.time_per_node,  # [N, 1] per-node time features
            create_graph=True,   # needed for second backward pass
            retain_graph=True,   # keep graph for subsequent loss computation
            allow_unused=False   # ERROR if time_per_node is not connected
        )[0]
        dy_dt = dy_dt.squeeze()
    except RuntimeError as e:
        if "not have been used in the graph" in str(e):
            raise ValueError(
                "Physics residual computation failed: y_pred is not connected to time_per_node. "
                "This indicates that the model predictions don't depend on time, which breaks the physics constraint. "
                "Ensure that y_pred comes from a model forward pass that uses the same data object, "
                "or that the model architecture properly utilizes the time feature."
            ) from e
        else:
            raise

    return dy_dt


def physics_residual(y_pred: torch.Tensor, data: Data):
    """Compute physics-informed residual term based on sucrose transport equations.

    Implements the governing equation for content-based sucrose transport in sieve-tubes:
    dS/dt = J_ax + (F_in - F_out)

    where:
    - J_ax is the axial sucrose flux
    - F_in is the phloem loading rate
    - F_out is the sucrose outflow

    IMPORTANT: y_pred MUST come from a model forward pass using the same data object,
    so that data.time_per_node is properly connected to y_pred in the computation graph.
    Without this connection, dy/dt cannot be computed and the physics constraint is meaningless.

    Args:
        y_pred: Predicted sucrose content [N, 1] -> MUST be connected to data.time_per_node
        data: Graph data containing topology, features, simulation parameters, and node fields

    Returns:
        tuple: (residual_loss, physics_components_dict) where physics_components_dict contains
            {'J_ax', 'F_in', 'F_out', 'dy_dt', 'dS_dt_from_flux'}

    Raises:
        ValueError: If y_pred is not connected to data.time_per_node in the computation graph
    """
    device = y_pred.device
    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

    # Inverse-transform node/edge features and targets
    node_feat_standardized = data.node_feat.to(device)
    node_feat_original = data.feature_scaler.inv_transform(node_feat_standardized)

    edge_feat_standardized = data.edge_feat.to(device)
    edge_feat_original = data.edge_scaler.inv_transform(edge_feat_standardized)

    edge_index = data.edge_index.to(device)
    vol_ST = node_feat_original[:, 1]

    S_ST = data.target_scaler.inv_transform(y_pred).squeeze(-1)

    # Units in HDF5:
    # - S_ST (Q_ST in HDF5) is in MILLIMOLES (mmol)
    # - vol_ST is in cm^3
    # - Kinetic parameters are calibrated for concentrations in mmol/cm^3
    # Therefore: NO conversion needed: use C_ST = S_ST / vol_ST directly
    C_ST = S_ST / vol_ST

    # Use true values for testing coherence of physics calculations
    if DEBUG:
        y_true = data.y.clone().detach()
        y_true.requires_grad_(True)
        S_ST_true = data.target_scaler.inv_transform(y_true).squeeze(-1)
        C_ST_true = S_ST_true / vol_ST

        # DEBUG: Print actual concentration values to check units
        with open(debug_path, "a") as f:
            msg = (
                f"\n{'='*60}\n"
                f"DEBUG OUTPUT\n"
                f"{'='*60}\n"
                f"\nNumber of graphs in batch: {torch.bincount(batch_vec).size(0)}\n"
                f"Number of nodes per graph: {torch.bincount(batch_vec).detach().cpu().numpy()}\n"
            )
            f.write(msg)

            msg = (
                f"\n--- CONCENTRATION VALUES (should be ~0.2-1.0 mol/L = 0.2-1.0 mmol/cm³) ---\n"
                f"S_ST_true (mmol): {S_ST_true[:10].detach().cpu().numpy()}\n"
                f"vol_ST (cm³): {vol_ST[:10].detach().cpu().numpy()}\n"
                f"C_ST_true (mmol/cm³): {C_ST_true[:10].detach().cpu().numpy()}\n"
                f"C_ST_pred (mmol/cm³): {C_ST[:10].detach().cpu().numpy()}\n"
            )
            f.write(msg)

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # Compute axial flux and its divergence (pass C_ST directly)
    J_ax = compute_axial_flux(C_ST, node_feat_original, edge_feat_original, edge_index, batch_vec, device, y_pred.dtype)
    dS_dt_from_flux = compute_flux_divergence(J_ax, edge_index, N, device)

    # Compute phloem loading and outflow (pass C_ST directly)
    F_in = compute_phloem_loading(C_ST, node_feat_original, params, node_fields, device)
    F_out = compute_sucrose_outflow(C_ST, node_feat_original, params, node_fields, device)

    # NO UNIT CONVERSION NEEDED:
    # - Time is now in HOURS (converted from days in dataset_loader.py)
    # - All flux parameters (Q_Rmmax, Q_Grmax, Q_Exudmax, conductivities) are in mmol/h
    # - Therefore dS/dt is naturally in mmol/h, matching all flux terms
    # - This matches the C++ simulation units exactly
    # Compute time derivative from model
    dy_dt = compute_time_derivative(y_pred, data)

    # Compute physics-based derivative in physical (original) space
    dS_dt_from_physics_original = dS_dt_from_flux + F_in - F_out

    # CRITICAL FIX: Work entirely in standardized space to avoid scale mismatch
    #
    # The issue: Converting dy_dt from standardized to physical space using (σ_S / σ_t)
    # creates a huge scale mismatch because:
    #   - σ_S ≈ 2.3e-5 mol (targets are very small)
    #   - σ_t ≈ 0.239 days (time has small variation)
    #   - Conversion factor ≈ 9.7e-5 (extremely small!)
    #   - This makes dy_dt_physical ~6 orders of magnitude smaller than physics terms
    #
    # The solution: Instead of converting dy_dt to physical space, convert physics terms
    # to standardized space. This keeps all quantities in the same scale.
    #
    # Standardize the physics-based derivative:
    #   dS_std/dt_std = (dS_phys/dt_phys) × (σ_t / σ_S)
    #
    # This is the inverse of the previous conversion, and now both sides are O(1)

    sigma_target = data.target_scaler.std.view(-1)[0].to(device)  # σ_S
    sigma_time = data.time_scaler.std.view(-1)[0].to(device)      # σ_t

    # Convert physics derivative to standardized space
    dS_dt_from_physics = dS_dt_from_physics_original * (sigma_time / sigma_target)

    # For logging: compute physical dy_dt (even though we don't use it for loss)
    dy_dt_physical = dy_dt * (sigma_target / sigma_time)

    if DEBUG:
        with open(debug_path, "a") as f:
            J_ax_true = compute_axial_flux(C_ST_true, node_feat_original, edge_feat_original, edge_index, batch_vec, device, y_pred.dtype)
            dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, edge_index, N, device)

            F_in_true = compute_phloem_loading(C_ST_true, node_feat_original, params, node_fields, device)
            F_out_true = compute_sucrose_outflow(C_ST_true, node_feat_original, params, node_fields, device)

            dS_dt_from_physics_true = dS_dt_from_flux_true + F_in_true - F_out_true

            # Show predictions in their native units (content)
            # Show both standardized and physical values
            msg = (
                f"\n--- STANDARDIZED VALUES ---\n"
                f"y_true (standardized):\n{y_true[batch_vec == 0].squeeze(-1).detach().cpu().numpy()}\n"
                f"y_pred (standardized):\n{y_pred[batch_vec == 0].squeeze(-1).detach().cpu().numpy()}\n"
            )
            f.write(msg)

            # Compute physical values using target_scaler
            S_ST_true_physical = data.target_scaler.inv_transform(y_true).squeeze(-1)[batch_vec == 0]
            S_ST_pred_physical = data.target_scaler.inv_transform(y_pred).squeeze(-1)[batch_vec == 0]

            msg = (
                f"\n--- PHYSICAL VALUES ---\n"
                f"S_ST_true (mol):\n{S_ST_true_physical.detach().cpu().numpy()}\n"
                f"S_ST_pred (mol):\n{S_ST_pred_physical.detach().cpu().numpy()}\n"

                # NOTE: All flux values are in mmol/h (C++ units)
                # Time is now in HOURS (converted from days in dataset_loader.py)
                # Therefore dS/dt is also in mmol/h, matching C++ simulation
                f"\n--- FLUX VALUES (mmol/h, for direct comparison with C++ output) ---\n"
                f"F_in_true (mmol/h):\n{F_in_true[batch_vec == 0].detach().cpu().numpy()[:10]}\n"
                f"F_in_pred (mmol/h):\n{F_in[batch_vec == 0].detach().cpu().numpy()[:10]}\n"

                f"\nF_out_true (mmol/h):\n{F_out_true[batch_vec == 0].detach().cpu().numpy()[:10]}\n"
                f"F_out_pred (mmol/h):\n{F_out[batch_vec == 0].detach().cpu().numpy()[:10]}\n"

                f"\ndS_dt_from_flux_true (mmol/h):\n{dS_dt_from_flux_true[batch_vec == 0].detach().cpu().numpy()[:10]}\n"
                f"dS_dt_from_flux_pred (mmol/h):\n{dS_dt_from_flux[batch_vec == 0].detach().cpu().numpy()[:10]}\n"

                f"\ndS_dt_from_physics_true (mmol/h):\n{dS_dt_from_physics_true[batch_vec == 0].detach().cpu().numpy()[:10]}\n"
                f"dS_dt_from_physics_pred (mmol/h):\n{dS_dt_from_physics_original[batch_vec == 0].detach().cpu().numpy()[:10]}\n"

                # Show model's time derivative in both standardized and physical space
                # Note: Cannot compute dy_dt for y_true since it's not connected to time_per_node in computation graph
                f"\ndS_dt_pred (standardized) [from model]:\n{dy_dt[batch_vec == 0].detach().cpu().numpy()[:10]}\n"
                f"dS_dt_from_physics (standardized) [converted from physical]:\n{dS_dt_from_physics[batch_vec == 0].detach().cpu().numpy()[:10]}\n"
                # f"\ndS_dt_pred (mmol/h) [from model, physical space]:\n{dy_dt_physical[batch_vec == 0].detach().cpu().numpy()[:10]}\n"
                f"\nConversion factor (sigma_S/sigma_t): {(sigma_target / sigma_time).item():.6e}\n"
                f"Inverse conversion factor (sigma_t/sigma_S): {(sigma_time / sigma_target).item():.6e}\n"

                f"{'='*60}\n"
            )
            f.write(msg)

    # Use adaptive normalization based on current residual scale
    #
    # Compute characteristic scale from this batch (robust to outliers using median-like measure)
    # Use 90th percentile of absolute differences as scale (more robust than max)
    #
    # Without this approach, the residual's behavior would be:
    # With small values, gradients vanish -> physics loss is ignored
    # With huge values (when model outputs noisy dS/dt) -> physics dominates and collapses the model
    #
    # CRITICAL FIX: Now comparing dy_dt with dS_dt_from_physics (both in standardized space)
    # This avoids the huge scale mismatch from the (σ_S / σ_t) conversion factor
    diff = dy_dt.squeeze() - dS_dt_from_physics

    with torch.no_grad():
        scale = diff.abs().quantile(0.9).clamp(min=0.1, max=10000.0)

    # Normalize residual by adaptive scale
    residual_node = (diff / scale).pow(2)

    # Average per graph first, then across graphs
    if batch_vec is not None:
        residual_per_graph = scatter_mean(residual_node, batch_vec, dim=0)
        loss = residual_per_graph.mean()
    else:
        loss = residual_node.mean()

    # Prepare detailed physics components for logging
    # Compute per-graph averages first, then average across graphs (consistent with loss computation)
    if batch_vec is not None:
        # Batched case: compute per-graph averages using scatter_mean
        # Node-level quantities: average per graph, then across graphs
        F_in_per_graph = scatter_mean(F_in.detach(), batch_vec, dim=0)
        F_out_per_graph = scatter_mean(F_out.detach(), batch_vec, dim=0)
        dS_dt_per_graph = scatter_mean(dy_dt.detach().abs(), batch_vec, dim=0)
        dS_dt_from_flux_per_graph = scatter_mean(dS_dt_from_flux.detach().abs(), batch_vec, dim=0)
        dS_dt_from_physics_per_graph = scatter_mean(dS_dt_from_physics.detach().abs(), batch_vec, dim=0)

        # Edge-level quantities: need edge-to-graph mapping
        if J_ax.size(0) > 0:
            # Create edge batch vector by mapping edges to their source node's graph
            edge_batch = batch_vec[data.edge_index[0].to(device)]
            J_ax_per_graph = scatter_mean(J_ax.detach().abs(), edge_batch, dim=0)
            J_ax_avg = J_ax_per_graph.mean()
        else:
            J_ax_avg = torch.tensor(0.0, device=device)

        loss_dict = {
            'J_ax': J_ax_avg,
            'F_in': F_in_per_graph.mean(),
            'F_out': F_out_per_graph.mean(),
            'dS_dt': dS_dt_per_graph.mean(),
            'dS_dt_from_flux': dS_dt_from_flux_per_graph.mean(),
            'dS_dt_from_physics': dS_dt_from_physics_per_graph.mean()
        }
    else:
        # Single graph case: simple mean across nodes/edges
        loss_dict = {
            'J_ax': J_ax.detach().abs().mean() if J_ax.size(0) > 0 else torch.tensor(0.0, device=device),
            'F_in': F_in.detach().mean(),
            'F_out': F_out.detach().mean(),
            'dS_dt': dy_dt.detach().abs().mean(),
            'dS_dt_from_flux': dS_dt_from_flux.detach().abs().mean(),
            'dS_dt_from_physics': dS_dt_from_physics.detach().abs().mean()
        }

    return loss, loss_dict