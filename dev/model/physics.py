from __future__ import annotations

import torch

from torch_scatter import scatter_mean
from torch_geometric.data import Data

from . import utils
from . import config

DEBUG = False  # Debug flag: set to True to enable detailed physics loss debugging

def denormalize_to_concentration(
    y_pred: torch.Tensor,
    vol_ST: torch.Tensor,
    data: Data,
    device: torch.device
) -> torch.Tensor:
    """Convert normalized predictions to concentrations.

    Handles denormalization and conversion from sucrose content to concentration,
    accounting for batched graphs where target_scale may vary per graph.

    Args:
        y_pred: Predicted sucrose content [N, 1] (normalized)
        vol_ST: Sieve-tube volume per node [N]
        data: Graph data containing target_scale and batch info
        device: Target device for computations

    Returns:
        torch.Tensor: Sucrose concentration C_ST per node [N] (mol/cm³)
    """
    # Get target scale for denormalization
    target_scale = getattr(
        data, 'target_scale',
        torch.tensor(1.0, device=device, dtype=y_pred.dtype)
    ).to(device)

    # Handle batched case: target_scale is [B] but we need [N]
    batch_vec = getattr(data, "batch", None)
    if batch_vec is not None and target_scale.numel() > 1:
        target_scale_per_node = target_scale[batch_vec]
    else:
        target_scale_per_node = target_scale

    # Denormalize and convert to concentration
    S_ST_normalized = y_pred.squeeze(-1)
    S_ST = S_ST_normalized * target_scale_per_node
    C_ST = S_ST / vol_ST

    return C_ST


def compute_axial_flux(y_pred: torch.Tensor, data: Data, device: torch.device) -> torch.Tensor:
    """Compute axial sucrose flux J_ax along edges.

    This implementation follows the C++ PiafMunch algorithm (external/PiafMunch/solve.cpp):
    1. Compute osmotic pressure P_ST = C_ST * RT for each node
    2. Compute water flux JW_ST based on pressure gradients
    3. Select upstream concentration based on flow direction
    4. Compute sugar flux JS_ST = JW_ST * C_upstream

    Args:
        y_pred: Predicted sucrose content [N, 1]
        data: Graph data containing topology and features
        device: Target device for computations

    Returns:
        torch.Tensor: Axial flux per edge [E]
    """
    # Get edge topology and features
    edge_index = data.edge_index.to(device)  # [2, E]

    # Handle empty graph case
    if edge_index.size(1) == 0:
        return torch.zeros(0, device=device, dtype=y_pred.dtype)

    src, dst = edge_index[0], edge_index[1]
    r_ST = data.edge_feat.to(device).squeeze(-1)  # [E, 1] -> [E]

    # Node features already in original space
    node_feat = data.node_feat.to(device)
    psi = node_feat[:, 0]  # hydraulic potential
    vol_ST = node_feat[:, 1]  # sieve-tube volume per node
    Temp = node_feat[:, 6]  # temperature [°C]

    # ---- Step 1: Get concentrations from predictions
    # Content: predictions are S_ST (normalized), need to denormalize and divide by vol_ST
    C_ST = denormalize_to_concentration(y_pred, vol_ST, data, device)

    # Osmotic pressure P_ST = C_ST * RT
    # In C++, TairK_phloem is global, but in batched case we need per-graph temperature
    batch_vec = getattr(data, "batch", None)
    RT = utils.compute_RT_per_node(
        Temp=Temp,
        batch_vec=batch_vec,
        R=config.R,
        device=device,
        dtype=y_pred.dtype,
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


def compute_phloem_loading(y_pred: torch.Tensor, data: Data, params: dict, node_fields: dict, device: torch.device) -> torch.Tensor:
    """Compute phloem loading rate F_in per node.

    Args:
        y_pred: Predicted sucrose content [N, 1]
        data: Graph data containing node features
        params: Simulation and step parameters
        node_fields: Node field values
        device: Target device for computations

    Returns:
        torch.Tensor: Phloem loading rate per node [N]
    """
    node_feat = data.node_feat.to(device)
    vol_ST = node_feat[:, 1]
    len_leaf = node_feat[:, 2]

    # Convert to concentration
    CSTi = denormalize_to_concentration(y_pred, vol_ST, data, device)

    CSTi_positive = torch.clamp(CSTi, min=0.0)

    # Phloem loading with feedback inhibition
    F_in = (params["Vmaxloading"] * len_leaf) * node_fields["C_meso"] / \
           (params["Mloading"] + node_fields["C_meso"]) * \
           torch.exp(-CSTi_positive * params["beta_loading"])

    return F_in


def compute_sucrose_outflow(y_pred: torch.Tensor, data: Data, params: dict, node_fields: dict, device: torch.device) -> torch.Tensor:
    """Compute sucrose outflow F_out per node.

    Args:
        y_pred: Predicted sucrose content [N, 1]
        data: Graph data containing node features
        params: Simulation and step parameters
        node_fields: Node field values
        device: Target device for computations

    Returns:
        torch.Tensor: Sucrose outflow rate per node [N]
    """
    node_feat = data.node_feat.to(device)
    vol_ST = node_feat[:, 1]
    Q_Rmmax = node_feat[:, 3]
    Q_Grmax = node_feat[:, 4]
    Q_Exudmax = node_feat[:, 5]
    Temp = node_feat[:, 6]

    # Convert to concentration
    CSTi = denormalize_to_concentration(y_pred, vol_ST, data, device)

    CSTi_positive = torch.clamp(CSTi, min=0.0)

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
            data.time_per_node,      # [N, 1] per-node time features
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

    # Get vol_ST for concentration-to-content conversion
    node_feat = data.node_feat.to(device)
    vol_ST = node_feat[:, 1]

    # TEMPORARY: Use true values for testing coherence of physics calculations
    if DEBUG:
        y_true = data.y.clone().detach()
        y_true.requires_grad_(True)

    # Extract parameters and node fields
    params = utils.extract_parameters(data, device, batch_vec, N if batch_vec is None else None)
    node_fields = utils.extract_node_fields(data, device)

    # Compute axial flux and its divergence
    J_ax = compute_axial_flux(y_pred, data, device)
    dS_dt_from_flux = compute_flux_divergence(J_ax, data.edge_index.to(device), N, device)

    # Compute phloem loading and outflow
    F_in = compute_phloem_loading(y_pred, data, params, node_fields, device)
    F_out = compute_sucrose_outflow(y_pred, data, params, node_fields, device)

    # Compute time derivative from model
    dy_dt = compute_time_derivative(y_pred, data)

    # Compute physics-based derivative (always compute dC/dt from physics)
    dS_dt_from_physics = dS_dt_from_flux + F_in - F_out
    dC_dt_from_physics = dS_dt_from_physics / vol_ST

    # Convert model's dy_dt to dC/dt and compute residual
    # Content mode: compare dS/dt directly
    dy_dt_from_physics = dS_dt_from_physics
    dC_dt = dy_dt / vol_ST  # for logging only

    if DEBUG:
        J_ax_true = compute_axial_flux(y_true, data, device)
        dS_dt_from_flux_true = compute_flux_divergence(J_ax_true, data.edge_index.to(device), N, device)
        F_in_true = compute_phloem_loading(y_true, data, params, node_fields, device)
        F_out_true = compute_sucrose_outflow(y_true, data, params, node_fields, device)
        dS_dt_from_physics_true = dS_dt_from_flux_true + F_in_true - F_out_true
        dC_dt_from_physics_true = dS_dt_from_physics_true / vol_ST

        print(f"\n{'='*60}")
        print(f"DEBUG OUTPUT")
        print(f"{'='*60}")
        print(f"\nNumber of graphs in batch: {torch.bincount(batch_vec).size(0)}")
        print(f"Number of nodes per graph: {torch.bincount(batch_vec).detach().cpu().numpy()}")

        # Get target_scale for denormalization
        target_scale_debug = getattr(data, 'target_scale', torch.tensor(1.0, device=device, dtype=y_pred.dtype)).to(device)
        if batch_vec is not None and target_scale_debug.numel() > 1:
            target_scale_per_node_debug = target_scale_debug[batch_vec]
        else:
            target_scale_per_node_debug = target_scale_debug

        # Show predictions in their native units (content)
        # Show both normalized [0,1] and physical values
        print(f"\n--- NORMALIZED VALUES [0,1] ---")
        print(f"y_true (normalized):\n{y_true[batch_vec == 0].squeeze(-1).detach().cpu().numpy()[:10]}")
        print(f"y_pred (normalized):\n{y_pred[batch_vec == 0].squeeze(-1).detach().cpu().numpy()[:10]}")

        # Compute physical values
        S_ST_true_physical = (y_true.squeeze(-1) * target_scale_per_node_debug)[batch_vec == 0]
        S_ST_pred_physical = (y_pred.squeeze(-1) * target_scale_per_node_debug)[batch_vec == 0]
        C_ST_true_physical = (S_ST_true_physical / vol_ST[batch_vec == 0])
        C_ST_pred_physical = (S_ST_pred_physical / vol_ST[batch_vec == 0])

        print(f"\n--- PHYSICAL VALUES ---")
        print(f"S_ST_true (mol):\n{S_ST_true_physical.detach().cpu().numpy()[:10]}")
        print(f"S_ST_pred (mol):\n{S_ST_pred_physical.detach().cpu().numpy()[:10]}")
        print(f"C_ST_true (mol/cm³):\n{C_ST_true_physical.detach().cpu().numpy()[:10]}")
        print(f"C_ST_pred (mol/cm³):\n{C_ST_pred_physical.detach().cpu().numpy()[:10]}")

        print(f"\n--- PHYSICS TERMS (always in physical units) ---")
        print(f"dS_dt_from_flux_true (mol/day):\n{dS_dt_from_flux_true[batch_vec == 0].detach().cpu().numpy()[:10]}")
        print(f"dS_dt_from_flux_pred (mol/day):\n{dS_dt_from_flux[batch_vec == 0].detach().cpu().numpy()[:10]}")

        print(f"\nF_in_true (mol/day):\n{F_in_true[batch_vec == 0].detach().cpu().numpy()[:10]}")
        print(f"F_in_pred (mol/day):\n{F_in[batch_vec == 0].detach().cpu().numpy()[:10]}")

        print(f"\nF_out_true (mol/day):\n{F_out_true[batch_vec == 0].detach().cpu().numpy()[:10]}")
        print(f"F_out_pred (mol/day):\n{F_out[batch_vec == 0].detach().cpu().numpy()[:10]}")

        print(f"\ndS_dt_from_physics_true (mol/day):\n{dS_dt_from_physics_true[batch_vec == 0].detach().cpu().numpy()[:10]}")
        print(f"dS_dt_from_physics_pred (mol/day):\n{dS_dt_from_physics[batch_vec == 0].detach().cpu().numpy()[:10]}")

        print(f"\ndC_dt_from_physics_true (mol/cm³/day):\n{dC_dt_from_physics_true[batch_vec == 0].detach().cpu().numpy()[:10]}")
        print(f"dC_dt_from_physics_pred (mol/cm³/day):\n{dC_dt_from_physics[batch_vec == 0].detach().cpu().numpy()[:10]}")

        # Show model's time derivative in the appropriate units (only for predictions)
        # Note: Cannot compute dy_dt for y_true since it's not connected to time_per_node in computation graph
        print(f"\ndS_dt_pred (mol/day) [from model]:\n{dy_dt[batch_vec == 0].detach().cpu().numpy()[:10]}")

        # Compute and show errors
        print(f"\n--- ERROR METRICS (first graph) ---")
        graph0_mask = batch_vec == 0
        # Show errors in normalized space (what the model sees)
        norm_error = (y_pred[graph0_mask] - y_true[graph0_mask]).squeeze(-1).detach().cpu().numpy()
        print(f"Normalized error (pred - true) [0,1]:\n  mean={norm_error.mean():.6f}, std={norm_error.std():.6f}")
        print(f"  min={norm_error.min():.6f}, max={norm_error.max():.6f}")

        # Show errors in physical space (what we care about)
        S_ST_error = (S_ST_pred_physical - S_ST_true_physical).detach().cpu().numpy()
        C_ST_error = (C_ST_pred_physical - C_ST_true_physical).detach().cpu().numpy()
        print(f"\nS_ST error (mol):\n  mean={S_ST_error.mean():.3e}, std={S_ST_error.std():.3e}")
        print(f"  min={S_ST_error.min():.3e}, max={S_ST_error.max():.3e}")
        print(f"\nC_ST error (mol/cm³):\n  mean={C_ST_error.mean():.3e}, std={C_ST_error.std():.3e}")
        print(f"  min={C_ST_error.min():.3e}, max={C_ST_error.max():.3e}")

        print(f"{'='*60}\n")

    # Use adaptive normalization based on current residual scale
    #
    # Compute characteristic scale from this batch (robust to outliers using median-like measure)
    # Use 90th percentile of absolute differences as scale (more robust than max)
    #
    # Without this approach, the residual's behavior would be:
    # With small values, gradients vanish -> physics loss is ignored
    # With huge values (when model outputs noisy dS/dt) -> physics dominates and collapses the model
    diff = dy_dt.squeeze() - dy_dt_from_physics

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
        dC_dt_per_graph = scatter_mean(dC_dt.detach().abs(), batch_vec, dim=0)
        dS_dt_from_flux_per_graph = scatter_mean(dS_dt_from_flux.detach().abs(), batch_vec, dim=0)
        dC_dt_from_physics_per_graph = scatter_mean(dC_dt_from_physics.detach(), batch_vec, dim=0)

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
            'dC_dt': dC_dt_per_graph.mean(),
            'dS_dt_from_flux': dS_dt_from_flux_per_graph.mean(),
            'dC_dt_from_physics': dC_dt_from_physics_per_graph.mean()
        }
    else:
        # Single graph case: simple mean across nodes/edges
        loss_dict = {
            'J_ax': J_ax.detach().abs().mean() if J_ax.size(0) > 0 else torch.tensor(0.0, device=device),
            'F_in': F_in.detach().mean(),
            'F_out': F_out.detach().mean(),
            'dC_dt': dC_dt.detach().abs().mean(),
            'dS_dt_from_flux': dS_dt_from_flux.detach().abs().mean(),
            'dC_dt_from_physics': dC_dt_from_physics.detach().mean()
        }

    return loss, loss_dict