from __future__ import annotations

import torch

from torch_scatter import scatter_mean
from torch_geometric.data import Data

from . import utils

# Global constants
R = 83.14  # universal gas constant

def compute_axial_flux(y_pred: torch.Tensor, data: Data, device: torch.device) -> torch.Tensor:
    """Compute axial sucrose flux J_ax along edges.

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
        return torch.zeros(0, device=device)

    src, dst = edge_index[0], edge_index[1]
    r_ST = data.edge_feat.to(device).squeeze(-1)  # [E, 1] -> [E]

    # Prevent division by zero in resistance
    r_ST = torch.clamp(r_ST, min=1e-12)

    # Node features already in original space
    node_feat = data.node_feat.to(device)
    psi = node_feat[:, 0]
    Temp = node_feat[:, 6]

    # Sucrose at endpoints
    s_i = y_pred[src, 0]
    s_j = y_pred[dst, 0]

    # Water potential differences
    psi_i = psi[src]
    psi_j = psi[dst]

    # Jw < 0: psi_i > psi_j (physical flow src -> dst (arrow direction))
    # Jw > 0: psi_j > psi_i (physical flow dst -> src (arrow direction))
    Jw = psi_j - psi_i

    # (Optional / debug) Incidence matrices aligned by physics; not used in residual here
    # delta2_physics = create_delta2(data, psi)

    # Use the upstream node’s sucrose for the advective term:
    # If Jw < 0 (src -> dst), upstream is src, thus pick s_i
    # If Jw > 0 (dst -> src), upstream is dst, thus pick s_j
    s_ij = torch.where(Jw < 0, s_i, s_j)

    # Sign choice below makes J_ax > 0 correspond to flow src -> dst (arrow direction)
    # Jax > 0: Flow along edge direction (src → dst)
    # Jax < 0: Flow opposite to edge direction (dst → src)
    # If psi_i > psi_j (and/or s_i > s_j), driving > 0, giving J_ax > 0 (src -> dst).
    RT_i = R * (Temp[src] + 273.15)  # Use temperature at source nodes for each edge
    J_ax = (1 / r_ST) * s_ij * (RT_i * (s_i - s_j) + (psi_i - psi_j))

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
    dS_dt_from_flux = torch.zeros(N, device=device)

    # Handle empty graph case
    if J_ax.size(0) == 0:
        return dS_dt_from_flux

    src, dst = edge_index[0], edge_index[1]

    # Divergence of flux -> net inflow per node
    # This computes the sum of incoming/outgoing fluxes for each node
    # dst node accumulates +J_ax   (incoming)
    # src node accumulates -J_ax   (outgoing)
    dS_dt_from_flux.scatter_add_(0, dst, J_ax)   # Add incoming fluxes
    dS_dt_from_flux.scatter_add_(0, src, -J_ax)  # Subtract outgoing fluxes

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

    # Sucrose concentration in sieve tube
    CSTi = y_pred.squeeze(-1) / vol_ST
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

    # Sucrose concentration in sieve tube
    CSTi = y_pred.squeeze(-1) / vol_ST
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
    """Compute time derivative of sucrose content from model predictions.

    Args:
        y_pred: Predicted sucrose content [N, 1] - must be connected to data.time_norm
        data: Graph data containing time features

    Returns:
        torch.Tensor: Time derivative ds/dt per node [N]

    Raises:
        ValueError: If y_pred is not connected to data.time_norm in computation graph
    """

    # We need ds/dt from the model with respect to a differentiable time feature
    # This is ESSENTIAL for physics-informed learning: without it, the physics constraint is meaningless
    if not hasattr(data, 'time_norm') or data.time_norm is None:
        raise ValueError("data.time_norm not found. Required for physics residual computation.")
    if not data.time_norm.requires_grad:
        raise ValueError("data.time_norm must have requires_grad=True for physics residual computation.")
    if not hasattr(data, "time_sigma") or data.time_sigma is None:
        raise ValueError("data.time_sigma missing; ensure model.forward() sets it.")

    # Compute gradient of predictions w.r.t. **scaled** time_norm τ [N,1]
    # We'll convert to real time derivative via  ∂/∂t = (1/σ_t) ∂/∂τ
    try:
        ds_dt = torch.autograd.grad(
            y_pred.sum(),        # sum to get scalar for gradient computation
            data.time_norm,      # [N, 1] per-node time features
            create_graph=True,   # needed for second backward pass
            retain_graph=True,   # keep graph for subsequent loss computation
            allow_unused=False   # ERROR if time_norm is not connected
        )[0]
        ds_dt = ds_dt.squeeze()
    except RuntimeError as e:
        if "not have been used in the graph" in str(e):
            raise ValueError(
                "Physics residual computation failed: y_pred is not connected to time_norm. "
                "This indicates that the model predictions don't depend on time, which breaks the physics constraint. "
                "Ensure that y_pred comes from a model forward pass that uses the same data object, "
                "or that the model architecture properly utilizes the time feature."
            ) from e
        else:
            raise

    # Convert ∂/∂τ to ∂/∂t using stored σ_t (τ = (t - μ_t)/σ_t)
    ds_dt = ds_dt / data.time_sigma.squeeze()

    return ds_dt


def physics_residual(y_pred: torch.Tensor, data: Data):
    """Compute physics-informed residual term based on sucrose transport equations.

    Implements the governing equation:
    ds_{st}/dt = J_ax + (F_in - F_out)

    where:
    - J_ax is the axial sucrose flux
    - F_in is the phloem loading rate
    - F_out is the sucrose outflow

    IMPORTANT: y_pred MUST come from a model forward pass using the same data object,
    so that data.time_norm is properly connected to y_pred in the computation graph.
    Without this connection, ds/dt cannot be computed and the physics constraint is meaningless.

    Args:
        y_pred: Predicted sucrose content [N, 1] - MUST be connected to data.time_norm
        data: Graph data containing topology, features, simulation parameters, and node fields

    Returns:
        tuple: (residual_loss, physics_components_dict) where physics_components_dict contains
            {'J_ax', 'F_in', 'F_out', 'ds_dt', 'dS_dt_from_flux'}

    Raises:
        ValueError: If y_pred is not connected to data.time_norm in the computation graph
    """
    device = y_pred.device
    batch_vec = getattr(data, "batch", None)
    N = y_pred.size(0)

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
    ds_dt = compute_time_derivative(y_pred, data)

    # Total rate of change from physics
    dS_dt_from_physics = dS_dt_from_flux + F_in - F_out

    # Compute residual as difference between model derivative and physics derivative
    residual_node = (ds_dt.squeeze() - dS_dt_from_physics).pow(2)

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
        ds_dt_per_graph = scatter_mean(ds_dt.detach().abs(), batch_vec, dim=0)
        dS_dt_from_flux_per_graph = scatter_mean(dS_dt_from_flux.detach().abs(), batch_vec, dim=0)
        dS_dt_from_physics_per_graph = scatter_mean(dS_dt_from_physics.detach(), batch_vec, dim=0)

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
            'ds_dt': ds_dt_per_graph.mean(),
            'dS_dt_from_flux': dS_dt_from_flux_per_graph.mean(),
            'dS_dt_from_physics': dS_dt_from_physics_per_graph.mean()
        }
    else:
        # Single graph case: simple mean across nodes/edges
        loss_dict = {
            'J_ax': J_ax.detach().abs().mean() if J_ax.size(0) > 0 else torch.tensor(0.0, device=device),
            'F_in': F_in.detach().mean(),
            'F_out': F_out.detach().mean(),
            'ds_dt': ds_dt.detach().abs().mean(),
            'dS_dt_from_flux': dS_dt_from_flux.detach().abs().mean(),
            'dS_dt_from_physics': dS_dt_from_physics.detach().mean()
        }
    return loss, loss_dict