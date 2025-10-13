import torch
from typing import Optional, List
from torch_geometric.data import Data

def create_delta2(data: Data, psi: torch.Tensor, align_to_upstream: bool = True,
                  device: Optional[torch.device] = None):
    """Create a per-graph flow-aligned incidence matrix Delta2 (rows = nodes, cols = edges)

    Each column corresponds to one edge and has two nonzero entries (+1 and -1)
    representing how that edge connects two nodes. The orientation of these signs
    is aligned with the physical flow direction inferred from the node potentials `psi`.

    Args:
        data: PyG Data object with `edge_index` [2, E] and `batch` [N].
        psi: Per-node potential (e.g., water potential) tensor [N].
        align_to_upstream:
            If True, +1 is placed on the upstream node (source of flow).
            If False, +1 is placed on the downstream node (sink of flow).
        device: Target device; inferred automatically if None.

    Returns:
        List[torch.sparse_coo_tensor]: One sparse incidence matrix per graph,
        each with shape (N_g, E_g).
    """
    if device is None:
        device = data.edge_index.device if hasattr(data, 'edge_index') else torch.device('cpu')

    edge_index = data.edge_index.to(device)
    psi = psi.to(device)

    # Total number of nodes across all graphs in the batch
    N_total = int(edge_index.max().item() + 1)

    src, dst = edge_index[0], edge_index[1] # edge_index[0] = source nodes, edge_index[1] = destination nodes

    # Jw per edge and its sign
    # If Jw > 0, flow from dst -> src (opposite arrow, dst is upstream)
    # If Jw < 0, flow from src -> dst (same as arrow, src is upstream)
    Jw = psi[dst] - psi[src]
    sign = torch.sign(Jw)

    # Replace zeros (no difference in potential) with +1 to keep a deterministic orientation
    sign[sign == 0.] = 1.

    # If align_to_upstream is True, multiply by sign so that +1 is at upstream node
    # If align_to_upstream=False, flip the sign to put +1 on downstream.
    col_multiplier = sign if align_to_upstream else -sign

    # Per-graph signed incidence
    batch = data.batch.to(device)               # [N_total], node -> graph id
    edge_graph = batch[src]                     # [E_total], edge -> graph id (edges never connect nodes from different graphs)
    num_graphs = int(batch.max().item() + 1)
    out = []

    for g in range(num_graphs):
        # Identify nodes and edges belonging to this graph g
        mask_nodes = (batch == g)
        mask_edges = (edge_graph == g)

        node_idx = torch.nonzero(mask_nodes, as_tuple=False).view(-1) # torch.nonzero returns the indices where mask_nodes is True
        N_g = node_idx.numel()
        E_g = int(mask_edges.sum().item())

        # Map global node indices -> local indices [0..N_g-1]
        local_map = torch.full((N_total,), -1, device=device, dtype=torch.long) # initializes an array of length N_total filled with -1
        local_map[node_idx] = torch.arange(N_g, device=device, dtype=torch.long) # torch.arange(N_g) generates [0, 1, 2, ..., N_g-1]

        src_g = local_map[src[mask_edges]]  # local source node index for each edge
        dst_g = local_map[dst[mask_edges]]  # local target node index for each edge
        cm_g = col_multiplier[mask_edges]   # column multiplier (+1 or -1) for each edge

        # Concatenate src_g and dst_g means listing all rows (node indices) where nonzeros will appear
        # The first E_g entries -> rows for the source nodes; the last E_g entries -> rows for the target nodes
        # row_indices = [src_0, src_1, ..., src_(E_g-1), dst_0, dst_1, ..., dst_(E_g-1)]
        # col_indices = [0, 1, ..., E_g-1, 0, 1, ..., E_g-1]
        row_indices = torch.cat([src_g, dst_g], dim=0)
        col_indices = torch.cat([torch.arange(E_g, device=device), torch.arange(E_g, device=device)], dim=0)

        vals = torch.cat([-cm_g, cm_g], dim=0).to(torch.float32)

        # Optionally sort by column (edge)
        perm = torch.argsort(col_indices)
        row_indices = row_indices[perm]
        col_indices = col_indices[perm]
        vals = vals[perm]

        indices = torch.stack([row_indices, col_indices], dim=0)  # [2, nnz]

        # Create sparse tensor with shape (N_g, E_g): rows=nodes, cols=edges
        sparse_g = torch.sparse_coo_tensor(indices, vals, size=(N_g, E_g), dtype=torch.float32, device=device)
        out.append(sparse_g)

        # print(f"[GNN][CREATE_DELTA2] Graph {g}: N_g={N_g}, E_g={E_g}, nnz: {sparse_g._nnz()}")
        # print(f"[GNN][CREATE_DELTA2] Indices sample: {sparse_g._indices()}")
        # print(f"[GNN][CREATE_DELTA2] Values sample: {sparse_g._values()}")

    return out