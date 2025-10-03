"""
Datasets for phloem flow prediction.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Sequence
import math
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Node/organ constants
ROOT, STEM, LEAF = 0, 1, 2
NODE_TYPE_NAME = {ROOT: "root", STEM: "stem", LEAF: "leaf"}
ORGAN_ID = {"root": 0, "stem": 1, "leaf": 2}

@dataclass
class Config:
    seed: int = 42

    # topology sizes
    n_stem: int = 6
    n_leaf: int = 8
    n_root: int = 6

    # geometry (layout only)
    stem_dx: float = 0.2
    stem_dy: float = 0.3
    leaf_span_x: float = 1.8
    root_span_x: float = 1.0
    stem_y0: float = 0.0
    root_y_max: float = -1.5
    jitter: float = 0.05

    # node features (hardcoded)
    vol_root: float = 1.0
    vol_stem: float = 1.0
    vol_leaf: float = 1.0
    psi_root0: float = -0.2
    psi_stem0: float = -0.5
    psi_leaf0: float = -0.8
    psi_jitter: float = 0.02

    # edge/segment resistance by edge organ
    R_leaf: float = 2.0
    R_stem: float = 1.0
    R_root: float = 0.8

    # sucrose model (sources/sinks)
    P_leaf_max: float = 0.30
    C_sat: float = 1.5
    k_sink_root: float = 0.15
    k_resp_stem: float = 0.01

    # daylight (0..1) sinusoid amplitude; set 0.0 for “night”
    daylight_amp: float = 1.0

    # integration
    dt: float = 0.01
    days: float = 3.0


def build_plant_graph(cfg: Config, rng: np.random.Generator) -> nx.Graph:
    G = nx.Graph()
    node_id = 0

    # stems
    stem_ids = []
    for i in range(cfg.n_stem):
        x = rng.normal(0.0, cfg.jitter) + (rng.random() - 0.5) * cfg.stem_dx
        y = cfg.stem_y0 + i * cfg.stem_dy + rng.normal(0.0, cfg.jitter)
        G.add_node(node_id, type=STEM, x=float(x), y=float(y))
        stem_ids.append(node_id)
        node_id += 1
    for i in range(len(stem_ids) - 1):
        G.add_edge(stem_ids[i], stem_ids[i+1])

    y_top = G.nodes[stem_ids[-1]]["y"]
    y_mid = G.nodes[stem_ids[len(stem_ids)//2]]["y"]
    y_bot = G.nodes[stem_ids[0]]["y"]

    # leaves
    leaf_ids = []
    top_stems = stem_ids[len(stem_ids)//2:]
    for _ in range(cfg.n_leaf):
        p = int(top_stems[rng.integers(0, len(top_stems))])
        px, py = G.nodes[p]["x"], G.nodes[p]["y"]
        x = px + (rng.random()*2 - 1) * cfg.leaf_span_x/2 + rng.normal(0, cfg.jitter)
        y = py + rng.random() * (y_top - y_mid + 0.6) + 0.4 + rng.normal(0, cfg.jitter)
        y = max(y, py + 0.2)
        G.add_node(node_id, type=LEAF, x=float(x), y=float(y))
        G.add_edge(p, node_id)
        leaf_ids.append(node_id)
        node_id += 1

    # roots
    root_ids = []
    bottom_stems = stem_ids[:len(stem_ids)//2]
    for _ in range(cfg.n_root):
        p = int(bottom_stems[rng.integers(0, len(bottom_stems))])
        px, py = G.nodes[p]["x"], G.nodes[p]["y"]
        x = px + (rng.random()*2 - 1) * cfg.root_span_x/2 + rng.normal(0, cfg.jitter)
        y = py - rng.random() * (abs(cfg.root_y_max) + (py - y_bot) + 0.6) - 0.4 + rng.normal(0, cfg.jitter)
        y = min(y, py - 0.2)
        G.add_node(node_id, type=ROOT, x=float(x), y=float(y))
        G.add_edge(p, node_id)
        root_ids.append(node_id)
        node_id += 1

    assert len(leaf_ids) >= 3 and len(root_ids) >= 3 and len(stem_ids) >= 3, "Need to have at least 3 elements of each node type."
    return G


def assign_node_features(G: nx.Graph, cfg: Config, rng: np.random.Generator):
    for n in G.nodes:
        t = G.nodes[n]["type"]
        if t == LEAF:
            vol = cfg.vol_leaf
            psi0 = cfg.psi_leaf0
        elif t == STEM:
            vol = cfg.vol_stem
            psi0 = cfg.psi_stem0
        else:
            vol = cfg.vol_root
            psi0 = cfg.psi_root0
        G.nodes[n]["volume"] = float(vol)
        G.nodes[n]["psi"] = float(psi0 + rng.normal(0.0, cfg.psi_jitter))


def assign_edge_features(G: nx.Graph, cfg: Config) -> np.ndarray:
    """
    Return array with columns: [u, v, R, organ_id, length]
    Segment organ rule: touches leaf -> leaf; else touches root -> root; else stem.
    """
    rows = []
    for u, v in G.edges():
        tu, tv = G.nodes[u]["type"], G.nodes[v]["type"]
        if LEAF in (tu, tv):
            organ = "leaf"
            R = cfg.R_leaf
        elif ROOT in (tu, tv):
            organ = "root"
            R = cfg.R_root
        else:
            organ = "stem"
            R = cfg.R_stem
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        length = float(np.hypot(x2 - x1, y2 - y1))
        rows.append([int(u), int(v), float(R), float(ORGAN_ID[organ]), length])
    return np.array(rows, dtype=np.float64)


def init_state_arrays(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (pos[N,2], x_node[N,2], C[N], node_type[N]) where x_node = [psi, volume]
    """
    N = G.number_of_nodes()
    pos = np.zeros((N, 2), dtype=np.float64)
    psi = np.zeros(N, dtype=np.float64)
    vol = np.zeros(N, dtype=np.float64)
    C = np.zeros(N, dtype=np.float64)
    node_type = np.zeros(N, dtype=np.int64)

    for n in range(N):
        pos[n, 0] = G.nodes[n]["x"]
        pos[n, 1] = G.nodes[n]["y"]
        psi[n] = G.nodes[n]["psi"]
        vol[n] = G.nodes[n]["volume"]
        t = G.nodes[n]["type"]
        node_type[n] = t
        C[n] = {LEAF: 0.6, STEM: 0.4, ROOT: 0.3}[t]

    x_node = np.stack([psi, vol], axis=1)
    return pos, x_node, C, node_type


def build_adjacency_Q(psi: np.ndarray, edges_feat: np.ndarray, N: int):
    """Adjacency with Q_ij = (psi_i - psi_j)/R_e."""

    # Each node stores its neighbors and the signed flux between them
    adj = [[] for _ in range(N)]
    for row in edges_feat:
        u, v, R = int(row[0]), int(row[1]), float(row[2])
        dp = psi[u] - psi[v]
        Q = dp / max(R, 1e-9)

        adj[u].append((v,  Q))  # from u -> v, the flow is positive
        adj[v].append((u, -Q))  # from v -> u, the flow is negative
    return adj


def step_sucrose(C: np.ndarray, psi: np.ndarray, vol: np.ndarray,
                 edges_feat: np.ndarray, cfg: Config, t: float,
                 node_type: np.ndarray) -> np.ndarray:
    """
    dC_i/dt = (sum_inflow - sum_outflow) / vol_i + S_i - U_i
    Transport: advection-only via upwind with Q = (psi_i - psi_j)/R_e
    """
    N = C.shape[0]
    adj = build_adjacency_Q(psi, edges_feat, N)

    # sources/sinks
    d = daylight(t, cfg.daylight_amp)
    S = np.zeros(N, dtype=np.float64)  # sink term
    U = np.zeros(N, dtype=np.float64)  # source term

    for i in range(N):
        typ = int(node_type[i])
        if typ == LEAF:
            # saturation-limited: high when C is far from C_sat, goes to 0 when C -> C_sat
            S[i] = cfg.P_leaf_max * d * max(0.0, 1.0 - C[i]/max(cfg.C_sat, 1e-6))
        elif typ == ROOT:
            U[i] = cfg.k_sink_root * C[i]
        else:
            U[i] = cfg.k_resp_stem * C[i]

    dC = np.zeros(N, dtype=np.float64)
    for i in range(N):
        for (j, Q_ij) in adj[i]:
            # outflow i -> j
            if Q_ij > 1e-12:
                adv_out = Q_ij * C[i]
                adv_in  = 0.0
            # inflow j -> i
            elif Q_ij < -1e-12:
                adv_out = 0.0
                adv_in  = (-Q_ij) * C[j]
            else:
                adv_out = 0.0
                adv_in  = 0.0
            dC[i] += (adv_in - adv_out)

    C_new = C + cfg.dt * ((dC + S - U) / vol)
    return np.maximum(C_new, 0.0)


def daylight(t_days: float, amp: float) -> float:
    """Sinusoidal 0..1 daily cycle with amplitude 'amp'."""
    base = 0.5 * (1.0 + math.sin(2.0 * math.pi * t_days))
    return float(max(0.0, min(1.0, amp * base)))


class DummyTemporalDataset(torch.utils.data.Dataset):
    """
    A temporal sequence of PyG Data snapshots for one synthetic plant graph.
    Each index returns Data for a timestep t = idx * dt.
    """

    def __init__(self, days: float = 3.0, dt: float = 0.01, seed: int = 42,
                 n_stem: int = 6, n_leaf: int = 8, n_root: int = 6,
                 cfg_override: Optional[Dict] = None):
        # Build config
        cfg = Config(seed=seed, dt=dt, days=days, n_stem=n_stem, n_leaf=n_leaf, n_root=n_root)
        if cfg_override:
            for k, v in cfg_override.items():
                setattr(cfg, k, v)
        self.cfg = cfg

        # Build one static graph + initial state
        rng = np.random.default_rng(cfg.seed)
        G = build_plant_graph(cfg, rng)
        assign_node_features(G, cfg, rng)
        edges_features = assign_edge_features(G, cfg)  # [u, v, R, organ_id, length]
        pos, x_node, C0, node_type = init_state_arrays(G)

        # Fixed tensors across time
        u = edges_features[:, 0].astype(np.int64)
        v = edges_features[:, 1].astype(np.int64)
        edge_index = np.vstack([u, v])
        edge_feat = edges_features[:, 2:3]  # just the resistance [R]

        edge_feat = edges_features[:, 2:3]  # just the resistance [R]
        edge_org = edges_features[:, 3]  # organ_id for each edge

        self.edge_index_t = torch.as_tensor(edge_index, dtype=torch.long)
        self.edge_feat_t = torch.as_tensor(edge_feat, dtype=torch.float32)
        self.edge_org_t = torch.as_tensor(edge_org, dtype=torch.long)
        self.pos_t = torch.as_tensor(pos, dtype=torch.float32)
        self.node_type_t = torch.as_tensor(node_type, dtype=torch.long)

        # Node features X are constant (psi, volume); sucrose C evolves
        self.x_node = x_node.copy()
        self.psi = x_node[:, 0]
        self.vol = x_node[:, 1]
        self.edges_features = edges_features
        self.C = C0.copy()

        # Precompute number of steps
        self.nsteps = int(math.ceil(cfg.days / cfg.dt)) + 1

    def __len__(self) -> int:
        return self.nsteps

    def __getitem__(self, idx: int) -> Data:
        """Return Data at time t = idx * dt (and advance an internal sim copy if needed)."""
        C = self._simulate_to_index(idx)

        t = idx * self.cfg.dt
        node_feat_t = torch.as_tensor(self.x_node, dtype=torch.float32)  # [psi, volume]
        y_t = torch.as_tensor(C.reshape(-1, 1), dtype=torch.float32)  # sucrose at time t

        return Data(
            node_feat=node_feat_t,
            edge_index=self.edge_index_t,
            edge_feat=self.edge_feat_t,
            edge_org=self.edge_org_t,
            y=y_t,
            time=torch.tensor([t], dtype=torch.float32),
            pos=self.pos_t,
            node_type=self.node_type_t,
        )

    def _simulate_to_index(self, idx: int) -> np.ndarray:
        """Simulate sucrose from t=0 to step idx (inclusive) starting from initial C."""
        C = self._initial_C()  # fresh copy each call
        t = 0.0
        for _ in range(idx):
            C = step_sucrose(C, self.psi, self.vol, self.edges_features, self.cfg, t, self.node_type_t.cpu().numpy())
            t += self.cfg.dt
        return C

    def _initial_C(self) -> np.ndarray:
        # reconstruct initial C from node_type (same mapping used in init_state_arrays)
        nt = self.node_type_t.cpu().numpy()
        C = np.zeros_like(nt, dtype=np.float64)
        C[nt == LEAF] = 0.6
        C[nt == STEM] = 0.4
        C[nt == ROOT] = 0.3
        return C


def visualize_graph(data: Data, title: str = "Plant graph", show_labels: bool = False,
                    figsize=(6, 6), save_path: Optional[str] = None, show: bool = True):
    """
    Plot a single snapshot data.
    - Nodes colored by node_type (root=blue, stem=gray, leaf=green)
    - Edges colored by organ_id (root=blue, stem=black, leaf=green)
    - Node size ~ volume
    """
    pos = data.pos.cpu().numpy()
    N = pos.shape[0]

    # Node features (use node_feat: [psi, volume])
    if not hasattr(data, "node_feat"):
        raise AttributeError("Expected 'node_feat' in Data (shape [N, 2] = [psi, volume]).")
    node_feat = data.node_feat.cpu().numpy()
    if node_feat.shape[1] < 2:
        raise ValueError(f"'node_feat' expected at least 2 columns, got shape {node_feat.shape}")
    vol = node_feat[:, 1]
    vol_scaled = 200 * (vol / (np.max(vol) + 1e-8)) + 50

    node_type = data.node_type.cpu().numpy() if hasattr(data, "node_type") else np.zeros(N, dtype=int)
    colors_nodes = np.array(["tab:blue", "tab:gray", "tab:green"])  # root, stem, leaf
    node_colors = colors_nodes[node_type]

    # edges
    ei = data.edge_index.cpu().numpy()
    E = ei.shape[1]
    if hasattr(data, "edge_org"):
        organ_id = data.edge_org.cpu().numpy().astype(int)
        if organ_id.shape[0] != E:
            raise ValueError(f"edge_org length {organ_id.shape[0]} != number of edges {E}")
    else:
        # Fallback: infer by touching node types (leaf if any leaf, else root if any root, else stem)
        organ_id = np.zeros(E, dtype=int)  # 0=root,1=stem,2=leaf
        for k in range(E):
            u, v = int(ei[0, k]), int(ei[1, k])
            tu, tv = node_type[u], node_type[v]
            if LEAF in (tu, tv):
                organ_id[k] = 2
            elif ROOT in (tu, tv):
                organ_id[k] = 0
            else:
                organ_id[k] = 1

    colors_edges = np.array(["tab:blue", "black", "tab:green"])  # root, stem, leaf
    edge_colors = colors_edges[organ_id]

    fig, ax = plt.subplots(figsize=figsize)

    # draw edges
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], color=edge_colors[k], alpha=0.85)

    # draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], s=vol_scaled, c=node_colors, edgecolors="k", zorder=3)

    if show_labels:
        for i in range(N):
            ax.text(pos[i, 0], pos[i, 1], str(i), fontsize=8, ha="center", va="center")

    t = float(data.t.item()) if hasattr(data, "t") else 0.0
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} (t={t:.2f} d)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def make_temporal_dataset(**kwargs) -> List[Data]:
    """Back-compat helper returning list[Data] like before."""
    ds = DummyTemporalDataset(**kwargs)
    return [ds[i] for i in range(len(ds))]


def plot_timeseries(dummy_dataset: DummyTemporalDataset,
                    data_type: str = "psi",
                    nodes: Optional[Sequence[int]] = None,
                    title: str = "",
                    show: bool = True,
                    save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray]:
    """
    Plots psi (water potential) per node vs time for a temporal PyG dataset whose
    Data objects have:
        - data.time: FloatTensor [1]  (graph-level scalar)
        - data.node_feat: FloatTensor [N, *] with psi in column 0

    Args:
        dummy_dataset: sequence of Data snapshots (supports __len__/__getitem__)
        nodes: optional subset of node indices to plot
        title: title for the figure
        show: whether to show the plot
        save_path: optional path to save the figure

    Returns:
        (fig, ax, times[T], psi[T, N])
    """
    T = len(dummy_dataset)
    if T == 0:
        raise ValueError("Dataset is empty.")

    if data_type not in ("psi", "sucrose"):
        raise ValueError("data_type must be 'psi' or 'sucrose'.")

    # Collect times and psi vectors
    times_list: List[float] = []
    data_rows: List[np.ndarray] = []
    for i in range(T):
        d = dummy_dataset[i]

        # time
        if not hasattr(d, "time"):
            raise AttributeError("Expected 'time' attribute on Data.")
        t = d.time
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        t_val = float(np.asarray(t).reshape(-1)[0])
        times_list.append(t_val)

        # choose source by data_type
        if data_type == "psi":
            if not hasattr(d, "node_feat"):
                raise AttributeError("Expected 'node_feat' attribute on Data.")
            node_feat = d.node_feat
            if isinstance(node_feat, torch.Tensor):
                node_feat = node_feat.detach().cpu().numpy()
            data_rows.append(np.asarray(node_feat[:, 0], dtype=float))
        elif data_type == "sucrose":
            if not hasattr(d, "y"):
                raise AttributeError("Expected 'y' attribute on Data.")
            y = d.y
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            data_rows.append(np.asarray(y[:, 0], dtype=float))

    # Assemble into arrays
    times = np.asarray(times_list, dtype=float)
    N = max(row.shape[0] for row in data_rows)

    series = np.zeros((T, N), dtype=float)
    for i, row in enumerate(data_rows):
        series[i, :row.shape[0]] = row

    # Node selection
    if nodes is None:
        nodes = list(range(N))
    else:
        nodes = list(nodes)
        if any(j < 0 or j >= N for j in nodes):
            raise IndexError(f"Some requested node indices are out of range 0..{N-1}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in nodes:
        ax.plot(times, series[:, j], lw=1.6, alpha=0.9)

    ax.set_xlabel("time (days)")
    if data_type == "psi":
        ax.set_ylabel("ψ (water potential)")
    elif data_type == "sucrose":
        ax.set_ylabel("sucrose")
    ax.set_ylabel("sucrose")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if len(nodes) <= 12:
        ax.legend([f"node {j}" for j in nodes], loc="best", fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, times, series


if __name__ == "__main__":
    ds = DummyTemporalDataset(days=2.0, dt=0.05, seed=7, n_stem=6, n_leaf=8, n_root=6)

    plot_timeseries(ds, nodes=[0, 3, 7], data_type="psi", title="ψ (water potential) per node over time", save_path="plot/psi_timeseries.png", show=False)
    plot_timeseries(ds, nodes=[0, 3, 7], data_type="sucrose", title="sucrose per node over time", save_path="plot/sst_timeseries.png", show=False)
    visualize_graph(ds[0], title="Plant structure", show_labels=True)
