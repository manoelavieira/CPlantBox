"""
Datasets for phloem flow prediction.
"""
import torch
from torch_geometric.data import Data

class DummyTemporalDataset(torch.utils.data.Dataset):
    """Creates random graphs to illustrate usage.
    Replace with your real dataset that loads per-timestep PyG `Data`.
    """
    def __init__(self, n_graphs: int = 64, n_min: int = 20, n_max: int = 60, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for _ in range(n_graphs):
            N = int(torch.randint(n_min, n_max+1, (1,), generator=g))

            # Build a random tree-like graph
            parents = torch.randint(0, N, (N-1,), generator=g)
            children = torch.arange(1, N)
            edge_index = torch.stack([torch.cat([parents, children]), torch.cat([children, parents])], dim=0)  # undirected
            E = edge_index.size(1)

            # Edge features
            edge_attr = torch.rand(E, 1, generator=g) * 5.0 + 0.1  # resistance r_st
            edge_org = torch.randint(0, 3, (E,), generator=g)  # Organ types (0,1,2) per edge

            # Node features
            psi = (torch.rand(N, 1, generator=g) - 0.5) * 2.0  # [-1, 1] MPa (scaled)
            vol = torch.rand(N, 1, generator=g) * 1.0 + 0.1
            time = torch.rand(N, 1, generator=g) * 5.0  # Random time between 0-5 days
            x_cont = torch.cat([psi, vol, time], dim=1)  # [psi, vol_st, time]

            # Target sucrose (mock): linear + time-dependent noise
            y = 0.2*psi - 0.1*vol + 0.1*time + 0.05*torch.randn(N, 1, generator=g)

            self.items.append(Data(edge_index=edge_index,
                                   edge_attr=edge_attr,
                                   edge_org=edge_org,
                                   x_cont=x_cont,
                                   y=y,
                                   num_nodes=N))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
