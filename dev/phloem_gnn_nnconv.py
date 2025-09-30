"""
Phloem GNN (baseline) — NNConv with edge features
-------------------------------------------------
Predicts sucrose concentration per node at timestep t, given:
- Plant topology (edge_index)
- Node features at time t: water potential (psi), sieve-tube volume (vol_st)
- Edge features at time t: sieve-tube resistance (r_st), organ type (categorical)


Expected `Data` fields per graph (per timestep)
----------------------------------------------
- data.edge_index: LongTensor [2, E]
- data.edge_attr:  FloatTensor [E, 1]      # r_st (resistance); can expand later
- data.x_cont:     FloatTensor [N, 3]      # [psi, vol_st, time]  (time in days)
- data.x_org:      LongTensor   [N]        # organ type indices (e.g., 0=LEAF,1=STEM,2=ROOT)
- data.y:          FloatTensor [N, 1]      # target sucrose at t+1 (or t, if you prefer)
- Optional: data.batch for mini-batching multiple graphs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv

# -----------------------------
# Small utilities
# -----------------------------
class Standardizer:
    """Feature-wise standardization (mean, std deviation) with safe inverse-transform.

    Call fit() on a Tensor [N, D], then use transform()/inv_transform().
    """
    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True).clamp_min(1e-8)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return X
        return (X - self.mean) / self.std

    def inv_transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return X
        return X * self.std + self.mean


# -----------------------------
# Model
# -----------------------------
@dataclass
class ModelConfig:
    """Configuration for PhloemNNConv model.

    Attributes:
        x_cont_dim: Dimension of continuous node features [psi, vol_st, time]
        n_org_types: Number of organ types [LEAF, STEM, ROOT]
        org_emb_dim: Dimension of organ type embeddings
        hidden_dim: Hidden dimension in neural networks
        n_layers: Number of NNConv layers
        edge_cont_dim: Dimension of continuous edge features [r_st]
        aggr: NNConv aggregator type ("add", "mean", or "max")
        dropout: Dropout probability
    """
    x_cont_dim: int = 3  # [psi, vol_st, time]
    n_org_types: int = 3
    org_emb_dim: int = 8
    hidden_dim: int = 64
    n_layers: int = 3
    edge_cont_dim: int = 1
    aggr: str = "add"
    dropout: float = 0.0

    def __post_init__(self):
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Dropout must be between 0 and 1, got {self.dropout}")
        if self.n_layers < 1:
            raise ValueError(f"Number of layers must be positive, got {self.n_layers}")
        if self.aggr not in ["add", "mean", "max"]:
            raise ValueError(f"Aggregator must be one of ['add', 'mean', 'max'], got {self.aggr}")
        if any(d < 1 for d in [self.x_cont_dim, self.n_org_types, self.org_emb_dim,
                              self.hidden_dim, self.edge_cont_dim]):
            raise ValueError("All dimensions must be positive integers")


class EdgeNet(nn.Module):
    """Edge MLP producing weight matrices for NNConv.

    Maps edge features (continuous + organ type) -> [E, out_channels * in_channels].
    MLP that turns per-edge features into a per-edge weight matrix W_e that NNConv
    will use to transform neighbor node features.
    """
    def __init__(self, edge_cont_dim: int, n_org_types: int, org_emb_dim: int,
                 in_channels: int, out_channels: int, hidden: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Organ type embedding
        self.org_emb = nn.Embedding(n_org_types, org_emb_dim)

        # Input dim is: continuous edge features + embedded organ type
        edge_feat_dim = edge_cont_dim + org_emb_dim

        # Combined MLP for both continuous and embedded features
        # It outputs a flattened weight matrix of size [out_channels * in_channels] per edge
        self.mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_channels * in_channels)
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        # edge_features: [E, D+1] where D is edge_cont_dim and last column is organ type
        edge_attr = edge_features[:, :-1]  # continuous features
        edge_org = edge_features[:, -1].long()  # organ type as long tensor

        # Combine continuous edge features with organ embeddings
        edge_emb = self.org_emb(edge_org)
        edge_feat_combined = torch.cat([edge_attr, edge_emb], dim=-1)
        return self.mlp(edge_feat_combined)


class PhloemNNConv(nn.Module):
    """Neural network model for phloem flow prediction using NNConv layers.

    Combines node features (continuous and organ type) with edge features
    through multiple NNConv layers to predict sucrose concentration.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Use continuous node features directly
        in_channels = cfg.x_cont_dim

        layers = []
        norms = []

        c_in = in_channels
        for _ in range(cfg.n_layers):
            edge_mlp = EdgeNet(edge_cont_dim=cfg.edge_cont_dim,
                             n_org_types=cfg.n_org_types,
                             org_emb_dim=cfg.org_emb_dim,
                             in_channels=c_in,
                             out_channels=cfg.hidden_dim,
                             hidden=cfg.hidden_dim)

            # EdgeNet returns [E, c_in * hidden_dim]
            # NNConv reshapes to [E, c_in, hidden_dim]
            conv = NNConv(c_in, cfg.hidden_dim, nn=edge_mlp, aggr=cfg.aggr)
            layers.append(conv)
            norms.append(nn.BatchNorm1d(cfg.hidden_dim))
            c_in = cfg.hidden_dim

        self.convs = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1)
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _validate_input(self, data: Data) -> None:
        """Validate input data dimensions and types."""
        if not hasattr(data, 'x_cont'):
            raise ValueError("Data must have x_cont attribute")
        if data.x_cont.size(1) != self.cfg.x_cont_dim:
            raise ValueError(f"Expected x_cont dim {self.cfg.x_cont_dim}, got {data.x_cont.size(1)}")
        if data.edge_attr.size(1) != self.cfg.edge_cont_dim:
            raise ValueError(f"Expected edge_attr dim {self.cfg.edge_cont_dim}, got {data.edge_attr.size(1)}")
        if not hasattr(data, 'edge_org'):
            raise ValueError("Data must have edge_org attribute")
        if data.edge_org.max() >= self.cfg.n_org_types:
            raise ValueError(f"Edge organ type index {data.edge_org.max()} >= n_org_types {self.cfg.n_org_types}")

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            data: Graph data object containing node features, edge features, and topology

        Returns:
            torch.Tensor: Predicted sucrose concentration for each node [N, 1]
        """
        self._validate_input(data)

        x: torch.Tensor = data.x_cont # [N, Dc] node features
        edge_index: torch.Tensor = data.edge_index # [2, E] graph connectivity (sources, targets indices)
        edge_attr: torch.Tensor = data.edge_attr  # [E, De] continuous edge features
        edge_org: torch.Tensor = data.edge_org  # [E] categorical organ type per edge

        # Combine edge features for message passing
        edge_features = torch.cat([edge_attr, edge_org.unsqueeze(-1).float()], dim=-1)

        # stack NNConv layers with residual connections
        for conv, bn in zip(self.convs, self.norms):
            # Process combined edge features through NNConv
            h = conv(x, edge_index, edge_features)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

            # residual if dimensions match (for layers after first)
            if h.shape == x.shape:
                x = x + h
            else:
                x = h

        out = self.head(x) # [N, 1]
        return out


# -----------------------------------------
# Physics hook (non operational for now)
# -----------------------------------------
def physics_residual(y_pred: torch.Tensor, data: Data) -> torch.Tensor:
    """Placeholder for future physics-informed residuals.
    Return 0.0 for now. When ready, compute residuals using `data` fields.
    """
    return torch.tensor(0.0, device=y_pred.device)


# -----------------------------
# Training / evaluation
# -----------------------------
def collate_graphs(batch):
    return Batch.from_data_list(batch)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, target_scaler: Optional[Standardizer] = None) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_nodes = 0
    for data in loader:
        data = data.to(next(model.parameters()).device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data) # [N,1]

        y = data.y # [N,1]
        if target_scaler is not None:
            y_t = target_scaler.transform(y)
        else:
            y_t = y

        mse = F.mse_loss(pred, y_t)
        phys = physics_residual(pred, data)
        loss = mse + phys
        loss.backward()

        total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # print(f"Batch grad norm: {total_norm:.4f}")
        optimizer.step()

        with torch.no_grad():
            # Report MAE in original units
            if target_scaler is not None:
                pred_un = target_scaler.inv_transform(pred)
            else:
                pred_un = pred

            mae = (pred_un - y).abs().sum() # (pred_un - y): per-node errors, shape [N, 1]
            total_mae += mae.item() # accumulates the sum of absolute errors across batches
            total_loss += loss.item() * y.size(0)
            n_nodes += y.size(0)

    if n_nodes == 0:
        raise RuntimeError("No training samples this epoch.")

    avg_loss = total_loss / n_nodes
    avg_mae = total_mae / n_nodes

    return avg_loss, avg_mae


def evaluate(model: nn.Module, loader: DataLoader, target_scaler: Optional[Standardizer] = None) -> Tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_nodes = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(next(model.parameters()).device)
            pred = model(data)
            y = data.y
            if target_scaler is not None:
                y_t = target_scaler.transform(y)
                mse = F.mse_loss(pred, y_t, reduction='sum')
                pred_un = target_scaler.inv_transform(pred)
            else:
                mse = F.mse_loss(pred, y, reduction='sum')
                pred_un = pred
            mae = (pred_un - y).abs().sum()
            total_mse += mse.item()
            total_mae += mae.item()
            n_nodes += y.size(0)
    return total_mse / max(n_nodes, 1), total_mae / max(n_nodes, 1)


# -----------------------------------
# Create dummy dataset for testing
# -----------------------------------
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


# -----------------------------
# Script entry
# -----------------------------
def main():
    """Main training function"""
    print(f"Using torch {torch.__version__}, torch_geometric {torch_geometric.__version__}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ModelConfig()
    model = PhloemNNConv(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset preparation
    ds = DummyTemporalDataset(n_graphs=80)
    print(f"\nTotal graphs in dataset: {len(ds)}")
    print(f"Example graph:")
    print(f"edge_index.shape: {ds[0].edge_index.shape}, x_cont.shape: {ds[0].x_cont.shape}, edge_attr.shape: {ds[0].edge_attr.shape}, y.shape: {ds[0].y.shape}")

    n_train = int(0.8 * len(ds))
    n_test = len(ds) - n_train
    train_set, val_set = torch.utils.data.random_split(ds, [n_train, n_test])
    print(f"Training graphs: {len(train_set)}, validation graphs: {len(val_set)}")

    # Target standardization
    target_scaler = Standardizer()
    with torch.no_grad():
        Ys = torch.cat([train_set[i].y for i in range(len(train_set))], dim=0)
        target_scaler.fit(Ys)

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_graphs)
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Training setup
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)

    # Create scheduler in a backward-compatible way.
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=5, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=5
        )

    # Training loop with early stopping
    best_val = float('inf')
    patience = 10
    patience_counter = 0
    best_epoch = 0

    print("\nStarting training...")
    for epoch in range(1, 101):
        # Training
        tr_loss, tr_mae = train_one_epoch(model, train_loader, optim, target_scaler=target_scaler)

        # Validation
        val_mse, val_mae = evaluate(model, val_loader, target_scaler=target_scaler)

        # Learning rate scheduling
        scheduler.step(val_mse)

        # Logging
        print(f"Epoch {epoch:03d} | "
              f"train_loss={tr_loss:.4f} train_MAE={tr_mae:.4f} | "
              f"val_MSE={val_mse:.4f} val_MAE={val_mae:.4f} | "
              f"lr={optim.param_groups[0]['lr']:.2e}")

        # Model saving and early stopping
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            patience_counter = 0
            # Save model
            torch.save({
                'epoch': epoch,
                'cfg': cfg.__dict__,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_mse': val_mse,
                'target_scaler': target_scaler,
            }, 'phloem_nnconv.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best validation MSE: {best_val:.4f} at epoch {best_epoch}")
                break

    print("\nTraining completed!")


if __name__ == "__main__":
    main()