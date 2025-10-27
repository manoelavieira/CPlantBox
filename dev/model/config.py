from __future__ import annotations
from dataclasses import dataclass

# Global constants
R = 83.14  # universal gas constant
cmH2O_to_hPa = 0.980638  # Conversion factor from cmH2O to hPa (from C++ runPM.cpp)

@dataclass
class ModelConfig:
    """Configuration for PhloemNNConv model.

    Attributes:
        node_feat_dim: Dimension of continuous node features
        num_org_types: Number of organ types [ORGAN, SEED, ROOT, STEM, LEAF] following CPlantBox mapping
        org_emb_size: Dimension of organ type embeddings
        hidden_size: Dimension of hidden layers in NNConv/MLPs
        num_layers: Number of NNConv layers
        edge_feat_dim: Dimension of continuous edge features [r_st]
        aggr: NNConv aggregator type ("add", "mean", or "max")
        dropout: Dropout probability
    """
    node_feat_dim: int = 7  # [psi, vol_st, len_leaf, Q_Rmmax, Q_Grmax, Q_Exudmax, Temp]
    edge_feat_dim: int = 1  # [r_st]
    num_org_types: int = 5  # ot_organ=0, ot_seed=1, ot_root=2, ot_stem=3, ot_leaf=4
    org_emb_size: int = 8   # embedding dimension for categorical organ type
    hidden_size: int = 64
    num_layers: int = 3
    aggr: str = "add"
    dropout: float = 0.0

    def __post_init__(self):
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Dropout must be between 0 and 1, got {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"Number of layers must be positive, got {self.num_layers}")
        if self.aggr not in ["add", "mean", "max"]:
            raise ValueError(f"Aggregator must be one of ['add', 'mean', 'max'], got {self.aggr}")
        if any(d < 1 for d in [self.node_feat_dim, self.edge_feat_dim, self.num_org_types,
                               self.org_emb_size, self.hidden_size]):
            raise ValueError("All dimensions must be positive integers")