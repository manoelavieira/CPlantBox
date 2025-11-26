# Quick Start: Using the Operator-Based GNN

You now have **two GNN architectures** for phloem flow modeling:

1. **Baseline (`nnconv`)**: Original NNConv model - predicts concentrations, reconstructs fluxes
2. **Operator (`operator`)**: New architecture - predicts fluxes directly via message passing

Both work with the same training code. Just add `--model-type operator` to use the new one!

## Quick Commands

### Train Operator Model
```bash
python train_gnn.py \
    --data-path data/your_simulation.h5 \
    --model-type operator \
    --loss-type physics \
    --batch-size 8 \
    --epochs 100
```

### Train Baseline Model (Default)
```bash
python train_gnn.py \
    --data-path data/your_simulation.h5 \
    --loss-type physics \
    --batch-size 8 \
    --epochs 100
```

## What Changed?

### New Classes
- `FluxMessagePassing`: Message passing layer that predicts edge fluxes
- `PhloemOperatorGNN`: Full model using flux-based message passing
- `physics_residual_operator()`: Physics loss using predicted fluxes

### Modified Functions
- `setup_model_and_scalers()`: Now accepts `model_type` parameter
- `run_forward()`: Returns tensor OR dict depending on model
- `compute_physics_residual_step()`: Auto-detects model type

### New Parameters
- `--model-type`: CLI argument to choose architecture
- `model_type`: Config parameter in `ModelConfig` and `TrainingConfig`

## Key Differences

| Aspect | Baseline | Operator |
|--------|----------|----------|
| Output | `Tensor[N,1]` | `Dict{predictions, edge_fluxes, divergences}` |
| Messages | Edge-conditioned features | Learned fluxes |
| Physics Loss | Reconstructs fluxes | Uses predicted fluxes |
| Use Case | Standard GNN | Physics-informed operator learning |

## Example Usage in Code

```python
from model.gnn import PhloemOperatorGNN
from model.config import ModelConfig

# Create operator model
config = ModelConfig(model_type="operator")
model = PhloemOperatorGNN(config)

# Forward pass
output = model(data)

# Access outputs
predictions = output['predictions']    # [N, 1] concentrations
edge_fluxes = output['edge_fluxes']    # [E] predicted fluxes
divergences = output['divergences']    # [N] divergence values
```

## Why Use the Operator Model?

- ✅ **Direct flux prediction** - no reconstruction needed
- ✅ **True discrete operator** - explicit divergence computation via scatter_add
- ✅ **Physically meaningful** - fluxes are actual transport quantities
- ✅ **Interpretable** - can visualize predicted flux networks
- ✅ **Conservation by design** - divergence = outflow - inflow, explicit in forward pass
- ✅ **Clear implementation** - no hidden abstractions, all operations visible

## When to Use Baseline vs Operator?

**Use Baseline (`nnconv`) when:**
- You want the proven, tested approach
- You're doing initial experiments
- You want simpler outputs (just predictions)

**Use Operator (`operator`) when:**
- You want to learn flux patterns directly
- You need interpretable edge-level outputs
- You're exploring physics-informed operator learning
- You want explicit conservation enforcement

## Troubleshooting

**Model not found?**
```python
from model.gnn import PhloemOperatorGNN  # Make sure imports work
```

**Wrong output type?**
```python
# Extract predictions from either model:
pred = output if isinstance(output, torch.Tensor) else output['predictions']
```

**Physics loss error?**
```python
# Use correct physics function:
from model.physics import physics_residual_operator
loss, metrics = physics_residual_operator(model_output, data)
```

## Next Steps

1. **Run tests**: `python test_operator_model.py`
2. **Train baseline**: `python train_gnn.py --model-type nnconv --data-path ...`
3. **Train operator**: `python train_gnn.py --model-type operator --data-path ...`
4. **Compare results**: Check TensorBoard logs
5. **Analyze fluxes**: Access `output['edge_fluxes']` from operator model

## Complete Example

```python
#!/usr/bin/env python3
"""Example: Training operator model programmatically."""

import torch
from model.gnn import PhloemOperatorGNN
from model.config import ModelConfig
from model.physics import physics_residual_operator
from training.config import TrainingConfig
from training.setup import setup_model_and_scalers, setup_environment
from training.utils import get_dataloaders

# Configuration
train_config = TrainingConfig(
    data_path="data/simulation.h5",
    model_type="operator",  # Use new operator model
    batch_size=8,
    epochs=100,
    loss_type="physics"
)

# Setup
device = setup_environment(train_config)
train_loader, val_loader, test_loader = get_dataloaders(train_config)
model_setup = setup_model_and_scalers(train_loader, device, train_config.model_type)

# Training loop (simplified)
model = model_setup.model
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

for epoch in range(train_config.epochs):
    for batch in train_loader:
        # Forward pass
        output = model(batch)

        # Extract predictions for metrics
        predictions = output['predictions']

        # Compute physics loss using operator function
        phys_loss, metrics = physics_residual_operator(output, batch)

        # Backward pass
        phys_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Optional: analyze fluxes
        edge_fluxes = output['edge_fluxes']
        print(f"Mean flux magnitude: {edge_fluxes.abs().mean():.6f}")
```

---

**Ready to use!** Both architectures are tested and ready for training. Start with the baseline to establish a reference, then try the operator model to explore direct flux learning.


# Operator-Based GNN Architecture

## Overview

This document describes the new operator-based GNN architecture that implements message passing as a discrete transport operator. This architecture complements the existing NNConv-based model and provides an alternative approach where fluxes are predicted directly rather than reconstructed from concentrations.

## Architecture Comparison

### Baseline (PhloemNNConv)
- **Predictions**: Node-wise sucrose concentrations
- **Message Passing**: NNConv with edge-conditioned weight matrices
- **Physics Integration**: Fluxes reconstructed from predicted concentrations
- **Output**: Tensor of shape `[N, 1]` (concentration predictions)

### Operator Model (PhloemOperatorGNN)
- **Predictions**: Edge-wise fluxes directly, plus node concentrations
- **Message Passing**: FluxMessagePassing layer where messages ARE fluxes
- **Physics Integration**: Uses predicted fluxes and divergences directly
- **Output**: Dictionary with:
  - `'predictions'`: `[N, 1]` concentration predictions
  - `'edge_fluxes'`: `[E]` predicted edge fluxes
  - `'divergences'`: `[N]` node-wise divergence values

## Key Components

### 1. FluxMessagePassing Layer (`model/gnn.py`)

A custom layer (inherits from `nn.Module`, not `MessagePassing`) that implements discrete transport explicitly:

```python
class FluxMessagePassing(nn.Module):
    """
    Operator-like message passing layer:
    - Step 1: Compute scalar flux J_ij for each edge
    - Step 2: Compute divergence per node via scatter_add
    - Step 3: Update node embeddings using divergence
    """
```

**Visual Flow**:
```
Input: x [N, d], edge_index [2, E], edge_features [E, d']
   │
   ├─> Extract x_src, x_dst from edge endpoints
   │
   ├─> Flux MLP: [x_src, x_dst, edge_feat] → edge_fluxes [E]
   │
   ├─> Divergence via scatter_add:
   │     divergence[src] += edge_fluxes   (outflow)
   │     divergence[dst] -= edge_fluxes   (inflow)
   │     → divergence [N]
   │
   └─> Node Update MLP: [x, divergence] → x_new [N, d']

Output: x_new [N, d'], edge_fluxes [E]
```

**Key Implementation**:
- **NO reliance on PyG's `propagate()`** - everything is explicit
- **Flux Computation**: MLP maps `[h_src, h_dst, edge_features]` → scalar flux
- **Divergence**: `div_i = sum_j J_ij - sum_k J_ki` via scatter_add
- **Node Update**: MLP maps `[h_old, divergence]` → updated embedding

**Architecture**:
```python
def forward(self, x, edge_index, edge_features):
    # Step 1: Compute edge fluxes
    x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
    edge_fluxes = flux_mlp([x_src, x_dst, edge_features])

    # Step 2: Compute divergence
    divergence = zeros(N)
    divergence.scatter_add_(0, src, +edge_fluxes)  # outflow
    divergence.scatter_add_(0, dst, -edge_fluxes)  # inflow

    # Step 3: Node update
    x_new = node_update_mlp([x, divergence])

    return x_new, edge_fluxes
```

This is a **true discrete operator**: the forward pass explicitly computes the discrete divergence operator applied to learned fluxes.

### FluxMessagePassing (Additional Information)


`FluxMessagePassing` is now a **true discrete operator** layer that explicitly computes:
1. Edge fluxes from node states
2. Divergence via scatter operations
3. Node updates using divergence

No hidden PyG abstractions - everything is explicit in `forward()`.

**The Three Steps**

**Step 1: Compute Edge Fluxes**
```python
# Extract node features at edge endpoints
x_src = x[edge_index[0]]  # Source node features [E, d]
x_dst = x[edge_index[1]]  # Destination node features [E, d]

# Compute scalar flux per edge
edge_fluxes = flux_mlp([x_src, x_dst, edge_features])  # [E]
```

**Step 2: Compute Divergence**
```python
# Initialize divergence
divergence = zeros(N)

# Accumulate outflows and inflows
divergence.scatter_add_(0, src, +edge_fluxes)  # +J at source
divergence.scatter_add_(0, dst, -edge_fluxes)  # -J at destination

# Result: div_i = sum(outgoing) - sum(incoming)
```

**Step 3: Update Nodes**
```python
# Concatenate old features with divergence
x_updated = node_update_mlp([x, divergence])  # [N, d']
```


### 2. PhloemOperatorGNN Model (`model/gnn.py`)

Stacks multiple `FluxMessagePassing` layers with normalization and residual connections:

```python
class PhloemOperatorGNN(nn.Module):
    """
    Operator-based GNN where message passing implements discrete transport.
    Returns dict with predictions, edge_fluxes, and divergences.
    """
```

**Output Structure**:
```python
{
    'predictions': torch.Tensor,  # [N, 1] sucrose content
    'edge_fluxes': torch.Tensor,  # [E] predicted fluxes from last layer
    'divergences': torch.Tensor   # [N] divergence from last layer
}
```

### 3. Physics Residual for Operator Model (`model/physics.py`)

New function `physics_residual_operator()` that uses predicted fluxes directly:

**Conservation Law**:
```
dS/dt = div(J) + F_in - F_out ≈ 0
```

**Key Difference**:
- **NNConv**: Reconstructs `J` from predicted concentrations using physical equations
- **Operator**: Uses `div(J)` directly from model's divergence output
- **Same**: `F_in` and `F_out` still computed from predicted concentrations

**Advantages**:
- No reconstruction needed - fluxes are model outputs
- More direct enforcement of discrete conservation law
- Potential for learning non-standard flux patterns

### 4. Model Configuration (`model/config.py`)

Extended `ModelConfig` with `model_type` parameter:

```python
@dataclass
class ModelConfig:
    model_type: str = "nnconv"  # 'nnconv' or 'operator'
    # ... other parameters unchanged
```

### 5. Training Updates

**Setup (`training/setup.py`)**:
- `setup_model_and_scalers()` now accepts `model_type` parameter
- Instantiates appropriate model based on configuration

**Training Loop (`training/train.py`)**:
- `run_forward()`: Returns tensor OR dict depending on model type
- `_extract_predictions()`: Helper to get prediction tensor from either output format
- `compute_physics_residual_step()`: Automatically detects model type and calls appropriate physics function

**CLI (`training/cli.py`)**:
- New `--model-type` argument: `nnconv` or `operator`

## Usage

### Training with Operator Model

```bash
python train_gnn.py \
    --data-path data/simulation_output.h5 \
    --model-type operator \
    --loss-type physics \
    --batch-size 8 \
    --epochs 100
```

### Training with Baseline Model (default)

```bash
python train_gnn.py \
    --data-path data/simulation_output.h5 \
    --model-type nnconv \
    --loss-type physics \
    --batch-size 8 \
    --epochs 100
```

## Backward Compatibility

✅ **Fully backward compatible**:
- Existing code defaults to `model_type='nnconv'`
- PhloemNNConv model unchanged
- Original physics_residual function unchanged
- All training scripts work with default settings

## Implementation Details

### Message as Flux

The operator model treats messages as physical quantities (fluxes) and computes divergence explicitly:

1. **Edge Flux Computation**:
   ```python
   x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
   flux_input = [x_src, x_dst, edge_features]
   edge_fluxes = MLP(flux_input)  # Scalar flux per edge [E]
   ```

2. **Divergence via scatter_add** (explicit, not via PyG's propagate):
   ```python
   divergence = zeros(N)
   divergence.scatter_add_(0, src, +edge_fluxes)  # +J at source (outflow)
   divergence.scatter_add_(0, dst, -edge_fluxes)  # -J at destination (inflow)
   ```

   This gives the true discrete divergence: `div_i = sum_{j: i→j} J_ij - sum_{k: k→i} J_ki`

3. **Node Update**:
   ```python
   x_new = MLP([x_old, divergence])
   ```

**Key Design Choice**: We do NOT use PyG's `MessagePassing.propagate()` abstraction. Instead, we compute fluxes and divergence explicitly in `forward()`, making the operator interpretation crystal clear.### Physics Loss Integration

**NNConv Model**:
```python
pred = model(data)  # [N, 1] tensor
phys_loss, _ = physics_residual(pred, data)
```

**Operator Model**:
```python
output = model(data)  # dict
phys_loss, _ = physics_residual_operator(output, data)
```

The training code handles both automatically by checking output type.


## References

This implementation is inspired by:
- Neural operator learning approaches (e.g., FNO, DeepONet)
- Physics-informed neural networks (PINNs)
- Discrete conservation laws in finite volume methods
- Message passing as computational operators
