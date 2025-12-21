# CPlantBox Phloem Simulation Scripts

This directory contains scripts for running CPlantBox phloem flow simulations, extracting data for GNN training, and analyzing results.

## Table of Contents

- [Quick Start](#quick-start)
- [Main Scripts](#main-scripts)
- [Directory Structure](#directory-structure)
- [Important Commands](#important-commands)
- [Workflow Examples](#workflow-examples)
- [Output Files](#output-files)

## Quick Start

### Generate Training Data

```bash
# Run simulation with automatic timing measurement
python3 sim_phloem_flow.py --phloem-dir data/sim_08 --seed 42

# Extract HDF5 data for GNN training
python3 extract_phloem_data.py \
    --input data/sim_08/phloem_simulation_00.h5 \
    --output data/sim_08/phloem_simulation.h5
```

### Compare Simulations

```bash
# Compare two simulation runs
python3 compare_simulations.py \
    data/sim_05/phloem_simulation.h5 \
    data/sim_06/phloem_simulation.h5
```

## Main Scripts

### 1. `sim_phloem_flow.py`

**Purpose**: Run CPlantBox phloem flow simulation with automatic performance timing.

**Key Features**:
- Simulates plant growth and phloem flow dynamics
- Automatically measures and logs timing per step
- Saves graph size information (nodes, edges) alongside timing
- Generates visualization images (optional)
- Supports custom weather/climate conditions

**Usage**:
```bash
python3 sim_phloem_flow.py [OPTIONS]
```

**Options**:
| Flag | Default | Description |
|------|---------|-------------|
| `--save-image` | False | Save visualization images |
| `--image-dir` | `images/tmp` | Directory for saving images |
| `--phloem-dir` | `data/tmp` | Directory for phloem output files |
| `--weather-file` | `climate/baseline.json` | Weather/climate configuration file |
| `--seed` | 42 | Random seed for reproducibility |

**Output Files**:
- `{phloem-dir}/phloem_simulation_00.h5` - Full simulation data
- `{phloem-dir}/phloem_timing.csv` - Timing and graph size per step
- `{image-dir}/phloem_*.png` - Visualization images (if `--save-image`)

**Example**:
```bash
# Basic simulation
python3 sim_phloem_flow.py --phloem-dir data/sim_08 --seed 42

# With custom weather and images
python3 sim_phloem_flow.py \
    --phloem-dir data/sim_08 \
    --weather-file climate/drought.json \
    --save-image \
    --image-dir images/sim_08 \
    --seed 42
```

**Timing CSV Format**:
```csv
step,time_s,nodes,edges
0,0.156,245,312
1,0.154,245,312
2,0.153,248,316
...
```

---

### 2. `extract_phloem_data.py`

**Purpose**: Extract and preprocess phloem simulation data for GNN training.

**Key Features**:
- Converts raw simulation output to GNN-ready format
- Extracts graph structure and node/edge features
- Handles temporal snapshots
- Validates data integrity

**Usage**:
```bash
python3 extract_phloem_data.py --input INPUT_FILE --output OUTPUT_FILE
```

**Typical Workflow**:
```bash
# After running sim_phloem_flow.py
python3 extract_phloem_data.py \
    --input data/sim_08/phloem_simulation_00.h5 \
    --output data/sim_08/phloem_simulation.h5
```

**Input**: Raw simulation HDF5 file from `sim_phloem_flow.py`
**Output**: Preprocessed HDF5 file ready for GNN training

---

### 3. `compare_simulations.py`

**Purpose**: Compare two phloem simulation files to identify differences.

**Key Features**:
- Detailed comparison of simulation parameters
- Graph structure comparison (nodes, edges)
- Time step validation
- Feature value comparison
- Statistical summaries

**Usage**:
```bash
python3 compare_simulations.py FILE1 FILE2 [--verbose]
```

**Arguments**:
| Argument | Required | Description |
|----------|----------|-------------|
| `file1` | Yes | Path to first HDF5 simulation file |
| `file2` | Yes | Path to second HDF5 simulation file |
| `--verbose`, `-v` | No | Show detailed differences |

**Example**:
```bash
# Basic comparison
python3 compare_simulations.py \
    data/sim_05/phloem_simulation.h5 \
    data/sim_06/phloem_simulation.h5

# Verbose mode with detailed output
python3 compare_simulations.py \
    data/sim_05/phloem_simulation.h5 \
    data/sim_06/phloem_simulation.h5 \
    --verbose
```

**Example Output**:
```
================================================================================
Comparing Phloem Simulations
================================================================================
File 1: data/sim_05/phloem_simulation.h5
File 2: data/sim_06/phloem_simulation.h5

Metadata Comparison:
  ✓ num_timesteps: 100 == 100
  ✓ num_nodes: 245 == 245
  ✓ num_edges: 312 == 312
  ✗ seed: 42 != 43

Graph Structure:
  ✓ All edge indices match
  ✓ Node features consistent

Time Steps:
  ✓ All 100 time steps present in both files
  ✓ Feature dimensions match

Summary: Files are structurally identical but used different seeds
================================================================================
```

---

### 4. `compare_climates.py`

**Purpose**: Compare different climate/weather configurations and their effects.

**Usage**:
```bash
python3 compare_climates.py [OPTIONS]
```

**Use Case**: Analyze how different weather conditions affect simulation results.

---

### 5. `analyze_results.py`

**Purpose**: Analyze and visualize phloem simulation results.

**Usage**:
```bash
python3 analyze_results.py [OPTIONS]
```

**Use Case**: Post-processing analysis of simulation outputs, statistics, and plots.

---

## Directory Structure

```
cplantbox/
├── README.md                    # This file
├── sim_phloem_flow.py          # Main simulation script
├── extract_phloem_data.py      # Data extraction for GNN
├── compare_simulations.py      # Compare simulation files
├── compare_climates.py         # Climate comparison tool
├── analyze_results.py          # Result analysis
├── data/                       # Simulation output data
│   ├── sim_00/
│   │   ├── phloem_simulation_00.h5    # Raw simulation
│   │   ├── phloem_simulation.h5       # Processed for GNN
│   │   └── phloem_timing.csv          # Performance timing
│   ├── sim_01/
│   ├── sim_02/
│   └── ...
├── images/                     # Visualization outputs
│   ├── sim_00/
│   ├── sim_01/
│   └── ...
├── climate/                    # Weather configuration files
│   ├── baseline.json
│   ├── drought.json
│   └── ...
└── structural/                 # Structural analysis scripts
```

## Important Commands

### Standard Workflow

#### 1. Generate Multiple Simulation Runs

```bash
# Create 5 different simulation runs with different seeds
for i in {0..4}; do
    python3 sim_phloem_flow.py \
        --phloem-dir data/sim_0${i} \
        --seed $((42 + i))
done
```

#### 2. Extract Data for GNN Training

```bash
# Extract all simulations
for i in {0..4}; do
    python3 extract_phloem_data.py \
        --input data/sim_0${i}/phloem_simulation_00.h5 \
        --output data/sim_0${i}/phloem_simulation.h5
done
```

#### 3. Validate Simulations

```bash
# Compare consecutive runs
python3 compare_simulations.py \
    data/sim_00/phloem_simulation.h5 \
    data/sim_01/phloem_simulation.h5
```

### Quality Checks

#### Check Timing Performance

```bash
# View timing data
head -n 10 data/sim_08/phloem_timing.csv

# Calculate average timing
awk -F',' 'NR>1 {sum+=$2; count++} END {print "Average:", sum/count, "s"}' \
    data/sim_08/phloem_timing.csv
```

#### Verify HDF5 File Structure

```bash
# List HDF5 contents
h5ls -r data/sim_08/phloem_simulation.h5

# Check file size
du -h data/sim_08/phloem_simulation.h5
```

### Data Organization

#### List All Simulation Directories

```bash
ls -d data/sim_*/
```

#### Count Total Simulations

```bash
ls data/sim_*/phloem_simulation.h5 | wc -l
```

#### Clean Up Temporary Files

```bash
# Remove images only
rm -rf images/tmp/*

# Remove specific simulation
rm -rf data/sim_tmp/
```

## Workflow Examples

### Example 1: Quick Test Run

```bash
# 1. Run simulation
python3 sim_phloem_flow.py --phloem-dir data/test --seed 42

# 2. Extract data
python3 extract_phloem_data.py \
    --input data/test/phloem_simulation_00.h5 \
    --output data/test/phloem_simulation.h5

# 3. Check timing
cat data/test/phloem_timing.csv
```

### Example 2: Generate Training Dataset

```bash
#!/bin/bash
# Script to generate 10 simulation runs for k-fold training

for i in {0..9}; do
    echo "Running simulation $i..."
    python3 sim_phloem_flow.py \
        --phloem-dir data/sim_$(printf "%02d" $i) \
        --seed $((42 + i))

    echo "Extracting data..."
    python3 extract_phloem_data.py \
        --input data/sim_$(printf "%02d" $i)/phloem_simulation_00.h5 \
        --output data/sim_$(printf "%02d" $i)/phloem_simulation.h5

    echo "Simulation $i complete!"
done

echo "All simulations complete. Ready for GNN training."
```

### Example 3: Performance Benchmarking

```bash
# 1. Run simulation with timing
python3 sim_phloem_flow.py --phloem-dir data/sim_08 --seed 42

# 2. Train GNN model (from dev/ directory)
cd ..
python3 train_gnn.py \
    --data-path cplantbox/data/sim_08/phloem_simulation.h5 \
    --epochs 100

# 3. Benchmark GNN
python3 benchmark_gnn.py \
    --model-path logs/model/sim_08_best_model.pt \
    --data-path cplantbox/data/sim_08/phloem_simulation.h5 \
    --batch-sizes 1 4 8 \
    --output-dir benchmarks

# 4. Compare performance
python3 compare_performance.py \
    --cplantbox-timing cplantbox/data/sim_08/phloem_timing.csv \
    --gnn-timing benchmarks/sim_08_gnn_latency.csv \
    --output-dir comparison \
    --plot

cd cplantbox
```

### Example 4: Climate Sensitivity Study

```bash
# Run simulations with different climate conditions
for climate in baseline drought flood; do
    python3 sim_phloem_flow.py \
        --phloem-dir data/climate_${climate} \
        --weather-file climate/${climate}.json \
        --seed 42
done

# Compare results
python3 compare_climates.py \
    --baseline data/climate_baseline \
    --variants data/climate_drought data/climate_flood
```

## Output Files

### Phloem Timing CSV

Generated automatically by `sim_phloem_flow.py`:

**Format**:
```csv
step,time_s,nodes,edges
0,0.156,245,312
1,0.154,245,312
```

**Columns**:
- `step`: Simulation time step
- `time_s`: Wall-clock time for this step (seconds)
- `nodes`: Number of nodes in graph at this step
- `edges`: Number of edges in graph at this step

**Use**: Performance benchmarking and GNN comparison

### HDF5 Simulation Files

**Raw file** (`phloem_simulation_00.h5`):
- Complete simulation state
- All time steps
- Raw graph structure

**Processed file** (`phloem_simulation.h5`):
- GNN-ready format
- Normalized features
- Graph snapshots

**Structure**:
```
phloem_simulation.h5
├── metadata/
│   ├── num_timesteps
│   ├── num_nodes
│   ├── num_edges
│   └── seed
├── graph/
│   ├── edge_index
│   ├── node_features
│   └── edge_features
└── timesteps/
    ├── 0/
    ├── 1/
    └── ...
```

## Tips and Best Practices

### Data Organization

1. **Naming convention**: Use `sim_XX` format for simulation directories (e.g., `sim_00`, `sim_01`)
2. **Consistent seeds**: For reproducibility, document seeds used in each simulation
3. **Keep raw files**: Always keep `phloem_simulation_00.h5` as backup

### Performance

1. **Timing accuracy**: Timing includes only the phloem solver, not I/O operations
2. **Graph size tracking**: Node/edge counts help normalize performance metrics
3. **Multiple runs**: Run same configuration multiple times to get timing variance

### Simulation Quality

1. **Validate structure**: Use `compare_simulations.py` to ensure consistency
2. **Check convergence**: Review timing CSV to identify convergence patterns
3. **Climate effects**: Document weather file used for each simulation

### Data for GNN Training

1. **Multiple simulations**: Generate 5-10 runs with different seeds for k-fold training
2. **Consistent parameters**: Keep plant structure/physics constant, vary only seed
3. **Extract after simulation**: Always run `extract_phloem_data.py` before training

## Troubleshooting

### Simulation Fails

**Error**: `Cannot create simulation directory`
```bash
# Solution: Create parent directory first
mkdir -p data/sim_08
```

**Error**: `Weather file not found`
```bash
# Solution: Check climate file exists
ls climate/baseline.json
# Or use absolute path
python3 sim_phloem_flow.py --weather-file /full/path/to/climate.json
```

### Data Extraction Issues

**Error**: `Input HDF5 file not found`
```bash
# Solution: Verify raw simulation file exists
ls data/sim_08/phloem_simulation_00.h5
```

**Error**: `Output file already exists`
```bash
# Solution: Remove old file or use different output path
rm data/sim_08/phloem_simulation.h5
```

### Timing CSV Missing

```bash
# Check if file was created
ls data/sim_08/phloem_timing.csv

# If missing, re-run simulation
python3 sim_phloem_flow.py --phloem-dir data/sim_08 --seed 42
```

### Comparison Shows Unexpected Differences

```bash
# Run in verbose mode for details
python3 compare_simulations.py file1.h5 file2.h5 --verbose

# Common causes:
# - Different seeds (expected)
# - Different plant parameters (check weather files)
# - Different simulation lengths
```

## Integration with GNN Training

### From Simulation to Trained Model

```bash
# 1. Generate data (in cplantbox/)
python3 sim_phloem_flow.py --phloem-dir data/sim_08 --seed 42
python3 extract_phloem_data.py \
    --input data/sim_08/phloem_simulation_00.h5 \
    --output data/sim_08/phloem_simulation.h5

# 2. Train GNN (in dev/)
cd ..
python3 train_gnn.py \
    --data-path cplantbox/data/sim_08/phloem_simulation.h5 \
    --epochs 100

# 3. Benchmark (in dev/)
python3 benchmark_gnn.py \
    --model-path logs/model/sim_08_best_model.pt \
    --data-path cplantbox/data/sim_08/phloem_simulation.h5 \
    --output-dir benchmarks

# 4. Compare (in dev/)
python3 compare_performance.py \
    --cplantbox-timing cplantbox/data/sim_08/phloem_timing.csv \
    --gnn-timing benchmarks/sim_08_gnn_latency.csv \
    --output-dir comparison \
    --plot
```

All output files will be automatically prefixed with `sim_08` for easy organization.

## Further Reading

- **GNN Training**: See `../README.md` for training options and k-fold cross-validation
- **Model Architecture**: See `../model/` for GNN implementation details
- **Physics Loss**: See `../model/physics.py` for physics-informed training

---

**For questions or issues, please check the main project README or contact the development team.**
