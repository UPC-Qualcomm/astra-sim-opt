# AstraSim Peak Memory Prediction ML Model

This directory contains tools to generate training data, train ML models, and predict peak memory usage for AstraSim simulations.

## Overview

The ML model predicts GPU peak memory usage based on:
- **Model parameters**: `din`, `dmodel`, `dff`, `batch`, `micro_batch`, `seq`, `head`, `num_stacks`
- **Parallelism strategy**: `dp`, `mp`, `sp`, `pp`, `fsdp`
- **Hardware**: `num_npus`

## Quick Start

### Test Mode (Fast)
```bash
cd /media/mohammad/extension/experiments/astra-sim/upc
export TEST_MODE=true
bash ml_model/run_all.sh
```

### Full Pipeline
```bash
cd /media/mohammad/extension/experiments/astra-sim/upc
export NPU_COUNTS="16,32,64,128,256,512,1024,2048"
export PARALLEL_JOBS=8
bash ml_model/run_all.sh
```

## Step-by-Step Usage

### 1. Generate Training Data

Generate simulation data for various configurations:

```bash
python ml_model/generate_training_data.py \
    --npu_counts "16,32,64,128,256" \
    --parallel_jobs 4 \
    --sim_type analytical_unaware \
    --output_csv ml_model/training_data.csv
```

**Arguments:**
- `--npu_counts`: Comma-separated NPU counts to test (e.g., "16,32,64,128")
- `--parallel_jobs`: Number of parallel simulation jobs (default: 4)
- `--output_csv`: Output CSV file path (default: ml_model/training_data.csv)
- `--sim_type`: Simulation type (analytical_unaware, analytical_aware, g2)
- `--test_mode`: Run with minimal configurations for testing

**Output:** `ml_model/training_data.csv` with columns:
- Model params: din, dmodel, dff, batch, micro_batch, seq, head, num_stacks
- Parallelism: dp, mp, sp, pp, fsdp, num_npus
- Results: avg_peak_memory_gb, avg_wall_time_cycles, avg_exposed_comm_cycles

### 2. Train ML Model

Train multiple models and select the best one:

```bash
python ml_model/train_model.py \
    --input_csv ml_model/training_data.csv \
    --output_dir ml_model/trained_models \
    --test_size 0.2
```

**Arguments:**
- `--input_csv`: Input training data CSV
- `--output_dir`: Directory to save trained models
- `--test_size`: Fraction of data for testing (default: 0.2)

**Output Files:**
- `best_model.pkl`: Best performing model
- `scaler.pkl`: Feature scaler
- `feature_names.txt`: Feature column names
- `model_results.csv`: Comparison of all models
- `model_comparison.png`: Performance comparison plot
- `predictions_vs_actual.png`: Prediction accuracy plots
- `feature_importance.png`: Feature importance analysis

**Models Trained:**
- Random Forest
- Gradient Boosting
- Ridge Regression
- Lasso Regression
- Multi-Layer Perceptron (MLP)

### 3. Make Predictions

Use the trained model to predict peak memory:

```bash
python ml_model/predict.py \
    --model_dir ml_model/trained_models \
    --din 50000 \
    --dmodel 4096 \
    --dff 16384 \
    --batch 512 \
    --micro_batch 32 \
    --seq 2048 \
    --head 32 \
    --num_stacks 32 \
    --dp 2 \
    --mp 8 \
    --sp 1 \
    --pp 4 \
    --fsdp 1 \
    --num_npus 64
```

**Arguments:**
- Model parameters: `--din`, `--dmodel`, `--dff`, `--batch`, `--micro_batch`, `--seq`, `--head`, `--num_stacks`
- Parallelism: `--dp`, `--mp`, `--sp`, `--pp`, `--fsdp` (0 or 1), `--num_npus`
- `--model_dir`: Directory containing trained model

**Constraint:** `dp × mp × sp × pp` must equal `num_npus`

## Features

The model uses the following features:

### Base Features
- `din`: Input vocabulary size
- `dmodel`: Model dimension
- `dff`: Feed-forward dimension
- `batch`: Global batch size
- `micro_batch`: Micro-batch size per iteration
- `seq`: Sequence length
- `head`: Number of attention heads
- `num_stacks`: Number of transformer layers
- `dp`: Data parallelism degree
- `mp`: Model/tensor parallelism degree
- `sp`: Spatial parallelism degree
- `pp`: Pipeline parallelism degree
- `fsdp`: Fully Sharded Data Parallel (0 or 1)
- `num_npus`: Total number of NPUs

### Derived Features
- `total_params_estimate`: Total model parameters
- `sharding_factor`: Combined sharding from mp and fsdp
- `micro_batch_per_npu`: Effective micro-batch per NPU
- `activation_size_estimate`: Estimated activation memory
- `params_per_npu`: Parameters per NPU after sharding
- `params_size_gb`: Parameter size in GB
- Log-transformed versions of key features

## Directory Structure

```
ml_model/
├── README.md                    # This file
├── generate_training_data.py    # Generate simulation data
├── train_model.py               # Train ML models
├── predict.py                   # Make predictions
├── run_all.sh                   # Complete pipeline script
├── training_data.csv            # Generated training data
├── trained_models/              # Trained model artifacts
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.txt
│   ├── model_results.csv
│   └── *.png                    # Visualization plots
├── networks/                    # Generated network YAML files
├── output/                      # Simulation output logs
└── network_log/                 # Network simulation logs
```

## Environment Variables

Set before running `run_all.sh`:

```bash
export TEST_MODE=true              # Run in test mode (faster, fewer configs)
export PARALLEL_JOBS=8             # Number of parallel simulation jobs
export NPU_COUNTS="16,32,64,128"   # NPU counts to test
export SIM_TYPE="analytical_unaware"  # Simulation type
```

## Network Configuration

The pipeline automatically generates network YAML files for each NPU count:
- First dimension: 8 (fixed)
- Second dimension: `num_npus / 8`

Example for 64 NPUs:
```yaml
topology: [Switch, Switch]
npus_count: [8, 8]
bandwidth: [900, 200.0]  # GB/s
latency: [500.0, 500.0]  # ns
```

## Performance Tips

1. **Start with test mode** to verify the pipeline works
2. **Adjust parallel_jobs** based on available CPU cores
3. **Filter NPU counts** based on your target hardware
4. **Monitor disk space** - simulations generate large log files

## Troubleshooting

**Issue: Simulations timeout**
- Reduce the number of configurations
- Increase timeout in `generate_training_data.py`
- Use smaller models in test mode

**Issue: Low model accuracy**
- Generate more training data
- Include more diverse model configurations
- Check for outliers in training data

**Issue: Out of memory during training**
- Reduce the number of samples
- Use simpler models (Ridge/Lasso instead of MLP)
- Increase system RAM or use data sampling

## Example Output

```
Configuration:
  Model: din=50000, dmodel=4096, dff=16384
         batch=512, micro_batch=32, seq=2048
         head=32, num_stacks=32
  Parallelism: dp=2, mp=8, sp=1, pp=4, fsdp=1
  NPUs: 64

Predicted Peak Memory: 42.35 GB per NPU
```

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- PyYAML

Install with:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib pyyaml
```
