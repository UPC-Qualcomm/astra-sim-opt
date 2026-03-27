# SLURM Optimization Suite - Experiments Guide

## Overview

This document describes the 28 DeepHyper optimization experiments organized across 4 LLM models and 7 objectives (3 original + 4 new).

**Total Experiments**: 28
- 12 original experiments (4 models × 3 initial objectives)  
- 16 new experiments (4 models × 4 new objectives)

## Models

| Model | Model Name | Model Num | NPUs | Search Space | Default Memory/Core |
|-------|-----------|-----------|------|--------------|-------------------|
| LLaMA-8B | LLaMA_8B | 17 | 32 | `parallelism_strategy_params_g2_intra.json` | 4 GB |
| GPT-60B | GPT_60B_56L | 19 | 128 | `parallelism_strategy_params_g2_intra.json` | 4 GB |
| LLaMA-70B | LLaMA_70B | 14 | 128 | `parallelism_strategy_params_g2_intra.json` | 4 GB |
| GPT-175B | GPT_175B | 10 | 1024 | `parallelism_strategy_params_g2_intra.json` | **10 GB** |

## Objective Functions

### Group 1: Original Objectives (12 experiments)

Directories: `experiments/<model>_*npus_<objective>/`

#### Objective 1: Minimize Training Time
- **Key**: `time`
- **Models**: All 4
- **Returns**: Single scalar (execution time in nanoseconds)
- **Direction**: Minimize
- **Use case**: Optimizing for speed

#### Objective 2: Minimize EDP (Energy-Delay Product)
- **Key**: `edp`
- **Models**: All 4
- **Returns**: Single scalar (log10(E × D))
- **Direction**: Minimize
- **Use case**: Joint energy-efficiency optimization at symmetric weight (1:1)
- **Requires**: `estimate_power=1` in net_sim_config

#### Objective 3: Minimize Energy + Time (2-objective)
- **Key**: `energy_and_time`
- **Models**: All 4
- **Returns**: Tuple `(log10(total_energy_J), log10(exec_time))`
- **Directions**: [Minimize, Minimize]
- **Use case**: Pareto-optimal front: find configs that reduce energy *without* sacrificing speed
- **Requires**: `estimate_power=1` in net_sim_config

---

### Group 2: Time + Bandwidth Objectives (8 new experiments)

Directories: `experiments/<model>_*npus_time_and_bw/`

#### Objective: Minimize Training Time + Total Network Bandwidth
- **Key**: `time_and_network_bw`
- **Models**: All 4
- **Returns**: Tuple `(log10(exec_time), log10(total_network_bw_GBps))`
- **Directions**: [Minimize, Minimize]
- **Use case**: Reduce execution time while constraining network over-provisioning; important for cost-aware design
- **Network BW**: Computed from bandwidth_config (intra-node/inter-node links)

#### Objective: Minimize EDP + Total Network Bandwidth
- **Key**: `edp_and_network_bw`
- **Models**: All 4
- **Returns**: Tuple `(log10(EDP), log10(total_network_bw_GBps))`
- **Directions**: [Minimize, Minimize]
- **Use case**: Energy-efficient configurations that don't over-provision network
- **Requires**: `estimate_power=1` in net_sim_config

---

### Group 3: Cluster Size Optimization (4 new experiments)

Directories: `experiments/<model>_*npus_time_and_throughput_per_energy/`

#### Objective: Minimize Training Time + Maximize Throughput/Energy Efficiency
- **Key**: `time_and_throughput_per_energy`
- **Models**: All 4
- **Returns**: Tuple `(log10(exec_time), log10(samples/sec/MJ))`
- **Directions**: [Minimize, Maximize]
- **Formula**: 
  - `throughput_per_energy = batch_size / (exec_time_sec × total_energy_MJ)`
  - Measures how many samples we process per second per unit of energy consumed
- **Use case**: Find optimal cluster configurations that balance speed and energy efficiency
- **Requires**: `estimate_power=1` in net_sim_config

---

### Group 4: Batch Size Optimization (4 new experiments)

Directories: `experiments/<model>_*npus_memory_and_time/`

#### Objective: Maximize GPU Memory Utilization + Minimize Training Time
- **Key**: `memory_and_time`
- **Models**: All 4
- **Returns**: Tuple `(log10(peak_memory_GB), log10(exec_time))`
- **Directions**: [Maximize, Minimize]
- **Use case**: Find batch sizes that efficiently use GPU memory capacity without increasing execution time too much
- **Peak Memory**: Extracted from metadata['peak_memory_bytes'], converted to GB

---

## Experiment Folder Structure

Each experiment folder contains:

```
<exp_name>/
├── config.env                          # Experiment configuration (model, objective, budget, etc.)
├── run_experiment.sh                   # SLURM job runner script (executable)
├── inputs/
│   └── search_space.json               # Search space definition (copied from shared location)
├── logs/
│   ├── optimization_TIMESTAMP.log      # Detailed optimization log per run
│   └── slurm-JOBID.{out,err}          # SLURM job stdout/stderr
└── outputs/
    └── run_TIMESTAMP/
        ├── deephyper_results_*.csv     # Optimization history + final results
        ├── pareto_front_*.{html,png}   # Pareto front visualization (2-objective only)
        └── hypervolume_*.csv           # Hypervolume progression
```

## Configuration Tuning Per Experiment

Edit `config.env` in each experiment folder to adjust:

| Setting | Purpose | Default |
|---------|---------|---------|
| `BUDGET` | Total number of optimization evaluations | 100 |
| `INIT_SAMPLES` | Random initialization samples | 20 |
| `N_WORKERS` | Parallel workers for DeepHyper | 8 |
| `TOP_K` | Keep top-K checkpoints | 10 |
| `CLEANUP_BATCH_SIZE` | Cleanup frequency (artifact management) | 20 |
| `CORES_PERCENT` | CPU cores as % of available | 32 |
| `CPUS_PER_TASK_OVERRIDE` | Explicit CPU request (overrides %) | (unset) |
| `MEM_PER_CPU_GB_OVERRIDE` | Explicit memory per CPU in GB | (unset, auto=4GB for <1024 NPU, 10GB for 1024 NPU) |
| `PARTITION_OVERRIDE` | Force SLURM partition | (unset) |

## How Tracker Works With New Objectives

The SimulationTracker in `core/simulation_tracker.py` has been enhanced to handle mixed min/max objectives:

1. **Objective Directions**: Each objective function can specify `objective_directions` attribute
   - `[True, False]` = minimize 1st, maximize 2nd
   - `[False, True]` = maximize 1st, minimize 2nd
   - `[True, True]` = minimize both

2. **Smart Kill Threshold**: For each objective, the tracker computes:
   - **Minimization**: Kill if `score > threshold × multiplier`
   - **Maximization**: Kill if `score < threshold / multiplier`

3. **Threshold Update**: After successful runs, threshold updates reflect per-objective semantics:
   - **Minimize**: Update if `new_score < old_threshold`
   - **Maximize**: Update if `new_score > old_threshold`

4. **Multi-Objective Pruning**: Kills a run only if it's worse than threshold on **all objectives** (Pareto-aware)

This ensures the tracker correctly manages early termination for heterogeneous objective sets.

## Launching All Experiments

### Dry Run (Show Job Assignments)
```bash
cd /scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite
bash launch_all.sh --dry-run
```

### Submit All 28 Experiments
```bash
cd /scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite
bash launch_all.sh
```

### Submit Only First 10 Experiments
```bash
bash launch_all.sh --max-jobs 10
```

### Allow Node Reuse (if more experiments than nodes)
```bash
bash launch_all.sh --allow-node-reuse
```

### Force Specific SLURM Partition
```bash
bash launch_all.sh --partition gpu_nodes
```

## Resource Allocation Policy

### Default Behavior

1. **Node Discovery**: Finds all idle/mixed nodes via `sinfo`
2. **Free Resource Check**: Queries each node for available CPUs/memory using `scontrol show node`
3. **One-to-One Assignment**: Each experiment → one unique node (by default)
4. **CPU Calculation**: `cpus_per_task = free_cores × CORES_PERCENT / 100`
5. **Memory Calculation** (auto):
   - NPUs < 1024: `mem_per_cpu = 4 GB`
   - NPUs ≥ 1024: `mem_per_cpu = 10 GB`
6. **Validation**: Ensures total memory request fits node capacity; reduces CPUs if needed

### Per-Model Defaults

| Model | NPUs | Default Memory/Core |
|-------|------|-------------------|
| LLaMA-8B | 32 | 4 GB |
| GPT-60B | 128 | 4 GB |
| LLaMA-70B | 128 | 4 GB |
| GPT-175B | 1024 | **10 GB** (auto-set in config) |

### Overriding Defaults

Edit `config.env` in an experiment folder:

```bash
# Force 16 CPUs
CPUS_PER_TASK_OVERRIDE=16

# Force 8 GB memory per CPU
MEM_PER_CPU_GB_OVERRIDE=8

# Force specific partition
PARTITION_OVERRIDE=memory_optimized
```

## Launch Log Outputs

After running `launch_all.sh`, check:

```bash
ls -la launch_logs/TIMESTAMP/
├── sinfo_nodes.txt        # Snapshot of available nodes
└── assignments.csv        # Job-to-node mapping with resource allocation
```

Headers in `assignments.csv`:
```
experiment,node,partition,node_cores_total,node_mem_total_mb,node_cores_free,
node_mem_free_mb,cpus_per_task,mem_per_cpu_gb,mem_per_cpu_slurm,job_name,
submit_status,job_id
```

## Monitoring Running Jobs

```bash
# Watch all jobs
squeue | grep -E "llama8b|gpt60b|llama70b|gpt175b"

# Check specific experiment logs
tail -f slurm_optimization_suite/experiments/llama8b_32npus_time/logs/slurm-*.out
```

## Results & Analysis

After optimization completes, results are in:

```bash
<exp_dir>/outputs/run_TIMESTAMP/deephyper_results_*.csv
```

Key columns:
- `point`: Configuration index
- `energy_J`: Total energy (if applicable)
- `exec_time`: Execution time in nanoseconds
- `peak_memory_bytes`: Peak GPU memory used
- `was_killed`: 1 if terminated early by tracker, 0 if completed
- Objective scores (2-objective files include both metrics)

## Example Workflow

### 1. Submit all 28 experiments
```bash
cd slurm_optimization_suite
bash launch_all.sh > launch_output.txt 2>&1
```

### 2. Monitor progress
```bash
watch -n 30 'squeue | grep -c llama'
tail -f experiments/llama8b_32npus_time/logs/slurm-*.out
```

### 3. After completion, analyze results
```bash
# Collect Pareto fronts
for d in experiments/*/outputs/run_*/; do
  [ -f "$d/pareto_front_*.html" ] && echo "Pareto: $d"
done

# Check kill statistics
grep "total_killed" experiments/*/logs/*.log | head -20
```

## Troubleshooting

### Issue: "Not enough available nodes for one-to-one assignment"
**Solution**: Use `--allow-node-reuse` flag or reduce jobs with `--max-jobs`

### Issue: Jobs stay in PENDING
**Solution**: Check memory requests fit available nodes:
```bash
# See what's needed
grep "MEM_PER_CPU_GB_OVERRIDE" experiments/*/config.env | sort -u

# Adjust CORES_PERCENT in config.env to reduce memory demand
```

### Issue: Tracker isn't killing slow simulations
**Solution**: Verify `--compress-and-clean` is in run script and enable_tracker=True in optimizer

### Issue: Multi-objective metrics not showing in Pareto plot
**Solution**: Check CSV column headers; ensure objective returns tuple not scalar

---

**Last Updated**: 2026-03-27  
**Suite Status**: 28/28 experiments ready for SLURM submission
