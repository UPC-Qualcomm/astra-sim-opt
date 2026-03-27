# SLURM Optimization Suite - Quick Experiment Reference

## All 28 Experiments Organized by Model

### LLaMA-8B (32 NPUs)

**Original Objectives**
1. `llama8b_32npus_time` → Minimize training time
2. `llama8b_32npus_edp` → Minimize EDP (Energy-Delay Product)
3. `llama8b_32npus_energy_and_time` → Minimize energy + time (multi-objective)

**New Objectives**
4. `llama8b_32npus_time_and_bw` → Minimize training time + network bandwidth
5. `llama8b_32npus_edp_and_bw` → Minimize EDP + network bandwidth
6. `llama8b_32npus_time_and_throughput_per_energy` → Minimize time, maximize throughput/energy (cluster size opt)
7. `llama8b_32npus_memory_and_time` → Maximize memory, minimize time (batch size opt)

---

### GPT-60B (128 NPUs, model_num=19, 56 layers)

**Original Objectives**
1. `gpt60b_128npus_time` → Minimize training time
2. `gpt60b_128npus_edp` → Minimize EDP
3. `gpt60b_128npus_energy_and_time` → Minimize energy + time

**New Objectives**
4. `gpt60b_128npus_time_and_bw` → Minimize training time + network bandwidth
5. `gpt60b_128npus_edp_and_bw` → Minimize EDP + network bandwidth
6. `gpt60b_128npus_time_and_throughput_per_energy` → Cluster size optimization
7. `gpt60b_128npus_memory_and_time` → Batch size optimization

---

### LLaMA-70B (128 NPUs)

**Original Objectives**
1. `llama70b_128npus_time` → Minimize training time
2. `llama70b_128npus_edp` → Minimize EDP
3. `llama70b_128npus_energy_and_time` → Minimize energy + time

**New Objectives**
4. `llama70b_128npus_time_and_bw` → Minimize training time + network bandwidth
5. `llama70b_128npus_edp_and_bw` → Minimize EDP + network bandwidth
6. `llama70b_128npus_time_and_throughput_per_energy` → Cluster size optimization
7. `llama70b_128npus_memory_and_time` → Batch size optimization

---

### GPT-175B (1024 NPUs)

**Original Objectives**
1. `gpt175b_1024npus_time` → Minimize training time
2. `gpt175b_1024npus_edp` → Minimize EDP
3. `gpt175b_1024npus_energy_and_time` → Minimize energy + time

**New Objectives**
4. `gpt175b_1024npus_time_and_bw` → Minimize training time + network bandwidth
5. `gpt175b_1024npus_edp_and_bw` → Minimize EDP + network bandwidth
6. `gpt175b_1024npus_time_and_throughput_per_energy` → Cluster size optimization
7. `gpt175b_1024npus_memory_and_time` → Batch size optimization

**Note**: GPT-175B experiments have 10 GB memory per core (auto-set)

---

## Grouped by Objective Type

### Type 1: Single-Objective (Training Time)
```
llama8b_32npus_time
gpt60b_128npus_time
llama70b_128npus_time
gpt175b_1024npus_time
```
**Count**: 4 experiments

### Type 2: Single-Objective (EDP)
```
llama8b_32npus_edp
gpt60b_128npus_edp
llama70b_128npus_edp
gpt175b_1024npus_edp
```
**Count**: 4 experiments (Requires power estimation)

### Type 3: Multi-Objective (Energy + Time)
```
llama8b_32npus_energy_and_time
gpt60b_128npus_energy_and_time
llama70b_128npus_energy_and_time
gpt175b_1024npus_energy_and_time
```
**Count**: 4 experiments (Requires power estimation)

### Type 4: Multi-Objective (Time + Network Bandwidth)
```
llama8b_32npus_time_and_bw
gpt60b_128npus_time_and_bw
llama70b_128npus_time_and_bw
gpt175b_1024npus_time_and_bw
```
**Count**: 4 experiments

### Type 5: Multi-Objective (EDP + Network Bandwidth)
```
llama8b_32npus_edp_and_bw
gpt60b_128npus_edp_and_bw
llama70b_128npus_edp_and_bw
gpt175b_1024npus_edp_and_bw
```
**Count**: 4 experiments (Requires power estimation)

### Type 6: Multi-Objective (Time + Throughput/Energy) — Cluster Size Optimization
```
llama8b_32npus_time_and_throughput_per_energy
gpt60b_128npus_time_and_throughput_per_energy
llama70b_128npus_time_and_throughput_per_energy
gpt175b_1024npus_time_and_throughput_per_energy
```
**Count**: 4 experiments (Requires power estimation)

### Type 7: Multi-Objective (Memory + Time) — Batch Size Optimization
```
llama8b_32npus_memory_and_time
gpt60b_128npus_memory_and_time
llama70b_128npus_memory_and_time
gpt175b_1024npus_memory_and_time
```
**Count**: 4 experiments

---

## Key Statistics

| Category | Count |
|----------|-------|
| Total Experiments | 28 |
| Single-Objective | 8 |
| Multi-Objective (2-objective) | 20 |
| Multi-Objective (3+ objectives) | 0 |
| Requiring Power Estimation | 16 |
| Not Requiring Power Estimation | 12 |

---

## Launch Scenarios

### Scenario 1: Test All Objectives on Single Model
```bash
# Test all 7 objectives on LLaMA-8B
bash launch_all.sh --max-jobs 7 --partition gpu_large
```

### Scenario 2: Focus on New Objectives Only
```bash
# Run only the 16 new experiments
find experiments -type d -name "*time_and_bw" -o -name "*edp_and_bw" \
  -o -name "*time_and_throughput*" -o -name "*memory_and_time" | wc -l
# Then submit with --max-jobs based on count
```

### Scenario 3: Compare Large Models Only
```bash
# Submit only GPT models (gpt60b and gpt175b)
find experiments -type d -name "gpt*" | wc -l
bash launch_all.sh --max-jobs 14
```

### Scenario 4: Iterative Testing
```bash
# Start with dry run
bash launch_all.sh --dry-run > assignments.txt

# Review assignments
cat launch_logs/TIMESTAMP/assignments.csv

# Submit first batch
bash launch_all.sh --max-jobs 5

# After monitoring, submit next batch later
bash launch_all.sh --max-jobs 10
```

---

## Default Settings Per Experiment

```bash
BUDGET=100              # 100 evaluations per experiment
INIT_SAMPLES=20         # 20 random initial samples
N_WORKERS=8             # 8 parallel workers
TOP_K=10                # Keep top 10 checkpoints
CLEANUP_BATCH_SIZE=20   # Cleanup every 20 evaluations
CORES_PERCENT=32        # Use 32% of available cores
```

All experiments use:
- Simulator: G2 (astra_g2)
- Topology: FoldedClos
- Routing: foldedclos_uniform
- Compression: Enabled (compress_and_clean)

---

## Objective Directions Summary

| Objective | Minimizes | Maximizes |
|-----------|-----------|-----------|
| time | ✓ execution time | — |
| edp | ✓ EDP (E×D) | — |
| energy_and_time | ✓ energy, ✓ time | — |
| time_and_bw | ✓ time, ✓ BW | — |
| edp_and_bw | ✓ EDP, ✓ BW | — |
| time_and_throughput_per_energy | ✓ time | ✓ throughput/energy |
| memory_and_time | — | ✓ memory (1st obj) |
|  | ✓ time (2nd obj) | — |

---

## Power Estimation Requirement

Experiments requiring `estimate_power=1`:
- All `*_edp*` objectives
- `*_energy_and_time`
- `*_time_and_throughput_per_energy`

**Configuration**: Power model set to `/scratch/nas/4/nasser/astra-sim/upc/power_model/a100_config.json` in script (can be overridden per run).

---

For detailed experiment configuration and monitoring, see [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md).
