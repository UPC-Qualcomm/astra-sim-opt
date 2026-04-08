# Quick Start Guide - SLURM Optimization Suite

## 5-Minute Setup

### 1. Navigate to Suite
```bash
cd /scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite_last_hope
```

### 2. Preview Job Assignment
```bash
bash launch_all.sh --dry-run | head -50
```

### 3. Submit All 28 Experiments
```bash
bash launch_all.sh
```

Output will show:
```
Submitted llama8b_32npus_time to node-01 (job 12345, cpus=8, mem/cpu=4G)
Submitted llama8b_32npus_edp to node-02 (job 12346, cpus=8, mem/cpu=4G)
...
Launch summary
- Total considered: 28
- Submitted: 28
- Failed/skipped: 0
- Node snapshot: launch_logs/20260327_215900/sinfo_nodes.txt
- Assignment file: launch_logs/20260327_215900/assignments.csv
```

### 4. Monitor Progress
```bash
# Watch jobs
watch -n 30 'squeue | grep -E "llama|gpt" | wc -l'

# Or check specific job
squeue -j 12345 -o "%.20j %.3t %.10M"

# Check result files as they arrive
tail -f experiments/llama8b_32npus_time/logs/slurm-*.out
```

### 5. Review Results
```bash
# After jobs complete, find results
for d in experiments/*/outputs/run_*/; do
  echo "=== $(basename $(dirname $d)) ==="
  head -5 "$d/deephyper_results_*.csv" 2>/dev/null | head -3
done
```

---

## Common Tasks

### Run Only Cluster Size Optimization (4 jobs)
```bash
bash launch_all.sh --max-jobs 4
```

Runs: llama8b, gpt60b, llama70b, gpt175b with cluster size objective

### Use Specific Partition
```bash
# Check available partitions
sinfo -o "%P %a"

# Submit to GPU nodes
bash launch_all.sh --partition gpu_nodes
```

### Adjust Per-Experiment Resources
Edit specific experiment config:
```bash
# Increase memory for batch size optimization
cd experiments/gpt175b_1024npus_memory_and_time
sed -i 's/MEM_PER_CPU_GB_OVERRIDE=.*/MEM_PER_CPU_GB_OVERRIDE=12/' config.env
cd ../..
bash launch_all.sh --max-jobs 1
```

### Skip Power-Heavy Experiments (Quick Test)
```bash
# Run only non-power-estimation experiments (12 jobs)
# Modify launcher to use --max-jobs N where N is count of basic objectives
bash launch_all.sh --max-jobs 12
```

---

## Objective Quick Lookup

| Want | Experiments | Command |
|------|-------------|---------|
| Speed only | 4 (time) | `--max-jobs 4` |
| Energy-aware | 4 (edp) | Skip to job 5-8 |
| Time + Energy | 4 (energy_and_time) | Skip to job 9-12 |
| Time + Bandwidth | 8 (time_and_bw) | Jobs 13-20 |
| EDP + Bandwidth | 8 (edp_and_bw) | Jobs 21-28 |
| Cluster Size | 4 (time_throughput_energy) | Jobs 1,5,9,13 of new subset |
| Batch Size | 4 (memory_time) | Jobs 2,6,10,14 of new subset |

- Original suite: 12 experiments (objectives 1-3)
- New suite: 16 experiments (objectives 4-7)

---

## Troubleshooting

### "Not enough available nodes"
Solution: Enable node reuse OR reduce experiments
```bash
# Option 1: Reuse nodes
bash launch_all.sh --allow-node-reuse

# Option 2: Submit in batches
bash launch_all.sh --max-jobs 5  # First batch
sleep 3600  # Wait 1 hour
bash launch_all.sh --max-jobs 10  # Second batch
```

### Jobs stuck in PENDING
Check memory availability:
```bash
sinfo -o "%N %m %g %c"  # Node, Memory, Group, CPUs
squeue -l | grep PENDING
```

Reduce per-core memory if nodes are undersized:
```bash
# In config.env
MEM_PER_CPU_GB_OVERRIDE=2  # Was 4 for small NPU models
```

### "start_py_311.sh not found"
Verify environment script exists:
```bash
ls -la /scratch/nas/4/nasser/start_py_311.sh
```

If missing, create a simpler version:
```bash
cat > /scratch/nas/4/nasser/start_py_311.sh <<'EOF'
#!/bin/bash
export PATH=/scratch/nas/4/nasser/python311/python/bin:$PATH
export PYTHONPATH=/media/mohammad/extension/astra-sim/upc:$PYTHONPATH
EOF
chmod +x /scratch/nas/4/nasser/start_py_311.sh
```

### Results folder is empty
Check if jobs are still running:
```bash
squeue | grep -c <JOBID>
```

If running, wait for completion. Check logs for errors:
```bash
tail -50 experiments/<exp>/logs/optimization_*.log
tail -50 experiments/<exp>/logs/slurm-*.out
```

---

## Understanding Results

**deephyper_results_*.csv columns**:
- `point`: Configuration ID
- Model parameters (dp, mp, ssp, pp, weight_sharded, etc.)
- `exec_time`: Execution time in nanoseconds
- `total_energy_J`, `total_power_W`: Energy metrics (if power enabled)
- `peak_memory_bytes`: GPU memory used
- `network_bw_gbps`: Total network bandwidth
- Objective scores (log-transformed)
- `was_killed`: 1 if tracker terminated early, 0 if completed normally

**Example analysis**:
```bash
# Show best configuration (by time)
python3 -c "
import pandas as pd
import glob

csv_file = glob.glob('experiments/llama8b_32npus_time/outputs/*/deephyper_results_*.csv')[0]
df = pd.read_csv(csv_file)
best = df.loc[df['exec_time'].idxmin()]
print('Best configuration:\n', best)
"
```

---

## Performance Tips

1. **Warm-up Runs**: Initial random samples can be slow; let them stabilize
2. **Batch Submission**: Submit all 28 at once (better cluster utilization)
3. **Off-Peak Scheduling**: Submit at night/weekend for lower contention
4. **Early Termination**: Tracker kills slow runs after ~50% of evaluations
5. **Parallelism**: Use N_WORKERS=8 but adjust for node capacity

---

## Documentation

- **Full Guide**: Read [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md)
- **Quick Ref**: See [EXPERIMENTS_QUICK_REF.md](EXPERIMENTS_QUICK_REF.md)
- **Technical Details**: Check [NEW_OBJECTIVES_IMPLEMENTATION.md](NEW_OBJECTIVES_IMPLEMENTATION.md)
- **Summary**: Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## Files & Locations

```
slurm_optimization_suite_last_hope/
├── launch_all.sh                          # Main launcher
├── experiments/
│   ├── llama8b_32npus_time/
│   ├── llama8b_32npus_edp/
│   ├── llama8b_32npus_energy_and_time/
│   ├── llama8b_32npus_time_and_bw/
│   ├── llama8b_32npus_edp_and_bw/
│   ├── llama8b_32npus_time_and_throughput_per_energy/
│   ├── llama8b_32npus_memory_and_time/
│   ├── gpt60b_128npus_*/
│   ├── llama70b_128npus_*/
│   └── gpt175b_1024npus_*/
└── launch_logs/
    └── TIMESTAMP/
        ├── assignments.csv
        └── sinfo_nodes.txt
```

---

Start here: `bash launch_all.sh --dry-run`
