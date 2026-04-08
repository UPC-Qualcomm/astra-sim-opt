# New Objectives Implementation Notes

## Summary of Changes

Two new custom objective functions have been added to support advanced optimization scenarios:

1. **MinimizeTimeMaximizeThroughputPerEnergy** (key: `time_and_throughput_per_energy`)
2. **MaximizeMemoryMinimizeTime** (key: `memory_and_time`)

These have been integrated into:
- `/scratch/nas/4/nasser/astra-sim/upc/Optimization/core/objective.py`
- Updated the objective registry in `create_objective()` function
- Updated `get_available_objective_types()` to include new objectives

---

## Objective 1: MinimizeTimeMaximizeThroughputPerEnergy

### Purpose
Find cluster configurations that are both fast and energy-efficient. The second objective maximizes throughput-per-energy, encouraging configurations with high computational density relative to energy consumption.

### Use Case
Cluster size optimization: Choose between different node counts/topologies to balance speed and energy efficiency.

### Implementation Details

**Returns**: Tuple of 2 values
```python
(log10(exec_time), log10(throughput_per_energy))
```

**Throughput/Energy Calculation**:
```python
throughput_per_energy = batch_size / (exec_time_sec × total_energy_MJ)
```
- `batch_size`: From config (samples per iteration)
- `exec_time_sec`: Execution time converted to seconds (input is nanoseconds)
- `total_energy_MJ`: Total energy consumed in megajoules (metadata['total_energy_J'] / 1e6)

**Objective Directions**: `[True, False]`
- Index 0 (exec_time): **Minimize** (True)
- Index 1 (throughput/energy): **Maximize** (False)

### Metadata Requirements
```python
metadata['total_energy_J']  # Must be provided by simulator
```

**Requires**: `estimate_power=1` in `net_sim_config`

### Tracker Behavior

When `SimulationTracker` encounters this objective:

1. **Threshold Reading**:
   - For 1st objective (time): Uses `< kill_val` (minimize)
   - For 2nd objective (throughput): Uses `> kill_val` (maximize)

2. **Kill Decision**:
   - Kills only if BOTH conditions met:
     - `exec_time > kill_threshold × 1.5` AND
     - `throughput_per_energy < initial_best_efficiency / 1.5`

3. **Threshold Update**:
   - 1st objective: Update if `new_time < old_threshold`
   - 2nd objective: Update if `new_throughput > old_threshold`

---

## Objective 2: MaximizeMemoryMinimizeTime

### Purpose
Find batch sizes that efficiently use available GPU memory without increasing execution time excessively. Higher memory utilization often indicates larger batches, which can improve throughput.

### Use Case
Batch size optimization: Choose batch size to maximize GPU capacity utilization while maintaining fast execution.

### Implementation Details

**Returns**: Tuple of 2 values
```python
(log10(peak_memory_GB), log10(exec_time))
```

**Memory Extraction**:
```python
peak_memory_GB = metadata['peak_memory_bytes'] / (1024**3)
```

**Objective Directions**: `[False, True]`
- Index 0 (peak_memory): **Maximize** (False)
- Index 1 (exec_time): **Minimize** (True)

### Metadata Requirements
```python
metadata['peak_memory_bytes']  # Peak GPU memory used during execution
```

### Tracker Behavior

When `SimulationTracker` encounters this objective:

1. **Threshold Reading**:
   - For 1st objective (memory): Uses `< kill_val` to mean "maximize" (lower values are bad)
   - For 2nd objective (time): Uses `> kill_val` (minimize)

2. **Kill Decision**:
   - Kills only if BOTH conditions met:
     - `peak_memory < best_memory / 1.5` (under-utilizing memory) AND
     - `exec_time > best_time × 1.5` (too slow)

3. **Threshold Update**:
   - 1st objective: Update if `new_memory > old_threshold` (maximize)
   - 2nd objective: Update if `new_time < old_threshold` (minimize)

---

## Tracker Integration

### Key Enhancement: Mixed Min/Max Support

The `SimulationTracker` class in `core/simulation_tracker.py` now correctly handles objectives with different optimization directions per component.

#### New Methods/Features

```python
# Check objective-specific direction
directions = self._objective_directions(n_objectives)
# Returns: [True, False] = [minimize_obj0, maximize_obj1]

# Safe Pareto-aware multi-objective pruning
for score_val, kill_val, is_min in zip(estimated_score, kill_threshold_vector, directions):
    if is_min:
        should_kill &= (score_val > kill_val)  # Minimize: kill if too high
    else:
        should_kill &= (score_val < kill_val)  # Maximize: kill if too low
```

#### Objective Integration Points

1. **Direction Discovery** (line ~395):
   ```python
   directions = self._objective_directions(len(score_vector))
   # Reads objective.objective_directions if available
   ```

2. **Threshold Update** (line ~215):
   ```python
   for idx, (value, old_value) in enumerate(zip(score_vector, threshold_vector)):
       if directions[idx]:  # If minimize
           if value < old_value:
               threshold_vector[idx] = value
       else:  # If maximize
           if value > old_value:
               threshold_vector[idx] = value
   ```

3. **Kill Logic** (line ~365):
   ```python
   should_kill = all(
       (score_val > kill_val if is_min else score_val < kill_val)
       for score_val, kill_val, is_min in zip(...)
   )
   ```

---

## Verification Checklist

- [x] New objectives added to `objective.py`
- [x] Classes inherit from `ObjectiveFunction`
- [x] `objective_directions` property set correctly
- [x] Both objectives handle OOM case (return penalty values)
- [x] Metadata extraction is safe (uses `.get()` with defaults)
- [x] Log transforms prevent division by zero (`max(1.0, value)`)
- [x] Registered in `create_objective()` registry
- [x] Added to `get_available_objective_types()` list
- [x] Syntax validated with `python3 -m py_compile`
- [x] 16 new experiments created (4 models × 4 new objectives, minus original 12)
- [x] Tracker correctly interprets mixed directions
- [x] SLURM launcher handles all 28 experiments

---

## Usage Examples

### Example 1: Using New Objective Directly
```python
from Optimization import create_objective

# Create cluster size optimization objective
objective = create_objective('time_and_throughput_per_energy')
print(objective.objective_directions)  # [True, False]

# Simulate a run result
score = objective.compute(
    exec_time=1e10,  # 10 billion nanoseconds
    is_oom=False,
    metadata={
        'total_energy_J': 500,
        'peak_memory_bytes': 8 * 1024**3,  # 8 GB
    },
    config={'batch_size': 32}
)
# Returns: (float, float) tuple of log10 values
```

### Example 2: Using in Optimizer
```python
from Optimization import (
    create_search_space,
    RandomSampler,
    SimulationRunner,
    DeepHyperOptimizer,
    create_objective,
)

objective = create_objective('memory_and_time')

optimizer = DeepHyperOptimizer(
    search_space=search_space,
    sampler=RandomSampler(),
    objective=objective,
    enable_tracker=True,
    tracker_kill_multiplier=1.5,
    # ... other args
)
```

### Example 3: Command-Line Usage (via sweep script)
```bash
python example_deephyper_opt_sweep.py \
  --objective time_and_throughput_per_energy \
  --model-num 10 \
  --num-npus 1024 \
  --budget 100 \
  --n-workers 8
```

---

## Performance Impact

### Tracking Overhead
- Mixed direction objectives add minimal overhead
- Direction check happens only at threshold update/kill decision (not per evaluation)
- Multi-objective Pareto pruning uses `all()` short-circuit (faster than `any()`)

### Memory Impact
- New objectives don't increase memory footprint (+~200 LOC)
- Tracker state file remains single JSON file (size unchanged)

---

## Future Extensions

Possible enhancements:

1. **Weighted aggregation** for > 2 objectives
   - Combine objectives with user-specified weights
   - Allow post-processing of Pareto front

2. **Dynamic threshold scaling** based on convergence rate
   - Adapt kill multiplier over time
   - More aggressive pruning early, conservative later

3. **Constraint handling** for bounded objectives
   - "Latency must be < 1000ms for any memory choice"
   - Feasibility-aware Pareto dominance

4. **Reference point-based** MOO (e.g., hypervolume)
   - Currently uses scalarization; could add indicator-based methods
   - Better handling of many-objective problems (> 4 objectives)

---

## Troubleshooting

### Q: "KeyError: 'total_energy_J'" when running on non-power-estimation simulations
**A**: The `time_and_throughput_per_energy` objective requires power modeling. Ensure:
```python
net_sim_config = {
    'estimate_power': 1,
    'power_config_path': '/path/to/a100_config.json'
}
```

### Q: "log(0)" error or NaN in results
**A**: All log transforms use `max(1.0, value)` protection. If still seeing NaN:
- Check metadata contains required keys
- Verify exec_time > 0 and energy > 0
- Look for `is_oom=True` cases (should return penalty)

### Q: Tracker says "Threshold: [inf, 0]" for new objective
**A**: This indicates the first successful run hasn't completed. Once first eval finishes:
- Threshold updates to actual scores
- Kill logic activates with proper multiplier

### Q: "Directions mismatch" warning in tracker logs
**A**: Objective returned wrong number of values. Check:
- Should return 2-tuple for these objectives
- Not mixing scalar returns with tuple returns in same run

---

## References

- **Objective Implementation**: `/scratch/nas/4/nasser/astra-sim/upc/Optimization/core/objective.py` (lines ~880-970)
- **Tracker Integration**: `/scratch/nas/4/nasser/astra-sim/upc/Optimization/core/simulation_tracker.py` (lines ~394-425)
- **Example Usage**: `/scratch/nas/4/nasser/astra-sim/upc/Optimization/examples/example_deephyper_opt_sweep.py`
- **Experiments**: `/scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite_last_hope/experiments/`

---

**Date**: 2026-03-27  
**Status**: Implemented and Validated ✓
