#!/usr/bin/env python3
"""
Custom Objective Functions for Multi-Objective Optimization

This module provides reusable objective functions for optimizing different
combinations of metrics in ASTRA-sim experiments.

Multiple normalization strategies are provided:
- Log normalization: Reduces scale differences but sensitive to small changes
- Min-max normalization: Linear scaling to [0, 1] range
- Square root: Less sensitive than log, good for power-law distributions
- Raw values: Let DeepHyper's built-in scalers handle normalization
"""

import math
import numpy as np


def obj_latency_total_network(exec_time, is_oom, metadata, config):
    """
    Multi-objective function optimizing both execution time and network bandwidth.
    
    Objective 0: Execution time (minimize) - Lower is better
    Objective 1: Total network bandwidth (minimize) - Lower is better
    
    Returns:
        tuple: (exec_time, total_network_bw) for minimization
    """
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)  # GB/s
    inter_node_bw = config.get('inter-node-bw', 0)  # GB/s
    npus_per_node = 8  
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    # Calculate total network bandwidth
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw * (npu_count - 1)  # Connections between NPUs
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    # Return tuple for multi-objective (both minimization)
    return exec_time, total_network_bw

def obj_latency_network(exec_time, is_oom, metadata, config):
    """
    Multi-objective function optimizing both execution time and network bandwidth.
    
    Uses log transformation for threshold-free normalization. This ensures both
    objectives have equal importance regardless of their absolute scale.
    
    Args:
        exec_time: Execution time in nanoseconds (typically 1e9 to 1e12)
        is_oom: Whether the configuration ran out of memory
        metadata: Additional simulation metadata
        config: Configuration dictionary with network parameters
    
    Returns:
        tuple: (log_exec_time, log_network_bw) for minimization
    
    Example:
        >>> obj_latency_network(1e10, False, {}, {'npu_count': 64, 'intra-node-bw': 450})
        (10.0, 2.653...)
    """
    # Handle OOM cases first
    if is_oom:
        return float('inf'), float('inf')
    
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)  # GB/s
    inter_node_bw = config.get('inter-node-bw', 0)  # GB/s
    npus_per_node = 8  
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    # Calculate total network bandwidth
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw * (npu_count - 1)
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    # Log transformation - no thresholds needed!
    # exec_time (1e9 to 1e12 ns) -> log10 ~ 9-12
    # network_bw (300 to 100000 GB/s) -> log10 ~ 2.5-5
    log_exec_time = math.log10(max(1, exec_time))
    log_network_bw = math.log10(max(1, total_network_bw))
    
    return exec_time, total_network_bw


def obj_latency_memory(exec_time, is_oom, metadata, config):
    """
    Multi-objective function optimizing both execution time and memory usage.
    
    Uses log transformation for threshold-free normalization. This ensures both
    objectives have equal importance regardless of their absolute scale.
    
    Args:
        exec_time: Execution time in nanoseconds (typically 1e9 to 1e12)
        is_oom: Whether the configuration ran out of memory
        metadata: Additional simulation metadata
        config: Configuration dictionary with memory parameters
    
    Returns:
        tuple: (log_exec_time, log_memory_usage) for minimization
    
    Example:
        >>> obj_latency_memory(1e10, False, {}, {'npu_count': 64, 'local-mem-size': 32})
        (10.0, 3.309...)
    """
    # Handle OOM cases first
    if is_oom:
        return float('inf'), float('inf')
    
    npu_count = config.get('npu_count', 1)
    local_mem_size = config.get('local-mem-size', 0)  # GB
    
    # Calculate total memory usage across all NPUs
    total_memory_usage = local_mem_size * npu_count
    
    # Log transformation - no thresholds needed!
    # exec_time (1e9 to 1e12 ns) -> log10 ~ 9-12
    # memory_usage (100 to 10000 GB) -> log10 ~ 2-4
    log_exec_time = math.log10(max(1, exec_time))
    log_memory_usage = math.log10(max(1, total_memory_usage))
    
    return log_exec_time, log_memory_usage


def obj_network_memory(exec_time, is_oom, metadata, config):
    """
    Multi-objective function optimizing both network bandwidth and memory usage.
    
    Uses log transformation for threshold-free normalization. This optimizes
    resource usage without considering execution time.
    
    Args:
        exec_time: Execution time (not used in this objective)
        is_oom: Whether the configuration ran out of memory
        metadata: Additional simulation metadata
        config: Configuration dictionary with network and memory parameters
    
    Returns:
        tuple: (log_network_bw, log_memory_usage) for minimization
    """
    # Handle OOM cases first
    if is_oom:
        return float('inf'), float('inf')
    
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)  # GB/s
    inter_node_bw = config.get('inter-node-bw', 0)  # GB/s
    local_mem_size = config.get('local-mem-size', 0)  # GB
    npus_per_node = 8
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    # Calculate total network bandwidth
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw * (npu_count - 1)
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    # Calculate total memory usage
    total_memory_usage = local_mem_size * npu_count
    
    # Log transformation
    log_network_bw = math.log10(max(1, total_network_bw))
    log_memory_usage = math.log10(max(1, total_memory_usage))
    
    return log_network_bw, log_memory_usage


def obj_latency_network_memory(exec_time, is_oom, metadata, config):
    """
    Three-objective function optimizing execution time, network bandwidth, and memory.
    
    Uses log transformation for threshold-free normalization across all three objectives.
    
    Args:
        exec_time: Execution time in nanoseconds
        is_oom: Whether the configuration ran out of memory
        metadata: Additional simulation metadata
        config: Configuration dictionary with all parameters
    
    Returns:
        tuple: (log_exec_time, log_network_bw, log_memory_usage) for minimization
    """
    # Handle OOM cases first
    if is_oom:
        return float('inf'), float('inf'), float('inf')
    
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)
    inter_node_bw = config.get('inter-node-bw', 0)
    local_mem_size = config.get('local-mem-size', 0)
    npus_per_node = 8
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    # Calculate metrics
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw * (npu_count - 1)
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    total_memory_usage = local_mem_size * npu_count
    
    # Log transformation
    log_exec_time = math.log10(max(1, exec_time))
    log_network_bw = math.log10(max(1, total_network_bw))
    log_memory_usage = math.log10(max(1, total_memory_usage))
    
    return log_exec_time, log_network_bw, log_memory_usage


# ============================================================================
# ALTERNATIVE NORMALIZATION STRATEGIES (Less Sensitive)
# ============================================================================

def obj_latency_network_raw(exec_time, is_oom, metadata, config):
    """
    Multi-objective: execution time and network bandwidth (RAW VALUES).
    
    Returns raw values - let DeepHyper's objective_scaler handle normalization.
    This is the RECOMMENDED approach as DeepHyper can apply "minmax" or 
    "standardize" scaling across all observed values.
    
    Args:
        exec_time: Execution time in nanoseconds
        is_oom: Whether the configuration ran out of memory
        metadata: Additional simulation metadata
        config: Configuration dictionary
    
    Returns:
        tuple: (exec_time_seconds, total_network_bw_gbps)
    """
    if is_oom:
        return float('inf'), float('inf')
    
    # Convert to more reasonable units
    exec_time_seconds = exec_time / 1e9  # nanoseconds -> seconds
    
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)
    inter_node_bw = config.get('inter-node-bw', 0)
    npus_per_node = 8
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw * (npu_count - 1)
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    # Return raw values - DeepHyper will scale them
    return exec_time_seconds, total_network_bw


def obj_latency_network_minmax(exec_time, is_oom, metadata, config):
    """
    Multi-objective with MIN-MAX normalization to [0, 1] range.
    
    LESS SENSITIVE than log - linear scaling based on expected ranges.
    Good when you know approximate min/max values beforehand.
    
    Args:
        exec_time: Execution time in nanoseconds
        is_oom: Whether the configuration ran out of memory
        metadata: Additional simulation metadata
        config: Configuration dictionary
    
    Returns:
        tuple: (normalized_time, normalized_network) in [0, 1]
    """
    if is_oom:
        return float('inf'), float('inf')
    
    exec_time_seconds = exec_time / 1e9
    
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)
    inter_node_bw = config.get('inter-node-bw', 0)
    npus_per_node = 8
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw * (npu_count - 1)
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    # Define expected ranges (adjust based on your problem)
    TIME_MIN, TIME_MAX = 1.0, 100.0  # seconds
    BW_MIN, BW_MAX = 100, 2000  # GB/s
    
    # Min-max normalization to [0, 1]
    norm_time = (exec_time_seconds - TIME_MIN) / (TIME_MAX - TIME_MIN)
    norm_network = (total_network_bw - BW_MIN) / (BW_MAX - BW_MIN)
    
    # Clip to [0, 1] to handle outliers
    norm_time = np.clip(norm_time, 0, 1)
    norm_network = np.clip(norm_network, 0, 1)
    
    return norm_time, norm_network


def obj_latency_network_sqrt(exec_time, is_oom, metadata, config):
    """
    Multi-objective with SQUARE ROOT scaling.
    
    LESS SENSITIVE than log - compresses range but with gentler curve.
    Good middle ground between raw values and log scaling.
    
    Args:
        exec_time: Execution time in nanoseconds
        is_oom: Whether the configuration ran out of memory
        metadata: Additional simulation metadata
        config: Configuration dictionary
    
    Returns:
        tuple: (sqrt_time, sqrt_network)
    """
    if is_oom:
        return float('inf'), float('inf')
    
    exec_time_seconds = exec_time / 1e9
    
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)
    inter_node_bw = config.get('inter-node-bw', 0)
    npus_per_node = 8
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw * (npu_count - 1)
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    # Square root scaling - smoother than log
    sqrt_time = np.sqrt(max(0, exec_time_seconds))
    sqrt_network = np.sqrt(max(0, total_network_bw))
    
    return sqrt_time, sqrt_network


def obj_latency_network_power(exec_time, is_oom, metadata, config, power=0.5):
    """
    Multi-objective with POWER transformation (configurable exponent).
    
    Allows fine-tuning sensitivity: 
    - power < 1: Compresses range (like sqrt when power=0.5)
    - power = 1: Linear (no transformation)
    - power > 1: Expands range
    
    Args:
        exec_time: Execution time in nanoseconds
        is_oom: Whether the configuration ran out of memory
        metadata: Additional simulation metadata
        config: Configuration dictionary
        power: Exponent for transformation (default: 0.5 = square root)
    
    Returns:
        tuple: (time^power, network^power)
    """
    if is_oom:
        return float('inf'), float('inf')
    
    exec_time_seconds = exec_time / 1e9
    
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)
    inter_node_bw = config.get('inter-node-bw', 0)
    npus_per_node = 8
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw * (npu_count - 1)
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    # Power transformation
    power_time = np.power(max(0, exec_time_seconds), power)
    power_network = np.power(max(0, total_network_bw), power)
    
    return power_time, power_network


# ============================================================================
# USAGE RECOMMENDATIONS
# ============================================================================
"""
WHICH OBJECTIVE FUNCTION SHOULD YOU USE?

1. **obj_latency_network_raw** (RECOMMENDED)
   - Use with DeepHyper's objective_scaler="minmax" or "standardize"
   - DeepHyper automatically normalizes based on observed values
   - No manual tuning needed
   - Most robust approach

2. **obj_latency_network_sqrt** (Good Alternative)
   - Less sensitive than log
   - No need to know value ranges beforehand
   - Good for power-law distributions

3. **obj_latency_network_minmax**
   - When you know expected value ranges
   - Predictable, linear scaling
   - Adjust TIME_MIN/MAX and BW_MIN/MAX based on your problem

4. **obj_latency_network** (Original - Most Sensitive)
   - Use only if you need extreme compression
   - Very sensitive to small changes
   - Can cause optimization issues

EXAMPLE USAGE:
```python
from custom_objectives import obj_latency_network_raw

# Let DeepHyper handle scaling
optimizer = DeepHyperOptimizer(
    ...
    objective=CustomObjective(
        obj_latency_network_raw,
        "time_network_raw",
        minimize=True,
        is_multi_objective=True
    ),
    objective_scaler="minmax",  # DeepHyper handles normalization
    ...
)
```
"""
