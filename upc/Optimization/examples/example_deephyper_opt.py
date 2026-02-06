#!/usr/bin/env python3
"""
Example: DeepHyper Bayesian Optimization

Demonstrates how to use the DeepHyper-based optimizer within the modular 
optimization framework. DeepHyper provides a mature, well-tested BO implementation
with advanced features and efficient parallel evaluation.
"""

import sys
import os

# Add grandparent directory to path to find Optimization package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Optimization import (
    create_search_space,
    RandomSampler,
    SimulationRunner,
    DeepHyperOptimizer,
    create_objective,
    CustomObjective
)
from Optimization.core.base_optimizer import format_score

PENALTY = 10_000_000_000

def obj_latency_network(exec_time, is_oom, metadata, config):
    """
    Multi-objective function optimizing both execution time and network bandwidth.
    
    Objective 0: Execution time (minimize) - Lower is better
    Objective 1: Total network bandwidth (minimize) - Lower is better
    
    Normalization: Network bandwidth is divided by 1000 to bring it closer to exec_time scale
    
    Returns:
        tuple: (exec_time, normalized_network_bw) for minimization
    """
    npu_count = config.get('npu_count', 1)
    intra_node_bw = config.get('intra-node-bw', 0)  # GB/s
    inter_node_bw = config.get('inter-node-bw', 0)  # GB/s
    npus_per_node = 8  
    
    num_nodes = max(1, (npu_count + npus_per_node - 1) // npus_per_node)
    
    # Calculate total network bandwidth
    if num_nodes == 1:
        # Single node: only intra-node bandwidth matters
        total_network_bw = intra_node_bw #* (npu_count - 1)  # Connections between NPUs
    else:
        # Multiple nodes: both intra-node and inter-node bandwidth
        intra_bw_total = intra_node_bw# * npus_per_node * num_nodes  # Intra-node links
        inter_bw_total = inter_node_bw# * (num_nodes - 1)  # Inter-node links
        total_network_bw = intra_bw_total + inter_bw_total
    
    # Normalize network bandwidth to similar scale as exec_time
    # Typical total_network_bw: 1000-30000 GB/s, exec_time: 1-1000s
    normalized_network_bw = total_network_bw #/ 100.0
    
    # Return tuple for multi-objective (both minimization)
    if is_oom:
        normalized_network_bw = PENALTY  # Penalize OOM configurations
        exec_time = PENALTY  # Penalize OOM configurations
    return exec_time, normalized_network_bw

def obj_latency_memory(exec_time, is_oom, metadata, config):
    """
    Multi-objective function optimizing both execution time and memory usage.
    
    Objective 0: Execution time (minimize) - Lower is better
    Objective 1: Total memory usage (minimize) - Lower is better
    
    Normalization: Memory usage is divided by 100 to bring it closer to exec_time scale
    
    Returns:
        tuple: (exec_time, normalized_memory_usage) for minimization
    """
    npu_count = config.get('npu_count', 1)
    local_mem_size = config.get('local-mem-size', 0)  # GB
    
    total_memory_usage = local_mem_size * npu_count  # Total memory usage across all NPUs
    
    # Normalize memory usage to similar scale as exec_time
    # Typical total_memory_usage: 100-10000 GB, exec_time: 1-1000s
    normalized_memory_usage = total_memory_usage / 100.0
    
    # Return tuple for multi-objective (both minimization)
    if is_oom:
        normalized_memory_usage = PENALTY  # Penalize OOM configurations
        exec_time = PENALTY  # Penalize OOM configurations
    return exec_time, normalized_memory_usage

def main():
    """Run DeepHyper Bayesian Optimization example."""
    
    # Configuration
    MODEL_NUM = 19 # GPT_40B (Model enum value)
    MODEL_NAME = "GPT_40B_analytical_sync_obj_latency_network_test"
    NUM_NPUS = 64
    NETWORK_NAME = "FoldedClos"
    BUDGET = 1310
    INIT_SAMPLES = 80
    N_WORKERS = 8
    
    

    print("="*70)
    print("EXAMPLE: DeepHyper Bayesian Optimization")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"NPUs: {NUM_NPUS}")
    print(f"Network: {NETWORK_NAME}")
    print(f"Budget: {BUDGET} evaluations")
    print(f"Workers: {N_WORKERS} (parallel evaluation)\n")
    
    # 1. Setup search space
    print("1. Creating search space...")
    search_space_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "search_space", 
        "parallelism_strategy_params_g2_intra.json" 
    )
    search_space = create_search_space(
        search_space_path,
        include_categories=['parallelism_strategy', 'network']
    )
    print(f"   Design space size: {search_space.get_design_space_size()}")
    
    # 2. Choose sampler (for fallback if needed)
    print("\n2. Creating sampler...")
    sampler = RandomSampler(seed=42)
    print(f"   Using: {sampler}")
    
    net_sim_config = {
        'sim_type': 'g2',
        'topology': 'FoldedClos',
        'paths_mode': 'Uniform',
        'topology_config': {
            'num_npus': search_space.num_npus,
            'npus_per_node': 8,
            'intra_node_topology': 'fully_connected',
            'bandwidth_config': {
                'host_edge': 100,
                'edge_agg': 100,
                'agg_core': 100,
                'intra_node': 450
            },
            'bw_unit': 'GB/s'
        }
    }

    # 3. Setup simulation runner
    print("\n3. Creating simulation runner...")
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        network_name=NETWORK_NAME,
        folder_prefix="EXAMPLE_DEEPHYPER",
        verbose=True,
        #net_sim_config=net_sim_config 
    )
    print(f"   Using: {sim_runner}")
    
    # 4. Create objective function
    print("\n4. Creating objective function...")
    
    objective = create_objective(
        objective_type='time'
    )
    objective = CustomObjective(
        obj_latency_network, 
        "MOO_time_network",
        minimize=True,
        is_multi_objective=True
    )
    print(f"   Using: {objective.name}")

    # 5. Create DeepHyper optimizer
    print("\n5. Creating DeepHyper optimizer...")
    optimizer = DeepHyperOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=BUDGET,
        objective=objective,
        init_samples=INIT_SAMPLES,
        n_workers=N_WORKERS,
        acq_func="UCB",
        surrogate_model="ET",
        acq_optimizer="mixedga",
        random_state=42,
        verbose=True,
        keep_top_k=20,
        profile_time=True,
        evaluator_method="process",
        acq_optimizer_kwargs={"max_total_failures": -1}
    )
    print(f"   Using: {optimizer}")
    
    # 6. Run optimization
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION")
    print("="*70)
    
    best_config, history = optimizer.run()
    
    # 7. Display results
    if best_config is not None:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        
        # Format configuration dynamically
        config_str = ", ".join([f"{k}={v}" for k, v in best_config.items()])
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   {config_str}")
        print(f"   Score: {format_score(optimizer.best_score)}")
        print(f"\n📊 History saved with {len(history)} evaluations")
        print(f"\n💡 TIP: Check deephyper_results.csv for detailed DeepHyper output")
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()
