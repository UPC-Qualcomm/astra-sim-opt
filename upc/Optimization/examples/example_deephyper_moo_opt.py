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
    CustomObjective,
    create_objective
)


def obj_latency_network(exec_time, is_oom, metadata, config):
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

def obj_latency_memory(exec_time, is_oom, metadata, config):
    """
    Multi-objective function optimizing both execution time and network bandwidth.
    
    Objective 0: Execution time (minimize) - Lower is better
    Objective 1: Total memory usage (minimize) - Lower is better
    Returns:
        tuple: (exec_time, total_memory_usage) for minimization
    """
    npu_count = config.get('npu_count', 1)
    local_mem_size = config.get('local-mem-size', 0)  # GB/s
    
    
    total_memory_usage = local_mem_size * npu_count  # Total memory usage across all NPUs
    
    # Return tuple for multi-objective (both minimization)
    return exec_time, total_memory_usage

def main():
    """Run DeepHyper Bayesian Optimization example."""
    
    # Configuration
    MODEL_NUM = 19 # GPT_40B (Model enum value)
    MODEL_NAME = "GPT_40B_moo_mem_exec"
    NUM_NPUS = 64
    NETWORK_NAME = "FoldedClos"
    BUDGET = 50
    INIT_SAMPLES = 20
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
        "parallelism_strategy_params.json" 
    )
    search_space = create_search_space(
        search_space_path,
        include_categories=['parallelism_strategy', 'network', 'system']
    )
    print(f"   Design space size: {search_space.get_design_space_size()}")
    
    # 2. Choose sampler (for fallback if needed)
    print("\n2. Creating sampler...")
    sampler = RandomSampler(seed=42)
    print(f"   Using: {sampler}")
    
    #net_sim_config = {
    #    'sim_type': 'g2',
    #    'topology': 'FoldedClos',
    #    'paths_mode': 'ECMP',
    #    'topology_config': {
    #        'num_npus': search_space.num_npus,
    #        'npus_per_node': 8,
    #        'intra_node_topology': 'fully_connected',
    #        'bandwidth_config': {
    #            'host_edge': 100,
    #            'edge_agg': 100,
    #            'agg_core': 100,
    #            'intra_node': 450
    #        },
    #        'bw_unit': 'GB/s'
    #    }
    #}

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
        acq_func="UCBd",
        acq_func_kwargs={"kappa": 10.0, "scheduler": {"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.01}},
        surrogate_model="ET",
        surrogate_model_kwargs={"max_features": "sqrt"},
        acq_optimizer="mixedga",
        random_state=42,
        verbose=True,
        keep_top_k=20,
        profile_time=True,
        evaluator_method="process",
        acq_optimizer_kwargs={"max_total_failures": -1, "acq_optimizer_freq": 2},
        moo_scalarization_strategy="AugChebyshev",
        moo_scalarization_weight="random",
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
        print(f"   Execution time: {optimizer.best_score:.2f}s")
        print(f"\n📊 History saved with {len(history)} evaluations")
        print(f"\n💡 TIP: Check deephyper_results.csv for detailed DeepHyper output")
        
        # 8. Generate visualization plots
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        # Plot Pareto front
        print("\n1. Plotting Pareto front...")
        pareto_path = optimizer.plot_results(objective_names=("Execution Time (s)", "Total Memory (GB)"))
        
        # Plot hypervolume indicator
        print("\n2. Plotting hypervolume indicator...")
        hv_path, hvi = optimizer.plot_hypervolume()
        
        if pareto_path or hv_path:
            print("\n" + "="*70)
            print("VISUALIZATION COMPLETE")
            print("="*70)
            print("\n📈 Generated plots:")
            if pareto_path:
                print(f"   - Pareto Front: {pareto_path}")
            if hv_path:
                print(f"   - Hypervolume: {hv_path}")
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()
