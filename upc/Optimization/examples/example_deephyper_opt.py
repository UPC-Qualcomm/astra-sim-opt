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
    create_objective
)


def main():
    """Run DeepHyper Bayesian Optimization example."""
    
    # Configuration
    MODEL_NUM = 19 # GPT_40B (Model enum value)
    MODEL_NAME = "GPT_40B_g2_2500"
    NUM_NPUS = 64
    NETWORK_NAME = "FoldedClos"
    BUDGET = 1000
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
        include_categories=['parallelism_strategy', 'system', 'network', 'collective']
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
        num_npus=search_space.num_npus,
        network_name=NETWORK_NAME,
        folder_prefix="EXAMPLE_DEEPHYPER",
        verbose=True,
        #net_sim_config=net_sim_config 
    )
    print(f"   Using: {sim_runner}")
    
    # 4. Create objective function
    print("\n4. Creating objective function...")
    
    objective = create_objective('time')
    
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
        acq_func="UCB",  # Acquisition function: "UCB", "EI", "PI", "gp_hedge"
        acq_optimizer="auto",  # "sampling", "lbfgs", "auto"
        filter_duplicates=True,
        random_state=42,
        verbose=True,
        keep_top_k=20,
        profile_time=True,
        evaluator_method="process"  # "process" or "thread"
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
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()
