#!/usr/bin/env python3
"""
Example: Random Search Optimization

Demonstrates how to use the random search optimizer.
"""

import sys
import os

# Add grandparent directory to path to find Optimization package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Optimization import (
    create_search_space,
    RandomSampler,
    SimulationRunner,
    RandomOptimizer,
    create_objective,
    CustomObjective
)
from Optimization.core.base_optimizer import format_score
from custom_objectives import obj_latency_network, obj_latency_memory


def main():
    """Run random search example."""
    
    # Configuration
    MODEL_NUM = 19  # GPT_70B (Model enum value)
    MODEL_NAME = "GPT_40B_analytical_sync_obj_latency_network_random_search"
    NUM_NPUS = 64
    NETWORK_NAME = "FoldedClos"
    BUDGET = 300
    N_WORKERS = 8
    
    print("="*70)
    print("EXAMPLE: Random Search Optimization")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"NPUs: {NUM_NPUS}")
    print(f"Network: {NETWORK_NAME}")
    print(f"Budget: {BUDGET} evaluations\n")
    
    # 1. Setup search space
    print("1. Creating search space...")
    search_space_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "search_space", 
        "parallelism_strategy_params_g2_intra.json"  # Note: Using actual filename with typo
    )
    # Note: num_npus is read from the JSON file (npu_count field)
    search_space = create_search_space(
        search_space_path,
        include_categories=['parallelism_strategy', 'network']
    )
    print(f"   Design space size: {search_space.get_design_space_size()}")
    
    # 2. Choose sampler
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
        folder_prefix="EXAMPLE_RS",
        verbose=True,
        #net_sim_config=net_sim_config 
    )
    print(f"   Using: {sim_runner}")
    
    # 4. Create objective function (MULTIPLICATIVE)
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
    # 5. Create optimizer
    print("\n5. Creating random search optimizer...")
    optimizer = RandomOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=BUDGET,
        objective=objective,  # Pass the objective
        verbose=True,
        keep_top_k=5,
        n_workers=N_WORKERS,
        profile_time=True  # Enable time profiling
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
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()
