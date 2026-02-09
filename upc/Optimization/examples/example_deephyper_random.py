#!/usr/bin/env python3
"""
Example: DeepHyper Random Search

Demonstrates how to use DeepHyper's built-in RandomSearch for baseline comparison.
Random search is useful for:
- Establishing performance baselines
- Validating that BO provides improvement over random sampling
- Quick exploration of design space
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
from custom_objectives import obj_latency_network


def main():
    """Run DeepHyper Random Search example."""
    
    # Configuration
    MODEL_NUM = 19  # GPT_40B (Model enum value)
    MODEL_NAME = "GPT_40B_g2_sync_obj_time_network_56_layer_50_50_random"
    NUM_NPUS = 64
    NETWORK_NAME = "FoldedClos"
    BUDGET = 300
    N_WORKERS = 8
    Objective_0_Name = "Execution Time (s)"
    Objective_1_Name = "Network Total BW (GB/s)"
    
    print("="*70)
    print("EXAMPLE: DeepHyper Random Search")
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
        folder_prefix="EXAMPLE_DEEPHYPER_RANDOM",
        verbose=True,
        net_sim_config=net_sim_config,
    )
    print(f"   Using: {sim_runner}")
    
    # 4. Create objective function
    print("\n4. Creating objective function...")
    
    objective = create_objective(
        objective_type='time_and_network_bw'
    )
    objective = CustomObjective(
        obj_latency_network,
        "MOO_time_network_total_bw",
        minimize=True,
        is_multi_objective=True
    )
    print(f"   Using: {objective.name}")

    # 5. Create DeepHyper Random Search optimizer
    print("\n5. Creating DeepHyper Random Search optimizer...")
    optimizer = DeepHyperOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=BUDGET,
        objective=objective,
        init_samples=0,  # No initialization phase for random search
        n_workers=N_WORKERS,
        search_type="random",  # Use RandomSearch instead of CBO
        random_state=42,
        verbose=True,
        keep_top_k=20,
        profile_time=True,
        evaluator_method="process",
        moo_scalarization_strategy="AugChebyshev",
        moo_scalarization_weight=[0.5, 0.5]
    )
    print(f"   Using: {optimizer}")
    print(f"   Note: Random search does not use surrogate models or acquisition functions")
    
    # 6. Run optimization
    print("\n" + "="*70)
    print("STARTING RANDOM SEARCH")
    print("="*70)
    
    best_config, history = optimizer.run()
    
    # 7. Display results
    if best_config is not None:
        print("\n" + "="*70)
        print("RANDOM SEARCH COMPLETE")
        print("="*70)
        
        # Format configuration dynamically
        config_str = ", ".join([f"{k}={v}" for k, v in best_config.items()])
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   {config_str}")
        print(f"   Score: {format_score(optimizer.best_score)}")
        print(f"\n📊 History saved with {len(history)} evaluations")
        print(f"\n💡 TIP: Check deephyper_results.csv for detailed DeepHyper output")
        
        # 8. Generate visualization plots
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        
        # Plot Pareto front using external script
        print("\n1. Plotting Pareto front...")
        try:
            from plot_pareto_front import plot_pareto_front
            
            # Get the CSV file path
            csv_path = os.path.join(optimizer.save_dir, optimizer.results_filename)
            model_name = getattr(optimizer.simulation_runner, 'model_name', 'model')
            output_base = os.path.join(optimizer.save_dir, f"pareto_front_{model_name}")
            
            # Generate both HTML and PNG plots
            pareto_plots = plot_pareto_front(
                results_file=csv_path,
                obj0_name=Objective_0_Name,
                obj1_name=Objective_1_Name,
                output_file=output_base,
                plot_format="both",
                show_labels=True,
                remove_outliers=False  # No outlier removal for random search baseline
            )
        except Exception as e:
            print(f"⚠️  Error plotting Pareto front: {e}")
            import traceback
            traceback.print_exc()
            pareto_plots = None
        
        # Plot hypervolume indicator
        print("\n2. Plotting hypervolume indicator...")
        hv_path, hvi = optimizer.plot_hypervolume()
        
        if pareto_plots or hv_path:
            print("\n" + "="*70)
            print("VISUALIZATION COMPLETE")
            print("="*70)
            print("\n📈 Generated plots:")
            if pareto_plots:
                for plot_path in pareto_plots:
                    if plot_path.endswith('.html'):
                        print(f"   - Pareto Front (Interactive): {plot_path}")
                    elif plot_path.endswith('.png'):
                        print(f"   - Pareto Front (Static): {plot_path}")
            if hv_path:
                print(f"   - Hypervolume: {hv_path}")
                print(f"   - Final HVI: {hvi:.4f}")
    else:
        print("\n❌ Random search failed")


if __name__ == "__main__":
    main()
