#!/usr/bin/env python3
"""
Example: Basic Bayesian Optimization

Demonstrates how to use the modular optimization framework for Bayesian Optimization.
"""

import sys
import os

# Add grandparent directory to path to find Optimization package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Optimization import (
    create_search_space,
    RandomSampler,
    SimulationRunner,
    MaternKernel,
    ExpectedImprovement,
    BayesianOptimizer
)


def main():
    """Run basic Bayesian Optimization example."""
    
    # Configuration
    MODEL_NUM = 19 # GPT_70B (Model enum value)
    MODEL_NAME = "GPT_40B"
    NUM_NPUS = 256
    NETWORK_NAME = "FoldedClos"
    BUDGET = 30
    INIT_SAMPLES = 5
    
    print("="*70)
    print("EXAMPLE: Bayesian Optimization")
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
        "parallelism_strategy_params.json" 
    )
    # Note: num_npus is read from the JSON file (npu_count field)
    search_space = create_search_space(
        search_space_path,
        include_categories=['parallelism_strategy']
    )
    print(f"   Design space size: {search_space.get_design_space_size()}")
    
    # 2. Choose sampler
    print("\n2. Creating sampler...")
    sampler = RandomSampler(seed=42)
    print(f"   Using: {sampler}")
    
    # 3. Setup simulation runner
    print("\n3. Creating simulation runner...")
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        num_npus=NUM_NPUS,
        network_name=NETWORK_NAME,
        folder_prefix="EXAMPLE_BO",
        verbose=True  # Enable verbose to see detailed error messages
    )
    print(f"   Using: {sim_runner}")
    
    # 4. Choose kernel and acquisition
    print("\n4. Creating GP kernel and acquisition function...")
    kernel = MaternKernel(nu=2.5, length_scale=1.0)
    acquisition = ExpectedImprovement(xi=0.01)
    print(f"   Kernel: {kernel}")
    print(f"   Acquisition: {acquisition}")
    
    # 5. Create optimizer
    print("\n5. Creating Bayesian optimizer...")
    optimizer = BayesianOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        kernel=kernel,
        acquisition=acquisition,
        budget=BUDGET,
        init_samples=INIT_SAMPLES,
        verbose=True,
        keep_top_k=5,
        n_workers = 6,
        batch_size= 12

    )
    print(f"   Using: {optimizer}")
    
    # 6. Run optimization
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION")
    print("="*70)
    
    best_config, history = optimizer.run()
    
    # 7. Display results
    if best_config:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        
        # Format configuration dynamically
        config_str = ", ".join([f"{k}={v}" for k, v in best_config.items()])
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   {config_str}")
        print(f"   Execution time: {optimizer.best_score:.2f}s")
        print(f"\n📊 History saved with {len(history)} evaluations")
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()
