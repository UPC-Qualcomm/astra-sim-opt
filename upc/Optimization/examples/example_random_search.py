#!/usr/bin/env python3
"""
Example: Random Search Optimization

Demonstrates how to use the random search optimizer.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search_space import SearchSpace
from core.sampler import RandomSampler
from core.simulation_runner import SimulationRunner
from optimizers.random_optimizer import RandomOptimizer


def main():
    """Run random search example."""
    
    # Configuration
    MODEL_NUM = 19  # GPT_40B (Model enum value)
    MODEL_NAME = "GPT_40B"
    NUM_NPUS = 128
    NETWORK_NAME = "FoldedClos"
    BUDGET = 20
    
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
        "parallelism_strategy_params.json"  # Note: Using actual filename with typo
    )
    search_space = SearchSpace(search_space_path, num_npus=NUM_NPUS)
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
        folder_prefix="EXAMPLE_RS",
        verbose=False
    )
    print(f"   Using: {sim_runner}")
    
    # 4. Create optimizer
    print("\n4. Creating random search optimizer...")
    optimizer = RandomOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=BUDGET,
        verbose=True
    )
    print(f"   Using: {optimizer}")
    
    # 5. Run optimization
    print("\n" + "="*70)
    print("STARTING OPTIMIZATION")
    print("="*70)
    
    best_config, history = optimizer.run()
    
    # 6. Display results
    if best_config:
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        dp, mp, sp, pp, sharded = best_config
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   dp={dp}, mp={mp}, sp={sp}, pp={pp}, sharded={sharded}")
        print(f"   Execution time: {optimizer.best_score:.2f}s")
        print(f"\n📊 History saved with {len(history)} evaluations")
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()
