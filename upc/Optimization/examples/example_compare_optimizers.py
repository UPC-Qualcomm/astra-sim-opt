#!/usr/bin/env python3
"""
Example: Comparing Multiple Optimizers

Demonstrates how to run and compare different optimizers on the same problem.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add grandparent directory to path to find Optimization package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Optimization import (
    create_search_space,
    RandomSampler,
    LatinHypercubeSampler,
    SimulationRunner,
    MaternKernel,
    RBFKernel,
    ExpectedImprovement,
    UpperConfidenceBound,
    RandomOptimizer,
    BayesianOptimizer
)


def run_optimizer(name, optimizer):
    """Run an optimizer and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}\n")
    
    best_config, history = optimizer.run()
    
    if best_config:
        print(f"\n✓ {name} completed")
        print(f"  Best score: {optimizer.best_score:.2f}s")
        return {
            'name': name,
            'best_config': best_config,
            'best_score': optimizer.best_score,
            'history': history
        }
    else:
        print(f"\n✗ {name} failed")
        return None


def compare_results(results):
    """Compare and visualize results from multiple optimizers."""
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    # Print summary table
    print("\n{:<25} {:<15} {:<20}".format("Optimizer", "Best Score", "Best Config"))
    print("-" * 70)
    
    for result in results:
        if result:
            name = result['name']
            score = result['best_score']
            config = result['best_config']
            
            # Format configuration dynamically
            config_str = ", ".join([f"{v}" for v in config.values()])
            config_str = f"({config_str})"
            
            print("{:<25} {:<15.2f} {:<20}".format(name, score, config_str))
    
    # Find best overall
    valid_results = [r for r in results if r is not None]
    if valid_results:
        best_overall = min(valid_results, key=lambda x: x['best_score'])
        print("\n" + "🏆 WINNER: " + best_overall['name'])
        print(f"   Score: {best_overall['best_score']:.2f}s")
        
        # Create convergence plot
        try:
            create_convergence_plot(valid_results)
        except Exception as e:
            print(f"\nNote: Could not create plot: {e}")


def create_convergence_plot(results):
    """Create convergence plot comparing optimizers."""
    plt.figure(figsize=(10, 6))
    
    for result in results:
        history = result['history']
        if not history.empty:
            # Plot best_so_far over iterations
            plt.plot(history['iteration'], history['best_so_far'], 
                    marker='o', label=result['name'], linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Execution Time (s)', fontsize=12)
    plt.title('Optimizer Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = 'optimizer_comparison.png'
    plt.savefig(output_file, dpi=300)
    print(f"\n📊 Convergence plot saved to: {output_file}")


def main():
    """Compare different optimizers."""
    
    # Configuration
    MODEL_NUM = 19  # GPT_40B (Model enum value)
    MODEL_NAME = "GPT_40B"
    NUM_NPUS = 64
    NETWORK_NAME = "FoldedClos"
    BUDGET = 25
    
    print("="*70)
    print("EXAMPLE: Comparing Multiple Optimizers")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"NPUs: {NUM_NPUS}")
    print(f"Network: {NETWORK_NAME}")
    print(f"Budget: {BUDGET} evaluations per optimizer\n")
    
    # Shared components
    search_space_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "search_space", 
        "parallelism_strategy_params.json"  # Note: Using actual filename with typo
    )
    # Note: num_npus is read from the JSON file (npu_count field)
    search_space = create_search_space(
        search_space_path,
        include_categories=['parallelism_strategy']
    )
    
    # Define optimizers to compare
    optimizers = []
    
    # 1. Random Search
    print("Setting up Random Search...")
    sim_runner_rs = SimulationRunner(
        MODEL_NUM, MODEL_NAME, NUM_NPUS, NETWORK_NAME,
        folder_prefix="COMPARE_RS", verbose=False
    )
    optimizers.append((
        "Random Search",
        RandomOptimizer(
            search_space=search_space,
            sampler=RandomSampler(seed=42),
            simulation_runner=sim_runner_rs,
            budget=BUDGET,
            verbose=False
        )
    ))
    
    # 2. Bayesian Opt with Matern + EI
    print("Setting up Bayesian (Matern + EI)...")
    sim_runner_bo1 = SimulationRunner(
        MODEL_NUM, MODEL_NAME, NUM_NPUS, NETWORK_NAME,
        folder_prefix="COMPARE_BO_MATERN_EI", verbose=False
    )
    optimizers.append((
        "Bayesian (Matern + EI)",
        BayesianOptimizer(
            search_space=search_space,
            sampler=LatinHypercubeSampler(seed=42),
            simulation_runner=sim_runner_bo1,
            kernel=MaternKernel(nu=2.5),
            acquisition=ExpectedImprovement(xi=0.01),
            budget=BUDGET,
            init_samples=5,
            verbose=False
        )
    ))
    
    # 3. Bayesian Opt with RBF + UCB
    print("Setting up Bayesian (RBF + UCB)...")
    sim_runner_bo2 = SimulationRunner(
        MODEL_NUM, MODEL_NAME, NUM_NPUS, NETWORK_NAME,
        folder_prefix="COMPARE_BO_RBF_UCB", verbose=False
    )
    optimizers.append((
        "Bayesian (RBF + UCB)",
        BayesianOptimizer(
            search_space=search_space,
            sampler=LatinHypercubeSampler(seed=42),
            simulation_runner=sim_runner_bo2,
            kernel=RBFKernel(length_scale=1.0),
            acquisition=UpperConfidenceBound(kappa=2.576),
            budget=BUDGET,
            init_samples=5,
            verbose=False
        )
    ))
    
    # Run all optimizers
    results = []
    for name, optimizer in optimizers:
        result = run_optimizer(name, optimizer)
        results.append(result)
    
    # Compare results
    compare_results(results)


if __name__ == "__main__":
    main()
