"""
Integration example: Using SearchSpaceBuilder with optimizers.

This example shows how to:
1. Create a search space with custom parameters
2. Sample initial configurations
3. Use with RandomOptimizer
4. Convert configurations for simulation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from search_space_builder import create_search_space


def example_with_random_optimizer():
    """Example: Using SearchSpaceBuilder with RandomOptimizer."""
    print("\n" + "=" * 70)
    print("INTEGRATION EXAMPLE: SearchSpaceBuilder + Optimizer")
    print("=" * 70)
    
    # 1. Create search space
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "search_space", "all_parameters.json")
    
    builder = create_search_space(
        config_path=config_path,
        num_npus=64,
        include_categories=['parallelism_strategy'],
        custom_constraints=["dp * mp <= 8"]  # Limit parallelism
    )
    
    print(builder.summary())
    
    # 2. Sample initial configurations
    print("\n" + "=" * 70)
    print("SAMPLING CONFIGURATIONS")
    print("=" * 70)
    
    # Use LHS for better initial coverage
    configs = builder.sample(n_samples=10, strategy='lhs', seed=42)
    
    print(f"\nSampled {len(configs)} configurations:")
    for i, config in enumerate(configs, 1):
        print(f"\n  Config {i}:")
        print(f"    dp={config['dp']}, mp={config['mp']}, "
              f"sp={config['sp']}, pp={config['pp']}, sharded={config['sharded']}")
    
    # 3. Convert to optimizer format (if needed)
    print("\n" + "=" * 70)
    print("CONVERTING TO OPTIMIZER FORMAT")
    print("=" * 70)
    
    # Option A: Use as dictionaries directly
    print("\nOption A: Dictionary format (recommended)")
    for i, config in enumerate(configs[:3], 1):
        print(f"  {i}. {config}")
    
    # Option B: Convert to tuples if optimizer requires it
    print("\nOption B: Tuple format (for legacy optimizers)")
    tuple_configs = [
        (c['dp'], c['mp'], c['sp'], c['pp'], c['sharded'])
        for c in configs
    ]
    for i, config in enumerate(tuple_configs[:3], 1):
        print(f"  {i}. {config}")
    
    # 4. Demonstrate constraint satisfaction
    print("\n" + "=" * 70)
    print("CONSTRAINT VERIFICATION")
    print("=" * 70)
    
    all_valid = True
    for i, config in enumerate(configs, 1):
        dp, mp, sp, pp = config['dp'], config['mp'], config['sp'], config['pp']
        
        # Check NPU constraint
        npu_product = dp * mp * sp * pp
        npu_valid = (npu_product == 64)
        
        # Check custom constraint
        custom_valid = (dp * mp <= 8)
        
        if not (npu_valid and custom_valid):
            all_valid = False
            print(f"❌ Config {i}: INVALID")
        else:
            print(f"✅ Config {i}: dp*mp*sp*pp={npu_product}, dp*mp={dp*mp}")
    
    if all_valid:
        print("\n✅ All configurations satisfy constraints!")
    
    return configs


def example_parameter_exploration():
    """Example: Exploring different parameter combinations."""
    print("\n" + "=" * 70)
    print("PARAMETER EXPLORATION")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "search_space", "all_parameters.json")
    
    # Explore network parameters
    print("\n1. Network Parameter Space:")
    print("-" * 70)
    
    network_builder = create_search_space(
        config_path=config_path,
        include_categories=['network']
    )
    
    print(f"Design space size: {network_builder.get_design_space_size()}")
    
    network_configs = network_builder.sample(5, strategy='grid', seed=42)
    for i, config in enumerate(network_configs, 1):
        print(f"  {i}. topology={config['topology']}, "
              f"inter_bw={config['inter_node_bw']}, "
              f"intra_bw={config['intra_node_bw']}")
    
    # Explore model parameters with constraints
    print("\n2. Model Parameter Space (with constraints):")
    print("-" * 70)
    
    model_builder = create_search_space(
        config_path=config_path,
        include_categories=['model'],
        custom_constraints=[
            "batch_size % micro_batch_size == 0",
            "batch_size >= 4096"
        ]
    )
    
    print(f"Design space size: {model_builder.get_design_space_size()}")
    
    model_configs = model_builder.sample(5, strategy='random', seed=42)
    for i, config in enumerate(model_configs, 1):
        print(f"  {i}. batch={config['batch_size']}, "
              f"micro_batch={config['micro_batch_size']}, "
              f"layers={config['num_layers']}")
    
    # Combined parameters
    print("\n3. Combined Parameter Space:")
    print("-" * 70)
    
    combined_builder = create_search_space(
        config_path=config_path,
        num_npus=128,
        include_categories=['parallelism_strategy', 'network', 'hardware'],
        max_configs=100
    )
    
    print(f"Design space size: {combined_builder.get_design_space_size()}")
    
    combined_configs = combined_builder.sample(3, strategy='lhs', seed=42)
    for i, config in enumerate(combined_configs, 1):
        print(f"\n  Config {i}:")
        for key, value in sorted(config.items()):
            print(f"    {key}: {value}")


def example_adaptive_sampling():
    """Example: Adaptive sampling based on results."""
    print("\n" + "=" * 70)
    print("ADAPTIVE SAMPLING SIMULATION")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "search_space", "all_parameters.json")
    
    builder = create_search_space(
        config_path=config_path,
        num_npus=64,
        include_categories=['parallelism_strategy']
    )
    
    print(f"Total design space: {builder.get_design_space_size()} configs\n")
    
    # Round 1: Initial exploration with LHS
    print("Round 1: Initial exploration (LHS)")
    print("-" * 70)
    round1_configs = builder.sample(10, strategy='lhs', seed=42)
    print(f"Sampled {len(round1_configs)} configs for initial exploration")
    
    # Simulate finding best configs
    best_configs = round1_configs[:3]
    print("\nBest 3 configs from Round 1:")
    for i, config in enumerate(best_configs, 1):
        print(f"  {i}. dp={config['dp']}, mp={config['mp']}, "
              f"sp={config['sp']}, pp={config['pp']}")
    
    # Round 2: Focused exploration with random
    print("\nRound 2: Continued exploration (Random)")
    print("-" * 70)
    round2_configs = builder.sample(10, strategy='random', seed=43)
    print(f"Sampled {len(round2_configs)} additional configs")
    
    # Round 3: Grid search in promising region
    print("\nRound 3: Fine-grained search (Grid)")
    print("-" * 70)
    round3_configs = builder.sample(10, strategy='grid', seed=44)
    print(f"Sampled {len(round3_configs)} configs for grid search")
    
    total_evaluated = len(round1_configs) + len(round2_configs) + len(round3_configs)
    coverage = (total_evaluated / builder.get_design_space_size()) * 100
    
    print(f"\n📊 Total evaluated: {total_evaluated}/{builder.get_design_space_size()} "
          f"({coverage:.1f}% coverage)")


def main():
    """Run all integration examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "INTEGRATION EXAMPLES" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Run examples
    _ = example_with_random_optimizer()
    example_parameter_exploration()
    example_adaptive_sampling()
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
✅ SearchSpaceBuilder successfully integrated
✅ Multiple sampling strategies demonstrated
✅ Constraint validation working
✅ Ready for use with optimizers

Next Steps:
1. Use builder.sample() to get initial configurations
2. Pass configurations to your simulator/evaluator
3. Use results to guide further sampling or optimization
4. Iterate and refine
""")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
