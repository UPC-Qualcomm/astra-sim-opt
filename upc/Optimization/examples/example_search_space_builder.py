"""
Example usage of SearchSpaceBuilder.

This example demonstrates:
1. Building search spaces with different parameter subsets
2. Applying dynamic constraints
3. Sampling configurations with different strategies
4. Working with partial parameter sets
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from search_space_builder import SearchSpaceBuilder, create_search_space


def example_1_full_parameters():
    """Example 1: Using all parameters from JSON."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Full Parameter Space")
    print("=" * 70)
    
    config_path = "../search_space/all_parameters.json"
    
    # Create builder
    builder = SearchSpaceBuilder(config_path, num_npus=128)
    
    # Parse all parameters
    builder.parse_parameters()
    
    # Apply constraints from file
    builder.apply_constraints()
    
    # Build design space (limit to 1000 configs for demo)
    builder.build(max_configs=1000)
    
    # Print summary
    print(builder.summary())
    
    # Sample 10 configurations using random sampling
    samples = builder.sample(n_samples=10, strategy='random', seed=42)
    
    print("\n📦 Sample Configurations:")
    for i, config in enumerate(samples[:3], 1):
        print(f"\n  Config {i}:")
        for key, value in config.items():
            print(f"    {key}: {value}")
    print(f"\n  ... and {len(samples) - 3} more")
    
    return builder


def example_2_parallelism_only():
    """Example 2: Using only parallelism parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Parallelism Parameters Only")
    print("=" * 70)
    
    config_path = "../search_space/all_parameters.json"
    
    # Create builder with only parallelism parameters
    builder = SearchSpaceBuilder(config_path, num_npus=64)
    builder.parse_parameters(include_categories=['parallelism_strategy'])
    builder.apply_constraints()
    builder.build()
    
    print(builder.summary())
    
    # Try different sampling strategies
    print("\n🎲 Sampling Strategies:")
    
    strategies = ['random', 'lhs', 'grid']
    for strategy in strategies:
        samples = builder.sample(n_samples=5, strategy=strategy, seed=42)
        print(f"\n  {strategy.upper()} sampling: {len(samples)} configs")
        print(f"    First config: {samples[0]}")
    
    return builder


def example_3_network_and_hardware():
    """Example 3: Network and hardware parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Network and Hardware Parameters")
    print("=" * 70)
    
    config_path = "../search_space/all_parameters.json"
    
    # Include only network and hardware
    builder = SearchSpaceBuilder(config_path)
    builder.parse_parameters(include_categories=['network', 'hardware'])
    builder.apply_constraints()  # No parallelism constraints apply here
    builder.build()
    
    print(builder.summary())
    
    # Sample with Latin Hypercube
    _ = builder.sample(n_samples=20, strategy='lhs', seed=42)
    
    print("\n📊 Parameter Statistics:")
    param_info = builder.get_parameter_info()
    for param, info in param_info.items():
        print(f"\n  {param}:")
        print(f"    Type: {info['type']}")
        print(f"    Count: {info['count']}")
        if info['min'] is not None:
            print(f"    Range: [{info['min']}, {info['max']}]")
    
    return builder


def example_4_custom_constraints():
    """Example 4: Adding custom constraints."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Constraints")
    print("=" * 70)
    
    config_path = "../search_space/all_parameters.json"
    
    # Create builder with model parameters
    builder = SearchSpaceBuilder(config_path)
    builder.parse_parameters(include_categories=['model'])
    
    # Add custom constraints
    custom_constraints = [
        "batch_size % micro_batch_size == 0",  # batch_size divisible by micro_batch_size
        "micro_batch_size <= batch_size",       # micro_batch_size not larger than batch_size
        "num_layers % 12 == 0",                 # num_layers divisible by 12
    ]
    
    builder.apply_constraints(custom_constraints)
    builder.build()
    
    print(builder.summary())
    
    # Sample configurations
    samples = builder.sample(n_samples=15, strategy='random', seed=42)
    
    print("\n✅ Validated Configurations:")
    print(f"   All {len(samples)} configs satisfy:")
    for constraint in custom_constraints:
        print(f"     - {constraint}")
    
    # Show a few examples
    print("\n   Example configs:")
    for i, config in enumerate(samples[:3], 1):
        print(f"\n   {i}. batch_size={config['batch_size']}, "
              f"micro_batch_size={config['micro_batch_size']}, "
              f"num_layers={config['num_layers']}")
    
    return builder


def example_5_convenience_function():
    """Example 5: Using the convenience function."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Convenience Function")
    print("=" * 70)
    
    config_path = "../search_space/all_parameters.json"
    
    # Create and build in one call
    builder = create_search_space(
        config_path=config_path,
        num_npus=128,
        include_categories=['parallelism_strategy', 'collective'],
        custom_constraints=["dp * mp <= 16"],  # Limit parallelism
        max_configs=500
    )
    
    print(builder.summary())
    
    # Sample with different strategies
    random_samples = builder.sample(10, strategy='random', seed=42)
    grid_samples = builder.sample(10, strategy='grid', seed=42)
    
    print("\n📊 Sampling Results:")
    print(f"   Random: {len(random_samples)} configs")
    print(f"   Grid: {len(grid_samples)} configs")
    
    return builder


def example_6_save_and_load():
    """Example 6: Saving design space."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Saving Design Space")
    print("=" * 70)
    
    config_path = "../search_space/all_parameters.json"
    
    # Create a small design space
    builder = SearchSpaceBuilder(config_path, num_npus=16)
    builder.parse_parameters(include_categories=['parallelism_strategy'])
    builder.apply_constraints()
    builder.build()
    
    print(builder.summary())
    
    # Save to file
    output_path = "/tmp/design_space_example.json"
    builder.save_design_space(output_path)
    
    print(f"\n💾 Design space saved to: {output_path}")
    print(f"   Total configs: {builder.get_design_space_size()}")
    
    return builder


def example_7_mixed_parameters():
    """Example 7: Mixed parameter types."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Mixed Parameter Types")
    print("=" * 70)
    
    config_path = "../search_space/all_parameters.json"
    
    # Mix different parameter categories
    builder = SearchSpaceBuilder(config_path, num_npus=64)
    builder.parse_parameters(
        include_categories=['parallelism_strategy', 'network', 'model']
    )
    
    # Apply both config and custom constraints
    builder.apply_constraints(custom_constraints=[
        "batch_size >= 2048",  # Minimum batch size
        "npus_per_node <= 8",  # Maximum NPUs per node
    ])
    
    builder.build(max_configs=100)
    
    print(builder.summary())
    
    # Sample and show parameter diversity
    samples = builder.sample(n_samples=10, strategy='lhs', seed=42)
    
    print("\n🌈 Parameter Diversity in Samples:")
    
    # Count unique values for each parameter
    unique_counts = {}
    for param in samples[0].keys():
        unique_values = set(config[param] for config in samples)
        unique_counts[param] = len(unique_values)
    
    for param, count in sorted(unique_counts.items(), key=lambda x: -x[1]):
        print(f"   {param}: {count} unique values")
    
    return builder


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SEARCH SPACE BUILDER EXAMPLES" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    
    examples = [
        example_1_full_parameters,
        example_2_parallelism_only,
        example_3_network_and_hardware,
        example_4_custom_constraints,
        example_5_convenience_function,
        example_6_save_and_load,
        example_7_mixed_parameters,
    ]
    
    results = []
    for example in examples:
        try:
            result = example()
            results.append((example.__name__, result, None))
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
            results.append((example.__name__, None, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, result, error in results:
        if error:
            print(f"❌ {name}: FAILED ({error})")
        else:
            print(f"✅ {name}: SUCCESS (space_size={result.get_design_space_size()})")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
