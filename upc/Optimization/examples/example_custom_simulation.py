#!/usr/bin/env python3
"""
Example: Custom Simulation Configuration

Demonstrates how to customize simulation parameters including:
- Custom memory configuration
- Custom output directories
- Custom system/network configurations
- Different simulation types
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search_space import SearchSpace
from core.sampler import LatinHypercubeSampler
from core.simulation_runner import SimulationRunner
from optimizers.random_optimizer import RandomOptimizer


def example_custom_paths():
    """Example: Using custom paths for simulation outputs."""
    
    print("="*70)
    print("EXAMPLE 1: Custom Output Directories")
    print("="*70)
    
    # Configuration
    MODEL_NUM = 19
    MODEL_NAME = "GPT_40B"
    NUM_NPUS = 128
    NETWORK_NAME = "FoldedClos"
    
    # Custom paths
    CUSTOM_OUTPUT = "/tmp/my_custom_output"
    CUSTOM_NETWORK_LOG = "/tmp/my_custom_network_logs"
    
    # Setup search space
    search_space_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "search_space", 
        "parallelism_startegy_params.json"
    )
    search_space = SearchSpace(search_space_path, num_npus=NUM_NPUS)
    
    # Setup sampler
    sampler = LatinHypercubeSampler(seed=42)
    
    # Setup simulation runner with CUSTOM PATHS
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        num_npus=NUM_NPUS,
        network_name=NETWORK_NAME,
        folder_prefix="CUSTOM_DIRS",
        # Custom directories
        output_dir=CUSTOM_OUTPUT,
        network_log_dir=CUSTOM_NETWORK_LOG,
        verbose=True
    )
    
    print(f"\n📁 Using custom directories:")
    print(f"   Output: {sim_runner.output_dir}")
    print(f"   Network logs: {sim_runner.network_log_dir}")
    
    # Create optimizer
    optimizer = RandomOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=5,
        verbose=True
    )
    
    # Run
    best_config, history = optimizer.run()
    
    if best_config:
        print(f"\n✓ Results saved to custom directories")
        print(f"   Check: {CUSTOM_OUTPUT}")


def example_custom_configs():
    """Example: Using custom system/network configurations."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom System/Network Configurations")
    print("="*70)
    
    # Configuration
    MODEL_NUM = 19
    MODEL_NAME = "GPT_40B"
    NUM_NPUS = 64
    
    # Custom configuration files
    CUSTOM_SYSTEM = "/media/mohammad/extension/experiments/astra-sim/upc/configuration/analytical_unaware/Switch_sys.json"
    CUSTOM_NETWORK = "/media/mohammad/extension/experiments/astra-sim/upc/configuration/analytical_unaware/FoldedClos_64_config.txt"
    
    # Setup search space
    search_space_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "search_space", 
        "parallelism_startegy_params.json"
    )
    search_space = SearchSpace(search_space_path, num_npus=NUM_NPUS)
    
    # Setup sampler
    sampler = LatinHypercubeSampler(seed=123)
    
    # Setup simulation runner with CUSTOM CONFIGS
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        num_npus=NUM_NPUS,
        network_name="FoldedClos",
        folder_prefix="CUSTOM_CONFIG",
        # Custom configuration files
        system_config=CUSTOM_SYSTEM,
        network_config=CUSTOM_NETWORK,
        verbose=True
    )
    
    print(f"\n📄 Using custom configurations:")
    print(f"   System: {sim_runner.system_config}")
    print(f"   Network: {sim_runner.network_config}")
    
    # Create optimizer
    optimizer = RandomOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=5,
        verbose=True
    )
    
    # Run
    best_config, history = optimizer.run()
    
    if best_config:
        print(f"\n✓ Optimization complete with custom configs")


def example_g2_backend():
    """Example: Using g2 simulation backend with custom settings."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: G2 Backend with Custom Memory Config")
    print("="*70)
    
    # Configuration
    MODEL_NUM = 10  # GPT_3_175B
    MODEL_NAME = "GPT_3_175B"
    NUM_NPUS = 128
    
    # Custom memory configuration
    CUSTOM_MEMORY = "./configuration/RemoteMemory_Custom.json"
    
    # Setup search space
    search_space_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "search_space", 
        "parallelism_startegy_params.json"
    )
    search_space = SearchSpace(search_space_path, num_npus=NUM_NPUS)
    
    # Setup sampler
    sampler = LatinHypercubeSampler(seed=456)
    
    # Setup simulation runner for G2 backend
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        num_npus=NUM_NPUS,
        network_name="FoldedClos",
        sim_type="g2",  # Use g2 backend
        folder_prefix="G2_CUSTOM",
        # Custom memory config (if you have one, otherwise uses default)
        # memory_config=CUSTOM_MEMORY,
        verbose=True
    )
    
    print(f"\n⚙️  Using g2 backend:")
    print(f"   Sim type: {sim_runner.sim_type}")
    print(f"   Memory config: {sim_runner.memory_config}")
    
    # Create optimizer
    optimizer = RandomOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=3,
        verbose=True
    )
    
    # Run
    best_config, history = optimizer.run()
    
    if best_config:
        print(f"\n✓ G2 backend optimization complete")


def example_full_customization():
    """Example: Full customization of all parameters."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Full Customization")
    print("="*70)
    
    # Configuration
    MODEL_NUM = 19
    MODEL_NAME = "GPT_40B"
    NUM_NPUS = 64
    
    # All custom paths
    CUSTOM_BASE_DIR = "/media/mohammad/extension/experiments/astra-sim/upc"
    CUSTOM_OUTPUT = f"{CUSTOM_BASE_DIR}/output/MY_EXPERIMENT"
    CUSTOM_NETWORK_LOG = f"{CUSTOM_BASE_DIR}/network_log/MY_EXPERIMENT"
    CUSTOM_SYSTEM = f"{CUSTOM_BASE_DIR}/configuration/analytical_unaware/Switch_sys.json"
    CUSTOM_NETWORK = f"{CUSTOM_BASE_DIR}/configuration/analytical_unaware/FoldedClos_64_config.txt"
    CUSTOM_MEMORY = "./configuration/RemoteMemory.json"
    
    # Setup search space
    search_space_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "search_space", 
        "parallelism_startegy_params.json"
    )
    search_space = SearchSpace(search_space_path, num_npus=NUM_NPUS)
    
    # Setup sampler
    sampler = LatinHypercubeSampler(seed=999)
    
    # Setup simulation runner with ALL CUSTOM PARAMETERS
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        num_npus=NUM_NPUS,
        network_name="FoldedClos",
        sim_type="analytical_unaware",
        base_dir=CUSTOM_BASE_DIR,
        folder_prefix="FULLY_CUSTOM",
        clean_on_init=True,
        verbose=True,
        # All custom paths
        memory_config=CUSTOM_MEMORY,
        output_dir=CUSTOM_OUTPUT,
        network_log_dir=CUSTOM_NETWORK_LOG,
        system_config=CUSTOM_SYSTEM,
        network_config=CUSTOM_NETWORK
    )
    
    print(f"\n🎯 Fully customized simulation runner:")
    print(f"   System config: {sim_runner.system_config}")
    print(f"   Network config: {sim_runner.network_config}")
    print(f"   Memory config: {sim_runner.memory_config}")
    print(f"   Output dir: {sim_runner.output_dir}")
    print(f"   Network log: {sim_runner.network_log_dir}")
    
    # Create optimizer
    optimizer = RandomOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=5,
        verbose=True
    )
    
    # Run
    best_config, history = optimizer.run()
    
    if best_config:
        print(f"\n✓ Fully customized optimization complete")


def main():
    """Run examples."""
    
    print("\n" + "="*70)
    print("CUSTOM SIMULATION CONFIGURATION EXAMPLES")
    print("="*70)
    print("\nThis example demonstrates how to customize simulation parameters.")
    print("\nAvailable examples:")
    print("  1. Custom output directories")
    print("  2. Custom system/network configurations")
    print("  3. G2 backend with custom settings")
    print("  4. Full customization")
    
    # Uncomment the example you want to run:
    
    # example_custom_paths()
    # example_custom_configs()
    # example_g2_backend()
    # example_full_customization()
    
    print("\n" + "="*70)
    print("ℹ️  Uncomment an example in main() to run it")
    print("="*70)


if __name__ == "__main__":
    main()
