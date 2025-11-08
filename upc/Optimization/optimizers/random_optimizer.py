"""
RandomOptimizer: Simple random search baseline.

Randomly samples configurations from the design space and evaluates them.
Good baseline to compare against more sophisticated methods.
"""

import sys
from typing import Tuple, Optional, Dict
import pandas as pd
import time

# Add parent directory to path for imports
sys.path.append('/media/mohammad/extension/experiments/astra-sim/upc/Optimization')
from ..core import BaseOptimizer, SearchSpaceBuilder
from ..helper import config_to_tuple, tuple_to_config


class RandomOptimizer(BaseOptimizer):
    """
    Random Search optimizer.
    
    Randomly samples configurations without replacement from the design space.
    Simple but effective baseline, especially for:
    - Small design spaces
    - Flat optimization landscapes
    - Establishing baseline performance
    
    Example:
        from core.search_space_builder import create_search_space
        from core.sampler import RandomSampler
        from core.simulation_runner import SimulationRunner
        
        search_space = create_search_space(
            "search_space/parallelism_strategy_params.json",
            num_npus=64,
            include_categories=['parallelism_strategy']
        )
        sampler = RandomSampler(seed=42)
        sim_runner = SimulationRunner(40, "GPT_40B", 64, "FoldedClos")
        
        optimizer = RandomOptimizer(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=sim_runner,
            budget=50
        )
        
        best_config, results_df = optimizer.run()
    """
    
    def __init__(
        self,
        search_space,
        sampler,
        simulation_runner,
        budget: int = 30,
        verbose: bool = True,
        save_dir: str = ".",
        keep_top_k: int = -1
    ):
        """
        Initialize Random Search optimizer.
        
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance for sampling
            simulation_runner: SimulationRunner instance
            budget: Total number of evaluations
            verbose: Whether to print progress
            save_dir: Directory to save results
            keep_top_k: Keep only top K results' files (-1 = keep all, 0 = keep none)
        """
        # Call parent constructor with init_samples = 0
        # (we don't need separate init phase for random search)
        super().__init__(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=simulation_runner,
            budget=budget,
            init_samples=0,  # No separate initialization for random search
            verbose=verbose,
            save_dir=save_dir,
            keep_top_k=keep_top_k
        )
    
    def initialize(self) -> bool:
        """
        Initialize optimizer.
        
        For random search, we don't need a separate initialization phase.
        
        Returns:
            Always True
        """
        if self.verbose:
            print("\n" + "="*70)
            print("RANDOM SEARCH OPTIMIZATION")
            print("="*70)
            print(f"Model: {self.simulation_runner.model_name}")
            print(f"NPUs: {self.simulation_runner.num_npus}")
            print(f"Network: {self.simulation_runner.network_name}")
            print(f"Budget: {self.budget} evaluations")
            print(f"Design space: {self.search_space.get_design_space_size()} configurations")
            print("="*70 + "\n")
        
        return True
    
    def optimize_step(self) -> Tuple[Optional[Dict], Optional[float]]:
        """
        Execute one random search step.
        
        Returns:
            (config, score) tuple if successful, (None, None) otherwise
        """
        # This method is not used in random search run() loop,
        # but implemented for interface compliance
        design_space = self.search_space.get_design_space()
        
        # Filter out already evaluated configs
        # Convert configs to tuples for set operations (dicts are unhashable)
        evaluated_set = set(config_to_tuple(c) for c in self.configs)
        unevaluated = [c for c in design_space 
                       if config_to_tuple(c) not in evaluated_set]
        
        if not unevaluated:
            return None, None
        
        # Sample one config
        config = self.sampler.sample(unevaluated, 1)[0]
        
        # Evaluate
        score = self.evaluate_config(config, verbose=self.verbose)
        
        return config, score
    
    def run(self) -> Tuple[Optional[Dict], pd.DataFrame]:
        """
        Run full random search optimization.
        
        Returns:
            (best_config, results_dataframe) tuple
        """
        self.start_time = time.time()
        
        # Initialize
        if not self.initialize():
            return None, pd.DataFrame()
        
        try:
            # Get design space
            design_space = self.search_space.get_design_space()
            
            if not design_space:
                self._log("No valid configurations in design space!", "error")
                return None, pd.DataFrame()
            
            # Sample configurations
            sample_size = min(self.budget, len(design_space))
            sampled_configs = self.sampler.sample(design_space, sample_size)
            
            if self.verbose:
                print(f"Sampled {len(sampled_configs)} configurations\n")
            
            # Evaluate all sampled configurations
            for i, config in enumerate(sampled_configs):
                self.current_iteration = i
                
                if self.verbose:
                    # Print configuration parameters
                    config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
                    print(f"[{i+1}/{len(sampled_configs)}] Testing: {config_str}")
                
                # Evaluate
                iter_start = time.time()
                exec_time = self.evaluate_config(config, verbose=True)
                iter_time = time.time() - iter_start
                
                self.iteration_times.append(iter_time)
                
                if exec_time is None:
                    if self.verbose:
                        print("  ❌ Evaluation failed, skipping...\n")
                    continue
                
                if self.verbose:
                    print(f"  ✓ Time: {exec_time:.2f}s")
                    if self.best_config is not None:
                        print(f"  Best so far: {self.best_score:.2f}s\n")
            
            # Print summary
            if self.verbose:
                self.print_summary()
            
            # Save results
            self.save_results()
            
            return self.best_config, self.get_history()
            
        except KeyboardInterrupt:
            self._log("\n\nOptimization interrupted by user", "warning")
            self._log("Saving intermediate results...", "info")
            self.save_results()
            return self.best_config, self.get_history()
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"RandomOptimizer(budget={self.budget}, "
                f"evaluated={len(self.configs)})")
