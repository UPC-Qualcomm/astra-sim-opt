"""
RandomOptimizer: Simple random search baseline.

Randomly samples configurations from the design space and evaluates them.
Good baseline to compare against more sophisticated methods.

Supports parallel evaluation when n_workers > 1.
"""

import sys
from typing import Tuple, Optional, Dict
import pandas as pd
import time
from multiprocessing import Pool
from functools import partial

# Add parent directory to path for imports
sys.path.append('/media/mohammad/extension/experiments/astra-sim/upc/Optimization')
from ..core import BaseOptimizer
from ..helper import config_to_tuple, evaluate_config_worker


class RandomOptimizer(BaseOptimizer):
    """
    Random Search optimizer with optional parallelization.
    
    Randomly samples configurations without replacement from the design space.
    Simple but effective baseline, especially for:
    - Small design spaces
    - Flat optimization landscapes
    - Establishing baseline performance
    
    When n_workers > 1, evaluates multiple configurations in parallel using
    multiprocessing for significant speedup.
    
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
        
        # Sequential (default)
        optimizer = RandomOptimizer(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=sim_runner,
            budget=50
        )
        
        # Parallel with 8 workers
        optimizer = RandomOptimizer(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=sim_runner,
            budget=50,
            n_workers=8
        )
        
        best_config, results_df = optimizer.run()
    """
    
    def __init__(
        self,
        search_space,
        sampler,
        simulation_runner,
        budget: int = 30,
        n_workers: int = 1,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        save_dir: str = ".",
        keep_top_k: int = -1,
        profile_time: bool = False
    ):
        """
        Initialize Random Search optimizer.
        
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance for sampling
            simulation_runner: SimulationRunner instance
            budget: Total number of evaluations
            n_workers: Number of parallel workers (1 = sequential, >1 = parallel)
            batch_size: Configs per batch when parallel (default: n_workers * 2)
            verbose: Whether to print progress
            save_dir: Directory to save results
            keep_top_k: Keep only top K results' files (-1 = keep all, 0 = keep none)
            profile_time: Whether to track and print detailed time statistics
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
            keep_top_k=keep_top_k,
            profile_time=profile_time
        )
        
        # Parallelization settings
        self.n_workers = max(1, n_workers)
        
        # Default batch size
        if (self.n_workers == 1):
            self.batch_size = 1  # Sequential
        else:
            self.batch_size = batch_size if batch_size is not None else self.n_workers * 2
    
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
            print(f"Workers: {self.n_workers}")
            print(f"Batch size: {self.batch_size}")
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
        
        Uses parallel evaluation if n_workers > 1, otherwise sequential.
        
        Returns:
            (best_config, results_dataframe) tuple
        """
        self.start_time = time.time()
        self.time_stats.start_total()
        
        # Initialize
        with self.time_stats.timer("initialization"):
            if not self.initialize():
                return None, pd.DataFrame()
        
        try:
            # Get design space
            with self.time_stats.timer("design_space_generation"):
                design_space = self.search_space.get_design_space()
            
            if not design_space:
                self._log("No valid configurations in design space!", "error")
                return None, pd.DataFrame()
            
            # Sample configurations
            with self.time_stats.timer("sampling"):
                sample_size = min(self.budget, len(design_space))
                sampled_configs = self.sampler.sample(design_space, sample_size)
            
            if self.verbose:
                print(f"Sampled {len(sampled_configs)} configurations")
                print(f"Evaluating in batches of {self.batch_size}...\n")
            
            # Run evaluation
            self._run_batched(sampled_configs)
            
            # Print summary
            if self.verbose:
                self.print_summary()
            
            # Save results
            with self.time_stats.timer("save_results"):
                self.save_results()
            
            self.time_stats.end_total()
            return self.best_config, self.get_history()
            
        except KeyboardInterrupt:
            self._log("\n\nOptimization interrupted by user", "warning")
            self._log("Saving intermediate results...", "info")
            self.save_results()
            self.time_stats.end_total()
            return self.best_config, self.get_history()
    
    def _run_batched(self, sampled_configs):
        """Run evaluation in batches using multiprocessing."""
        n_batches = (len(sampled_configs) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(sampled_configs))
            batch_configs = sampled_configs[batch_start:batch_end]
            
            if self.verbose:
                print(f"Batch {batch_idx + 1}/{n_batches}: Evaluating {len(batch_configs)} configs in parallel...")
            
            batch_start_time = time.time()
            
            # Evaluate batch in parallel
            with self.time_stats.timer("batch_evaluation"):
                with Pool(processes=self.n_workers) as pool:
                    # Create partial function with simulation_runner bound
                    eval_func = partial(evaluate_config_worker, 
                                      simulation_runner=self.simulation_runner)
                    
                    # Map configs to workers
                    results = pool.map(eval_func, batch_configs)
            
            batch_time = time.time() - batch_start_time
            
            # Process results
            with self.time_stats.timer("result_processing"):
                successful = 0
                failed = 0
                for config, exec_time, file_paths, metadata in results:
                    self.current_iteration = len(self.configs)
                    
                    if exec_time is not None:
                        # Compute objective score
                        score = self.objective.compute(exec_time, metadata)
                        
                        # Record results
                        self.configs.append(config)
                        self.scores.append(score)
                        self.file_paths.append(file_paths)
                        self.metadata.append(metadata)
                        
                        # Update best
                        if self.objective.is_better(score, self.best_score):
                            self.best_score = score
                            self.best_config = config
                            self.best_iteration = self.current_iteration
                        
                        successful += 1
                    else:
                        self.file_paths.append({})
                        self.metadata.append({})
                        failed += 1
            
            # Cleanup if needed
            if self.keep_top_k >= 0:
                with self.time_stats.timer("file_cleanup"):
                    self._cleanup_files()
            
            if self.verbose:
                print(f"  ✓ Batch completed in {batch_time:.1f}s")
                print(f"  Successful: {successful}/{len(batch_configs)}")
                if failed > 0:
                    print(f"  Failed: {failed}")
                if self.best_config:
                    print(f"  Best so far: {self.best_score:.4f}")
                print()
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"RandomOptimizer(budget={self.budget}, "
                f"n_workers={self.n_workers}, "
                f"batch_size={self.batch_size}, "
                f"evaluated={len(self.configs)})")
