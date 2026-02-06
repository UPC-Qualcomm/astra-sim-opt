"""
BaseOptimizer: Abstract base class for all optimizers.

Defines the common interface that all optimizers must implement.
Provides shared functionality for result tracking, logging, and I/O.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any
import pandas as pd
import numpy as np
import time
import os

from .time_statistics import TimeStatistics
from .objective import ObjectiveFunction, MinimizeExecutionTime


def format_score(score) -> str:
    """Format score for display, handling both floats and tuples."""
    if isinstance(score, tuple):
        return f"({', '.join([f'{s:.4f}' for s in score])})"
    elif score is None or score == float('inf'):
        return "N/A"
    else:
        return f"{score:.4f}"


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    All optimizers must implement:
    - initialize(): Setup initial samples
    - optimize_step(): Single optimization iteration
    - run(): Main optimization loop
    
    Shared functionality:
    - Result tracking and history
    - Best configuration tracking
    - Results saving to CSV
    - Progress logging
    """
    
    def __init__(
        self,
        search_space,
        sampler,
        simulation_runner,
        budget: int = 30,
        init_samples: int = 5,
        objective: 'ObjectiveFunction' = MinimizeExecutionTime(),
        verbose: bool = True,
        save_dir: str = ".",
        keep_top_k: int = -1,
        profile_time: bool = False
    ):
        """
        Initialize base optimizer.
        
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance for initial sampling
            simulation_runner: SimulationRunner instance
            budget: Total number of evaluations
            init_samples: Number of initial random samples
            objective: ObjectiveFunction to optimize (default: MinimizeExecutionTime)
            verbose: Whether to print progress
            save_dir: Directory to save results
            keep_top_k: Keep only top K results' files (-1 = keep all, 0 = keep none)
            profile_time: Whether to track and print detailed time statistics
        """
        self.search_space = search_space
        self.sampler = sampler
        self.simulation_runner = simulation_runner
        self.budget = budget
        self.init_samples = init_samples
        self.objective = objective 
        self.verbose = verbose
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.profile_time = profile_time
        
        # Time profiling
        self.time_stats = TimeStatistics(enabled=profile_time)
        
        # Result tracking
        self.configs: List[Dict] = []  # Evaluated configurations (now dicts)
        self.scores: List[float] = []   # Execution times (lower is better)
        self.iteration_times: List[float] = []  # Time per iteration
        self.history: List[Dict] = []  # Detailed history
        self.file_paths: List[Dict[str, str]] = []  # Track workload and output files
        self.metadata: List[Dict] = []  # Track simulation metadata (model, network, hardware params)
        
        # Best tracking
        self.best_config: Optional[Dict] = None
        self.best_score: float = float('inf')
        self.best_iteration: int = -1
        
        # State
        self.current_iteration = 0
        self.start_time = None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize optimizer with initial samples.
        
        Should generate init_samples configurations and evaluate them.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def optimize_step(self) -> Tuple[Optional[Dict], Optional[float]]:
        """
        Execute one optimization iteration.
        
        Returns:
            (config, score) tuple if successful, (None, None) otherwise
        """
        pass
    
    @abstractmethod
    def run(self) -> Tuple[Optional[Dict], pd.DataFrame]:
        """
        Run full optimization loop.
        
        Returns:
            (best_config, results_dataframe) tuple
        """
        pass
    
    def evaluate_config(self, config: Dict, verbose: bool = False) -> Optional[float]:
        """
        Evaluate a single configuration.
        
        Args:
            config: Configuration dictionary (e.g., {'dp': 2, 'mp': 4, ...})
            verbose: Whether to print evaluation details
        
        Returns:
            Objective score, or None if evaluation failed
        """
        try:
            # Enrich config with cluster info if cluster parameter exists
            if 'cluster' in config and hasattr(self, 'search_space'):
                config = self.search_space.enrich_config_with_cluster_info(config)
            
            # Run simulation and get execution time + file paths + metadata
            result = self.simulation_runner.run_simulation(config, return_paths=True)
            
            if result is not None:
                if isinstance(result, tuple) and len(result) == 4:
                    # From evaluate_config_worker: (exec_time, is_oom, file_paths, metadata)
                    exec_time, is_oom, file_paths, metadata = result
                elif isinstance(result, tuple) and len(result) == 3:
                    exec_time, is_oom, file_paths = result
                    metadata = {}
                else:
                    exec_time, is_oom = result
                    file_paths = {}
                    metadata = {}
                
                # Compute objective score (pass config as well)
                score = self.objective.compute(exec_time, is_oom, metadata, config)
                
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
                    
                    if self.verbose and verbose:
                        print(f"    🏆 NEW BEST! Score: {score:.4f} (exec_time: {exec_time:.2f}s)")
                
                # Cleanup if needed
                if self.keep_top_k >= 0:
                    self._cleanup_files()
                
                return score
            else:
                # Track empty file paths and metadata for failed runs
                self.file_paths.append({})
                self.metadata.append({})
                if verbose:
                    print("    ⚠️  Evaluation failed")
                return None
                
        except Exception as e:
            # Track empty file paths and metadata for failed runs
            self.file_paths.append({})
            self.metadata.append({})
            if verbose:
                print(f"    ⚠️  Error evaluate config: {e}")
            return None
    
    def get_best_config(self) -> Tuple[Optional[Dict], float]:
        """
        Get the best configuration found so far.
        
        Returns:
            (best_config, best_score) tuple
        """
        return self.best_config, self.best_score
    
    def get_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.
        
        Returns:
            DataFrame with columns: iteration, config parameters, exec_time, metadata
        """
        if not self.configs:
            return pd.DataFrame()
        
        history = []
        for i, (config, score, metadata) in enumerate(zip(self.configs, self.scores, self.metadata)):
            # Create record with iteration
            record = {'iteration': i + 1}
            
            # Add optimized configuration parameters (parallelism strategy)
            record.update(config)
            
            # Add results immediately after optimization parameters
            record['exec_time_seconds'] = score
            record['best_so_far'] = min(self.scores[:i+1])
            
            # Add simulation metadata (model, network, hardware parameters)
            record.update(metadata)
            
            history.append(record)
        
        return pd.DataFrame(history)
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save optimization results to CSV.
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            optimizer_name = self.__class__.__name__.replace('Optimizer', '').lower()
            filename = (f"{optimizer_name}_results_"
                       f"{self.simulation_runner.model_name}_"
                       f"{self.simulation_runner.num_npus}npus.csv")
        
        filepath = os.path.join(self.save_dir, filename)
        
        df = self.get_history()
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"\n✓ Results saved to: {filepath}")
        
        return filepath
    
    def print_summary(self):
        """Print optimization summary."""
        if not self.scores:
            print("No results to summarize")
            return
        
        print("\n" + "="*70)
        print(f"OPTIMIZATION SUMMARY - {self.__class__.__name__}")
        print("="*70)
        
        # Objective information
        print(f"\n🎯 OBJECTIVE: {self.objective.name}")
        print(f"   Direction: {'Minimize' if self.objective.minimize else 'Maximize'}")
        
        # Statistics
        scores_array = np.array(self.scores)
        print("\n📊 STATISTICS:")
        print(f"   Total evaluations: {len(self.scores)}")
        if isinstance(self.scores[0], tuple):
            # Multi-objective: show statistics for each objective
            n_objectives = len(self.scores[0])
            for i in range(n_objectives):
                obj_scores = [s[i] for s in self.scores]
                print(f"\n   Objective {i+1}:")
                print(f"     Best: {min(obj_scores):.4f}")
                print(f"     Worst: {max(obj_scores):.4f}")
                print(f"     Mean: {np.mean(obj_scores):.4f}")
                print(f"     Std Dev: {np.std(obj_scores):.4f}")
        else:
            # Single objective
            print(f"   Best score: {scores_array.min():.4f}")
            print(f"   Worst score: {scores_array.max():.4f}")
            print(f"   Mean score: {scores_array.mean():.4f}")
            print(f"   Std Dev: {scores_array.std():.4f}")
        
        # Improvement
        if len(self.scores) > 1 and self.init_samples > 0:
            initial_best = self.objective.get_best_score(self.scores[:self.init_samples])
            print("\n📈 IMPROVEMENT:")
            print(f"   Initial best: {format_score(initial_best)}")
            print(f"   Final best: {format_score(self.best_score)}")
            if not isinstance(self.best_score, tuple):
                improvement_pct = abs((initial_best - self.best_score) / initial_best * 100)
                print(f"   Improvement: {improvement_pct:.1f}%")
        
        # Best configuration
        if self.best_config:
            print("\n🏆 BEST CONFIGURATION:")
            # Print all parameters in the config
            config_str = ", ".join([f"{k}={v}" for k, v in self.best_config.items()])
            print(f"   {config_str}")
            print(f"   Score: {format_score(self.best_score)}")
            print(f"   Found at iteration: {self.best_iteration + 1}")
            
            # Configuration profile (only if parallelism params exist)
            if all(k in self.best_config for k in ['dp', 'mp', 'sp', 'pp']):
                dp = self.best_config['dp']
                mp = self.best_config['mp']
                sp = self.best_config['sp']
                pp = self.best_config['pp']
                sharded = self.best_config.get('sharded', False)
                
                total_npus = dp * mp * sp * pp
                print("\n📋 CONFIGURATION PROFILE:")
                print(f"   Total NPUs: {total_npus}/{self.simulation_runner.num_npus}")
                print(f"   DP/MP ratio: {dp/mp:.2f}")
                print(f"   SP enabled: {'Yes' if sp > 1 else 'No'}")
                print(f"   PP enabled: {'Yes' if pp > 1 else 'No'}")
                print(f"   FSDP enabled: {'Yes' if sharded else 'No'}")
        
        # Timing
        if self.start_time:
            elapsed = time.time() - self.start_time
            print("\n⏱️  TIMING:")
            print(f"   Total time: {elapsed:.1f}s")
            print(f"   Time per evaluation: {elapsed/len(self.scores):.1f}s")
        
        # Time profiling statistics
        if self.profile_time:
            self.time_stats.print_summary()
    
    def _log(self, message: str, level: str = "info"):
        """
        Log message if verbose.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        if self.verbose:
            prefix = {
                'info': '',
                'warning': '⚠️  ',
                'error': '❌ '
            }.get(level, '')
            
            print(f"{prefix}{message}")
    
    def _cleanup_files(self):
        """
        Clean up files, keeping only the top K results.
        
        Removes workload and simulation output files for configurations
        that are not in the top K performers. CSV results are always kept.
        """
        import glob
        
        if self.keep_top_k < 0:
            # No cleanup
            if self.verbose:
                self._log(f"Cleanup disabled (keep_top_k={self.keep_top_k})", "info")
            return
        
        if len(self.scores) <= self.keep_top_k:
            # Not enough evaluations yet
            if self.verbose:
                self._log(f"Cleanup skipped: only {len(self.scores)} evaluations, need > {self.keep_top_k}", "info")
            return
        
        # Find indices of top K configurations (lowest scores)
        top_k_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i])[:self.keep_top_k]
        top_k_set = set(top_k_indices)
        
        if self.verbose:
            self._log(f"Cleanup: Keeping top {self.keep_top_k} out of {len(self.scores)} evaluations", "info")
        
        # Clean up files not in top K
        files_removed = 0
        output_files_removed = 0
        for i, file_paths in enumerate(self.file_paths):
            if i not in top_k_set and file_paths:
                # Remove all workload files matching the pattern (for multi-NPU setups)
                # file_paths['workload'] is the base path without numbered extension
                # e.g., "/path/to/4_8_2_1_1.seq_2048.batch_2048"
                # We need to remove 4_8_2_1_1.seq_2048.batch_2048.0.et, .1.et, .2.et, etc.
                if 'workload' in file_paths:
                    workload_pattern = file_paths['workload'] + ".*"
                    matching_files = glob.glob(workload_pattern)
                    for workload_file in matching_files:
                        if os.path.exists(workload_file):
                            try:
                                os.remove(workload_file)
                                files_removed += 1
                            except Exception as e:
                                if self.verbose:
                                    self._log(f"Warning: Could not remove {workload_file}: {e}", "warning")
                    
                    if self.verbose and matching_files:
                        self._log(f"  Removed {len(matching_files)} workload files: {os.path.basename(file_paths['workload'])}.*.et", "info")
                
                # Remove simulation output files (all files with the config basename)
                # e.g., 4_8_2_1_1.seq_2048.batch_2048.log, .csv, etc.
                if 'output_pattern' in file_paths:
                    output_pattern = file_paths['output_pattern'] + "*"
                    matching_outputs = glob.glob(output_pattern)
                    for output_file in matching_outputs:
                        if os.path.exists(output_file):
                            try:
                                os.remove(output_file)
                                output_files_removed += 1
                            except Exception as e:
                                if self.verbose:
                                    self._log(f"Warning: Could not remove {output_file}: {e}", "warning")
                    
                    if self.verbose and matching_outputs:
                        self._log(f"  Removed {len(matching_outputs)} output files: {os.path.basename(file_paths['output_pattern'])}*", "info")
        
        if self.verbose:
            self._log(f"Cleanup complete: Removed {files_removed} workload files, {output_files_removed} output files", "info")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"budget={self.budget}, "
                f"init_samples={self.init_samples}, "
                f"evaluated={len(self.configs)})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        info = [
            f"{self.__class__.__name__}",
            f"Budget: {self.budget} evaluations",
            f"Initialization: {self.init_samples} samples",
            f"Evaluated: {len(self.configs)} configs",
        ]
        
        if self.best_config:
            info.append(f"Best score: {format_score(self.best_score)}")
        
        return "\n".join(info)
