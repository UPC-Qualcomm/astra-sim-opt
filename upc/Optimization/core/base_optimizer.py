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
        verbose: bool = True,
        save_dir: str = "."
    ):
        """
        Initialize base optimizer.
        
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance for initial sampling
            simulation_runner: SimulationRunner instance
            budget: Total number of evaluations
            init_samples: Number of initial random samples
            verbose: Whether to print progress
            save_dir: Directory to save results
        """
        self.search_space = search_space
        self.sampler = sampler
        self.simulation_runner = simulation_runner
        self.budget = budget
        self.init_samples = init_samples
        self.verbose = verbose
        self.save_dir = save_dir
        
        # Result tracking
        self.configs: List[Tuple] = []  # Evaluated configurations
        self.scores: List[float] = []   # Execution times (lower is better)
        self.iteration_times: List[float] = []  # Time per iteration
        self.history: List[Dict] = []  # Detailed history
        
        # Best tracking
        self.best_config: Optional[Tuple] = None
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
    def optimize_step(self) -> Tuple[Optional[Tuple], Optional[float]]:
        """
        Execute one optimization iteration.
        
        Returns:
            (config, score) tuple if successful, (None, None) otherwise
        """
        pass
    
    @abstractmethod
    def run(self) -> Tuple[Optional[Tuple], pd.DataFrame]:
        """
        Run full optimization loop.
        
        Returns:
            (best_config, results_dataframe) tuple
        """
        pass
    
    def evaluate_config(self, config: Tuple, verbose: bool = False) -> Optional[float]:
        """
        Evaluate a single configuration.
        
        Args:
            config: Configuration tuple (dp, mp, sp, pp, sharded)
            verbose: Whether to print evaluation details
        
        Returns:
            Execution time in seconds, or None if evaluation failed
        """
        try:
            exec_time = self.simulation_runner.run_simulation(config)
            
            if exec_time is not None:
                # Record results
                self.configs.append(config)
                self.scores.append(exec_time)
                
                # Update best
                if exec_time < self.best_score:
                    self.best_score = exec_time
                    self.best_config = config
                    self.best_iteration = self.current_iteration
                    
                    if self.verbose and verbose:
                        print(f"    🏆 NEW BEST! Time: {exec_time:.2f}s")
                
                return exec_time
            else:
                if verbose:
                    print("    ⚠️  Evaluation failed")
                return None
                
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Error: {e}")
            return None
    
    def get_best_config(self) -> Tuple[Optional[Tuple], float]:
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
            DataFrame with columns: iteration, dp, mp, sp, pp, sharded, exec_time
        """
        if not self.configs:
            return pd.DataFrame()
        
        history = []
        for i, (config, score) in enumerate(zip(self.configs, self.scores)):
            dp, mp, sp, pp, sharded = config
            history.append({
                'iteration': i + 1,
                'dp': dp,
                'mp': mp,
                'sp': sp,
                'pp': pp,
                'sharded': sharded,
                'exec_time_seconds': score,
                'best_so_far': min(self.scores[:i+1])
            })
        
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
        
        # Statistics
        scores_array = np.array(self.scores)
        print("\n📊 STATISTICS:")
        print(f"   Total evaluations: {len(self.scores)}")
        print(f"   Best time: {scores_array.min():.2f}s")
        print(f"   Worst time: {scores_array.max():.2f}s")
        print(f"   Mean time: {scores_array.mean():.2f}s")
        print(f"   Std Dev: {scores_array.std():.2f}s")
        
        # Improvement
        if len(self.scores) > 1 and self.init_samples > 0:
            initial_best = min(self.scores[:self.init_samples])
            improvement = (initial_best - self.best_score) / initial_best * 100
            print(f"\n📈 IMPROVEMENT:")
            print(f"   Initial best: {initial_best:.2f}s")
            print(f"   Final best: {self.best_score:.2f}s")
            print(f"   Improvement: {improvement:.1f}%")
        
        # Best configuration
        if self.best_config:
            dp, mp, sp, pp, sharded = self.best_config
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   dp={dp}, mp={mp}, sp={sp}, pp={pp}, sharded={sharded}")
            print(f"   Execution time: {self.best_score:.2f}s")
            print(f"   Found at iteration: {self.best_iteration + 1}")
            
            # Configuration profile
            total_npus = dp * mp * sp * pp
            print(f"\n📋 CONFIGURATION PROFILE:")
            print(f"   Total NPUs: {total_npus}/{self.simulation_runner.num_npus}")
            print(f"   DP/MP ratio: {dp/mp:.2f}")
            print(f"   SP enabled: {'Yes' if sp > 1 else 'No'}")
            print(f"   PP enabled: {'Yes' if pp > 1 else 'No'}")
            print(f"   FSDP enabled: {'Yes' if sharded else 'No'}")
        
        # Timing
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"\n⏱️  TIMING:")
            print(f"   Total time: {elapsed:.1f}s")
            print(f"   Time per evaluation: {elapsed/len(self.scores):.1f}s")
    
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
            info.append(f"Best score: {self.best_score:.2f}s")
        
        return "\n".join(info)
