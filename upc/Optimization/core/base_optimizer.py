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
import glob

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
        cleanup_batch_size: int = -1,
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
            keep_top_k: Keep only top K results' files during cleanup (-1 = disable cleanup, 0 = keep none)
            cleanup_batch_size: Run cleanup every N successful evaluations (-1 = disable periodic cleanup).
                               Must be greater than keep_top_k when cleanup is enabled.
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
        self.cleanup_batch_size = cleanup_batch_size
        self.profile_time = profile_time

        if self.keep_top_k >= 0 and self.cleanup_batch_size > 0 and self.cleanup_batch_size <= self.keep_top_k:
            raise ValueError(
                f"cleanup_batch_size ({self.cleanup_batch_size}) must be greater than keep_top_k ({self.keep_top_k})"
            )
        
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
        self._next_cleanup_at = self.cleanup_batch_size if self._periodic_cleanup_enabled() else None
        self._cleaned_file_indices = set()
        self.cleanup_stats: Dict[str, int] = {
            'simulations_cleaned': 0,
            'files_deleted': 0,
        }
        self.cleanup_print_deleted_files: bool = False
        
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

                # Failed run that still produced files: clean immediately.
                if exec_time is None:
                    self._cleanup_single_simulation_files(file_paths, reason="failed")
                    if verbose:
                        print("    ⚠️  Evaluation failed")
                    return None
                
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

                # Conservative cleanup for high-cost runs
                was_killed = bool(metadata.get('was_killed', False)) if isinstance(metadata, dict) else False
                if was_killed or bool(is_oom):
                    reason = "killed" if was_killed else "oom"
                    if self._cleanup_single_simulation_files(file_paths, reason=reason):
                        self._cleaned_file_indices.add(len(self.file_paths) - 1)

                self._maybe_run_periodic_cleanup()
                
                return score
            else:
                if verbose:
                    print("    ⚠️  Evaluation failed")
                return None
                
        except Exception as e:
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

    def _periodic_cleanup_enabled(self) -> bool:
        """Return True when periodic cleanup is configured."""
        return self.keep_top_k >= 0 and self.cleanup_batch_size > 0 and self.cleanup_batch_size > self.keep_top_k

    def _maybe_run_periodic_cleanup(self, force: bool = False):
        """Run periodic cleanup when the configured successful-evaluation threshold is reached."""
        if not self._periodic_cleanup_enabled():
            return False

        record_count = len(self.scores)
        if record_count <= self.keep_top_k:
            return False

        if not force:
            if self._next_cleanup_at is None or record_count < self._next_cleanup_at:
                return False

        if self.verbose:
            self._log(
                f"Periodic cleanup: keeping top {self.keep_top_k} out of {record_count} successful evaluations",
                "info"
            )

        self.cleanup_records(
            scores=self.scores,
            file_paths=self.file_paths,
            keep_top_k=self.keep_top_k,
            minimize=self.objective.minimize,
            cleaned_indices=self._cleaned_file_indices,
            verbose=self.verbose,
            log_fn=self._log,
            print_deleted_files=self.cleanup_print_deleted_files,
            cleanup_counters=self.cleanup_stats,
        )

        if self.cleanup_batch_size > 0:
            self._next_cleanup_at = ((record_count // self.cleanup_batch_size) + 1) * self.cleanup_batch_size

        return True

    def set_cleanup_debug(self, print_deleted_files: bool = True):
        """Enable/disable detailed printing of deleted files during cleanup."""
        self.cleanup_print_deleted_files = bool(print_deleted_files)

    def get_cleanup_status(self) -> Dict[str, int]:
        """Return cleanup counters for verification/debugging."""
        return {
            'simulations_cleaned': int(self.cleanup_stats.get('simulations_cleaned', 0)),
            'files_deleted': int(self.cleanup_stats.get('files_deleted', 0)),
        }

    def _cleanup_single_simulation_files(self, file_paths: Optional[Dict[str, str]], reason: str = "failed") -> bool:
        """Immediately clean files generated by a single simulation."""
        if not file_paths:
            return False

        result = self.cleanup_path_bundle(
            tracked_paths=file_paths,
            verbose=self.verbose,
            log_fn=self._log,
            print_deleted_files=self.cleanup_print_deleted_files,
            reason=reason,
        )

        has_any_path = isinstance(file_paths, dict) and bool(file_paths)
        if has_any_path:
            self.cleanup_stats['simulations_cleaned'] += 1
        if result['total_removed'] > 0:
            self.cleanup_stats['files_deleted'] += result['total_removed']
        return has_any_path

    @staticmethod
    def cleanup_path_bundle(
        tracked_paths: Dict[str, str],
        verbose: bool = False,
        log_fn=None,
        print_deleted_files: bool = False,
        reason: str = "cleanup",
    ) -> Dict[str, Any]:
        """Delete all files associated with one simulation record."""
        def emit(message: str, level: str = "info"):
            if not verbose:
                return
            if log_fn is not None:
                log_fn(message, level)
            else:
                prefix = {
                    'info': '',
                    'warning': '⚠️  ',
                    'error': '❌ '
                }.get(level, '')
                print(f"{prefix}{message}")

        deleted_files: List[str] = []
        workload_files_removed = 0
        output_files_removed = 0

        workload_base = tracked_paths.get('workload') if isinstance(tracked_paths, dict) else None
        if workload_base:
            for workload_file in glob.glob(workload_base + ".*"):
                if os.path.exists(workload_file):
                    try:
                        os.remove(workload_file)
                        deleted_files.append(workload_file)
                        workload_files_removed += 1
                    except Exception as exc:
                        emit(f"Warning: Could not remove {workload_file}: {exc}", "warning")

        output_base = tracked_paths.get('output_pattern') if isinstance(tracked_paths, dict) else None
        if output_base:
            for output_file in glob.glob(output_base + "*"):
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                        deleted_files.append(output_file)
                        output_files_removed += 1
                    except Exception as exc:
                        emit(f"Warning: Could not remove {output_file}: {exc}", "warning")

        total_removed = workload_files_removed + output_files_removed
        if total_removed > 0 and verbose:
            emit(
                f"Immediate cleanup ({reason}): removed {total_removed} files "
                f"({workload_files_removed} workload + {output_files_removed} output)",
                "info",
            )
            if print_deleted_files:
                for path in deleted_files:
                    emit(f"  deleted: {path}", "info")

        return {
            'deleted_files': deleted_files,
            'workload_files_removed': workload_files_removed,
            'output_files_removed': output_files_removed,
            'total_removed': total_removed,
        }

    @staticmethod
    def cleanup_records(
        scores,
        file_paths,
        keep_top_k: int,
        minimize: bool = True,
        cleaned_indices=None,
        verbose: bool = False,
        log_fn=None,
        print_deleted_files: bool = False,
        cleanup_counters: Optional[Dict[str, int]] = None,
    ):
        """
        Remove tracked files for all non-top-K scored records.

        Args:
            scores: Sequence of recorded objective scores.
            file_paths: Sequence of tracked file-path dictionaries aligned with ``scores``.
            keep_top_k: Number of best-scoring records to preserve.
            minimize: Whether lower scores are better.
            cleaned_indices: Mutable set-like or dict-like object used to avoid repeated cleanup.
            verbose: Whether to emit cleanup logs.
            log_fn: Optional logger callable ``log_fn(message, level)``.
        """
        def emit(message: str, level: str = "info"):
            if not verbose:
                return
            if log_fn is not None:
                log_fn(message, level)
            else:
                prefix = {
                    'info': '',
                    'warning': '⚠️  ',
                    'error': '❌ '
                }.get(level, '')
                print(f"{prefix}{message}")

        def is_cleaned(idx: int) -> bool:
            if cleaned_indices is None:
                return False
            if hasattr(cleaned_indices, 'get'):
                return bool(cleaned_indices.get(idx, False))
            return idx in cleaned_indices

        def mark_cleaned(idx: int):
            if cleaned_indices is None:
                return
            if hasattr(cleaned_indices, '__setitem__'):
                cleaned_indices[idx] = True
            else:
                cleaned_indices.add(idx)

        if keep_top_k < 0 or len(scores) <= keep_top_k:
            return False

        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=not minimize,
        )
        top_k_set = set(sorted_indices[:keep_top_k])

        files_removed = 0
        output_files_removed = 0
        cleaned_records = 0

        def bump_counter(name: str, amount: int):
            if cleanup_counters is None or amount == 0:
                return
            if hasattr(cleanup_counters, 'get') and hasattr(cleanup_counters, '__setitem__'):
                cleanup_counters[name] = int(cleanup_counters.get(name, 0)) + int(amount)

        for idx, tracked_paths in enumerate(file_paths):
            if idx in top_k_set or not tracked_paths or is_cleaned(idx):
                continue

            result = BaseOptimizer.cleanup_path_bundle(
                tracked_paths=tracked_paths,
                verbose=verbose,
                log_fn=log_fn,
                print_deleted_files=print_deleted_files,
                reason="periodic_topk",
            )

            cleaned_records += 1
            if result['total_removed'] > 0:
                files_removed += result['workload_files_removed']
                output_files_removed += result['output_files_removed']

            mark_cleaned(idx)

        bump_counter('simulations_cleaned', cleaned_records)
        bump_counter('files_deleted', files_removed + output_files_removed)

        if verbose:
            emit(
                f"Cleanup complete: pruned {cleaned_records} records, removed {files_removed} workload files and {output_files_removed} output files",
                "info"
            )

            if cleanup_counters is not None:
                emit(
                    f"Cleanup totals so far: simulations_cleaned={int(cleanup_counters.get('simulations_cleaned', 0))}, "
                    f"files_deleted={int(cleanup_counters.get('files_deleted', 0))}",
                    "info",
                )

        return True
    
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
