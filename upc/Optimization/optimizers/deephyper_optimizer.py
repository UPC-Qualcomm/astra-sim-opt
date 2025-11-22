"""
DeepHyperOptimizer: Bayesian Optimization using the DeepHyper library.

Uses DeepHyper's CBO (Centralized Bayesian Optimization) to explore the design space.
DeepHyper provides a mature, well-tested BO implementation with a set of advanced features.
"""

import sys
import os
from typing import Tuple, Optional, Dict
import pandas as pd
import time
import tempfile

# Add parent directory to path for imports
sys.path.append(os.environ['ASTRA_SIM_ROOT'] + '/upc/Optimization')
from ..core import BaseOptimizer
from ..helper import evaluate_config_worker
import ConfigSpace as cs

try:
    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator
    import ConfigSpace as cs
    DEEPHYPER_AVAILABLE = True
except ImportError:
    DEEPHYPER_AVAILABLE = False


def _deephyper_evaluate_wrapper(job, optimizer_state):
    """
    DeepHyper evaluation wrapper using evaluate_config_worker.
    
    This adapts the existing evaluate_config_worker function for DeepHyper's interface.
    DeepHyper requires a function that takes a job and returns only a float (objective value).
    
    Uses runtime validation to reject invalid configurations.
    Invalid configs return 'F' but STILL COUNT toward max_evals budget in DeepHyper.
    
    IMPORTANT: DeepHyper's max_evals counts ALL evaluations (including failed).
    The filter_failures parameter only affects GP surrogate model, not budget counting.
    
    Args:
        job: DeepHyper job object with job.parameters dict
        optimizer_state: Dictionary with optimizer state (simulation_runner, objective, etc.)
        
    Returns:
        float: objective value for valid configs, 'F' for invalid (DeepHyper requirement)
    """
    config = dict(job.parameters)
    simulation_runner = optimizer_state['simulation_runner']
    objective = optimizer_state['objective']
    
    # Runtime validation to check constraint satisfaction
    # Invalid configs return 'F' which tells DeepHyper this eval FAILED
    # NOTE: Failed evals still COUNT toward max_evals budget!
    valid_configs_set = optimizer_state.get('valid_configs_set', None)
    if valid_configs_set is not None:
        param_names = optimizer_state.get('param_names', sorted(config.keys()))
        config_tuple = tuple(config[k] for k in param_names)
        if config_tuple not in valid_configs_set:
            return "F"  
    
    # Use the existing evaluate_config_worker function
    try:
        returned_config, exec_time, is_oom, file_paths, metadata = evaluate_config_worker(
            config, simulation_runner
        )
        
        # Check if evaluation failed (simulation error, not OOM)
        if exec_time is None:
            return "F"
        
        # Compute objective score
        score = objective.compute(exec_time, is_oom, metadata, config)
        
        # Handle None score from objective
        if score is None:
            return "F"
        
        # Check if this is an OOM case with penalty
        # We want OOM configs to COUNT toward budget (they provide learning signal)
        
        # DeepHyper maximizes by default, so negate if we're minimizing
        if objective.minimize:
            objective_value = -score if score != float('inf') else -1e10
        else:
            objective_value = score if score != float('inf') else -1e10
        
        return objective_value
        
    except Exception as e:
        print(f"    ⚠️  Evaluation error: {e}")
        return "F" 


class DeepHyperOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using DeepHyper's CBO (Centralized Bayesian Optimization).

    Example:
        from core.search_space_builder import create_search_space
        from core.sampler import RandomSampler
        from core.simulation_runner import SimulationRunner
        
        search_space = create_search_space(
            "search_space/parallelism_strategy_params.json",
            include_categories=['parallelism_strategy']
        )
        sampler = RandomSampler(seed=42)
        sim_runner = SimulationRunner(40, "GPT_40B", 64, "FoldedClos")
        
        optimizer = DeepHyperOptimizer(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=sim_runner,
            budget=100,
            init_samples=20,
            n_workers=4,
            acq_func="UCB",  # or "EI", "PI", "gp_hedge"
            random_state=42
        )
        
        best_config, results_df = optimizer.run()
    """
    
    def __init__(
        self,
        search_space,
        sampler,
        simulation_runner,
        budget: int = 30,
        init_samples: int = 20,
        n_workers: int = 1,
        acq_func: str = "UCB",
        acq_optimizer: str = "auto",
        filter_duplicates: bool = True,
        random_state: Optional[int] = None,
        log_dir: Optional[str] = None,
        verbose: bool = True,
        save_dir: str = ".",
        keep_top_k: int = -1,
        profile_time: bool = False,
        evaluator_method: str = "process",
        filter_failures: str = "mean",
        max_total_failures: int = -1,
        **cbo_kwargs
    ):
        """
        Initialize DeepHyper Bayesian Optimizer.
        
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance (used for fallback if needed)
            simulation_runner: SimulationRunner instance
            budget: Total number of evaluations
            init_samples: Number of initial random samples
            n_workers: Number of parallel workers (1 = sequential, >1 = parallel)
            acq_func: Acquisition function ("UCB", "EI", "PI", "gp_hedge")
            acq_optimizer: Strategy to optimize acquisition ("sampling", "lbfgs", "auto")
            filter_duplicates: Whether to filter duplicate configurations
            random_state: Random seed for reproducibility
            log_dir: Directory for DeepHyper logs (default: temporary directory)
            verbose: Whether to print progress
            save_dir: Directory to save results
            keep_top_k: Keep only top K results' files (-1 = keep all, 0 = keep none)
            profile_time: Whether to track and print detailed time statistics
            evaluator_method: Method for parallel evaluation ("process" or "thread")
            filter_failures: The way to deal with failures
            max_total_failures: The number of failures to accept
            **cbo_kwargs: Additional keyword arguments passed to CBO constructor
        """
        if not DEEPHYPER_AVAILABLE:
            raise ImportError(
                "DeepHyper is required for DeepHyperOptimizer. "
                "Install with: pip install deephyper"
            )
        
        super().__init__(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=simulation_runner,
            budget=budget,
            init_samples=init_samples,
            verbose=verbose,
            save_dir=save_dir,
            keep_top_k=keep_top_k,
            profile_time=profile_time
        )
        
        self.n_workers = max(1, n_workers)
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.filter_duplicates = filter_duplicates
        self.random_state = random_state
        self.evaluator_method = evaluator_method
        self.filter_failures = filter_failures
        self.max_total_failures = max_total_failures
        self.cbo_kwargs = cbo_kwargs
        
        # Setup log directory
        if log_dir is None:
            self.log_dir = tempfile.mkdtemp(prefix="deephyper_")
        else:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
        
        # DeepHyper components (initialized in initialize())
        self.hp_problem = None
        self.evaluator = None
        self.cbo = None
        
        # Track DeepHyper results
        self.deephyper_results = None
        
        # Evaluation callback for result collection
        self._eval_callback_data = []
    
    def initialize(self) -> bool:
        """
        Initialize DeepHyper components.
        
        1. Convert SearchSpace to DeepHyper HpProblem
        2. Create evaluation function wrapper
        3. Setup parallel evaluator
        4. Create CBO instance
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.verbose:
            print("\n" + "="*70)
            print("DEEPHYPER BAYESIAN OPTIMIZATION")
            print("="*70)
            print(f"Model: {self.simulation_runner.model_name}")
            print(f"NPUs: {self.simulation_runner.num_npus}")
            print(f"Network: {self.simulation_runner.network_name}")
            print(f"Budget: {self.budget} evaluations")
            print(f"Initial samples: {self.init_samples}")
            print(f"Workers: {self.n_workers}")
            print(f"Acquisition function: {self.acq_func}")
            print(f"Design space: {self.search_space.get_design_space_size()} configurations")
            print(f"Log directory: {self.log_dir}")
            print("="*70 + "\n")
        
        try:
            # 1. Create HpProblem from SearchSpace
            with self.time_stats.timer("hp_problem_creation"):
                self.hp_problem = self._create_hp_problem()
            
            if self.verbose:
                print(f"✓ Created HpProblem with {len(self.hp_problem.space)} hyperparameters")
            
            # 2. Create evaluator for parallel execution
            with self.time_stats.timer("evaluator_creation"):
                self.evaluator = self._create_evaluator()
            
            if self.verbose:
                print(f"✓ Created evaluator with {self.n_workers} workers ({self.evaluator_method} method)")
            
            # 3. Create CBO instance
            with self.time_stats.timer("cbo_creation"):
                self.cbo = self._create_cbo()
            
            if self.verbose:
                print("✓ Created CBO optimizer")
                print()
            
            return True
            
        except Exception as e:
            self._log(f"Initialization failed: {e}", "error")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False
    
    def _create_hp_problem(self) -> HpProblem:
        """
        Convert SearchSpace to DeepHyper HpProblem.
        
        Strategy for handling constraints:
        1. Add individual parameters as categorical hyperparameters
        2. Store valid configurations for runtime validation
        3. Use penalty returns for invalid configs during optimization
        
        This approach is efficient because:
        - ConfigSpace Cartesian product would create exponential forbidden clauses
        - Runtime validation is fast (O(1) set lookup)
        - GP can still learn from the penalty landscape
        
        Returns:
            HpProblem instance
        """
        problem = HpProblem()
        
        # Check if design space is built (contains pre-filtered valid configurations)
        if self.search_space.design_space is None or len(self.search_space.design_space) == 0:
            raise ValueError(
                "Search space design_space is empty or not built. "
            )
        
        # Get all valid configurations (already constraint-filtered)
        valid_configs = self.search_space.design_space
        
        if self.verbose:
            print(f"✓ Using {len(valid_configs)} pre-filtered valid configurations from search space")
            if self.search_space.constraint_strings:
                print("  Applied constraints:")
                for constraint in self.search_space.constraint_strings:
                    print(f"    - {constraint}")
        
        # Extract parameter names and values
        param_names = list(valid_configs[0].keys())
        param_values_map = {name: [] for name in param_names}
        
        for config in valid_configs:
            for name in param_names:
                if config[name] not in param_values_map[name]:
                    param_values_map[name].append(config[name])
        
        # Add hyperparameters with default values
        for param_name in param_names:
            unique_values = param_values_map[param_name]
            if not unique_values:
                continue
            
            # Use first value as default
            default_value = unique_values[0]
            
            # Add as categorical parameter
            problem.add_hyperparameter(
                unique_values,
                param_name,
                default_value=default_value
            )
        
        # Store valid configurations for runtime validation
        # This is much more efficient than adding exponential forbidden clauses
        valid_tuples = set()
        for config in valid_configs:
            config_tuple = tuple(config[name] for name in sorted(param_names))
            valid_tuples.add(config_tuple)
        
        self._valid_configs_set = valid_tuples
        self._param_names = sorted(param_names)
        
        if self.verbose:
            print(f"✓ Exposed {len(param_names)} parameters to GP for learning")
            print(f"✓ Using runtime validation with {len(valid_tuples)} valid configurations")
            
            # Calculate how many forbidden clauses would be needed
            total_combinations = 1
            for values in param_values_map.values():
                total_combinations *= len(values)
            invalid_count = total_combinations - len(valid_tuples)
            print(f"  (Avoided adding {invalid_count} forbidden clauses to ConfigSpace)")
            print("  ⚠️  NOTE: Invalid configs still count toward max_evals budget in DeepHyper")
            print(f"  ⚠️  To get {self.budget} valid evals, may need 2-3x higher budget")
        
        return problem
    
    def _create_evaluator(self) -> Evaluator:
        """
        Create DeepHyper evaluator for parallel execution.
        
        Returns:
            Evaluator instance
        """
        # Create a wrapper that passes optimizer state
        from functools import partial
        
        optimizer_state = {
            'simulation_runner': self.simulation_runner,
            'objective': self.objective,
            'valid_configs_set': getattr(self, '_valid_configs_set', None),
            'param_names': getattr(self, '_param_names', None),
            'verbose': self.verbose
        }
        
        # Use partial to bind optimizer_state to the module-level function
        eval_func = partial(_deephyper_evaluate_wrapper, optimizer_state=optimizer_state)
        
        # Create evaluator
        evaluator = Evaluator.create(
            eval_func,
            method=self.evaluator_method,
            method_kwargs={
                "num_workers": self.n_workers,
            }
        )
        
        return evaluator
    
    def _collect_results_from_deephyper(self):
        """
        Collect results from DeepHyper's results dataframe.
        
        DeepHyper returns a dataframe with all evaluations. We need to
        extract the configurations, scores, and update our tracking.
        
        """
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            return
        
        # Get parameter names
        param_cols = [col for col in self.deephyper_results.columns 
                     if col.startswith('p:')]
        param_names = [col[2:] for col in param_cols]  # Remove 'p:' prefix
        
        PENALTY_THRESHOLD = 1e9  # Any score >= 1e9 is considered a penalty (OOM)
        
        # Track progress for logging
        n_failed = 0
        n_success = 0
        
        # Extract configurations and scores
        # Note: Failed configs (constraint violations) have objective='F'
        for idx, row in self.deephyper_results.iterrows():
            objective_value = row['objective']
            
            # Skip failed evaluations (constraint violations, simulation errors)
            # These show up as 'F' or string starting with 'F' in the objective column
            # They already counted toward DeepHyper's max_evals, but we don't learn from them
            if isinstance(objective_value, str):
                if objective_value == 'F' or objective_value.startswith('F'):
                    n_failed += 1
                    continue  # Skip for learning, but they consumed budget
            
            # Build config dict from parameters
            config = {name: row[f'p:{name}'] for name in param_names}
            
            # Convert objective_value to float
            try:
                objective_value = float(objective_value)
            except (ValueError, TypeError):
                n_failed += 1
                continue
            
            # Convert back to score (undo negation if we minimized)
            if self.objective.minimize:
                score = -objective_value
            else:
                score = objective_value
            
            # Check if this is a penalty score
            is_penalty = score >= PENALTY_THRESHOLD
            
            # Store results (including OOM configs - they count toward budget and provide info)
            self.configs.append(config)
            self.scores.append(score)
            
            # Reconstruct file paths from config for cleanup support
            # TODO: This is not working yet
            file_paths = self._reconstruct_file_paths(config)
            self.file_paths.append(file_paths)
        
            # Reconstruct metadata from simulation_runner
            metadata = self._get_simulation_metadata()
            self.metadata.append(metadata)
            n_success += 1
            
            # Log progress
            if self.verbose and n_success <= 10:
                config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
                print(f"  Iteration {n_success}: {config_str}")
                print(f"    Score: {score:.4f}")
            
            # Update best ONLY if not a penalty score
            if not is_penalty and self.objective.is_better(score, self.best_score):
                self.best_score = score
                self.best_config = config
                self.best_iteration = len(self.configs) - 1
                
                if self.verbose:
                    print(f"    🏆 NEW BEST: {score:.4f}")
                    
            # Cleanup files if keep_top_k is enabled
            # TODO: This is not working yet
            if self.keep_top_k >= 0:
                self._cleanup_files()
        
        if self.verbose:
            print(f"\nCollected {n_success} successful evaluations ({n_failed} failed/invalid)")
            print(f"Total DeepHyper evaluations attempted: {len(self.deephyper_results)}")
            if n_failed > 0:
                failed_pct = 100.0 * n_failed / len(self.deephyper_results)
                print(f"Failed rate: {failed_pct:.1f}% (due to constraint violations)")
    
    def _reconstruct_file_paths(self, config: Dict) -> Dict:
        """
        Reconstruct file paths from configuration for cleanup support.
        
        Since DeepHyper evaluations happen in separate processes, we don't get
        file paths back. We reconstruct them here based on the naming convention
        used by SimulationRunner.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with 'workload' and 'output_pattern' keys
        """
        # TODO: This is not working yet
        sr = self.simulation_runner
        
        # Build config string from parallelism parameters
        # Format: dp_mp_sp_pp_sharded
        parallelism_parts = []
        for key in ['dp', 'mp', 'sp', 'pp']:
            if key in config:
                parallelism_parts.append(str(config[key]))
        
        if 'sharded' in config:
            parallelism_parts.append('1' if config['sharded'] else '0')
        
        config_str = '_'.join(parallelism_parts)
        
        # Get model parameters for seq and batch
        try:
            import sys
            sys.path.insert(0, os.environ['ASTRA_SIM_ROOT'] + '/upc')
            from Model import Model
            model = Model(sr.model_num)
            _, _, _, _, batch, _, seq, _, _ = model.get_model_params()
            
            # Build full filename pattern
            # Format: dp_mp_sp_pp_sharded.seq_SEQ.batch_BATCH
            filename_base = f"{config_str}.seq_{seq}.batch_{batch}"
            
            # Workload path
            workload_dir = sr.workload_dir
            workload_path = os.path.join(workload_dir, filename_base)
            
            # Output pattern
            output_dir = sr.output_dir
            output_pattern = os.path.join(output_dir, filename_base)
            
            return {
                'workload': workload_path,
                'output_pattern': output_pattern
            }
            
        except Exception:
            # If reconstruction fails, return empty dict (no cleanup for this config)
            return {}
    
    def _get_simulation_metadata(self) -> Dict:
        """
        Get simulation metadata from simulation_runner.
        
        Since DeepHyper evaluations happen in separate processes,
        we reconstruct the metadata here for consistency with other optimizers.
        
        Returns:
            Dictionary with simulation metadata
        """
        # TODO: Not required, to update or delete
        sr = self.simulation_runner
        
        # Get model parameters
        try:
            # Import Model from correct path
            import sys
            sys.path.insert(0, os.environ['ASTRA_SIM_ROOT'] + '/upc')
            from Model import Model
            
            model = Model(sr.model_num)
            din, dout, dmodel, dff, batch, micro_batch, seq, head, num_stacks = model.get_model_params()
            
            metadata = {
                'model_name': sr.model_name,
                'model_num': sr.model_num,
                'vocab_size_in': din,
                'vocab_size_out': dout,
                'hidden_size': dmodel,
                'ffn_hidden_size': dff,
                'batch_size': batch,
                'sequence_length': seq,
                'num_attention_heads': head,
                'num_layers': num_stacks,
                'num_npus': sr.num_npus,
                'sim_type': sr.sim_type,
            }
            
            # Add system metadata from search space
            if hasattr(sr, 'system_metadata'):
                metadata.update(sr.system_metadata)
            
            # Add network metadata
            # TODO: Leave this as place holder
            #metadata.update({
            #    'net_bandwidth_l0': None,
            #    'net_bandwidth_l1': None,
            #    'net_bandwidth_unit': 'GB/s',
            #    'net_header_size': 48,
            #    'net_latency_l0': 0.0,
            #    'net_latency_l1': 0.0,
            #    'net_npus_count_l0': None,
            #    'net_npus_count_l1': None,
            #    'net_packet_size': 1500,
            #    'net_topology_l0': 'Switch',
            #    'net_topology_l1': 'Switch',
            #    'mem_memory-type': 'NO_MEMORY_EXPANSION',
            #})
            
            return metadata
            
        except Exception:
            # Fallback to minimal metadata if reconstruction fails
            return {
                'model_name': sr.model_name,
                'model_num': sr.model_num,
                'num_npus': sr.num_npus,
                'sim_type': sr.sim_type,
            }
    
    def _create_cbo(self) -> CBO:
        """
        Create DeepHyper CBO (Centralized Bayesian Optimization) instance.
        
        Returns:
            CBO instance
        """
        # Prepare CBO arguments
        cbo_args = {
            "problem": self.hp_problem,
            "evaluator": self.evaluator,
            "log_dir": self.log_dir,
            "random_state": self.random_state,
            "acq_func": self.acq_func,
            "acq_optimizer": self.acq_optimizer,
            "filter_duplicated": self.filter_duplicates,
            "n_initial_points": self.init_samples,
            "verbose": 1 if self.verbose else 0,
            "filter_failures": self.filter_failures,
            "max_total_failures": self.max_total_failures, 
        }
        
        # Add any additional kwargs
        cbo_args.update(self.cbo_kwargs)
        
        # Create CBO
        cbo = CBO(**cbo_args)
        
        return cbo
    
    def optimize_step(self) -> Tuple[Optional[Dict], Optional[float]]:
        """
        Execute one optimization iteration.
        
        Note: This method is not typically used with DeepHyper,

        Returns:
            (config, score) tuple if successful, (None, None) otherwise
        """
        self._log("optimize_step() not supported for DeepHyperOptimizer", "warning")
        return None, None
    
    def run(self) -> Tuple[Optional[Dict], pd.DataFrame]:
        """
        Run full Bayesian Optimization using DeepHyper CBO.
        
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
            # Run CBO optimization
            if self.verbose:
                print("-"*70)
                print("RUNNING OPTIMIZATION")
                print("-"*70)
                print()
            
            with self.time_stats.timer("cbo_search"):
                self.deephyper_results = self.cbo.search(max_evals=self.budget)
            
            if self.verbose:
                print("\n" + "-"*70)
                print("COLLECTING RESULTS")
                print("-"*70)
            
            # Collect results from DeepHyper output
            self._collect_results_from_deephyper()
            
            # Final cleanup to keep only top K results
            if self.keep_top_k >= 0:
                if self.verbose:
                    print("\n" + "-"*70)
                    print("FINAL CLEANUP")
                    print("-"*70)
                self._cleanup_files()

            
            if self.verbose:
                print("\n" + "-"*70)
                print("OPTIMIZATION COMPLETE")
                print("-"*70)
            
            # Print summary
            if self.verbose:
                self.print_summary()
            
            # Save results
            with self.time_stats.timer("save_results"):
                self.save_results()
                
                # Also save DeepHyper's native results
                dh_results_path = os.path.join(self.save_dir, "deephyper_results.csv")
                self.deephyper_results.to_csv(dh_results_path, index=False)
                if self.verbose:
                    print(f"✓ DeepHyper results saved to: {dh_results_path}")
            
            self.time_stats.end_total()
            return self.best_config, self.get_history()
            
        except KeyboardInterrupt:
            self._log("\n\nOptimization interrupted by user", "warning")
            self._log("Saving intermediate results...", "info")
            self.save_results()
            self.time_stats.end_total()
            return self.best_config, self.get_history()
        
        except Exception as e:
            self._log(f"Optimization failed: {e}", "error")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return None, pd.DataFrame()
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"DeepHyperOptimizer(budget={self.budget}, "
                f"init_samples={self.init_samples}, "
                f"n_workers={self.n_workers}, "
                f"acq_func='{self.acq_func}', "
                f"evaluated={len(self.configs)})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        info = [
            "DeepHyperOptimizer",
            f"Budget: {self.budget} evaluations",
            f"Initialization: {self.init_samples} samples",
            f"Workers: {self.n_workers}",
            f"Acquisition: {self.acq_func}",
            f"Evaluated: {len(self.configs)} configs",
        ]
        
        if self.best_config:
            info.append(f"Best score: {self.best_score:.2f}")
        
        return "\n".join(info)
