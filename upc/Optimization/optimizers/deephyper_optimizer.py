"""DeepHyper Bayesian Optimization wrapper for AstraSim."""

import sys
import os
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
import time
import tempfile

# Add parent directory to path for imports
sys.path.append(os.environ['ASTRA_SIM_ROOT'] + '/upc/Optimization')
from ..core import BaseOptimizer
from ..helper import evaluate_config_worker, workload_generator

try:
    from deephyper.hpo import HpProblem, CBO
    from deephyper.evaluator import Evaluator
    DEEPHYPER_AVAILABLE = True
except ImportError:
    DEEPHYPER_AVAILABLE = False


def _deephyper_evaluate_wrapper(job, optimizer_state):
    """Wrapper for DeepHyper evaluation. Returns objective value or 'F' for failures."""
    from ..helper.config_utils import enrich_config_with_clusters
    
    config = dict(job.parameters)
    simulation_runner = optimizer_state['simulation_runner']
    objective = optimizer_state['objective']
    clusters = optimizer_state.get('clusters')
    
    if clusters and 'cluster' in config:
        config = enrich_config_with_clusters(config, clusters)
    
    try:
        returned_config, exec_time, is_oom, file_paths, metadata = evaluate_config_worker(
            config, simulation_runner
        )
        
        if exec_time is None:
            return "F"
        
        score = objective.compute(exec_time, is_oom, metadata, config)
        if score is None:
            return "F"
        
        if objective.minimize:
            objective_value = -score if score != float('inf') else -1e10
        else:
            objective_value = score if score != float('inf') else -1e10
        
        return objective_value
        
    except Exception as e:
        print(f"    ⚠️  Evaluation error: {e}")
        return "F" 


class DeepHyperOptimizer(BaseOptimizer):
    """Bayesian Optimization using DeepHyper's CBO."""
    
    def __init__(
        self,
        search_space,
        sampler,
        simulation_runner,
        objective,
        budget: int = 30,
        init_samples: int = 20,
        n_workers: int = 1,
        # Core CBO parameters
        random_state: Optional[int] = None,
        log_dir: Optional[str] = None,
        verbose: bool = True,
        stopper=None,
        checkpoint_history_to_csv: bool = True,
        solution_selection: Optional[str] = None,
        checkpoint_restart: bool = False,
        # Surrogate model parameters
        surrogate_model: str = "ET",
        surrogate_model_kwargs: Optional[Dict] = None,
        # Acquisition function parameters
        acq_func: str = "UCB",
        acq_func_kwargs: Optional[Dict] = None,
        acq_optimizer: str = "mixedga",
        acq_optimizer_kwargs: Optional[Dict] = None,
        # Multi-point strategy
        multi_point_strategy: str = "cl_max",
        # Initial points parameters
        n_initial_points: Optional[int] = None,
        initial_point_generator: str = "random",
        initial_points: Optional[list] = None,
        # Multi-objective optimization parameters
        moo_lower_bounds=None,
        moo_scalarization_strategy: str = "Chebyshev",
        moo_scalarization_weight=None,
        objective_scaler: str = "minmax",
        # Framework parameters
        save_dir: str = ".",
        keep_top_k: int = -1,
        profile_time: bool = False,
        evaluator_method: str = "process",
        # Additional kwargs
        problem_kwargs: Optional[Dict] = None,
        cbo_kwargs: Optional[Dict] = None
    ):
        """
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance
            simulation_runner: SimulationRunner instance
            objective: Objective instance
            budget: Total number of evaluations
            init_samples: Number of initial random samples (overrides n_initial_points)
            n_workers: Number of parallel workers
            
            # Core CBO parameters
            random_state: Random seed for reproducibility
            log_dir: Directory for DeepHyper logs (None = temp dir)
            verbose: Print progress
            stopper: Custom stopper for early termination
            checkpoint_history_to_csv: Save search history to CSV
            solution_selection: How to select best solution ("argmax_obs", "argmax_est")
            checkpoint_restart: Restart from checkpoint
            
            # Surrogate model parameters
            surrogate_model: Surrogate model ("RF", "ET", "GP", "DUMMY")
            surrogate_model_kwargs: Additional surrogate model arguments
            
            # Acquisition function parameters
            acq_func: Acquisition function ("UCB", "EI", "PI", "gp_hedge", "UCBd")
            acq_func_kwargs: Additional acquisition function arguments
            acq_optimizer: Acquisition optimizer ("mixedga", "sampling", "lbfgs", "auto")
            acq_optimizer_kwargs: Acquisition optimizer arguments (e.g., {"max_total_failures": -1})
            
            # Multi-point strategy
            multi_point_strategy: Strategy for parallel evaluations ("cl_max", "cl_min", "cl_mean", "qUCB")
            
            # Initial points parameters
            n_initial_points: Number of random initial samples (None = use init_samples)
            initial_point_generator: Initial point generation strategy ("random", "sobol", "halton", "hammersly", "lhs", "grid")
            initial_points: Pre-specified initial configurations
            
            # Multi-objective optimization parameters
            moo_lower_bounds: Lower bounds for multi-objective optimization
            moo_scalarization_strategy: Scalarization strategy ("Chebyshev", "Linear", "AugChebyshev")
            moo_scalarization_weight: Weights for scalarization
            objective_scaler: Objective scaler ("minmax", "standardize", "identity")
            
            # Framework parameters
            save_dir: Directory to save results
            keep_top_k: Keep top K results files (-1 = all, 0 = none)
            profile_time: Track detailed timing statistics
            evaluator_method: Parallel evaluation method ("process" or "thread")
            
            # Additional overrides
            problem_kwargs: Additional HpProblem arguments (advanced)
            cbo_kwargs: Additional CBO arguments (advanced, overrides above)
        """
        if not DEEPHYPER_AVAILABLE:
            raise ImportError("DeepHyper required: pip install deephyper")
        
        super().__init__(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=simulation_runner,
            objective=objective,
            budget=budget,
            init_samples=init_samples,
            verbose=verbose,
            save_dir=save_dir,
            keep_top_k=keep_top_k,
            profile_time=profile_time
        )
        
        # Framework parameters
        self.n_workers = max(1, n_workers)
        self.evaluator_method = evaluator_method
        
        # Core CBO parameters
        self.random_state = random_state
        self.stopper = stopper
        self.checkpoint_history_to_csv = checkpoint_history_to_csv
        self.solution_selection = solution_selection
        self.checkpoint_restart = checkpoint_restart
        
        # Surrogate model parameters
        self.surrogate_model = surrogate_model
        self.surrogate_model_kwargs = surrogate_model_kwargs or {}
        
        # Acquisition function parameters
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs or {}
        self.acq_optimizer = acq_optimizer
        self.acq_optimizer_kwargs = acq_optimizer_kwargs or {"max_total_failures": -1}
        
        # Multi-point strategy
        self.multi_point_strategy = multi_point_strategy
        
        # Initial points parameters
        # If initial_points not provided and we have constraints, sample from valid configs
        if initial_points is None and hasattr(search_space, 'design_space') and search_space.design_space:
            # Sample initial points from pre-computed valid configurations
            import random
            n_init = min(init_samples if n_initial_points is None else n_initial_points, len(search_space.design_space))
            self.initial_points = random.sample(search_space.design_space, n_init)
            # Set n_initial_points to match actual initial_points to avoid DeepHyper sampling more
            self.n_initial_points = len(self.initial_points)
            if self.verbose:
                print(f"✓ Sampled {n_init} initial points from valid configurations")
        else:
            self.initial_points = initial_points
            self.n_initial_points = n_initial_points if n_initial_points is not None else init_samples
        
        self.initial_point_generator = initial_point_generator
        
        # Multi-objective optimization parameters
        self.moo_lower_bounds = moo_lower_bounds
        self.moo_scalarization_strategy = moo_scalarization_strategy
        self.moo_scalarization_weight = moo_scalarization_weight
        self.objective_scaler = objective_scaler
        
        # Additional kwargs
        self.problem_kwargs = problem_kwargs or {}
        self.cbo_kwargs = cbo_kwargs or {}
        
        # Setup log directory
        if log_dir is None:
            self.log_dir = tempfile.mkdtemp(prefix="deephyper_")
        else:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
        
        # DeepHyper components
        self.hp_problem = None
        self.evaluator = None
        self.cbo = None
        self.deephyper_results = None
    
    def initialize(self) -> bool:
        """Initialize DeepHyper components (HpProblem, Evaluator, CBO)."""
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
            with self.time_stats.timer("hp_problem_creation"):
                self.hp_problem = self._create_hp_problem()
            if self.verbose:
                print(f"✓ Created HpProblem with {len(self.hp_problem.space)} hyperparameters")
            
            with self.time_stats.timer("evaluator_creation"):
                self.evaluator = self._create_evaluator()
            if self.verbose:
                print(f"✓ Created evaluator with {self.n_workers} workers ({self.evaluator_method} method)")
            
            with self.time_stats.timer("cbo_creation"):
                self.cbo = self._create_cbo()
            if self.verbose:
                print("✓ Created CBO optimizer\n")
            
            return True
            
        except Exception as e:
            self._log(f"Initialization failed: {e}", "error")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False
    
    def _create_hp_problem(self) -> HpProblem:
        """Convert SearchSpace to DeepHyper HpProblem with constraint function."""
        problem = HpProblem(**self.problem_kwargs)
        
        if self.search_space.design_space is None or len(self.search_space.design_space) == 0:
            raise ValueError("Search space design_space is empty or not built.")
        
        valid_configs = self.search_space.design_space
        
        if self.verbose:
            print(f"✓ Using {len(valid_configs)} valid configurations for reference")
        
        # Get all unique parameter values from valid configs
        param_names = list(valid_configs[0].keys())
        param_values_map = {name: [] for name in param_names}
        
        for config in valid_configs:
            for name in param_names:
                if config[name] not in param_values_map[name]:
                    param_values_map[name].append(config[name])
        
        # Sort parameter values for consistency
        for name in param_names:
            param_values_map[name] = sorted(param_values_map[name], key=lambda x: (x is None, x))
        
        # Add hyperparameters in alphabetical order (DeepHyper sorts parameters alphabetically)
        sorted_param_names = sorted(param_names)
        for param_name in sorted_param_names:
            unique_values = param_values_map[param_name]
            if unique_values:
                problem.add(unique_values, param_name, default_value=unique_values[0])
        

        def constraint_fn(s: pd.DataFrame):
            """Validate parallelism constraint: dp * mp * sp * pp = npu_count"""
            is_valid = np.ones(len(s), dtype=bool)
            
            for idx, row in s.iterrows():
                # Get NPU count for this specific cluster
                # DeepHyper prefixes parameter names with 'p:' in the DataFrame
                cluster_name = row['cluster']  # Access by column name with 'p:' prefix
                if cluster_name and hasattr(self.search_space, 'clusters'):
                    npu_count = self.search_space.clusters[cluster_name]['npu_count']
                else:
                    npu_count = self.search_space.num_npus
                    print("⚠️  Warning: 'cluster' not specified or clusters not defined; using total num_npus")
                
                # Check constraint - DataFrame columns are prefixed with 'p:'
                product = row['dp'] * row['mp'] * row['sp'] * row['pp']
                is_valid[idx] = (product == npu_count)
            
            # Return as pandas Series to avoid AttributeError in DeepHyper
            return pd.Series(is_valid, index=s.index)
        
        problem.set_constraint_fn(constraint_fn)
        
        if self.verbose:
            total_combinations = 1
            for values in param_values_map.values():
                total_combinations *= len(values)
        
        return problem
    
    def _create_evaluator(self) -> Evaluator:
        """Create DeepHyper evaluator for parallel execution."""
        from functools import partial
        
        optimizer_state = {
            'simulation_runner': self.simulation_runner,
            'objective': self.objective,
            'clusters': getattr(self.search_space, 'clusters', None),
            'verbose': self.verbose
        }
        
        eval_func = partial(_deephyper_evaluate_wrapper, optimizer_state=optimizer_state)
        evaluator = Evaluator.create(
            eval_func,
            method=self.evaluator_method,
            method_kwargs={"num_workers": self.n_workers}
        )
        
        return evaluator
    
    def _collect_results_from_deephyper(self):
        """Collect and process results from DeepHyper's output dataframe."""
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            return
        
        param_cols = [col for col in self.deephyper_results.columns if col.startswith('p:')]
        param_names = [col[2:] for col in param_cols]
        
        PENALTY_THRESHOLD = 1e9
        n_failed = 0
        n_success = 0
        n_infeasible = 0
        
        for idx, row in self.deephyper_results.iterrows():
            objective_value = row['objective']
            
            # Check if configuration is infeasible (constraint violation)
            if 'constraint' in row and not row['constraint']:
                n_infeasible += 1
                continue
            
            # Check for evaluation failures
            if isinstance(objective_value, str):
                if objective_value == 'F' or objective_value.startswith('F'):
                    n_failed += 1
                    continue
            
            config = {name: row[f'p:{name}'] for name in param_names}
            
            try:
                objective_value = float(objective_value)
            except (ValueError, TypeError):
                n_failed += 1
                continue
            
            score = -objective_value if self.objective.minimize else objective_value
            is_penalty = score >= PENALTY_THRESHOLD
            
            self.configs.append(config)
            self.scores.append(score)
            self.file_paths.append(self._reconstruct_file_paths(config))
            self.metadata.append(self._get_simulation_metadata())
            n_success += 1
            
            if self.verbose and n_success <= 10:
                config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
                print(f"  Iteration {n_success}: {config_str} | Score: {score:.4f}")
            
            if not is_penalty and self.objective.is_better(score, self.best_score):
                self.best_score = score
                self.best_config = config
                self.best_iteration = len(self.configs) - 1
                if self.verbose:
                    print(f"    🏆 NEW BEST: {score:.4f}")
            
            if self.keep_top_k >= 0:
                self._cleanup_files()
        
        if self.verbose:
            print(f"\nCollected {n_success} successful evaluations")
            print(f"Total DeepHyper evaluations: {len(self.deephyper_results)}")
            if n_infeasible > 0:
                print(f"  - Infeasible (constraint violations): {n_infeasible}")
            if n_failed > 0:
                print(f"  - Failed (simulation errors): {n_failed}")
    
    def _reconstruct_file_paths(self, config: Dict) -> Dict:
        """Reconstruct file paths from config for cleanup support."""
        sr = self.simulation_runner
        parallelism_parts = [str(config[key]) for key in ['dp', 'mp', 'sp', 'pp'] if key in config]
        if 'sharded' in config:
            parallelism_parts.append('1' if config['sharded'] else '0')
        config_str = '_'.join(parallelism_parts)
        
        try:
            import sys
            sys.path.insert(0, os.environ['ASTRA_SIM_ROOT'] + '/upc')
            model = workload_generator.Model(self.model_num)
            _, _, _, _, batch, _, seq, _, _ = model.get_model_params()
            filename_base = f"{config_str}.seq_{seq}.batch_{batch}"
            return {
                'workload': os.path.join(sr.workload_dir, filename_base),
                'output_pattern': os.path.join(sr.output_dir, filename_base)
            }
        except Exception:
            return {}
    
    def _get_simulation_metadata(self) -> Dict:
        """Get simulation metadata from simulation_runner."""
        sr = self.simulation_runner
        try:
            import sys
            sys.path.insert(0, os.environ['ASTRA_SIM_ROOT'] + '/upc')
            model = workload_generator.Model(self.model_num)
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
                'sim_type': sr.net_sim_config.get('sim_type', 'analytical_unaware'),
            }
            if hasattr(sr, 'system_metadata'):
                metadata.update(sr.system_metadata)
            return metadata
        except Exception:
            return {
                'model_name': sr.model_name,
                'model_num': sr.model_num,
                'num_npus': sr.num_npus,
                'sim_type': sr.net_sim_config.get('sim_type', 'analytical_unaware'),
            }
    
    def _create_cbo(self) -> CBO:
        """Create DeepHyper CBO instance."""
        cbo_args = {
            # Required
            "problem": self.hp_problem,
            # Core parameters
            "random_state": self.random_state,
            "log_dir": self.log_dir,
            "verbose": 1 if self.verbose else 0,
            "stopper": self.stopper,
            "checkpoint_history_to_csv": self.checkpoint_history_to_csv,
            "solution_selection": self.solution_selection,
            "checkpoint_restart": self.checkpoint_restart,
            # Surrogate model
            "surrogate_model": self.surrogate_model,
            "surrogate_model_kwargs": self.surrogate_model_kwargs,
            # Acquisition function
            "acq_func": self.acq_func,
            "acq_func_kwargs": self.acq_func_kwargs,
            "acq_optimizer": self.acq_optimizer,
            "acq_optimizer_kwargs": self.acq_optimizer_kwargs,
            # Multi-point strategy
            "multi_point_strategy": self.multi_point_strategy,
            # Initial points
            "n_initial_points": self.n_initial_points,
            "initial_point_generator": self.initial_point_generator,
            "initial_points": self.initial_points,
            # Multi-objective optimization
            "moo_lower_bounds": self.moo_lower_bounds,
            "moo_scalarization_strategy": self.moo_scalarization_strategy,
            "moo_scalarization_weight": self.moo_scalarization_weight,
            "objective_scaler": self.objective_scaler,
        }
        
        # Apply additional overrides from cbo_kwargs
        cbo_args.update(self.cbo_kwargs)
        
        return CBO(**cbo_args)
    
    def optimize_step(self) -> Tuple[Optional[Dict], Optional[float]]:
        """Not supported for DeepHyper (uses batch search instead)."""
        self._log("optimize_step() not supported", "warning")
        return None, None
    
    def run(self) -> Tuple[Optional[Dict], pd.DataFrame]:
        """Run Bayesian Optimization using DeepHyper CBO."""
        self.start_time = time.time()
        self.time_stats.start_total()
        
        with self.time_stats.timer("initialization"):
            if not self.initialize():
                return None, pd.DataFrame()
        
        try:
            if self.verbose:
                print("-"*70 + "\nRUNNING OPTIMIZATION\n" + "-"*70 + "\n")
            
            with self.time_stats.timer("cbo_search"):
                self.deephyper_results = self.cbo.search(
                    evaluator=self.evaluator,
                    max_evals=self.budget
                )
            
            if self.verbose:
                print("\n" + "-"*70 + "\nCOLLECTING RESULTS\n" + "-"*70)
            
            self._collect_results_from_deephyper()
            
            if self.keep_top_k >= 0:
                if self.verbose:
                    print("\n" + "-"*70 + "\nFINAL CLEANUP\n" + "-"*70)
                self._cleanup_files()
            
            if self.verbose:
                print("\n" + "-"*70 + "\nOPTIMIZATION COMPLETE\n" + "-"*70)
                self.print_summary()
            
            with self.time_stats.timer("save_results"):
                self.save_results()
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
