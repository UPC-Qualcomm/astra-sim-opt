"""DeepHyper Bayesian Optimization wrapper for AstraSim."""

import sys
import os
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
import time
import tempfile
import matplotlib.pyplot as plt
import json
import yaml

# Add parent directory to path for imports
sys.path.append(os.environ['ASTRA_SIM_ROOT'] + '/upc/Optimization')
from ..core import BaseOptimizer, ArtifactCleanupManager
from ..core.simulation_tracker import SimulationTracker
from ..helper import evaluate_config_worker, workload_generator

try:
    from deephyper.hpo import HpProblem, CBO, RandomSearch
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
    
    # Create cache key BEFORE enriching config (so it matches the DataFrame params)
    config_key = tuple(config.items())
    
    if clusters and 'cluster' in config:
        config = enrich_config_with_clusters(config, clusters)

    def _bump_cleanup_counter(name: str, amount: int):
        cleanup_state = optimizer_state.get('periodic_cleanup')
        if cleanup_state is None or amount == 0:
            return
        counters = cleanup_state.get('cleanup_counters')
        if counters is None:
            return
        counters[name] = int(counters.get(name, 0)) + int(amount)
    
    try:
        returned_config, exec_time, is_oom, file_paths, metadata = evaluate_config_worker(
            config, simulation_runner
        )

        cleanup_reason = ArtifactCleanupManager.get_immediate_cleanup_reason(exec_time, is_oom, metadata)
        was_killed = cleanup_reason == "killed"

        # Failed evaluation with generated files: clean immediately.
        if exec_time is None:
            cleanup_result = ArtifactCleanupManager.cleanup_path_bundle(
                tracked_paths=file_paths,
                verbose=optimizer_state.get('periodic_cleanup', {}).get('verbose', False) if optimizer_state.get('periodic_cleanup') is not None else False,
                print_deleted_files=optimizer_state.get('periodic_cleanup', {}).get('print_deleted_files', False) if optimizer_state.get('periodic_cleanup') is not None else False,
                reason=cleanup_reason,
            )
            if isinstance(file_paths, dict) and file_paths:
                _bump_cleanup_counter('simulations_cleaned', 1)
            if cleanup_result['total_removed'] > 0:
                _bump_cleanup_counter('files_deleted', cleanup_result['total_removed'])
            return "F"
        
        # Update tracker threshold if exec_time improved
        tracker = optimizer_state.get('tracker')
        if tracker and exec_time:
            tracker.update_threshold(exec_time)
        
        # Cache exec_time and config files for enrichment
        if 'extra_data_cache' in optimizer_state:
            config_files = {}
            for key in ['system_config', 'network_config', 'memory_config']:
                path = file_paths.get(key)
                if path:
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f) if path.endswith('.json') else yaml.safe_load(f)
                        config_files[key] = json.dumps(data)
                    except Exception as e:
                        print(f"⚠️  Failed to read {key}: {e}")
                        config_files[key] = None
            
            optimizer_state['extra_data_cache'][config_key] = {
                'exec_time': exec_time,
                'config_files': config_files,
                'was_killed': bool(metadata.get('was_killed', False)) if isinstance(metadata, dict) else False,
            }
            print(f"✓ Cached data for config_key with {len(config_files)} files, exec_time={exec_time}")
        
        score = objective.compute(exec_time, is_oom, metadata, config)
        if score is None:
            return "F"

        cleanup_state = optimizer_state.get('periodic_cleanup')
        if cleanup_state is not None:
            with cleanup_state['lock']:
                cleanup_state['scores'].append(score)
                cleanup_state['file_paths'].append(dict(file_paths))
                current_idx = len(cleanup_state['scores']) - 1

                # Immediate cleanup for killed/OOM runs.
                if cleanup_reason in {"killed", "oom"}:
                    cleanup_result = ArtifactCleanupManager.cleanup_path_bundle(
                        tracked_paths=dict(file_paths),
                        verbose=cleanup_state.get('verbose', False),
                        print_deleted_files=cleanup_state.get('print_deleted_files', False),
                        reason=cleanup_reason,
                    )
                    if isinstance(file_paths, dict) and file_paths:
                        cleanup_state['cleaned_indices'][current_idx] = True
                        _bump_cleanup_counter('simulations_cleaned', 1)
                    if cleanup_result['total_removed'] > 0:
                        _bump_cleanup_counter('files_deleted', cleanup_result['total_removed'])

                ArtifactCleanupManager.run_periodic_cleanup_for_state(
                    cleanup_state=cleanup_state,
                    minimize=objective.minimize,
                    force=False,
                )
        
        # Handle multi-objective returns (tuple of scores)
        if isinstance(score, (tuple, list)):
            objective_values = []
            for s in score:
                if objective.minimize:
                    obj_val = -s if s != float('inf') else -1e10
                else:
                    obj_val = s if s != float('inf') else -1e10
                objective_values.append(obj_val)
            return tuple(objective_values)
        else:
            # Single objective
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
        # Search type selection
        search_type: str = "cbo",  # "cbo" or "random"
        # Core CBO parameters
        random_state: Optional[int] = 42,
        log_dir: Optional[str] = None,
        verbose: bool = True,
        stopper=None,
        checkpoint_history_to_csv: bool = True,
        solution_selection: Optional[str] = None,
        checkpoint_restart: bool = False,
        # Surrogate model parameters (CBO only)
        surrogate_model: str = "ET",
        surrogate_model_kwargs: Optional[Dict] = None,
        # Acquisition function parameters (CBO only)
        acq_func: str = "UCB",
        acq_func_kwargs: Optional[Dict] = None ,
        acq_optimizer: str = "mixedga",
        acq_optimizer_kwargs: Optional[Dict] = None,
        # Multi-point strategy (CBO only)
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
        save_dir: str = "./experiments",
        keep_top_k: int = -1,
        cleanup_batch_size: int = -1,
        profile_time: bool = False,
        evaluator_method: str = "process",
        results_filename: Optional[str] = None,
        # Simulation tracking
        enable_tracker: bool = True,
        tracker_kill_multiplier: float = 1.5,
        tracker_initial_threshold: float = 1e15,
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
            
            # Search type
            search_type: Type of search ("cbo" for Bayesian Optimization, "random" for Random Search)
            
            # Core CBO parameters
            random_state: Random seed for reproducibility
            log_dir: Directory for DeepHyper logs (None = temp dir)
            verbose: Print progress
            stopper: Custom stopper for early termination
            checkpoint_history_to_csv: Save search history to CSV
            solution_selection: How to select best solution ("argmax_obs", "argmax_est")
            checkpoint_restart: Restart from checkpoint
            
            # Surrogate model parameters (CBO only)
            surrogate_model: Surrogate model ("RF", "ET", "GP", "DUMMY")
            surrogate_model_kwargs: Additional surrogate model arguments
            
            # Acquisition function parameters (CBO only)
            acq_func: Acquisition function ("UCB", "EI", "PI", "gp_hedge", "UCBd")
            acq_func_kwargs: Additional acquisition function arguments
            acq_optimizer: Acquisition optimizer ("mixedga", "sampling", "lbfgs", "auto")
            acq_optimizer_kwargs: Acquisition optimizer arguments (e.g., {"max_total_failures": -1})
            
            # Multi-point strategy (CBO only)
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
            keep_top_k: Keep top K results files during periodic cleanup (-1 = disable cleanup)
            cleanup_batch_size: Run cleanup every N successful evaluations (-1 = disable periodic cleanup)
            profile_time: Track detailed timing statistics
            evaluator_method: Parallel evaluation method ("process" or "thread")
            
            # Simulation tracking
            enable_tracker: Enable early termination of slow simulations (default True)
            tracker_kill_multiplier: Kill simulations exceeding threshold * multiplier (default 1.5)
            tracker_initial_threshold: Initial threshold in cycles (default 1e15)
            
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
            cleanup_batch_size=cleanup_batch_size,
            profile_time=profile_time
        )
        
        # Framework parameters
        self.n_workers = max(1, n_workers)
        self.evaluator_method = evaluator_method
        self.search_type = search_type.lower()
        
        if self.search_type not in ["cbo", "random"]:
            raise ValueError(f"search_type must be 'cbo' or 'random', got '{search_type}'")
        
        # Core CBO parameters
        self.random_state = random_state
        self.stopper = stopper
        self.checkpoint_history_to_csv = checkpoint_history_to_csv
        self.solution_selection = solution_selection
        self.checkpoint_restart = checkpoint_restart
        
        # Surrogate model parameters (CBO only)
        self.surrogate_model = surrogate_model
        self.surrogate_model_kwargs = surrogate_model_kwargs or {}
        
        # Acquisition function parameters (CBO only)
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs or {}
        self.acq_optimizer = acq_optimizer
        self.acq_optimizer_kwargs = acq_optimizer_kwargs or {"max_total_failures": -1}
        
        # Multi-point strategy (CBO only)
        self.multi_point_strategy = multi_point_strategy
        
        # Initial points parameters
        # If initial_points not provided and we have constraints, sample from valid configs
        if initial_points is None and hasattr(search_space, 'design_space') and search_space.design_space:
            # Sample initial points from pre-computed valid configurations
            import random as py_random
            # Seed Python's random module for reproducibility
            if random_state is not None:
                py_random.seed(random_state)
            
            # For RandomSearch, don't provide initial_points - let it sample randomly
            if search_type.lower() == "random":
                self.initial_points = None
                self.n_initial_points = 0
                if self.verbose:
                    print(f"✓ Random search will sample from {len(search_space.design_space)} valid configurations")
            else:
                # Sample initial points for CBO
                n_init = min(init_samples if n_initial_points is None else n_initial_points, len(search_space.design_space))
                self.initial_points = py_random.sample(search_space.design_space, n_init)
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
        
        # Results filename - use model name if not specified
        if results_filename is None:
            model_name = getattr(simulation_runner, 'model_name', 'deephyper')
            self.results_filename = f"deephyper_results_{model_name}.csv"
        else:
            self.results_filename = results_filename
        
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
        # Use Manager dict for multiprocess-safe cache sharing
        from multiprocessing import Manager
        self._manager = Manager()
        self.extra_data_cache = self._manager.dict()
        self.periodic_cleanup_state = None
        if self._periodic_cleanup_enabled():
            self.periodic_cleanup_state = {
                'scores': self._manager.list(),
                'file_paths': self._manager.list(),
                'cleaned_indices': self._manager.dict(),
                'control': self._manager.dict({'next_cleanup_at': self.cleanup_batch_size}),
                'cleanup_counters': self._manager.dict({'simulations_cleaned': 0, 'files_deleted': 0}),
                'lock': self._manager.Lock(),
                'keep_top_k': self.keep_top_k,
                'cleanup_batch_size': self.cleanup_batch_size,
                'verbose': self.verbose,
                'print_deleted_files': self.cleanup_print_deleted_files,
            }
        # Note: self.file_paths is already initialized as [] in BaseOptimizer
        
        # Initialize simulation tracker for early termination
        self.enable_tracker = enable_tracker
        if enable_tracker:
            self.tracker = SimulationTracker(
                initial_threshold=tracker_initial_threshold,
                kill_multiplier=tracker_kill_multiplier,
                verbose=verbose
            )
            # Set tracker in simulation runner
            self.simulation_runner.tracker = self.tracker
            if self.verbose:
                print(f"✓ Simulation tracker enabled (kill at {tracker_kill_multiplier}x threshold)")
        else:
            self.tracker = None
    
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
            
            with self.time_stats.timer("search_creation"):
                if self.search_type == "cbo":
                    self.search = self._create_cbo()
                    search_name = "CBO optimizer"
                else:  # random
                    self.search = self._create_random_search()
                    search_name = "RandomSearch optimizer"
            if self.verbose:
                print(f"✓ Created {search_name}\n")
            
            # Keep backward compatibility
            self.cbo = self.search
            
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
        problem.set_seed(self.random_state)
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
        # TODO: Report a bug to DeepHyper about this behavior
        # Note: we do the sorting because DeepHyper sorts parameters alphabetically but they don't sort values as key value pairs.
        #for name in param_names:
        #    param_values_map[name] = sorted(param_values_map[name], key=lambda x: (x is None, x))
        
        # Add hyperparameters in alphabetical order (DeepHyper sorts parameters alphabetically)
        for param_name in param_names:
            unique_values = param_values_map[param_name]
            if unique_values:
                problem.add_hyperparameter(unique_values, param_name, default_value=unique_values[0])
        

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
        
        def custom_sampling_fn(n_samples: int):
            """Sample directly from valid configurations to avoid constraint violations.
            
            This function is used by:
            1. CBO's acquisition optimizer (mixedga) for constraint-satisfying sampling
            2. RandomSearch's overridden _ask() method for uniform random sampling
            
            Uses numpy's RandomState for reproducibility with thread-safety.
            """
            # Use numpy for reproducible random sampling
            # Each call advances the state, giving different samples while maintaining reproducibility
            sampled = []
            for _ in range(n_samples):
                # Randomly select a valid configuration using numpy
                idx = np.random.randint(0, len(valid_configs))
                config = valid_configs[idx]
                sampled.append(config)
            return sampled
        
        # Seed numpy's global random state for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Set constraint and sampling functions
        problem.set_constraint_fn(constraint_fn)
        problem.set_sampling_fn(custom_sampling_fn)
        
        # Store custom_sampling_fn for reuse in RandomSearch
        self._custom_sampling_fn = custom_sampling_fn
        
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
            'extra_data_cache': self.extra_data_cache,
            'tracker': self.tracker if self.enable_tracker else None,
            'periodic_cleanup': self.periodic_cleanup_state,
        }
        
        eval_func = partial(_deephyper_evaluate_wrapper, optimizer_state=optimizer_state)
        evaluator = Evaluator.create(
            eval_func,
            method=self.evaluator_method,
            method_kwargs={"num_workers": self.n_workers}
        )
        
        return evaluator
    
    def _enrich_results_with_config_files(self):
        """Enrich DeepHyper results dataframe with config file information.
        
        This method reads the system, network, and memory config files for each
        successful evaluation and appends the information as new columns in the
        results dataframe.
        """
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            return
        
        # Create a temporary optimizer state to re-evaluate configs and get file_paths
        param_cols = [col for col in self.deephyper_results.columns if col.startswith('p:')]
        param_names = [col[2:] for col in param_cols]
        
        config_file_data = []
        
        for idx, row in self.deephyper_results.iterrows():
            # Skip failed or infeasible evaluations
            if 'objective' in row:
                if isinstance(row['objective'], str) and row['objective'] == 'F':
                    config_file_data.append({})
                    continue
            elif 'objective_0' in row:
                if isinstance(row['objective_0'], str) and row['objective_0'] == 'F':
                    config_file_data.append({})
                    continue
            
            if 'constraint' in row and not row['constraint']:
                config_file_data.append({})
                continue
            
            # Reconstruct config from row
            config = {name: row[f'p:{name}'] for name in param_names}
            
            try:
                # Retrieve exec_time and config files from cache
                config_key = tuple(config.items())
                if self.verbose and idx < 3:
                    print(f"  Looking up config_key for row {idx}: {config_key}")
                    print(f"  Cache has {len(self.extra_data_cache)} entries")
                    if config_key in self.extra_data_cache:
                        print(f"  ✓ Found in cache")
                    else:
                        print(f"  ✗ NOT found in cache")
                        print(f"  Available keys: {list(self.extra_data_cache.keys())[:2]}")
                
                cached_data = self.extra_data_cache.get(config_key, {})
                if cached_data:
                    config_data = dict(cached_data.get('config_files', {}))
                    config_data['exec_time'] = cached_data.get('exec_time')
                    config_data['was_killed'] = bool(cached_data.get('was_killed', False))
                else:
                    config_data = {}
                config_file_data.append(config_data)
            
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️  Warning: Could not enrich row {idx} with config files: {e}")
                config_file_data.append({})
        
        # Append config file data to results dataframe
        if config_file_data:
            # Add columns for config files (system_config, network_config, memory_config)
            config_file_keys = ['system_config', 'network_config', 'memory_config']
            
            for key in config_file_keys:
                self.deephyper_results[key] = [data.get(key, None) for data in config_file_data]
            
            # Add exec_time column
            self.deephyper_results['exec_time'] = [data.get('exec_time', None) for data in config_file_data]
            self.deephyper_results['was_killed'] = [bool(data.get('was_killed', False)) for data in config_file_data]
            
            if self.verbose:
                n_enriched = sum(1 for data in config_file_data if data)
                print(f"✓ Enriched {n_enriched}/{len(self.deephyper_results)} rows with config file information")
                print(f"  Added columns: {', '.join(config_file_keys)}, exec_time, was_killed")
    
    def _collect_results_from_deephyper(self):
        """Collect and process results from DeepHyper's output dataframe."""
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            return
        
        param_cols = [col for col in self.deephyper_results.columns if col.startswith('p:')]
        param_names = [col[2:] for col in param_cols]
        
        # Detect if multi-objective by checking for objective_0 column
        is_multi_objective = 'objective_0' in self.deephyper_results.columns
        
        PENALTY_THRESHOLD = float('inf')
        n_failed = 0
        n_success = 0
        n_infeasible = 0
        n_config_files_read = 0
        
        for idx, row in self.deephyper_results.iterrows():
            # Get objective value(s) - handle both single and multi-objective
            if is_multi_objective:
                # For MOO, use first objective as primary score for tracking "best"
                objective_value = row.get('objective_0', None)
                if objective_value is None:
                    n_failed += 1
                    continue
            else:
                objective_value = row.get('objective', None)
                if objective_value is None:
                    n_failed += 1
                    continue
            
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
            metadata = self._get_simulation_metadata()
            metadata['was_killed'] = bool(row.get('was_killed', False))
            self.metadata.append(metadata)
            n_success += 1
            
            # Try to read config files if job_id exists (to retrieve file_paths from job metadata)
            if 'job_id' in row:
                try:
                    # Access job metadata to get file_paths
                    # Note: This requires DeepHyper to store the file_paths in job metadata
                    # For now, we'll add this data directly to the dataframe after the search
                    pass
                except Exception:
                    pass
            
            if self.verbose and n_success <= 10:
                config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
                if is_multi_objective and 'objective_1' in row:
                    obj1_val = -row['objective_1'] if self.objective.minimize else row['objective_1']
                    print(f"  Iteration {n_success}: {config_str} | Obj0: {score:.4f}, Obj1: {obj1_val:.4f}")
                else:
                    print(f"  Iteration {n_success}: {config_str} | Score: {score:.4f}")
            
            if not is_penalty and self.objective.is_better(score, self.best_score):
                self.best_score = score
                self.best_config = config
                self.best_iteration = len(self.configs) - 1
                if self.verbose:
                    print(f"    🏆 NEW BEST: {score:.4f}")
        
        if self.verbose:
            print(f"\nCollected {n_success} successful evaluations")
            print(f"Total DeepHyper evaluations: {len(self.deephyper_results)}")
            if n_infeasible > 0:
                print(f"  - Infeasible (constraint violations): {n_infeasible}")
            if n_failed > 0:
                print(f"  - Failed (simulation errors): {n_failed}")
            if is_multi_objective:
                print(f"  - Multi-objective optimization detected")
            if n_config_files_read > 0:
                print(f"  - Config files read: {n_config_files_read}")
    
    def _read_config_files(self, file_paths: Dict) -> Dict:
        """Read config files and return as JSON strings."""
        config_data = {}
        
        for key in ['system_config', 'network_config', 'memory_config']:
            path = file_paths.get(key)
            if path:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f) if path.endswith('.json') else yaml.safe_load(f)
                    config_data[key] = json.dumps(data)
                except Exception as e:
                    if self.verbose:
                        print(f"    ⚠️  Could not read {key}: {e}")
                    config_data[key] = None
        
        return config_data
    
    def _get_simulation_metadata(self) -> Dict:
        """Get simulation metadata from simulation_runner."""
        sr = self.simulation_runner
        try:
            import sys
            sys.path.insert(0, os.environ['ASTRA_SIM_ROOT'] + '/upc')
            model = workload_generator.Model(sr.model_num)
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
    
    def _create_random_search(self) -> RandomSearch:
        """Create DeepHyper RandomSearch instance.
        
        Note: RandomSearch doesn't support multi-objective scalarization parameters.
        For multi-objective optimization with random search, the objective function
        should handle scalarization internally.
        
        Important: RandomSearch samples from the ConfigSpace, which may generate
        invalid combinations even if we restrict individual parameter values.
        To ensure only valid configurations are evaluated, we:
        1. Set initial_points to all valid configurations (done in __init__)
        2. Override the _ask method to sample from initial_points only
        """
        random_args = {
            # Required
            "problem": self.hp_problem,
            # Core parameters
            "random_state": self.random_state,
            "log_dir": self.log_dir,
            "verbose": 1 if self.verbose else 0,
            "stopper": self.stopper,
            "checkpoint_history_to_csv": self.checkpoint_history_to_csv,
            "solution_selection": self.solution_selection,
        }
        
        # Apply additional overrides from cbo_kwargs (reused for random search)
        # Filter out CBO-specific parameters
        if self.cbo_kwargs:
            valid_params = {"problem", "random_state", "log_dir", "verbose", "stopper", 
                          "checkpoint_history_to_csv", "solution_selection"}
            filtered_kwargs = {k: v for k, v in self.cbo_kwargs.items() if k in valid_params}
            random_args.update(filtered_kwargs)
        
        search = RandomSearch(**random_args)
        
        # Override _ask to sample only from valid configurations
        # Reuse the same custom_sampling_fn that CBO uses (stored during _create_hp_problem)
        # This ensures RandomSearch and CBO sample from the same valid design space
        if hasattr(self, '_custom_sampling_fn'):
            original_ask = search._ask
            
            def custom_ask(n: int = 1):
                """Sample from valid configurations using the same logic as CBO."""
                return self._custom_sampling_fn(n)
            
            search._ask = custom_ask
        
        return search
    
    def _finalize_and_save_results(self, enrichment_verbosity: bool = True) -> bool:
        """Finalize results by enriching with config files and saving to CSV.
        
        Args:
            enrichment_verbosity: Whether to print enrichment progress messages
            
        Returns:
            True if results were saved successfully, False otherwise
        """
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            if self.verbose:
                self._log("No results to save (optimization interrupted too early)", "warning")
            return False
        
        try:
            # Enrich results with config file information
            if enrichment_verbosity and self.verbose:
                print("\n" + "-"*70 + "\nENRICHING RESULTS WITH CONFIG FILES\n" + "-"*70)
            self._enrich_results_with_config_files()
            
            # Save to CSV
            dh_results_path = os.path.join(self.save_dir, self.results_filename)
            self.deephyper_results.to_csv(dh_results_path, index=False)
            
            if self.verbose:
                print(f"✓ Results saved to: {dh_results_path}")
                print(f"  {len(self.deephyper_results)} evaluations saved")
            
            # Collect results for BaseOptimizer tracking
            if enrichment_verbosity and self.verbose:
                print("\n" + "-"*70 + "\nCOLLECTING RESULTS\n" + "-"*70)
            self._collect_results_from_deephyper()
            
            return True
            
        except Exception as e:
            if self.verbose:
                self._log(f"Warning: Could not save results: {e}", "warning")
            return False
    
    def optimize_step(self) -> Tuple[Optional[Dict], Optional[float]]:
        """Not supported for DeepHyper (uses batch search instead)."""
        self._log("optimize_step() not supported", "warning")
        return None, None
    
    def run(self) -> Tuple[Optional[Dict], pd.DataFrame]:
        """Run optimization using DeepHyper (CBO or RandomSearch)."""
        self.start_time = time.time()
        self.time_stats.start_total()
        
        with self.time_stats.timer("initialization"):
            if not self.initialize():
                return None, pd.DataFrame()
        
        try:
            search_label = "CBO" if self.search_type == "cbo" else "RANDOM SEARCH"
            if self.verbose:
                print("-"*70 + f"\nRUNNING {search_label}\n" + "-"*70 + "\n")

            if self.periodic_cleanup_state is not None:
                self.periodic_cleanup_state['print_deleted_files'] = self.cleanup_print_deleted_files
            
            with self.time_stats.timer("search"):
                self.deephyper_results = self.search.search(
                    evaluator=self.evaluator,
                    max_evals=self.budget
                )

            # Final cleanup pass to ensure only top-K artifacts remain.
            if self.periodic_cleanup_state is not None:
                with self.periodic_cleanup_state['lock']:
                    ArtifactCleanupManager.run_periodic_cleanup_for_state(
                        cleanup_state=self.periodic_cleanup_state,
                        minimize=self.objective.minimize,
                        force=True,
                    )

                self.cleanup_manager.sync_counters(self.periodic_cleanup_state.get('cleanup_counters'))

            # Compress top-K artifacts and delete the originals.
            self.compress_and_clean()

            # Finalize and save results
            with self.time_stats.timer("save_results"):
                self._finalize_and_save_results(enrichment_verbosity=True)
            
            if self.verbose:
                print("\n" + "-"*70 + f"\n{search_label} COMPLETE\n" + "-"*70)
                print(
                    f"Cleanup summary: simulations_cleaned={self.cleanup_stats.get('simulations_cleaned', 0)}, "
                    f"files_deleted={self.cleanup_stats.get('files_deleted', 0)}"
                )
                self.print_summary()
            
            self.time_stats.end_total()
            return self.best_config, self.get_history()
            
        except KeyboardInterrupt:
            self._log("\n\nOptimization interrupted by user", "warning")
            self._log("Saving intermediate results...", "info")

            if self.periodic_cleanup_state is not None:
                with self.periodic_cleanup_state['lock']:
                    ArtifactCleanupManager.run_periodic_cleanup_for_state(
                        cleanup_state=self.periodic_cleanup_state,
                        minimize=self.objective.minimize,
                        force=True,
                    )
                self.cleanup_manager.sync_counters(self.periodic_cleanup_state.get('cleanup_counters'))
            
            # Finalize and save any completed results
            self._finalize_and_save_results(enrichment_verbosity=False)
            
            self.time_stats.end_total()
            return self.best_config, self.get_history()
        
        except Exception as e:
            self._log(f"Optimization failed: {e}", "error")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return None, pd.DataFrame()
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, columns: list, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """Remove outliers using Interquartile Range (IQR) method.
        
        Args:
            df: DataFrame to filter
            columns: List of column names to check for outliers
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
            
        Returns:
            DataFrame with outliers removed
        """
        # First, remove failure values (like -10000000000.0 = -1e10)
        mask = pd.Series([True] * len(df), index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Remove failure markers (-1e10) and NaN values
            # Using abs() to catch both positive and negative failure markers
            valid_mask = (df[col].notna()) & (df[col].abs() < 9e9)
            mask = mask & valid_mask
        
        # Now apply IQR filtering on the valid values
        df_valid = df[mask].copy()
        
        if len(df_valid) == 0:
            return df_valid
        
        for col in columns:
            if col not in df_valid.columns:
                continue
            
            values = df_valid[col]
            
            if len(values) < 4:  # Need at least 4 points for IQR
                continue
            
            # Calculate IQR
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:  # All values are the same
                continue
            
            # Define outlier bounds
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            # Update mask to keep only points within bounds
            df_valid = df_valid[(df_valid[col] >= lower_bound) & (df_valid[col] <= upper_bound)]
        
        return df_valid
    
    def plot_hypervolume(self, save_path: Optional[str] = None):
        """Plot hypervolume indicator over evaluations for multi-objective optimization.
        
        The hypervolume indicator measures the volume of objective space dominated by the
        current Pareto front relative to a reference point. It increases as the optimization
        finds better solutions, providing a single metric to track MOO progress.
        
        A higher hypervolume indicates:
        - Better overall solution quality
        - More diverse Pareto front coverage
        - Improved convergence toward optimal trade-offs
        
        Args:
            save_path: Path to save the plot. If None, uses save_dir/hypervolume.png
        """
        if self.deephyper_results is None or len(self.deephyper_results) == 0:
            print("⚠️  No results to plot hypervolume")
            return
        
        # Check if we have multi-objective results
        if "objective_0" not in self.deephyper_results.columns or "objective_1" not in self.deephyper_results.columns:
            print("⚠️  Not a multi-objective optimization - no hypervolume to compute")
            return
        
        try:
            from deephyper.analysis._matplotlib import update_matplotlib_rc
            from deephyper.sklearn.moo import MOOScalarBenchmark
            
            # Update matplotlib style for better plots
            update_matplotlib_rc()
            
            # Create benchmark for scoring
            bench = MOOScalarBenchmark(
                moo_lower_bounds=self.moo_lower_bounds,
                scalarization_strategy=self.moo_scalarization_strategy
            )
            
            # Compute hypervolume over time
            results = self.deephyper_results[["objective_0", "objective_1"]].values
            scorer = bench.scorer
            hvi = scorer.hypervolume(results)
            
            # Create plot
            x = list(range(1, len(hvi) + 1))
            fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
            
            _ = ax.plot(x, hvi, linewidth=2, color="#2E86AB", marker="o", markersize=4, markevery=max(1, len(x)//20))
            _ = ax.fill_between(x, hvi, alpha=0.3, color="#2E86AB")
            _ = ax.grid(alpha=0.3, linestyle="--")
            _ = ax.set_xlabel("Number of Evaluations", fontsize=12)
            _ = ax.set_ylabel("Hypervolume Indicator", fontsize=12)
            _ = ax.set_title("Hypervolume Indicator Progress", fontsize=14, fontweight="bold")
            
            # Add annotation for final hypervolume
            final_hv = hvi[-1]
            _ = ax.annotate(
                f"Final HV: {final_hv:.2f}",
                xy=(len(x), final_hv),
                xytext=(-60, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
            )
            
            # Save figure with model_name
            if save_path is None:
                model_name = getattr(self.simulation_runner, 'model_name', 'model')
                save_path = os.path.join(self.save_dir, f"hypervolume_{model_name}.png")
            
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            if self.verbose:
                print(f"✓ Hypervolume plot saved to: {save_path}")
                print(f"  Final hypervolume: {final_hv:.4f}")
                print(f"  Initial hypervolume: {hvi[0]:.4f}")
                print(f"  Improvement: {((final_hv - hvi[0]) / hvi[0] * 100):.2f}%")
            
            plt.close(fig)
            return save_path, hvi
            
        except ImportError as e:
            print(f"⚠️  Cannot plot hypervolume: {e}")
            print("   Install deephyper with: pip install deephyper[analytics]")
            return None, None
        except Exception as e:
            print(f"⚠️  Error computing hypervolume: {e}")
            return None, None

    def __repr__(self) -> str:
        """String representation."""
        return (f"DeepHyperOptimizer(budget={self.budget}, "
                f"init_samples={self.init_samples}, "
                f"n_workers={self.n_workers}, "
                f"acq_func='{self.acq_func}', "
                f"evaluated={len(self.configs)})")
    
    def __str__(self) -> str:
        """Human-readable string."""
        search_name = "CBO" if self.search_type == "cbo" else "RandomSearch"
        info = [
            f"DeepHyperOptimizer ({search_name})",
            f"Budget: {self.budget} evaluations",
            f"Workers: {self.n_workers}",
        ]
        
        if self.search_type == "cbo":
            info.extend([
                f"Initialization: {self.init_samples} samples",
                f"Acquisition: {self.acq_func}",
            ])
        
        info.append(f"Evaluated: {len(self.configs)} configs")
        
        if self.best_config:
            info.append(f"Best score: {self.best_score:.2f}")
        
        return "\n".join(info)
