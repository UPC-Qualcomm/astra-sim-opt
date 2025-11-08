"""
BayesianOptimizer: Bayesian Optimization using Gaussian Processes.

Uses a Gaussian Process to model the objective function and an acquisition
function to select the next point to evaluate. Efficiently explores the
search space by balancing exploration and exploitation.
"""

import sys
from typing import Tuple, Optional, List, Dict
import pandas as pd
import numpy as np
import time

# Add parent directory to path for imports
sys.path.append('/media/mohammad/extension/experiments/astra-sim/upc/Optimization')
from ..core import BaseOptimizer, SearchSpaceBuilder
from ..helper import config_to_tuple, tuple_to_config

# Check for sklearn availability
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian Optimization using Gaussian Process surrogate model.
    
    Workflow:
    1. Initialize with random samples
    2. Fit Gaussian Process to observed data
    3. Use acquisition function to find next promising configuration
    4. Evaluate configuration
    5. Repeat from step 2
    
    Key Features:
    - Pluggable kernel (Matern, RBF, custom)
    - Pluggable acquisition function (EI, UCB, PI)
    - Adaptive search strategy (exhaustive or sampling)
    - Detailed logging and visualization
    
    Example:
        from core.search_space_builder import create_search_space
        from core.sampler import LatinHypercubeSampler
        from core.simulation_runner import SimulationRunner
        from core.kernels import MaternKernel
        from core.acquisition import ExpectedImprovement
        
        search_space = create_search_space(
            "search_space/parallelism_strategy_params.json",
            num_npus=64,
            include_categories=['parallelism_strategy']
        )
        sampler = LatinHypercubeSampler(seed=42)
        sim_runner = SimulationRunner(40, "GPT_40B", 64, "FoldedClos")
        kernel = MaternKernel(nu=2.5)
        acquisition = ExpectedImprovement(xi=0.01)
        
        optimizer = BayesianOptimizer(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=sim_runner,
            kernel=kernel,
            acquisition=acquisition,
            budget=30,
            init_samples=5
        )
        
        best_config, results_df = optimizer.run()
    """
    
    def __init__(
        self,
        search_space,
        sampler,
        simulation_runner,
        kernel,
        acquisition,
        budget: int = 30,
        init_samples: int = 5,
        exhaustive_threshold: int = 5000,
        gp_alpha: float = 1e-6,
        gp_n_restarts: int = 5,
        verbose: bool = True,
        save_dir: str = ".",
        keep_top_k: int = -1
    ):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            search_space: SearchSpace instance
            sampler: Sampler instance for initial sampling
            simulation_runner: SimulationRunner instance
            kernel: Kernel instance (e.g., MaternKernel, RBFKernel)
            acquisition: AcquisitionFunction instance (e.g., ExpectedImprovement)
            budget: Total number of evaluations
            init_samples: Number of initial random samples
            exhaustive_threshold: Max design space size for exhaustive acquisition search
            gp_alpha: GP noise parameter (regularization)
            gp_n_restarts: Number of GP hyperparameter optimization restarts
            verbose: Whether to print progress
            save_dir: Directory to save results
            keep_top_k: Keep only top K results' files (-1 = keep all, 0 = keep none)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Bayesian Optimization. "
                            "Install with: pip install scikit-learn scipy")
        
        super().__init__(
            search_space=search_space,
            sampler=sampler,
            simulation_runner=simulation_runner,
            budget=budget,
            init_samples=init_samples,
            verbose=verbose,
            save_dir=save_dir,
            keep_top_k=keep_top_k
        )
        
        self.kernel = kernel
        self.acquisition = acquisition
        self.exhaustive_threshold = exhaustive_threshold
        self.gp_alpha = gp_alpha
        self.gp_n_restarts = gp_n_restarts
        
        # GP model storage
        self.gps: List[GaussianProcessRegressor] = []
        self.ei_scores: List[float] = []  # Acquisition scores per iteration
        
        # Number of BO iterations (after initialization)
        self.n_iterations = budget - init_samples
    
    def initialize(self) -> bool:
        """
        Initialize with random samples.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.verbose:
            print("\n" + "="*70)
            print("BAYESIAN OPTIMIZATION")
            print("="*70)
            print(f"Model: {self.simulation_runner.model_name}")
            print(f"NPUs: {self.simulation_runner.num_npus}")
            print(f"Network: {self.simulation_runner.network_name}")
            print(f"Kernel: {self.kernel}")
            print(f"Acquisition: {self.acquisition}")
            print(f"Budget: {self.budget} evaluations ({self.init_samples} init + {self.n_iterations} BO)")
            print(f"Design space: {self.search_space.get_design_space_size()} configurations")
            print("="*70)
            
            print("\n" + "-"*70)
            print("STAGE 1: INITIALIZATION")
            print("-"*70)
        
        # Get design space
        design_space = self.search_space.get_design_space()
        
        if not design_space:
            self._log("No valid configurations in design space!", "error")
            return False
        
        if self.verbose:
            print(f"Sampling {self.init_samples} initial configurations...\n")
        
        # Sample initial configurations
        init_configs = self.sampler.sample(design_space, self.init_samples)
        
        # Evaluate initial configurations
        for i, config in enumerate(init_configs):
            self.current_iteration = i
            
            # Format configuration for display
            config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
            
            if self.verbose:
                print(f"  [{i+1}/{self.init_samples}] Config: {config_str}")
            
            exec_time = self.evaluate_config(config, verbose=True)
            
            if exec_time is None:
                if self.verbose:
                    print("    ⚠️  Evaluation failed, skipping...\n")
                continue
        
        if not self.scores:
            self._log("All initial evaluations failed!", "error")
            return False
        
        # Summary
        if self.verbose:
            best_idx = np.argmin(self.scores)
            print(f"\nInitialization complete!")
            print(f"  Successful: {len(self.scores)}/{self.init_samples}")
            print(f"  Best init time: {self.scores[best_idx]:.1f}s")
            print(f"  Mean time: {np.mean(self.scores):.1f}s")
            print(f"  Std Dev: {np.std(self.scores):.1f}s\n")
        
        return True
    
    def optimize_step(self) -> Tuple[Optional[Dict], Optional[float]]:
        """
        Execute one Bayesian Optimization iteration.
        
        Returns:
            (config, score) tuple if successful, (None, None) otherwise
        """
        # Fit Gaussian Process
        gp = self._fit_gp()
        
        if gp is None:
            return None, None
        
        # Find next configuration using acquisition function
        next_config = self._optimize_acquisition(gp)
        
        if next_config is None:
            return None, None
        
        # Evaluate configuration
        score = self.evaluate_config(next_config, verbose=self.verbose)
        
        return next_config, score
    
    def run(self) -> Tuple[Optional[Dict], pd.DataFrame]:
        """
        Run full Bayesian Optimization.
        
        Returns:
            (best_config, results_dataframe) tuple
        """
        self.start_time = time.time()
        
        # Initialize
        if not self.initialize():
            return None, pd.DataFrame()
        
        try:
            # Bayesian Optimization loop
            for iteration in range(self.n_iterations):
                self.current_iteration = self.init_samples + iteration
                
                if self.verbose:
                    print(f"\n{'='*70}")
                    print(f"BO ITERATION {iteration + 1}/{self.n_iterations}")
                    print(f"{'='*70}\n")
                
                # Fit GP
                if self.verbose:
                    print("-" * 70)
                    print("STAGE 2: FITTING GAUSSIAN PROCESS")
                    print("-" * 70)
                
                gp = self._fit_gp()
                
                if gp is None:
                    self._log("GP fitting failed, stopping...", "error")
                    break
                
                # Optimize acquisition
                if self.verbose:
                    print(f"\n{'-' * 70}")
                    print("STAGE 3: ACQUISITION (Find Next Config)")
                    print("-" * 70)
                
                next_config = self._optimize_acquisition(gp)
                
                if next_config is None:
                    self._log("Acquisition optimization failed, stopping...", "error")
                    break
                
                # Evaluate
                if self.verbose:
                    print(f"\n{'-' * 70}")
                    print("STAGE 4: EVALUATION")
                    print("-" * 70)
                
                # Format configuration for display
                config_str = ", ".join([f"{k}={v}" for k, v in next_config.items()])
                if self.verbose:
                    print(f"Evaluating: {config_str}")
                
                exec_time = self.evaluate_config(next_config, verbose=True)
                
                if exec_time is None:
                    if self.verbose:
                        print("⚠️  Evaluation failed, continuing...\n")
                else:
                    if self.verbose:
                        print(f"\nBest so far: {self.best_score:.2f}s\n")
            
            # Print summary
            if self.verbose:
                print("\n" + "="*70)
                print("STAGE 5: ANALYSIS")
                print("-" * 70)
                self.print_summary()
            
            # Save results
            self.save_results()
            
            return self.best_config, self.get_history()
            
        except KeyboardInterrupt:
            self._log("\n\nOptimization interrupted by user", "warning")
            self._log("Saving intermediate results...", "info")
            self.save_results()
            return self.best_config, self.get_history()
    
    def _fit_gp(self) -> Optional[GaussianProcessRegressor]:
        """
        Fit Gaussian Process to observed data.
        
        Returns:
            Fitted GP model, or None if fitting failed
        """
        # Prepare training data
        X_train = np.array(self._configs_to_features(self.configs))
        y_train = np.array(self.scores)
        
        if self.verbose:
            print(f"Training data: {len(X_train)} observations, {X_train.shape[1]} dimensions")
            if self.configs:
                param_names = sorted(self.configs[0].keys())
                print(f"Features: {param_names}")
        
        # Get sklearn kernel
        sklearn_kernel = self.kernel.get_sklearn_kernel()
        
        # Create GP
        gp = GaussianProcessRegressor(
            kernel=sklearn_kernel,
            alpha=self.gp_alpha,
            normalize_y=True,
            n_restarts_optimizer=self.gp_n_restarts
        )
        
        # Fit GP
        try:
            if self.verbose:
                print("Fitting GP model...")
            
            gp.fit(X_train, y_train)
            self.gps.append(gp)
            
            if self.verbose:
                print("  Learned hyperparameters:")
                # Extract hyperparameters safely
                if hasattr(gp.kernel_, 'k1') and hasattr(gp.kernel_.k1, 'constant_value'):
                    print(f"    Constant amplitude: {gp.kernel_.k1.constant_value:.4f}")
                
                if hasattr(gp.kernel_, 'k2') and hasattr(gp.kernel_.k2, 'length_scale'):
                    length_scale = gp.kernel_.k2.length_scale
                    if np.isscalar(length_scale):
                        print(f"    Length scale: {length_scale:.4f}")
                    else:
                        print(f"    Length scale: {length_scale[0]:.4f}")
                
                # Compute training RMSE
                train_pred = gp.predict(X_train)
                rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
                print(f"  RMSE on training data: {rmse:.2f}s")
            
            return gp
            
        except Exception as e:
            self._log(f"GP fitting error: {e}", "error")
            return None
    
    def _optimize_acquisition(self, gp: GaussianProcessRegressor) -> Optional[Dict]:
        """
        Find next configuration by optimizing acquisition function.
        
        Args:
            gp: Fitted Gaussian Process
        
        Returns:
            Next configuration to evaluate
        """
        # Get design space
        design_space = self.search_space.get_design_space()
        
        # Filter out already evaluated configs
        # Convert configs to tuples for set operations (dicts are unhashable)
        evaluated_set = set(config_to_tuple(c) for c in self.configs)
        unevaluated_candidates = [c for c in design_space 
                                   if config_to_tuple(c) not in evaluated_set]
        
        if not unevaluated_candidates:
            self._log("All configurations evaluated!", "warning")
            # Re-evaluate best by EI
            unevaluated_candidates = design_space
        
        if self.verbose:
            print(f"Design space: {len(design_space)} total, "
                  f"{len(unevaluated_candidates)} unevaluated")
        
        # Choose strategy based on space size
        if len(unevaluated_candidates) <= self.exhaustive_threshold:
            if self.verbose:
                print(f"✓ Using EXHAUSTIVE search (space size ≤ {self.exhaustive_threshold})")
            
            next_config, next_ei = self._acquisition_exhaustive(gp, unevaluated_candidates)
        else:
            if self.verbose:
                print(f"✓ Using SMART SAMPLING (space size > {self.exhaustive_threshold})")
            
            next_config, next_ei = self._acquisition_sampling(gp, unevaluated_candidates)
        
        self.ei_scores.append(next_ei)
        
        return next_config
    
    def _acquisition_exhaustive(
        self,
        gp: GaussianProcessRegressor,
        candidates: List[Dict]
    ) -> Tuple[Dict, float]:
        """
        Exhaustive acquisition optimization.
        
        Evaluates acquisition function for ALL candidates.
        
        Args:
            gp: Fitted Gaussian Process
            candidates: List of candidate configurations
        
        Returns:
            (best_config, best_acquisition_score) tuple
        """
        X_candidates = np.array(self._configs_to_features(candidates))
        
        # Current best
        best_y = np.min(self.scores)
        if self.verbose:
            print(f"Current best: {best_y:.2f}s")
        
        # Compute acquisition for all candidates
        if self.verbose:
            print(f"Computing acquisition for all {len(candidates)} configs...")
        
        mu, sigma = gp.predict(X_candidates, return_std=True)
        acquisition_scores = self.acquisition.compute(mu, sigma, best_y)
        
        # Find global maximum
        best_idx = np.argmax(acquisition_scores)
        next_config = candidates[best_idx]
        next_acquisition = acquisition_scores[best_idx]
        
        # Statistics
        if self.verbose:
            print("  Acquisition landscape:")
            print(f"    Max: {next_acquisition:.6f} (← selecting this)")
            print(f"    Mean: {np.mean(acquisition_scores):.6f}")
            top_5 = sorted(acquisition_scores, reverse=True)[:5]
            print(f"    Top-5: {[f'{x:.6f}' for x in top_5]}")
            
            print("  Selected config:")
            config_str = ", ".join([f"{k}={v}" for k, v in next_config.items()])
            print(f"    {config_str}")
            print(f"    Predicted: {mu[best_idx]:.2f}s ± {sigma[best_idx]:.2f}s")
            print("\n  ⏱️  Now running simulation for this config...\n")
        
        return next_config, next_acquisition
    
    def _acquisition_sampling(
        self,
        gp: GaussianProcessRegressor,
        candidates: List[Dict],
        n_samples: int = 10000,
        n_top: int = 20
    ) -> Tuple[Dict, float]:
        """
        Smart sampling acquisition optimization for large spaces.
        
        Args:
            gp: Fitted Gaussian Process
            candidates: List of candidate configurations
            n_samples: Number of random samples
            n_top: Number of top candidates to evaluate acquisition on
        
        Returns:
            (best_config, best_acquisition_score) tuple
        """
        # Sample candidates
        sample_size = min(n_samples, len(candidates))
        import random
        sample_indices = random.sample(range(len(candidates)), sample_size)
        sampled_candidates = [candidates[i] for i in sample_indices]
        
        if self.verbose:
            print(f"Sampled {sample_size} candidates for evaluation")
        
        X_sampled = np.array(self._configs_to_features(sampled_candidates))
        
        # Current best
        best_y = np.min(self.scores)
        if self.verbose:
            print(f"Current best: {best_y:.2f}s")
        
        # Get GP predictions
        mu, sigma = gp.predict(X_sampled, return_std=True)
        
        # Score by UCB (for pre-selection)
        ucb_scores = -mu + 2.0 * sigma
        
        # Select top candidates
        top_indices = np.argsort(ucb_scores)[-n_top:][::-1]
        if self.verbose:
            print(f"Selected top {n_top} candidates by UCB score")
        
        # Compute acquisition for top candidates
        acquisition_scores = self.acquisition.compute(
            mu[top_indices], sigma[top_indices], best_y
        )
        
        # Find best
        best_in_top = np.argmax(acquisition_scores)
        best_idx = top_indices[best_in_top]
        next_config = sampled_candidates[best_idx]
        next_acquisition = acquisition_scores[best_in_top]
        
        # Statistics
        if self.verbose:
            # Compute EI for all samples for statistics
            all_ei = self.acquisition.compute(mu, sigma, best_y)
            
            print("  Sampling statistics:")
            print(f"    Sample size: {sample_size}/{len(candidates)}")
            print(f"    Coverage: {100*sample_size/len(candidates):.1f}%")
            print("  Acquisition landscape:")
            print(f"    Max (from top-{n_top}): {next_acquisition:.6f}")
            print(f"    Mean (all samples): {np.mean(all_ei):.6f}")
            
            print("  Selected config:")
            config_str = ", ".join([f"{k}={v}" for k, v in next_config.items()])
            print(f"    {config_str}")
            print(f"    Predicted: {mu[best_idx]:.2f}s ± {sigma[best_idx]:.2f}s")
            print(f"    UCB score: {ucb_scores[best_idx]:.2f}")
            print("\n  ⏱️  Now running simulation for this config...\n")
        
        return next_config, next_acquisition
    
    def _configs_to_features(self, configs: List[Dict]) -> List[List[float]]:
        """
        Convert configuration dictionaries to feature vectors for GP.
        
        Args:
            configs: List of configuration dictionaries
        
        Returns:
            List of feature vectors
        """
        if not configs:
            return []
        
        # Get sorted parameter names for consistent ordering
        param_names = sorted(configs[0].keys())
        
        features = []
        for config in configs:
            feature = []
            for param in param_names:
                value = config[param]
                # Convert boolean to float
                if isinstance(value, bool):
                    feature.append(float(value))
                else:
                    feature.append(float(value))
            features.append(feature)
        return features
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"BayesianOptimizer(budget={self.budget}, "
                f"init_samples={self.init_samples}, "
                f"kernel={self.kernel}, "
                f"acquisition={self.acquisition}, "
                f"evaluated={len(self.configs)})")
