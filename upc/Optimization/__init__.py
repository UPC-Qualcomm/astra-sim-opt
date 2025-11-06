"""
AstraSim Optimization Framework

A modular, scalable optimization framework for tuning AstraSim configurations.

Quick Start:
    >>> from Optimization import SearchSpace, BayesianOptimizer, SimulationRunner
    >>> from Optimization import LatinHypercubeSampler, MaternKernel, ExpectedImprovement
    
    >>> # Setup
    >>> search_space = SearchSpace("search_space/parallelism_strategy_params.json", 64)
    >>> sampler = LatinHypercubeSampler()
    >>> sim_runner = SimulationRunner(40, "GPT_40B", 64, "FoldedClos")
    >>> kernel = MaternKernel(nu=2.5)
    >>> acquisition = ExpectedImprovement(xi=0.01)
    
    >>> # Run
    >>> optimizer = BayesianOptimizer(
    ...     search_space, sampler, sim_runner, kernel, acquisition, budget=30
    ... )
    >>> best_config, history = optimizer.run()

See README.md for full documentation.
"""

# Core modules
from core.search_space import SearchSpace
from core.simulation_runner import SimulationRunner
from core.base_optimizer import BaseOptimizer

# Samplers
from core.sampler import (
    BaseSampler,
    RandomSampler,
    LatinHypercubeSampler,
    SobolSampler,
    GridSampler,
    StratifiedSampler,
    get_sampler
)

# Optimizers
from optimizers.random_optimizer import RandomOptimizer

# Try to import Bayesian optimization components
try:
    from core.kernels import (
        BaseKernel,
        MaternKernel,
        RBFKernel,
        CustomKernel,
        CompositeKernel,
        get_kernel
    )
    from core.acquisition import (
        BaseAcquisitionFunction,
        ExpectedImprovement,
        UpperConfidenceBound,
        ProbabilityOfImprovement,
        ThompsonSampling,
        get_acquisition
    )
    from optimizers.bayesian_optimizer import BayesianOptimizer
    
    BO_AVAILABLE = True
    
    __all__ = [
        # Core
        'SearchSpace',
        'SimulationRunner',
        'BaseOptimizer',
        # Samplers
        'BaseSampler',
        'RandomSampler',
        'LatinHypercubeSampler',
        'SobolSampler',
        'GridSampler',
        'StratifiedSampler',
        'get_sampler',
        # Optimizers
        'RandomOptimizer',
        'BayesianOptimizer',
        # Kernels
        'BaseKernel',
        'MaternKernel',
        'RBFKernel',
        'CustomKernel',
        'CompositeKernel',
        'get_kernel',
        # Acquisition
        'BaseAcquisitionFunction',
        'ExpectedImprovement',
        'UpperConfidenceBound',
        'ProbabilityOfImprovement',
        'ThompsonSampling',
        'get_acquisition',
    ]
    
except ImportError:
    BO_AVAILABLE = False
    
    __all__ = [
        # Core
        'SearchSpace',
        'SimulationRunner',
        'BaseOptimizer',
        # Samplers
        'BaseSampler',
        'RandomSampler',
        'LatinHypercubeSampler',
        'SobolSampler',
        'GridSampler',
        'StratifiedSampler',
        'get_sampler',
        # Optimizers
        'RandomOptimizer',
    ]

__version__ = '1.0.0'
__author__ = 'AstraSim Optimization Team'
