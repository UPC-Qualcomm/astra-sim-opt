"""
AstraSim Optimization Framework

A modular, scalable optimization framework for tuning AstraSim configurations.

Quick Start:
    >>> from core.search_space_builder import create_search_space
    >>> from optimizers.random_optimizer import RandomOptimizer
    >>> from optimizers.bayesian_optimizer import BayesianOptimizer
    >>> from core.sampler import RandomSampler
    >>> from core.simulation_runner import SimulationRunner
    
    >>> # Setup
    >>> search_space = create_search_space("search_space/parallelism_strategy_params.json")
    >>> sampler = RandomSampler(seed=42)
    >>> sim_runner = SimulationRunner(40, "GPT_40B", 128, "FoldedClos", "my_exp")
    
    >>> # Run
    >>> optimizer = RandomOptimizer(search_space, sampler, sim_runner, budget=20)
    >>> best_config, history = optimizer.run()

See examples/ directory for full documentation.
"""

# Core modules
from .core.search_space_builder import SearchSpaceBuilder, create_search_space
from .core.simulation_runner import SimulationRunner
from .core.base_optimizer import BaseOptimizer
from .helper import config_to_tuple, tuple_to_config

# Objective functions
from .core.objective import (
    ObjectiveFunction,
    MinimizeExecutionTime,
    CustomObjective,
    create_objective
)

# Samplers
from .core.sampler import (
    BaseSampler,
    RandomSampler,
    LatinHypercubeSampler,
    SobolSampler,
    GridSampler,
    StratifiedSampler,
    get_sampler
)

# Optimizers
from .optimizers import RandomOptimizer

# Try to import Bayesian optimization components
try:
    from .core.kernels import (
        BaseKernel,
        MaternKernel,
        RBFKernel,
        CustomKernel,
        CompositeKernel,
        get_kernel
    )
    from .core.acquisition import (
        BaseAcquisitionFunction,
        ExpectedImprovement,
        UpperConfidenceBound,
        ProbabilityOfImprovement,
        ThompsonSampling,
        get_acquisition
    )
    from .optimizers import BayesianOptimizer
    
    BO_AVAILABLE = True
    
    __all__ = [
        # Core
        'SearchSpaceBuilder',
        'create_search_space',
        'SimulationRunner',
        'BaseOptimizer',
        # Helpers
        'config_to_tuple',
        'tuple_to_config',
        # Objectives
        'ObjectiveFunction',
        'MinimizeExecutionTime',
        'CustomObjective',
        'create_objective',
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
        'SearchSpaceBuilder',
        'create_search_space',
        'SimulationRunner',
        'BaseOptimizer',
        # Helpers
        'config_to_tuple',
        'tuple_to_config',
        # Objectives
        'ObjectiveFunction',
        'MinimizeExecutionTime',
        'CustomObjective',
        'create_objective',
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
