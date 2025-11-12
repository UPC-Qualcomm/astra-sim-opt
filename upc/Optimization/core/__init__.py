"""
Core modules for the optimization framework.
"""

from .search_space_builder import SearchSpaceBuilder, create_search_space
from .sampler import (
    BaseSampler,
    RandomSampler,
    LatinHypercubeSampler,
    SobolSampler,
    GridSampler
)
from .base_optimizer import BaseOptimizer
from .simulation_runner import SimulationRunner
from .time_statistics import TimeStatistics

# GP-specific modules (optional, only needed for Bayesian optimization)
try:
    from .kernels import (
        BaseKernel,
        MaternKernel,
        RBFKernel,
        CustomKernel
    )
    from .acquisition import (
        BaseAcquisitionFunction,
        ExpectedImprovement,
        UpperConfidenceBound,
        ProbabilityOfImprovement
    )
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

__all__ = [
    'SearchSpaceBuilder',
    'create_search_space',
    'BaseSampler',
    'RandomSampler', 
    'LatinHypercubeSampler',
    'SobolSampler',
    'GridSampler',
    'BaseOptimizer',
    'SimulationRunner',
    'TimeStatistics',
]

if GP_AVAILABLE:
    __all__.extend([
        'BaseKernel',
        'MaternKernel',
        'RBFKernel',
        'CustomKernel',
        'BaseAcquisitionFunction',
        'ExpectedImprovement',
        'UpperConfidenceBound',
        'ProbabilityOfImprovement',
    ])
