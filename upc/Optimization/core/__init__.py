"""
Core modules for the optimization framework.
"""

from .search_space import SearchSpace
from .sampler import (
    BaseSampler,
    RandomSampler,
    LatinHypercubeSampler,
    SobolSampler,
    GridSampler
)
from .base_optimizer import BaseOptimizer
from .simulation_runner import SimulationRunner

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
    'SearchSpace',
    'BaseSampler',
    'RandomSampler', 
    'LatinHypercubeSampler',
    'SobolSampler',
    'GridSampler',
    'BaseOptimizer',
    'SimulationRunner',
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
