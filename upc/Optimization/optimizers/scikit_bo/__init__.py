"""
Scikit-learn based Bayesian Optimization components.

This package contains the scikit-learn Gaussian Process implementation
for Bayesian Optimization, including kernels and acquisition functions.
"""

from .kernels import (
    BaseKernel,
    MaternKernel,
    RBFKernel,
    CustomKernel,
    CompositeKernel,
    get_kernel
)

from .acquisition import (
    BaseAcquisitionFunction,
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    ThompsonSampling,
    get_acquisition
)

__all__ = [
    # Kernels
    'BaseKernel',
    'MaternKernel',
    'RBFKernel',
    'CustomKernel',
    'CompositeKernel',
    'get_kernel',
    # Acquisition Functions
    'BaseAcquisitionFunction',
    'ExpectedImprovement',
    'UpperConfidenceBound',
    'ProbabilityOfImprovement',
    'ThompsonSampling',
    'get_acquisition',
]
