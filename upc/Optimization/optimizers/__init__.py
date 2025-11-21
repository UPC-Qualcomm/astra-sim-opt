"""
Optimizer implementations.
"""

from .random_optimizer import RandomOptimizer

# Bayesian optimizer requires sklearn
import importlib.util

BO_AVAILABLE = importlib.util.find_spec('sklearn') is not None
DEEPHYPER_AVAILABLE = importlib.util.find_spec('deephyper') is not None

__all__ = ['RandomOptimizer']

if BO_AVAILABLE:
    from .bayesian_optimizer import BayesianOptimizer
    __all__.append('BayesianOptimizer')

if DEEPHYPER_AVAILABLE:
    from .deephyper_optimizer import DeepHyperOptimizer
    __all__.append('DeepHyperOptimizer')
