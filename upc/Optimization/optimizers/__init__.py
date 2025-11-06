"""
Optimizer implementations.
"""

from .random_optimizer import RandomOptimizer

# Bayesian optimizer requires sklearn
try:
    from .bayesian_optimizer import BayesianOptimizer
    BO_AVAILABLE = True
except ImportError:
    BO_AVAILABLE = False

__all__ = ['RandomOptimizer']

if BO_AVAILABLE:
    __all__.append('BayesianOptimizer')
