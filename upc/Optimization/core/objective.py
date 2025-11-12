"""
Objective function module for flexible optimization.

This module provides a modular system for defining optimization objectives.
- Use built-in objectives (time, memory, energy, throughput, etc.)
- Create weighted multi-objective optimizations
- Define completely custom objective functions

Example:
    # Default: minimize execution time
    objective = MinimizeExecutionTime()
    
    # Weighted multi-objective
    objective = create_objective(
        "weighted",
        weights={"exec_time": 0.7, "peak_memory_bytes": 0.3}
    )
    
    # Use in optimizer
    optimizer = BayesianOptimizer(..., objective=objective)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.
    
    An objective function computes a scalar score from simulation results.
    """
    
    def __init__(self, name: Optional[str] = None, minimize: bool = True):
        """
        Initialize objective function.
        
        Args:
            name: Human-readable name for this objective
            minimize: Whether to minimize (True) or maximize (False) the score
        """
        self.name = name or self.__class__.__name__
        self.minimize = minimize
    
    @abstractmethod
    def compute(self, exec_time: float, metadata: Dict[str, Any]) -> float:
        """
        Compute objective score from simulation results.
        
        Args:
            exec_time: Execution time in seconds (from simulation)
            metadata: Additional simulation metadata (model params, hardware config, etc.)
        
        Returns:
            Scalar score to optimize
        """
        pass
    
    def is_better(self, score1: float, score2: float) -> bool:
        """
        Check if score1 is better than score2.
        
        Args:
            score1: First score
            score2: Second score
        
        Returns:
            True if score1 is better than score2
        """
        if self.minimize:
            return score1 < score2
        else:
            return score1 > score2
    
    def get_best_score(self, scores: list) -> float:
        """
        Get the best score from a list.
        
        Args:
            scores: List of scores
        
        Returns:
            Best score (minimum if minimize=True, maximum if minimize=False)
        """
        if not scores:
            return float('inf') if self.minimize else float('-inf')
        return min(scores) if self.minimize else max(scores)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return self.name


class MinimizeExecutionTime(ObjectiveFunction):
    """
    Minimize execution time (default objective).
    
    Simply returns the execution time as the objective score.
    This is the standard objective for performance optimization.
    """
    
    def __init__(self):
        super().__init__("Minimize Execution Time")
    
    def compute(self, exec_time: float, metadata: Dict[str, Any]) -> float:
        """Return execution time as objective."""
        return exec_time


class CustomObjective(ObjectiveFunction):
    """
    Custom objective function from user-provided callable.
    
    Allows defining arbitrary objective functions.
    
    Example:
        # Minimize time + 0.1 * sqrt(memory_gb)
        def my_objective(exec_time, metadata):
            memory_gb = metadata['peak_memory_bytes'] / (1024**3)
            return exec_time + 0.1 * (memory_gb ** 0.5)
        
        objective = CustomObjective(my_objective, "My Custom Objective")
    """
    
    def __init__(
        self, 
        compute_fn: Callable[[float, Dict[str, Any]], float], 
        name: str = "Custom Objective",
        minimize: bool = True
    ):
        """
        Initialize custom objective.
        
        Args:
            compute_fn: Callable that takes (exec_time, metadata) and returns score
            name: Name for this objective
            minimize: Whether to minimize (True) or maximize (False)
        """
        super().__init__(name, minimize)
        self.compute_fn = compute_fn
    
    def compute(self, exec_time: float, metadata: Dict[str, Any]) -> float:
        """Call user-provided compute function."""
        return self.compute_fn(exec_time, metadata)


# Convenience factory function
def create_objective(objective_type: str, **kwargs) -> ObjectiveFunction:
    """
    Factory function to create objective functions by name.
    
    Args:
        objective_type: Type of objective ('time', 'throughput', 'weighted', etc.)
        **kwargs: Additional arguments for the objective
    
    Returns:
        ObjectiveFunction instance
    
    Example:
        objective = create_objective('time')
        objective = create_objective('weighted', weights={'exec_time': 0.7, 'peak_memory_bytes': 0.3})
    """
    objectives = {
        'time': MinimizeExecutionTime,
    }
    
    if objective_type not in objectives:
        raise ValueError(f"Unknown objective type: {objective_type}. "
                        f"Available: {list(objectives.keys())}")
    
    return objectives[objective_type](**kwargs)
