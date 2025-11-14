"""
Configuration utilities for handling dictionary configurations.

This module provides common utilities for working with configuration dictionaries,
particularly for converting them to hashable tuples for use in sets and other
data structures that require hashable types.

Also provides worker functions for parallel evaluation using multiprocessing.
"""

from typing import Dict, Tuple, Optional


def config_to_tuple(config: Dict) -> tuple:
    """
    Convert a configuration dictionary to a hashable tuple.
    
    This is useful when configs need to be stored in sets or used as dictionary keys.
    The tuple contains sorted (key, value) pairs to ensure consistent ordering.
    
    Args:
        config: Configuration dictionary (e.g., {'dp': 1, 'mp': 2, 'sp': 8, ...})
    
    Returns:
        Hashable tuple of sorted (key, value) pairs
        
    Example:
        >>> config = {'dp': 1, 'mp': 2, 'sp': 8, 'pp': 8, 'sharded': True}
        >>> config_to_tuple(config)
        (('dp', 1), ('mp', 2), ('pp', 8), ('sharded', True), ('sp', 8))
    """
    return tuple(sorted(config.items()))


def tuple_to_config(config_tuple: tuple) -> Dict:
    """
    Convert a configuration tuple back to a dictionary.
    
    This is the inverse operation of config_to_tuple().
    
    Args:
        config_tuple: Tuple of (key, value) pairs
    
    Returns:
        Configuration dictionary
        
    Example:
        >>> config_tuple = (('dp', 1), ('mp', 2), ('pp', 8), ('sharded', True), ('sp', 8))
        >>> tuple_to_config(config_tuple)
        {'dp': 1, 'mp': 2, 'pp': 8, 'sharded': True, 'sp': 8}
    """
    return dict(config_tuple)


def evaluate_config_worker(config: Dict, simulation_runner) -> Tuple[Dict, Optional[float], Dict, Dict]:
    """
    Worker function for parallel configuration evaluation.
    
    This is a module-level function so it can be pickled by multiprocessing.
    Used by optimizers that support parallel evaluation.
    
    Note: This function does NOT record results in the optimizer's history.
    It only evaluates and returns the raw results. The calling optimizer
    is responsible for storing results.
    
    Args:
        config: Configuration dictionary to evaluate
        simulation_runner: SimulationRunner instance to run the simulation
    
    Returns:
        Tuple of (config, exec_time, file_paths, metadata):
            - config: The input configuration
            - exec_time: Execution time in seconds (None if evaluation failed)
            - file_paths: Dict of file paths generated during simulation
            - metadata: Dict of additional metadata from simulation
    
    Example:
        >>> from functools import partial
        >>> from multiprocessing import Pool
        >>> from Optimization.helper import evaluate_config_worker
        >>> 
        >>> worker = partial(evaluate_config_worker, 
        ...                  simulation_runner=optimizer.simulation_runner)
        >>> with Pool(4) as pool:
        >>>     results = pool.map(worker, configs)
    """
    try:
        result = simulation_runner.run_simulation(config, return_paths=True)
        
        if result is not None:
            if isinstance(result, tuple) and len(result) == 4:
                exec_time, is_oom, file_paths, metadata = result
            elif isinstance(result, tuple) and len(result) == 3:
                exec_time, is_oom, file_paths, metadata = result
            elif isinstance(result, tuple) and len(result) == 2:
                exec_time, is_oom, file_paths = result
                metadata = {}
            else:
                exec_time, is_oom = result
                file_paths = {}
                metadata = {}
            
            return config, exec_time, is_oom, file_paths, metadata
        else:
            return config, None, {}, {}
            
    except Exception as e:
        print(f"⚠️  Evaluation error for config {config}: {e}")
        return config, None, {}, {}
