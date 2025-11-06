"""
Configuration utilities for handling dictionary configurations.

This module provides common utilities for working with configuration dictionaries,
particularly for converting them to hashable tuples for use in sets and other
data structures that require hashable types.
"""

from typing import Dict


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
