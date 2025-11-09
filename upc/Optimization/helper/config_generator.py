"""
Config File Generator: Single source of truth for all AstraSim configuration files.

This module is the ONLY way to get configuration file paths in the optimization framework.
The optimizer and simulation runner ALWAYS use this generator - never user-provided paths.

If custom parameters (net_*, sys_*, mem_*) are provided in the search space config, they
override the defaults. Otherwise, configs are generated from defaults.

All generated configs are stored in: ./Optimization/temp/config/

Usage:
    from helper import config_generator
    
    # Generate configs (always, with or without custom params)
    system_path = config_generator.generate_system_config(config)
    network_path = config_generator.generate_network_config(config)
    memory_path = config_generator.generate_memory_config(config)
"""

import json
import yaml
import os
from typing import Dict, Any


# Get the Optimization directory (parent of helper directory)
_HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
_OPTIMIZATION_DIR = os.path.dirname(_HELPER_DIR)

# Config output directory (absolute path)
CONFIG_OUTPUT_DIR = os.path.join(_OPTIMIZATION_DIR, "temp", "config")

# Caches for config paths - reuse configs when parameters don't change
_SYSTEM_CONFIG_CACHE = {}  # Maps config hash to file path
_NETWORK_CONFIG_CACHE = {}  # Maps config hash to file path
_MEMORY_CONFIG_CACHE = {}  # Maps config hash to file path


def _hash_config(config_dict: Dict[str, Any]) -> str:
    """Create a hash of config dict for caching purposes."""
    import hashlib
    # Convert dict to sorted JSON string for consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


# Default configurations
DEFAULT_SYSTEM_CONFIG = {
    "scheduling-policy": "LIFO",
    "endpoint-delay": 10,
    "active-chunks-per-dimension": 1,
    "preferred-dataset-splits": 1,
    "all-reduce-implementation": ["halvingDoubling", "halvingDoubling"],
    "all-gather-implementation": ["halvingDoubling", "halvingDoubling"],
    "reduce-scatter-implementation": ["halvingDoubling", "halvingDoubling"],
    "all-to-all-implementation": ["halvingDoubling", "halvingDoubling"],
    "collective-optimization": "localBWAware",
    "local-mem-bw": 3350,
    "local-mem-size": 80,
    "enable_network_logger": 1,
    "boost-mode": 0,
    "peak-perf": 989,
    "roofline-enabled": 1,
    "trace-enabled": 1,
    "track-local-mem": 0,
    "local-mem-trace-filename": "mem_trace.json"
}

DEFAULT_NETWORK_CONFIG = {
    "topology": ["Switch", "Switch"],
    "npus_count": [8, 16],
    "bandwidth": [450.0, 100.0],
    "bandwidth_unit": "GB/s",
    "latency": [0.0, 0.0],
    "packet_size": 1500,
    "header_size": 48
}

DEFAULT_MEMORY_CONFIG = {
    "memory-type": "NO_MEMORY_EXPANSION"
}


def generate_system_config(config: Dict[str, Any]) -> str:
    """
    Generate system configuration JSON file.
    
    Caches and reuses configs when parameters don't change.
    If config contains sys_* parameters, they override defaults.
    Otherwise, uses DEFAULT_SYSTEM_CONFIG.
    
    Args:
        config: Configuration dictionary (may contain sys_* prefixed parameters)
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _SYSTEM_CONFIG_CACHE
    
    # Start with defaults
    system_config = DEFAULT_SYSTEM_CONFIG.copy()
    
    # Override with custom values from config if present
    param_mapping = {
        'sys_scheduling-policy': 'scheduling-policy',
        'sys_endpoint-delay': 'endpoint-delay',
        'sys_active-chunks-per-dimension': 'active-chunks-per-dimension',
        'sys_preferred-dataset-splits': 'preferred-dataset-splits',
        'sys_collective-optimization': 'collective-optimization',
        'sys_local-mem-bw': 'local-mem-bw',
        'sys_local-mem-size': 'local-mem-size',
        'sys_boost-mode': 'boost-mode',
        'sys_peak-perf': 'peak-perf'
    }
    
    for config_key, system_key in param_mapping.items():
        if config_key in config:
            system_config[system_key] = config[config_key]
    
    # Check cache - reuse if same config exists
    config_hash = _hash_config(system_config)
    if config_hash in _SYSTEM_CONFIG_CACHE:
        cached_path = _SYSTEM_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with hash (not timestamp)
    output_path = os.path.join(CONFIG_OUTPUT_DIR, f"system_{config_hash}.json")
    
    # Write JSON file
    with open(output_path, 'w') as f:
        json.dump(system_config, f, indent=4)
    
    # Cache the path
    _SYSTEM_CONFIG_CACHE[config_hash] = output_path
    
    return output_path


def generate_network_config(config: Dict[str, Any]) -> str:
    """
    Generate network configuration YAML file.
    
    Caches and reuses configs when parameters don't change.
    If config contains net_* parameters, they override defaults.
    Otherwise, uses DEFAULT_NETWORK_CONFIG.
    
    Args:
        config: Configuration dictionary (may contain net_* prefixed parameters)
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _NETWORK_CONFIG_CACHE
    
    # Start with defaults
    network_config = DEFAULT_NETWORK_CONFIG.copy()
    
    # Override with custom values from config if present
    # Handle list parameters (bandwidth, latency, npus_count have _l0, _l1 suffixes)
    if 'net_bandwidth_l0' in config or 'net_bandwidth_l1' in config:
        bandwidth = network_config['bandwidth'].copy()
        if 'net_bandwidth_l0' in config:
            bandwidth[0] = config['net_bandwidth_l0']
        if 'net_bandwidth_l1' in config:
            bandwidth[1] = config['net_bandwidth_l1']
        network_config['bandwidth'] = bandwidth
    
    if 'net_latency_l0' in config or 'net_latency_l1' in config:
        latency = network_config['latency'].copy()
        if 'net_latency_l0' in config:
            latency[0] = config['net_latency_l0']
        if 'net_latency_l1' in config:
            latency[1] = config['net_latency_l1']
        network_config['latency'] = latency
    
    if 'net_npus_count_l0' in config or 'net_npus_count_l1' in config:
        npus_count = network_config['npus_count'].copy()
        if 'net_npus_count_l0' in config:
            npus_count[0] = config['net_npus_count_l0']
        if 'net_npus_count_l1' in config:
            npus_count[1] = config['net_npus_count_l1']
        network_config['npus_count'] = npus_count
    
    # Handle scalar parameters
    if 'net_packet_size' in config:
        network_config['packet_size'] = config['net_packet_size']
    
    if 'net_header_size' in config:
        network_config['header_size'] = config['net_header_size']
    
    # Check cache - reuse if same config exists
    config_hash = _hash_config(network_config)
    if config_hash in _NETWORK_CONFIG_CACHE:
        cached_path = _NETWORK_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with hash (not timestamp)
    output_path = os.path.join(CONFIG_OUTPUT_DIR, f"network_{config_hash}.yml")
    
    # Write YAML file with flow style for lists (matches FoldedClos.yml format)
    with open(output_path, 'w') as f:
        yaml.dump(network_config, f, default_flow_style=None)
    
    # Cache the path
    _NETWORK_CONFIG_CACHE[config_hash] = output_path
    
    return output_path


def generate_memory_config(config: Dict[str, Any]) -> str:
    """
    Generate memory configuration JSON file.
    
    Caches and reuses configs when parameters don't change.
    If config contains mem_* parameters, they override defaults.
    Otherwise, uses DEFAULT_MEMORY_CONFIG.
    
    Args:
        config: Configuration dictionary (may contain mem_* prefixed parameters)
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _MEMORY_CONFIG_CACHE
    
    # Start with defaults
    memory_config = DEFAULT_MEMORY_CONFIG.copy()
    
    # Override with custom values from config if present
    if 'mem_memory-type' in config:
        memory_config['memory-type'] = config['mem_memory-type']
    
    # Check cache - reuse if same config exists
    config_hash = _hash_config(memory_config)
    if config_hash in _MEMORY_CONFIG_CACHE:
        cached_path = _MEMORY_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with hash (not timestamp)
    output_path = os.path.join(CONFIG_OUTPUT_DIR, f"memory_{config_hash}.json")
    
    # Write JSON file
    with open(output_path, 'w') as f:
        json.dump(memory_config, f, indent=4)
    
    # Cache the path
    _MEMORY_CONFIG_CACHE[config_hash] = output_path
    
    return output_path


def generate_all_configs(config: Dict[str, Any]) -> tuple[str, str, str]:
    """
    Generate all three configuration files at once.
    
    This is the recommended way to get config paths for simulation.
    Always generates fresh configs with unique timestamps.
    
    Args:
        config: Configuration dictionary (may contain net_*, sys_*, mem_* prefixed parameters)
    
    Returns:
        Tuple of (system_path, network_path, memory_path) - all absolute paths
    
    Example:
        system, network, memory = config_generator.generate_all_configs(config)
        # Pass these paths to SimulationRunner
    """
    system_path = generate_system_config(config)
    network_path = generate_network_config(config)
    memory_path = generate_memory_config(config)
    
    return system_path, network_path, memory_path
