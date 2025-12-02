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
    "preferred-dataset-splits": 4,
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
    "track-local-mem": 1,
    "local-mem-trace-filename": "mem_trace.json",
    "dump-local-mem-trace": 0
}

DEFAULT_G2_SYSTEM_CONFIG = {
    "scheduling-policy": "LIFO",
    "endpoint-delay": 10,
    "active-chunks-per-dimension": 1,
    "preferred-dataset-splits": 4,
    "all-reduce-implementation": ["halvingDoubling"],
    "all-gather-implementation": ["halvingDoubling"],
    "reduce-scatter-implementation": ["halvingDoubling"],
    "all-to-all-implementation": ["halvingDoubling"],
    "collective-optimization": "localBWAware",
    "local-mem-bw": 3350,
    "local-mem-size": 80,
    "enable_network_logger": 1,
    "boost-mode": 0,
    "peak-perf": 989,
    "roofline-enabled": 1,
    "trace-enabled": 1,
    "track-local-mem": 1,
    "local-mem-trace-filename": "mem_trace.json",
    "dump-local-mem-trace": 0
}

DEFAULT_NETWORK_CONFIG = {
    "topology": ["Switch", "Switch"],
    "npus_count": [8, 2],
    "bandwidth": [450.0, 100.0],
    "bandwidth_unit": "GB/s",
    "latency": [0.0, 0.0],
    "packet_size": 1500,
    "header_size": 48
}

DEFAULT_G2_NETWORK_CONFIG = {
    "npus_count": [ 16 ],
    "bandwidth_unit": "GB/s",
    "packet_size": 1500,
    "header_size": 44,
    "topology_file": os.environ['ASTRA_SIM_ROOT'] + "/upc/configuration/g2/FoldedClos_128_topology.json"
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
    param_mapping = [
        'scheduling-policy',
        'endpoint-delay',
        'active-chunks-per-dimension',
        'preferred-dataset-splits',
        'collective-optimization',
        'local-mem-bw',
        'local-mem-size',
        'boost-mode',
        'peak-perf'
    ]
    
    for key in param_mapping:
        if key in config:
            system_config[key] = config[key]
    
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
    If config contains network parameters or cluster info, they override defaults.
    Otherwise, uses DEFAULT_NETWORK_CONFIG.
    
    Args:
        config: Configuration dictionary (may contain bandwidth, cluster, npus_per_dim, and npu_count parameters)
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _NETWORK_CONFIG_CACHE
    
    # Start with defaults
    network_config = DEFAULT_NETWORK_CONFIG.copy()
    
    # Override npus_count if npus_per_dim is provided (from enriched cluster config)
    if 'npus_per_dim' in config:
        network_config['npus_count'] = config['npus_per_dim']
    elif 'npu_count' in config and isinstance(config['npu_count'], (list, tuple)):
        network_config['npus_count'] = list(config['npu_count'])

    # Override with bandwidth values from config if present
    if 'intra-node-bw' in config or 'inter-node-bw' in config:
        bandwidth = network_config['bandwidth'].copy()
        if 'intra-node-bw' in config:
            bandwidth[0] = config['intra-node-bw']
        if 'inter-node-bw' in config:
            for i in range(1, len(network_config['npus_count'])):
                bandwidth[i] = config['inter-node-bw']
        network_config['bandwidth'] = bandwidth
    
    
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


def generate_g2_system_config(config: Dict[str, Any]) -> str:
    """
    Generate G2-specific system configuration JSON file.
    
    Differs from analytical system config by using 1D lists for collective algorithms
    instead of 2D lists (only one level of hierarchy in G2).
    
    Args:
        config: Configuration dictionary (may contain sys_* prefixed parameters)
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _SYSTEM_CONFIG_CACHE
    
    # Start with G2 defaults (1D lists for collective algorithms)
    system_config = DEFAULT_G2_SYSTEM_CONFIG.copy()
    
    # Override with custom values from config if present
    param_mapping = [
        'scheduling-policy',
        'endpoint-delay',
        'active-chunks-per-dimension',
        'preferred-dataset-splits',
        'collective-optimization',
        'local-mem-bw',
        'local-mem-size',
        'boost-mode',
        'peak-perf'
    ]
    
    for key in param_mapping:
        if key in config:
            system_config[key] = config[key]
    
    # Ensure collective algorithms are 1D lists (G2 requirement)
    for algo_key in ['all-reduce-implementation', 'all-gather-implementation', 
                     'reduce-scatter-implementation', 'all-to-all-implementation']:
        if algo_key in system_config and isinstance(system_config[algo_key], list):
            # If it's a 2D list, take the first element; otherwise keep as is
            if len(system_config[algo_key]) > 0 and isinstance(system_config[algo_key][0], list):
                system_config[algo_key] = system_config[algo_key][0]
    
    # Check cache - reuse if same config exists
    config_hash = _hash_config(system_config)
    if config_hash in _SYSTEM_CONFIG_CACHE:
        cached_path = _SYSTEM_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate filename with hash
    output_path = os.path.join(CONFIG_OUTPUT_DIR, f"system_g2_{config_hash}.json")
    
    # Write JSON file
    with open(output_path, 'w') as f:
        json.dump(system_config, f, indent=4)
    
    # Cache the path
    _SYSTEM_CONFIG_CACHE[config_hash] = output_path
    
    return output_path



def generate_g2_network_config(config: Dict[str, Any], net_sim_config: Dict[str, Any]) -> str:
    """
    Generate G2-specific network configuration JSON file.
    
    Uses the topology generator to create FoldedClos, Jellyfish, or Dragonfly topologies.
    Extracts topology parameters from net_sim_config and bandwidth from config.
    
    Args:
        config: Configuration dictionary containing bandwidth parameters:
                - 'intra-node-bw': Intra-node bandwidth
                - 'inter-node-bw': Inter-node bandwidth
                - 'npu_count': Total number of NPUs (optional, used if not in topology_config)
                - 'npus_per_dim': NPUs per dimension (optional, used to calculate num_npus)
        net_sim_config: Simulation configuration dictionary:
                - 'topology': 'FoldedClos', 'Jellyfish', or 'Dragonfly'
                - 'paths_mode': 'ECMP', 'Uniform', etc.
                - 'topology_config': Topology-specific parameters
                  
                  Common parameter (all topologies):
                  - 'num_npus': Total number of NPUs (will deduce topology params)
                  
                  FoldedClos-specific (if not using num_npus):
                  - 'K': K-port switches (alternative to num_npus)
                  - 'npus_per_node': NPUs per node (default: 1)
                  - 'intra_node_topology': 'fully_connected', 'ring', or 'switch'
                  
                  Dragonfly-specific (if not using num_npus):
                  - 'G': Number of groups
                  - 'A': Switches per group  
                  - 'h': Inter-group links
                  - 'concentration': NPUs per switch
                  
                  Jellyfish-specific (if not using num_npus):
                  - 'num_switches': Total switches
                  - 'degree': Ports per switch
                  - 'num_hosts_per_switch': NPUs per switch
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _NETWORK_CONFIG_CACHE
    
    # Import here to avoid circular dependency
    import sys
    create_topology_path = os.path.join(os.environ.get('ASTRA_SIM_ROOT', ''), 'upc', 'create_topology')
    if create_topology_path not in sys.path:
        sys.path.insert(0, create_topology_path)
    
    from create_topology_main import generate_topology_files
    
    # Extract topology configuration
    topology = net_sim_config.get('topology', 'FoldedClos')
    paths_mode = net_sim_config.get('paths_mode', 'ECMP')
    topology_config = net_sim_config.get('topology_config', {}).copy()
    
    # If num_npus is not in topology_config, try to get it from config
    if 'num_npus' not in topology_config:
        if 'npu_count' in config:
            topology_config['num_npus'] = config['npu_count']
        elif 'npus_per_dim' in config:
            # Calculate num_npus from npus_per_dim
            import math
            npus_per_dim = config['npus_per_dim']
            topology_config['num_npus'] = math.prod(npus_per_dim) if hasattr(math, 'prod') else eval('*'.join(map(str, npus_per_dim)))
    
    # Build bandwidth configuration from config parameters
    bw_config = topology_config.get('bandwidth_config', {}).copy()
    bw_unit = topology_config.get('bw_unit', 'GB/s')
    
    # Deduce topology parameters from desired NPU count if needed
    if topology == 'FoldedClos':
        # If num_npus is provided instead of K, calculate K
        if 'num_npus' in topology_config and 'K' not in topology_config:
            num_npus = topology_config['num_npus']
            npus_per_node = topology_config.get('npus_per_node', 1)
            # Formula: num_npus = (K^3 / 4) * npus_per_node
            # Solve for K: K = (4 * num_npus / npus_per_node)^(1/3)
            total_nodes = num_npus / npus_per_node
            K = round((4 * total_nodes) ** (1/3))
            
            # Verify the calculation and adjust K if needed
            actual_nodes = int(K**3 / 4)
            actual_npus = actual_nodes * npus_per_node
            
            # Keep incrementing K until actual_npus is >= num_npus AND is a power of 2
            def is_power_of_2(n):
                return n > 0 and (n & (n - 1)) == 0
            
            while actual_npus < num_npus or not is_power_of_2(actual_npus):
                K += 1
                actual_nodes = int(K**3 / 4)
                actual_npus = actual_nodes * npus_per_node
            
            topology_config['K'] = K
        
        # Map bandwidth parameters
        if 'intra-node-bw' in config:
            bw_config['intra_node'] = config['intra-node-bw']
        if 'inter-node-bw' in config:
            bw_config['host_edge'] = config['inter-node-bw']
            bw_config['edge_agg'] = config['inter-node-bw']
            bw_config['agg_core'] = config['inter-node-bw']
    
    elif topology == 'Dragonfly':
        # If num_npus is provided, calculate G, A, or concentration
        if 'num_npus' in topology_config:
            num_npus = topology_config['num_npus']
            # Formula: num_npus = G * A * concentration
            if 'G' not in topology_config and 'A' not in topology_config:
                # Default: G=4, A=4, deduce concentration
                G = 4
                A = 4
                concentration = num_npus // (G * A)
                topology_config['G'] = G
                topology_config['A'] = A
                topology_config['concentration'] = concentration
            elif 'G' in topology_config and 'A' in topology_config:
                # Deduce concentration
                G = topology_config['G']
                A = topology_config['A']
                topology_config['concentration'] = num_npus // (G * A)
        
        # Map bandwidth parameters
        if 'intra-node-bw' in config:
            bw_config['intra_group'] = config['intra-node-bw']
        if 'inter-node-bw' in config:
            bw_config['host_switch'] = config['inter-node-bw']
            bw_config['inter_group'] = config['inter-node-bw']
    
    elif topology == 'Jellyfish':
        # If num_npus is provided, calculate num_switches or hosts_per_switch
        if 'num_npus' in topology_config:
            num_npus = topology_config['num_npus']
            # Formula: num_npus = num_switches * num_hosts_per_switch
            if 'num_switches' in topology_config:
                topology_config['num_hosts_per_switch'] = num_npus // topology_config['num_switches']
            elif 'num_hosts_per_switch' in topology_config:
                topology_config['num_switches'] = num_npus // topology_config['num_hosts_per_switch']
            else:
                # Default: 1 host per switch
                topology_config['num_switches'] = num_npus
                topology_config['num_hosts_per_switch'] = 1
        
        # Map bandwidth parameters
        if 'inter-node-bw' in config:
            bw_config['host_switch'] = config['inter-node-bw']
            bw_config['switch_switch'] = config['inter-node-bw']
    
    topology_config['bandwidth_config'] = bw_config
    topology_config['bw_unit'] = bw_unit
    
    # Create cache key from all parameters
    cache_key = {
        'topology': topology,
        'paths_mode': paths_mode,
        'topology_config': topology_config
    }
    config_hash = _hash_config(cache_key)
    
    # Check cache - reuse if same config exists
    if config_hash in _NETWORK_CONFIG_CACHE:
        cached_path = _NETWORK_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            # Also check if the topology file still exists
            with open(cached_path, 'r') as f:
                cached_config = yaml.safe_load(f)
            if os.path.exists(cached_config['topology_file']):
                return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate topology file with hash-based name in temp/config
    topology_file_name = f"topology_g2_{config_hash}"
    topology_file_path = os.path.join(CONFIG_OUTPUT_DIR, topology_file_name)
        
    # Generate topology files (this creates G2 JSON file in current directory)
    generate_topology_files(
        topology=topology,
        paths_mode=paths_mode,
        config=topology_config,
        output_dir=CONFIG_OUTPUT_DIR,
        base_filename=topology_file_name
    )

    # Create the G2 network config that references this topology file
    g2_network_config = {
        "npus_count": [topology_config['num_npus']],
        "bandwidth_unit": bw_unit,
        "packet_size": 1500,
        "header_size": 44,
        "topology_file": topology_file_path + ".json"
    }
    
    # Generate network config YML file that points to the topology
    network_config_path = os.path.join(CONFIG_OUTPUT_DIR, f"network_g2_{config_hash}.yml")
    
    # Write the network config YML file (matching FoldedClos_16_config.yml format)
    with open(network_config_path, 'w') as f:
        yaml.dump(g2_network_config, f, default_flow_style=None)
    
    # Cache the path
    _NETWORK_CONFIG_CACHE[config_hash] = network_config_path
    
    return network_config_path


def generate_g2_network_config_old(config: Dict[str, Any], net_sim_config: Dict[str, Any]) -> str:
    """
    Generate G2-specific network configuration JSON file.
    
    Uses the topology generator to create FoldedClos, Jellyfish, or Dragonfly topologies.
    Extracts topology parameters from net_sim_config and bandwidth from config.
    
    Args:
        config: Configuration dictionary containing bandwidth parameters:
                - 'intra-node-bw': Intra-node bandwidth
                - 'inter-node-bw': Inter-node bandwidth
        net_sim_config: Simulation configuration dictionary:
                - 'topology': 'FoldedClos', 'Jellyfish', or 'Dragonfly'
                - 'paths_mode': 'ECMP', 'Uniform', etc.
                - 'topology_config': Topology-specific parameters
                  
                  Common parameter (all topologies):
                  - 'num_npus': Total number of NPUs (will deduce topology params)
                  
                  FoldedClos-specific (if not using num_npus):
                  - 'K': K-port switches (alternative to num_npus)
                  - 'npus_per_node': NPUs per node (default: 1)
                  - 'intra_node_topology': 'fully_connected', 'ring', or 'switch'
                  
                  Dragonfly-specific (if not using num_npus):
                  - 'G': Number of groups
                  - 'A': Switches per group  
                  - 'h': Inter-group links
                  - 'concentration': NPUs per switch
                  
                  Jellyfish-specific (if not using num_npus):
                  - 'num_switches': Total switches
                  - 'degree': Ports per switch
                  - 'num_hosts_per_switch': NPUs per switch
    
    Returns:
        Absolute path to config file (reused if same parameters)
    """
    global _NETWORK_CONFIG_CACHE
    
    # Import here to avoid circular dependency
    import sys
    create_topology_path = os.path.join(os.environ.get('ASTRA_SIM_ROOT', ''), 'upc', 'create_topology')
    if create_topology_path not in sys.path:
        sys.path.insert(0, create_topology_path)
    
    from create_topology_main import generate_topology_files
    
    # Extract topology configuration
    topology = net_sim_config.get('topology', 'FoldedClos')
    paths_mode = net_sim_config.get('paths_mode', 'ECMP')
    topology_config = net_sim_config.get('topology_config', {}).copy()
    
    # Build bandwidth configuration from config parameters
    bw_config = topology_config.get('bandwidth_config', {}).copy()
    bw_unit = topology_config.get('bw_unit', 'GB/s')
    
    # Deduce topology parameters from desired NPU count if needed
    if topology == 'FoldedClos':
        # If num_npus is provided instead of K, calculate K
        if 'num_npus' in topology_config and 'K' not in topology_config:
            num_npus = topology_config['num_npus']
            npus_per_node = topology_config.get('npus_per_node', 1)
            # Formula: num_npus = (K^3 / 4) * npus_per_node
            # Solve for K: K = (4 * num_npus / npus_per_node)^(1/3)
            total_nodes = num_npus / npus_per_node
            K = round((4 * total_nodes) ** (1/3))
            
            # Verify the calculation and adjust K if needed
            actual_nodes = int(K**3 / 4)
            actual_npus = actual_nodes * npus_per_node
            
            # Keep incrementing K until actual_npus is >= num_npus AND is a power of 2
            def is_power_of_2(n):
                return n > 0 and (n & (n - 1)) == 0
            
            while actual_npus < num_npus or not is_power_of_2(actual_npus):
                K += 1
                actual_nodes = int(K**3 / 4)
                actual_npus = actual_nodes * npus_per_node
            
            topology_config['K'] = K
        
        # Map bandwidth parameters
        if 'intra-node-bw' in config:
            bw_config['intra_node'] = config['intra-node-bw']
        if 'inter-node-bw' in config:
            bw_config['host_edge'] = config['inter-node-bw']
            bw_config['edge_agg'] = config['inter-node-bw']
            bw_config['agg_core'] = config['inter-node-bw']
    
    elif topology == 'Dragonfly':
        # If num_npus is provided, calculate G, A, or concentration
        if 'num_npus' in topology_config:
            num_npus = topology_config['num_npus']
            # Formula: num_npus = G * A * concentration
            if 'G' not in topology_config and 'A' not in topology_config:
                # Default: G=4, A=4, deduce concentration
                G = 4
                A = 4
                concentration = num_npus // (G * A)
                topology_config['G'] = G
                topology_config['A'] = A
                topology_config['concentration'] = concentration
            elif 'G' in topology_config and 'A' in topology_config:
                # Deduce concentration
                G = topology_config['G']
                A = topology_config['A']
                topology_config['concentration'] = num_npus // (G * A)
        
        # Map bandwidth parameters
        if 'intra-node-bw' in config:
            bw_config['intra_group'] = config['intra-node-bw']
        if 'inter-node-bw' in config:
            bw_config['host_switch'] = config['inter-node-bw']
            bw_config['inter_group'] = config['inter-node-bw']
    
    elif topology == 'Jellyfish':
        # If num_npus is provided, calculate num_switches or hosts_per_switch
        if 'num_npus' in topology_config:
            num_npus = topology_config['num_npus']
            # Formula: num_npus = num_switches * num_hosts_per_switch
            if 'num_switches' in topology_config:
                topology_config['num_hosts_per_switch'] = num_npus // topology_config['num_switches']
            elif 'num_hosts_per_switch' in topology_config:
                topology_config['num_switches'] = num_npus // topology_config['num_hosts_per_switch']
            else:
                # Default: 1 host per switch
                topology_config['num_switches'] = num_npus
                topology_config['num_hosts_per_switch'] = 1
        
        # Map bandwidth parameters
        if 'inter-node-bw' in config:
            bw_config['host_switch'] = config['inter-node-bw']
            bw_config['switch_switch'] = config['inter-node-bw']
    
    topology_config['bandwidth_config'] = bw_config
    topology_config['bw_unit'] = bw_unit
    
    # Create cache key from all parameters
    cache_key = {
        'topology': topology,
        'paths_mode': paths_mode,
        'topology_config': topology_config
    }
    config_hash = _hash_config(cache_key)
    
    # Check cache - reuse if same config exists
    if config_hash in _NETWORK_CONFIG_CACHE:
        cached_path = _NETWORK_CONFIG_CACHE[config_hash]
        if os.path.exists(cached_path):
            # Also check if the topology file still exists
            with open(cached_path, 'r') as f:
                cached_config = yaml.safe_load(f)
            if os.path.exists(cached_config['topology_file']):
                return cached_path
    
    # Create directory if needed
    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    # Generate topology file with hash-based name in temp/config
    topology_file_name = f"topology_g2_{config_hash}"
    topology_file_path = os.path.join(CONFIG_OUTPUT_DIR, f"{topology_file_name}.json")
    
   
    
    # Generate topology files directly in CONFIG_OUTPUT_DIR with hash-based name
    generate_topology_files(
        topology=topology,
        paths_mode=paths_mode,
        config=topology_config,
        output_dir=CONFIG_OUTPUT_DIR,
        base_filename=topology_file_name
    )
    
    # Verify the topology file was created
    if not os.path.exists(topology_file_path):
        raise FileNotFoundError(f"Failed to generate topology file: {topology_file_path}")
    
    # Create the G2 network config that references this topology file
    g2_network_config = {
        "available_npus": actual_npus,
        "npus_count": [64],
        "bandwidth_unit": bw_unit,
        "packet_size": 1500,
        "header_size": 44,
        "topology_file": os.environ['ASTRA_SIM_ROOT'] + "/upc/configuration/g2/FoldedClos_128_topology.json"
    }
    
    # Generate network config YML file that points to the topology
    network_config_path = os.path.join(CONFIG_OUTPUT_DIR, f"network_g2_{config_hash}.yml")
    
    # Write the network config YML file (matching FoldedClos_16_config.yml format)
    with open(network_config_path, 'w') as f:
        yaml.dump(g2_network_config, f, default_flow_style=None)
    
    # Cache the path
    _NETWORK_CONFIG_CACHE[config_hash] = network_config_path
    
    return network_config_path

def generate_all_configs(config: Dict[str, Any], net_sim_config: Dict[str, Any] = {"sim_type": "analytical_unaware"}) -> tuple[str, str, str]:
    """
    Generate all three configuration files at once.
    
    This is the recommended way to get config paths for simulation.
    Always generates fresh configs with unique timestamps.
    
    Args:
        config: Configuration dictionary (may contain net_*, sys_*, mem_* prefixed parameters)
        net_sim_config: Network simulation configuration dict with 'sim_type' and topology info
    
    Returns:
        Tuple of (system_path, network_path, memory_path) - all absolute paths
    
    Example:
        system, network, memory = config_generator.generate_all_configs(config)
        # Pass these paths to SimulationRunner
    """
    memory_path = generate_memory_config(config)
    if net_sim_config.get('sim_type') == "analytical_unaware":
        system_path = generate_system_config(config)
        network_path = generate_network_config(config)
    elif net_sim_config.get('sim_type') == "g2":
        try:
            print(f"[config_generator] Generating G2 system config...")
            system_path = generate_g2_system_config(config)
            print(f"[config_generator] G2 system config generated: {system_path}")
        except Exception as e:
            print(f"[config_generator] ERROR in generate_g2_system_config: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        try:
            print(f"[config_generator] Generating G2 network config...")
            print(f"[config_generator] Config: {config}")
            print(f"[config_generator] Net sim config: {net_sim_config}")
            network_path = generate_g2_network_config(config, net_sim_config)
            print(f"[config_generator] G2 network config generated: {network_path}")
        except Exception as e:
            print(f"[config_generator] ERROR in generate_g2_network_config: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return system_path, network_path, memory_path
