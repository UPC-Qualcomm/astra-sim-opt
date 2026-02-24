#!/usr/bin/env python3
"""
Generate training data for ML model to predict AstraSim peak memory usage.
This script generates various model configurations and parallelism strategies,
runs simulations, and collects results.
"""

import os
import sys
import subprocess
import argparse
import json
import yaml
import pandas as pd
import multiprocessing
from itertools import product
from pathlib import Path
import re
import time
from functools import partial
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Configuration choices for random model generation
VOCAB_CHOICES = [
    16000, 32000, 48000, 50000,
    64000, 100000, 128000,
    200000, 256000
]

DMODEL_CHOICES = [
    1024, 1536, 2048, 2560,
    3072, 4096, 5120, 6144,
    8192, 10240, 12288,
    16384, 20480, 24576
]

NUM_LAYERS_CHOICES = [
    12, 16, 24, 32, 40,
    48, 64, 80, 96,
    120, 144, 160
]

HEAD_CHOICES = [
    8, 12, 16, 20, 24,
    32, 40, 48, 64,
    80, 96, 128, 160
]

SEQ_CHOICES = [
    512, 1024, 2048,
    4096, 8192,
    16384, 32768,
    65536, 131072
]

BATCH_CHOICES = [
    64, 128, 256
]

MICRO_BATCH_CHOICES = [
    1, 2, 4, 8, 16, 32, 64
]

NPUS_CHOICES = [
    16, 32,
    64, 128, 256, 512,
    1024, 2048
]


def create_network_yaml(num_npus, output_path):
    """Create network YAML with adjusted topology for given NPU count."""
    if num_npus < 8 or num_npus % 8 != 0:
        raise ValueError(f"num_npus must be >= 8 and divisible by 8, got {num_npus}")
    
    dim1 = 8
    dim2 = num_npus // 8
    
    network_config = {
        'topology': ['Switch', 'Switch'],
        'npus_count': [dim1, dim2],
        'bandwidth': [900, 200.0],  # GB/s
        'latency': [500.0, 500.0],  # ns
        'packet_size': 1500,  # bytes
        'header_size': 48  # bytes
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(network_config, f, default_flow_style=False)
    
    return output_path

def get_valid_parallelism_strategies(num_npus, num_stacks, max_dp=None, max_mp=None, max_sp=None):
    """Generate valid parallelism strategies where dp*mp*sp*pp = num_npus.
    
    Args:
        num_npus: Total number of NPUs
        num_stacks: Number of model layers/stacks (constrains PP)
        max_dp: Maximum data parallelism (default: num_npus)
        max_mp: Maximum model/tensor parallelism (default: min(64, num_npus))
        max_sp: Maximum sequence parallelism (default: min(16, num_npus))
    """
    strategies = []
    
    # Set defaults if not provided
    if max_dp is None:
        max_dp = num_npus
    if max_mp is None:
        max_mp = min(64, num_npus)  # TP typically doesn't exceed 64
    if max_sp is None:
        max_sp = min(16, num_npus)  # SP typically doesn't exceed 16
    
    # PP is constrained by number of layers
    max_pp = min(num_stacks, num_npus)
    
    # Define possible values for each dimension
    dp_values = [d for d in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] if d <= min(max_dp, num_npus)]
    mp_values = [m for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] if m <= min(max_mp, num_npus)]
    sp_values = [s for s in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] if s <= min(max_sp, num_npus)]
    pp_values = [p for p in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] if p <= max_pp]
    fsdp_values = [0, 1]  # weight_sharded
    
    for dp, mp, sp, pp, fsdp in product(dp_values, mp_values, sp_values, pp_values, fsdp_values):
        if dp * mp * sp * pp == num_npus:
            strategies.append({
                'dp': dp,
                'mp': mp,
                'sp': sp,
                'pp': pp,
                'fsdp': fsdp
            })
    
    return strategies


def get_boundary_parallelism_strategies(num_npus, num_stacks):
    """Generate boundary case parallelism strategies where all NPUs are assigned to one dimension.
    
    Returns strategies where:
    - All NPUs in DP: (num_npus, 1, 1, 1)
    - All NPUs in MP: (1, num_npus, 1, 1) if num_npus <= 64
    - All NPUs in SP: (1, 1, num_npus, 1) if num_npus <= 16
    - All NPUs in PP: (1, 1, 1, num_npus) if num_npus <= num_stacks
    """
    boundary_strategies = []
    fsdp_values = [0, 1]
    
    for fsdp in fsdp_values:
        # All in DP
        boundary_strategies.append({
            'dp': num_npus,
            'mp': 1,
            'sp': 1,
            'pp': 1,
            'fsdp': fsdp
        })
        
        # All in MP (if reasonable)
        if num_npus <= 64:
            boundary_strategies.append({
                'dp': 1,
                'mp': num_npus,
                'sp': 1,
                'pp': 1,
                'fsdp': fsdp
            })
        
        # All in SP (if reasonable)
        if num_npus <= 16:
            boundary_strategies.append({
                'dp': 1,
                'mp': 1,
                'sp': num_npus,
                'pp': 1,
                'fsdp': fsdp
            })
        
        # All in PP (if valid)
        if num_npus <= num_stacks:
            boundary_strategies.append({
                'dp': 1,
                'mp': 1,
                'sp': 1,
                'pp': num_npus,
                'fsdp': fsdp
            })
    
    return boundary_strategies




def estimate_params(dmodel, layers, vocab):
    """Rough GPT-style parameter estimate."""
    transformer_params = 12 * layers * (dmodel ** 2)
    embedding_params = vocab * dmodel
    return transformer_params + embedding_params


def sample_heads(dmodel):
    """Return valid head counts that divide dmodel."""
    return [h for h in HEAD_CHOICES if dmodel % h == 0]


def generate_random_config(config_id, allow_equal_batch_prob=0.25, enforce_micro_batch_ratio=True):
    """Generate one random but valid config.
    
    Args:
        config_id: Unique identifier for the config
        allow_equal_batch_prob: Probability of allowing batch == micro_batch
        enforce_micro_batch_ratio: If True and batch != micro_batch, enforce micro_batch >= batch/4
    """
    max_attempts = 100000
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        
        din = random.choice(VOCAB_CHOICES)
        dmodel = random.choice(DMODEL_CHOICES)
        num_stacks = random.choice(NUM_LAYERS_CHOICES)
        
        valid_heads = sample_heads(dmodel)
        if not valid_heads:
            continue
        
        head = random.choice(valid_heads)
        
        # dff ratio 3x–8x
        dff_ratio = random.uniform(3.0, 8.0)
        dff = int(dff_ratio * dmodel)
        
        seq = random.choice(SEQ_CHOICES)
        batch = random.choice(BATCH_CHOICES)
        
        # allow some cases batch == micro_batch
        if random.random() < allow_equal_batch_prob:
            micro_batch = batch
        else:
            possible_micro = [
                mb for mb in MICRO_BATCH_CHOICES
                if mb <= batch and batch % mb == 0
            ]
            
            # Apply micro_batch ratio rule: if enforce_micro_batch_ratio is True,
            # micro_batch must be at least batch/4 (i.e., max 4x ratio)
            if enforce_micro_batch_ratio:
                min_micro_batch = batch // 4
                possible_micro = [
                    mb for mb in possible_micro
                    if mb >= min_micro_batch
                ]
            
            if not possible_micro:
                continue
            micro_batch = random.choice(possible_micro)
        
        params = estimate_params(dmodel, num_stacks, din)
        
        # restrict to 1B–1T
        if params < 1e9 or params > 1e12:
            continue
        
        return {
            'name': f'random_model_{config_id}',
            'din': din,
            'dmodel': dmodel,
            'dff': dff,
            'batch': batch,
            'micro_batch': micro_batch,
            'seq': seq,
            'head': head,
            'num_stacks': num_stacks,
            'estimated_params': int(params)
        }
    
    # If we fail to generate after max_attempts, return a safe default
    return {
        'name': f'random_model_{config_id}_default',
        'din': 50000,
        'dmodel': 4096,
        'dff': 16384,
        'batch': 2048,
        'micro_batch': 32,
        'seq': 2048,
        'head': 32,
        'num_stacks': 32,
        'estimated_params': int(estimate_params(4096, 32, 50000))
    }


def generate_model_configs(num_models=100, enforce_micro_batch_ratio=True):
    """Generate random model configurations with diversity."""
    configs = []
    
    # Set seed for reproducibility of model generation
    random.seed(42)
    
    for i in range(num_models):
        config = generate_random_config(i + 1, enforce_micro_batch_ratio=enforce_micro_batch_ratio)
        configs.append(config)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_models} random models...")
    
    return configs

def generate_workload(model_config, parallelism, num_npus, base_dir):
    """Generate a single workload configuration."""
    dp = parallelism['dp']
    mp = parallelism['mp']
    sp = parallelism['sp']
    pp = parallelism['pp']
    fsdp = parallelism['fsdp']
    
    folder_name = f"ml_data_npus{num_npus}_{model_config['name']}_dp{dp}_mp{mp}_sp{sp}_pp{pp}_fsdp{fsdp}"
    workload_dir = os.path.join(base_dir, 'workload', folder_name)
    output_name = f"{dp}_{mp}_{sp}_{pp}_{fsdp}.%d.et"
    
    # Call main.py directly from symbolic_tensor_graph
    main_py_dir = os.path.join(base_dir, '..', 'extern', 'symbolic_tensor_graph')
    
    cmd = [
        'python', 'main.py',
        '--output_dir', workload_dir,
        '--output_name', output_name,
        '--dp', str(dp),
        '--tp', str(mp),
        '--sp', str(sp),
        '--pp', str(pp),
        '--dvocal', str(model_config['din']),
        '--dmodel', str(model_config['dmodel']),
        '--dff', str(model_config['dff']),
        '--batch', f"[{model_config['batch']}]",
        '--micro_batch', str(model_config['micro_batch']),
        '--seq', str(model_config['seq']),
        '--head', str(model_config['head']),
        '--kvhead', str(model_config['head']),
        '--num_stacks', str(model_config['num_stacks']),
        '--weight_sharded', 'True' if fsdp else 'False',
        '--activation_recompute', 'False',
        '--tpsp', 'True',
        '--model_type', 'dense',
        '--mixed_precision', 'False',
        '--print_gpu_vram', 'True',
        '--ep', '1',
        '--experts', '1',
        '--kexperts', '1',
        '--chakra_schema_version', 'v0.0.4'
    ]
    
    return {
        'cmd': cmd,
        'cwd': main_py_dir,
        'folder_name': folder_name,
        'workload_dir': workload_dir,
        'model_config': model_config,
        'parallelism': parallelism,
        'num_npus': num_npus
    }

def run_single_simulation(config, base_dir, sim_type='analytical_unaware'):
    """Run a single AstraSim simulation and extract results."""
    folder_name = config['folder_name']
    num_npus = config['num_npus']
    
    try:
        # Step 1: Generate workload
        print(f"Generating workload for {folder_name}...")
        result = subprocess.run(
            config['cmd'],
            cwd=config['cwd'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"Failed to generate workload for {folder_name}")
            print(f"  Command: {' '.join(config['cmd'])}")
            print(f"  STDERR: {result.stderr[:500]}")
            print(f"  STDOUT: {result.stdout[:500]}")
            return None
        
        # Check if workload directory was created
        if not os.path.exists(config['workload_dir']):
            print(f"Workload directory not created: {config['workload_dir']}")
            print(f"  STDOUT: {result.stdout[:300]}")
            return None
        
        # Step 2: Create network YAML
        network_yaml = os.path.join(base_dir, 'ml_model', 'networks', f'network_{num_npus}npus.yml')
        create_network_yaml(num_npus, network_yaml)
        
        # Step 3: Run simulation
        workload_dir = config['workload_dir']
        output_dir = os.path.join(base_dir, 'ml_model', 'output', folder_name)
        network_log_dir = os.path.join(base_dir, 'ml_model', 'network_log', folder_name)
        system_config = os.path.join(base_dir, 'configuration', 'FoldedClos_my_BARRIER_sys.json')
        memory_config = os.path.join(base_dir, 'configuration', 'RemoteMemory.json')
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(network_log_dir, exist_ok=True)
        
        print(f"Running simulation for {folder_name}...")
        sim_cmd = [
            'python', 'run_astrasim.py',
            '--workload_dir', workload_dir,
            '--system', system_config,
            '--network', network_yaml,
            '--memory', memory_config,
            '--output_dir', output_dir,
            '--network_log', network_log_dir,
            '--sim_type', sim_type
        ]
        
        result = subprocess.run(
            sim_cmd,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode != 0:
            print(f"Failed to run simulation for {folder_name}")
            print(f"  STDERR: {result.stderr[:500]}")
            print(f"  STDOUT: {result.stdout[:500]}")
            return None
        
        # Debug: Check if output files were created
        log_files_created = list(Path(output_dir).glob('*.log'))
        print(f"  Simulation completed. Created {len(log_files_created)} log files")
        
        # Step 4: Parse results
        print(f"Parsing results for {folder_name}...")
        results = parse_simulation_results(output_dir, config)
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"Timeout for {folder_name}")
        return None
    except Exception as e:
        print(f"Error processing {folder_name}: {e}")
        return None

def cleanup_simulation_files(config, base_dir):
    """Remove workload and output files to save storage."""
    import shutil
    
    folder_name = config['folder_name']
    
    # Remove workload directory
    workload_dir = config['workload_dir']
    if os.path.exists(workload_dir):
        try:
            shutil.rmtree(workload_dir)
            print(f"  Cleaned up workload: {workload_dir}")
        except Exception as e:
            print(f"  Warning: Could not remove {workload_dir}: {e}")
    
    # Remove output directory
    output_dir = os.path.join(base_dir, 'ml_model', 'output', folder_name)
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"  Cleaned up output: {output_dir}")
        except Exception as e:
            print(f"  Warning: Could not remove {output_dir}: {e}")
    
    # Remove network log directory
    network_log_dir = os.path.join(base_dir, 'ml_model', 'network_log', folder_name)
    if os.path.exists(network_log_dir):
        try:
            shutil.rmtree(network_log_dir)
            print(f"  Cleaned up network logs: {network_log_dir}")
        except Exception as e:
            print(f"  Warning: Could not remove {network_log_dir}: {e}")

def parse_simulation_results(output_dir, config):
    """Parse simulation output to extract peak memory, cycles, and exposed communication."""
    results = []
    
    # Find all .log files in output directory
    log_files = list(Path(output_dir).glob('*.log'))
    
    if not log_files:
        print(f"  WARNING: No log files found in {output_dir}")
        return results
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            if not content:
                print(f"  WARNING: Empty log file {log_file}")
                continue
            
            # Extract peak memory for each sys  
            peak_memory_pattern = r'sys\[(\d+)\] peak memory usage: ([\d.]+) ([MG])B'
            cycles_pattern = r'sys\[(\d+)\].*?Wall time: (\d+)'
            exposed_comm_pattern = r'sys\[(\d+)\] finished, (\d+) cycles, exposed communication (\d+) cycles'
            
            peak_memories = re.findall(peak_memory_pattern, content)
            cycles = re.findall(cycles_pattern, content)
            exposed_comms = re.findall(exposed_comm_pattern, content)
            
            if not peak_memories:
                print(f"  WARNING: No peak memory data found in {log_file.name}")
                # Try to find what patterns exist
                if 'peak memory' in content.lower():
                    print(f"  DEBUG: File contains 'peak memory' text but pattern didn't match")
                    # Show a sample line
                    for line in content.split('\n'):
                        if 'peak memory' in line.lower():
                            print(f"    Sample line: {line[:150]}")
                            break
            
            # Create mapping of sys_id to metrics
            sys_metrics = {}
            
            for sys_id, peak_mem, unit in peak_memories:
                sys_id = int(sys_id)
                if sys_id not in sys_metrics:
                    sys_metrics[sys_id] = {}
                # Convert MB to GB if needed
                peak_mem_gb = float(peak_mem) / 1024 if unit == 'M' else float(peak_mem)
                sys_metrics[sys_id]['peak_memory_gb'] = peak_mem_gb
            
            for sys_id, wall_time in cycles:
                sys_id = int(sys_id)
                if sys_id not in sys_metrics:
                    sys_metrics[sys_id] = {}
                sys_metrics[sys_id]['wall_time_cycles'] = int(wall_time)
            
            for sys_id, total_cycles, exposed_cycles in exposed_comms:
                sys_id = int(sys_id)
                if sys_id not in sys_metrics:
                    sys_metrics[sys_id] = {}
                sys_metrics[sys_id]['total_cycles'] = int(total_cycles)
                sys_metrics[sys_id]['exposed_comm_cycles'] = int(exposed_cycles)
            
            # If we have any results, take the average across all NPUs
            if sys_metrics:
                avg_peak_memory = sum(m.get('peak_memory_gb', 0) for m in sys_metrics.values()) / len(sys_metrics)
                avg_wall_time = sum(m.get('wall_time_cycles', 0) for m in sys_metrics.values()) / len(sys_metrics)
                avg_exposed_comm = sum(m.get('exposed_comm_cycles', 0) for m in sys_metrics.values()) / len(sys_metrics)
                
                result = {
                    # Model parameters
                    'din': config['model_config']['din'],
                    'dmodel': config['model_config']['dmodel'],
                    'dff': config['model_config']['dff'],
                    'batch': config['model_config']['batch'],
                    'micro_batch': config['model_config']['micro_batch'],
                    'seq': config['model_config']['seq'],
                    'head': config['model_config']['head'],
                    'num_stacks': config['model_config']['num_stacks'],
                    
                    # Parallelism strategy
                    'dp': config['parallelism']['dp'],
                    'mp': config['parallelism']['mp'],
                    'sp': config['parallelism']['sp'],
                    'pp': config['parallelism']['pp'],
                    'fsdp': config['parallelism']['fsdp'],
                    'num_npus': config['num_npus'],
                    
                    # Results
                    'avg_peak_memory_gb': avg_peak_memory,
                    'avg_wall_time_cycles': avg_wall_time,
                    'avg_exposed_comm_cycles': avg_exposed_comm,
                    
                    # Metadata
                    'log_file': str(log_file),
                    'num_systems': len(sys_metrics)
                }
                
                results.append(result)
                
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate ML training data for AstraSim peak memory prediction')
    parser.add_argument('--npu_counts', type=str, default='16,32,64,128,256,512,1024,2048',
                        help='Comma-separated list of NPU counts to test')
    parser.add_argument('--parallel_jobs', type=int, default=4,
                        help='Number of parallel simulation jobs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Number of simulations to run before cleanup (saves storage)')
    parser.add_argument('--output_csv', type=str, default='ml_model/training_data.csv',
                        help='Output CSV file for training data')
    parser.add_argument('--sim_type', type=str, default='analytical_unaware',
                        choices=['analytical_unaware', 'analytical_aware', 'g2'],
                        help='Simulation type')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with minimal configurations')
    parser.add_argument('--strategy_sample_rate', type=float, default=0.2,
                        help='Fraction of parallelism strategies to sample (0.0-1.0, default: 0.2)')
    parser.add_argument('--num_models', type=int, default=100,
                        help='Number of random models to generate (default: 100)')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing CSV file instead of overwriting')
    parser.add_argument('--enforce_micro_batch_ratio', action='store_true', default=True,
                        help='Enforce micro_batch >= batch/4 ratio (max 4x ratio) when batch != micro_batch (default: True)')
    parser.add_argument('--no_enforce_micro_batch_ratio', dest='enforce_micro_batch_ratio', action='store_false',
                        help='Disable micro_batch ratio enforcement')
    
    args = parser.parse_args()
    
    # Get base directory (upc/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Parse NPU counts and filter NPUS_CHOICES to only include specified counts
    npu_counts_arg = [int(n) for n in args.npu_counts.split(',')]
    allowed_npus = [n for n in NPUS_CHOICES if n in npu_counts_arg]
    
    if not allowed_npus:
        print(f"ERROR: No valid NPU counts found. Requested: {npu_counts_arg}, Available: {NPUS_CHOICES}")
        sys.exit(1)
    
    print(f"\nFiltered NPU choices: {allowed_npus}")
    
    # Generate random model configurations
    print(f"\nGenerating {args.num_models} random model configurations...")
    if args.enforce_micro_batch_ratio:
        print("  Enforcing micro_batch >= batch/4 ratio (max 4x ratio)")
    else:
        print("  No micro_batch ratio enforcement")
    
    model_configs = generate_model_configs(
        num_models=args.num_models,
        enforce_micro_batch_ratio=args.enforce_micro_batch_ratio
    )
    
    if args.test_mode:
        print("Running in TEST MODE - using minimal configurations")
        model_configs = model_configs[:2]  # Only first 2 models
    
    print(f"\nGenerating training data for:")
    print(f"  - Model configs: {len(model_configs)}")
    print(f"  - Allowed NPU counts: {allowed_npus}")
    print(f"  - Parallel jobs: {args.parallel_jobs}")
    print(f"  - Strategy sampling rate: {args.strategy_sample_rate:.0%}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Including boundary cases for all models")
    
    # Generate all simulation configurations
    all_configs = []
    model_index = 0  # Track model index for seed generation
    
    for model_config in model_configs:
        # Use allowed_npus directly (user's NPU_COUNTS overrides NPUS_CHOICES)
        valid_npus = allowed_npus
        
        if not valid_npus:
            print(f"  - Skipping {model_config['name']} - no valid NPU counts")
            continue
        
        # Use different seed for each model
        seed_base = 42 + model_index * 10000
        random.seed(seed_base)
        
        # In test mode, use ALL valid NPUs for comprehensive but controlled testing
        # In normal mode, randomly select 1-3 NPU counts per model for diversity
        if args.test_mode:
            selected_npus = valid_npus  # Test all specified NPU counts
        else:
            # Randomly select NPU counts for this model (1-3 NPU counts per model)
            num_npu_configs = random.randint(1, min(3, len(valid_npus)))
            selected_npus = random.sample(valid_npus, num_npu_configs)
        
        for num_npus in selected_npus:
            # Get all valid strategies
            all_strategies = get_valid_parallelism_strategies(num_npus, model_config['num_stacks'])
            
            # Get boundary strategies
            boundary_strategies = get_boundary_parallelism_strategies(num_npus, model_config['num_stacks'])
            
            if args.test_mode:
                # Test mode: ALL boundary cases + 2 random non-boundary
                selected_strategies = boundary_strategies[0:4]
                if len(all_strategies) > len(boundary_strategies):
                    remaining = [s for s in all_strategies if s not in boundary_strategies]
                    selected_strategies.extend(random.sample(remaining, min(2, len(remaining))))
            else:
                # Use different seed for strategy sampling
                seed = seed_base + num_npus
                random.seed(seed)
                
                # Determine sampling rate based on num_npus
                # Use 50% for smaller NPU counts (<=128), default rate for larger
                if num_npus <= 128:
                    sampling_rate = 0.5
                else:
                    sampling_rate = args.strategy_sample_rate
                
                # Sample strategies from non-boundary set
                non_boundary = [s for s in all_strategies if s not in boundary_strategies]
                n_sample = max(1, int(len(non_boundary) * sampling_rate))
                sampled_strategies = random.sample(non_boundary, min(n_sample, len(non_boundary)))
                
                # Combine boundary + sampled strategies
                selected_strategies = boundary_strategies + sampled_strategies
            
            print(f"\n  {'='*70}")
            print(f"  Model: {model_config['name']}")
            print(f"  {'='*70}")
            print(f"  Architecture:")
            print(f"    Vocab size (din): {model_config['din']:,}")
            print(f"    Hidden size (dmodel): {model_config['dmodel']:,}")
            print(f"    FF size (dff): {model_config['dff']:,}")
            print(f"    Num layers: {model_config['num_stacks']}")
            print(f"    Num heads: {model_config['head']}")
            print(f"    Estimated params: {model_config['estimated_params']/1e9:.2f}B")
            print(f"  Training config:")
            print(f"    Batch size: {model_config['batch']}")
            print(f"    Micro batch: {model_config['micro_batch']}")
            print(f"    Sequence length: {model_config['seq']:,}")
            print(f"  NPU config: {num_npus} NPUs")
            print(f"  Parallelism strategies: {len(selected_strategies)} ({len(boundary_strategies)} boundary + {len(selected_strategies) - len(boundary_strategies)} sampled)")
            print(f"  {'='*70}")
            
            for strategy in selected_strategies:
                config = generate_workload(model_config, strategy, num_npus, base_dir)
                all_configs.append(config)
        
        model_index += 1
    
    print(f"\nTotal simulations to run: {len(all_configs)}")
    print(f"Processing in batches of {args.batch_size}")
    
    # Run simulations in batches
    print("\nStarting simulations...")
    start_time = time.time()
    
    # Check if appending and load existing data
    output_path = os.path.join(base_dir, args.output_csv)
    existing_data = []
    if args.append and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            existing_data = existing_df.to_dict('records')
            print(f"Appending to existing file with {len(existing_data)} samples")
        except Exception as e:
            print(f"Warning: Could not read existing file for append: {e}")
            print("Starting fresh...")
    
    run_func = partial(run_single_simulation, base_dir=base_dir, sim_type=args.sim_type)
    
    flat_results = []
    batch_num = 0
    
    for i in range(0, len(all_configs), args.batch_size):
        batch_configs = all_configs[i:i + args.batch_size]
        batch_num += 1
        
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_num}/{(len(all_configs) + args.batch_size - 1) // args.batch_size}")
        print(f"Simulations {i+1} to {min(i + args.batch_size, len(all_configs))} of {len(all_configs)}")
        print(f"{'='*60}")
        
        # Run batch in parallel
        with multiprocessing.Pool(args.parallel_jobs) as pool:
            batch_results = pool.map(run_func, batch_configs)
        
        # Flatten and collect batch results
        for result_list in batch_results:
            if result_list:
                flat_results.extend(result_list)
        
        print(f"\nBatch {batch_num} completed. Collected {len(flat_results)} total data points so far.")
        
        # Clean up batch files to save storage
        print(f"Cleaning up batch {batch_num} files...")
        for config in batch_configs:
            cleanup_simulation_files(config, base_dir)
        
        # Save intermediate results after each batch
        if flat_results:
            # Combine with existing data if appending
            all_data = existing_data + flat_results if args.append else flat_results
            df = pd.DataFrame(all_data)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Intermediate results saved to: {output_path} ({len(all_data)} total samples)")
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All simulations completed in {elapsed_time:.2f} seconds")
    print(f"Total data points collected: {len(flat_results)}")
    print(f"{'='*60}")
    
    # Save to CSV
    if flat_results:
        # Combine with existing data if appending
        all_data = existing_data + flat_results if args.append else flat_results
        df = pd.DataFrame(all_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        if args.append:
            print(f"\nTraining data appended to: {output_path}")
            print(f"  New samples: {len(flat_results)}")
            print(f"  Total samples: {len(all_data)}")
        else:
            print(f"\nTraining data saved to: {output_path}")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  NPU counts: {sorted(df['num_npus'].unique())}")
        print(f"  Peak memory range: {df['avg_peak_memory_gb'].min():.2f} - {df['avg_peak_memory_gb'].max():.2f} GB")
        print(f"  Mean peak memory: {df['avg_peak_memory_gb'].mean():.2f} GB")
        print(f"  Std peak memory: {df['avg_peak_memory_gb'].std():.2f} GB")
    else:
        print("\nNo results collected!")

if __name__ == '__main__':
    main()
