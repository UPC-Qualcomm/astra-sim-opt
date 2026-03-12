#!/usr/bin/env python3
"""
Experiment 8: Multi-scale workload generation for simulator comparison.

Generates diverse training workloads across:
  - NPU counts: {2, 4, 8, 16, 32, 64, 128}
  - Model shapes: d_model, num_stacks, seq_len
  - Parallelization strategies: DP × TP × SP × PP = N
  - Batch/micro-batch variations

Also generates per-NPU-count configuration files (logical dims, sys, yml)
for FoldedClos128 topology.
"""

import os
import sys
import json
import copy
import random
import shutil
import subprocess
import argparse
import math
import multiprocessing
from functools import partial
from tqdm import tqdm

# ===================================================================
# Constants & Search Space
# ===================================================================

NPU_COUNTS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Logical dimension factorizations (max 8 per dimension)
LOGICAL_DIMS = {
    2:   [2],
    4:   [4],
    8:   [8],
    16:  [8, 2],
    32:  [8, 4],
    64:  [8, 8],
    128: [8, 16],
    256: [8, 16, 2],
    512: [8, 16, 2, 2],
    1024: [8, 16, 2, 4]
}
#d1024_L32_seq2048_b16_mb16_16_1_16_1_1
# Model parameter search space
D_MODEL_VALUES = [512, 1024, 2048, 4096]
NUM_STACKS_VALUES = [2, 4, 8, 16, 32]
SEQ_LEN_VALUES = [512, 1024, 2048, 4096]

# Batch sizes: powers of 2 from 2 to 2048
BATCH_VALUES = [2**i for i in range(2,11)]

# Micro-batch options
MICRO_BATCH_VALUES = [2**i for i in range(2, 11)]

# Fixed model parameters
VOCAB_SIZE = 32000
HEAD_DIM = 64  # So num_heads = d_model / HEAD_DIM

WEIGHT_SHARDED_OPTIONS = [True, False]

# Network parameters (from FoldedClos128 reference)
BW_FIRST = 100.0   # GB/s (intra-pod)
BW_OTHER = 25.0     # GB/s (inter-pod)
LATENCIES = [200, 700, 1200, 1700]  # ns for up to 4 dims


# ===================================================================
# Configuration Generation
# ===================================================================

def generate_logical_dims_json(npu_count, config_dir):
    dims = LOGICAL_DIMS[npu_count]
    data = {"logical-dims": [str(d) for d in dims]}
    path = os.path.join(config_dir, f"FoldedClos{npu_count}_logical_dims.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path


def generate_sys_json(npu_count, config_dir):
    ndims = len(LOGICAL_DIMS[npu_count])
    ring_impl = ["ring"] * ndims
    data = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 1,
        "all-reduce-implementation": ring_impl,
        "all-gather-implementation": ring_impl,
        "reduce-scatter-implementation": ring_impl,
        "all-to-all-implementation": ring_impl,
        "collective-optimization": "localBWAware",
        "sync_mode": "ASTRASIM_BARRIER",
        "local-mem-bw": 3350,
        "local-mem-size": 80,
        "enable_network_logger": 0,
        "boost-mode": 0,
        "peak-perf": 989,
        "roofline-enabled": 0,
        "trace-enabled": 1,
        "trace-mem": 1,
        "mixed-percision": 1,
    }
    path = os.path.join(config_dir, f"FoldedClos{npu_count}_sys.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path


def generate_yml(npu_count, config_dir):
    dims = LOGICAL_DIMS[npu_count]
    ndims = len(dims)

    topology_list = ", ".join(["Switch"] * ndims)
    npus_list = ", ".join(str(d) for d in dims)
    bw_values = [BW_FIRST] + [BW_OTHER] * (ndims - 1) if ndims > 1 else [BW_FIRST]
    bw_list = ", ".join(str(b) for b in bw_values)
    lat_values = LATENCIES[:ndims]
    lat_list = ", ".join(str(l) for l in lat_values)

    content = (
        f"topology: [ {topology_list} ]\n"
        f"npus_count: [ {npus_list} ]\n"
        f"bandwidth: [ {bw_list} ]\n"
        f"latency: [ {lat_list} ]\n"
        f"packet_size: 9000\n"
        f"header_size: 36\n"
        f"ecmp_seed: 42\n"
        f"bandwidth_unit: GB/s\n"
    )
    path = os.path.join(config_dir, f"FoldedClos{npu_count}.yml")
    with open(path, 'w') as f:
        f.write(content)
    return path


def generate_all_configs(experiment_dir, source_ns3_config, source_ns3_topology):
    """Generate all configuration files for every NPU count."""
    config_dir = os.path.join(experiment_dir, "configuration")
    ns3_config_dir = os.path.join(config_dir, "ns3", "configs")
    ns3_topo_dir = os.path.join(config_dir, "ns3", "topologies")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(ns3_config_dir, exist_ok=True)
    os.makedirs(ns3_topo_dir, exist_ok=True)

    for npu_count in NPU_COUNTS:
        generate_logical_dims_json(npu_count, config_dir)
        generate_sys_json(npu_count, config_dir)
        generate_yml(npu_count, config_dir)

        # NS3 config file (same content for all NPU counts, different name)
        dest_cfg = os.path.join(ns3_config_dir, f"FoldedClos_{npu_count}_config3.txt")
        if not os.path.exists(dest_cfg):
            shutil.copy(source_ns3_config, dest_cfg)

    # NS3 topology (single copy, same for all)
    dest_topo = os.path.join(ns3_topo_dir, "FoldedClosECMP128")
    if not os.path.exists(dest_topo):
        shutil.copy(source_ns3_topology, dest_topo)

    print(f"Generated configurations for NPU counts: {NPU_COUNTS}")
    print(f"  Config dir: {config_dir}")


# ===================================================================
# Strategy & Workload Generation
# ===================================================================

def get_num_heads(d_model):
    return d_model // HEAD_DIM


def get_valid_strategies(npu_count, num_heads, num_stacks, seq_len):
    """
    All valid (dp, tp, sp, pp) with dp*tp*sp*pp = npu_count,
    tp | num_heads, pp | num_stacks, sp | seq_len.
    """
    strategies = []
    for tp in range(1, npu_count + 1):
        if npu_count % tp != 0:
            continue
        if num_heads % tp != 0:
            continue
        remaining_tp = npu_count // tp
        for sp in range(1, remaining_tp + 1):
            if remaining_tp % sp != 0:
                continue
            if seq_len % sp != 0:
                continue
            remaining_sp = remaining_tp // sp
            for pp in range(1, remaining_sp + 1):
                if remaining_sp % pp != 0:
                    continue
                if num_stacks % pp != 0:
                    continue
                dp = remaining_sp // pp
                strategies.append((dp, tp, sp, pp))
    return strategies


def get_valid_micro_batches(batch):
    return [mb for mb in MICRO_BATCH_VALUES if batch >= mb and batch % mb == 0 and batch // mb <= 2**4]


def sample_search_space(num_samples, seed=42):
    """Randomly sample valid configurations from the search space.

    Samples an equal number of configurations per NPU count so that every
    NPU count is represented the same number of times.  The quota per NPU
    count is num_samples // len(NPU_COUNTS); any remainder is distributed
    one-by-one to the first NPU counts in NPU_COUNTS order.
    """
    rng = random.Random(seed)
    configs = []
    total_attempts = 0

    n_npu = len(NPU_COUNTS)
    base_quota = num_samples // n_npu
    remainder = num_samples % n_npu
    # Build per-NPU quotas
    quotas = {npu: base_quota + (1 if i < remainder else 0)
              for i, npu in enumerate(NPU_COUNTS)}

    for npu_count in NPU_COUNTS:
        quota = quotas[npu_count]
        seen = set()
        collected = 0
        max_attempts = quota * 200
        attempts = 0

        while collected < quota and attempts < max_attempts:
            attempts += 1
            total_attempts += 1

            d_model = rng.choice(D_MODEL_VALUES)
            num_stacks = rng.choice(NUM_STACKS_VALUES)
            seq_len = rng.choice(SEQ_LEN_VALUES)
            num_heads = get_num_heads(d_model)
            dff = 4 * d_model
            batch = rng.choice(BATCH_VALUES)

            strategies = get_valid_strategies(npu_count, num_heads, num_stacks, seq_len)
            if not strategies:
                continue

            dp, tp, sp, pp = rng.choice(strategies)

            valid_mbs = get_valid_micro_batches(batch)
            if not valid_mbs:
                continue
            micro_batch = rng.choice(valid_mbs)

            weight_sharded = rng.choice(WEIGHT_SHARDED_OPTIONS) if dp > 1 else False

            # Deduplicate within this NPU count
            ws = 1 if weight_sharded else 0
            key = (d_model, num_stacks, seq_len, batch, micro_batch, dp, tp, sp, pp, ws)
            if key in seen:
                continue
            seen.add(key)

            configs.append({
                'npu_count': npu_count,
                'd_model': d_model,
                'dff': dff,
                'num_stacks': num_stacks,
                'seq_len': seq_len,
                'num_heads': num_heads,
                'batch': batch,
                'micro_batch': micro_batch,
                'dp': dp,
                'tp': tp,
                'sp': sp,
                'pp': pp,
                'weight_sharded': weight_sharded,
            })
            collected += 1

        if collected < quota:
            print(f"  Warning: only found {collected}/{quota} unique configs for NPU={npu_count}")

    print(f"Sampled {len(configs)} unique configs from {total_attempts} attempts")
    return configs


def generate_single_workload(config, experiment_dir):
    """Generate one workload via the symbolic tensor graph."""
    npu = config['npu_count']
    dm = config['d_model']
    dff = config['dff']
    ns = config['num_stacks']
    seq = config['seq_len']
    nh = config['num_heads']
    batch = config['batch']
    mb = config['micro_batch']
    dp = config['dp']
    tp = config['tp']
    pp = config['pp']
    sp = config['sp']
    ws = 1 if config['weight_sharded'] else 0

    workload_name = f"d{dm}_L{ns}_seq{seq}_b{batch}_mb{mb}_{dp}_{tp}_{sp}_{pp}_{ws}"
    workload_dir = os.path.join(experiment_dir, "workload", f"npu_{npu}", workload_name)
    os.makedirs(workload_dir, exist_ok=True)

    # Skip if already generated
    if any(f.endswith('.et') for f in os.listdir(workload_dir)):
        return workload_dir

    stg_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "extern", "symbolic_tensor_graph",
    ))

    cmd = (
        f"python main.py "
        f"--output_dir {workload_dir} "
        f"--output_name {dp}_{tp}_{sp}_{pp}_{ws}.%d.et "
        f"--dp {dp} "
        f"--tp {tp} "
        f"--sp {sp} "
        f"--pp {pp} "
        f"--dvocal {VOCAB_SIZE} "
        f"--dmodel {dm} "
        f"--dff {dff} "
        f"--batch '[{batch}]' "
        f"--micro_batch '{mb}' "
        f"--seq {seq} "
        f"--head {nh} "
        f"--num_stacks {ns} "
        f"--weight_sharded {config['weight_sharded']} "
        f"--chakra_schema_version v0.0.4"
    )

    result = subprocess.run(cmd, shell=True, cwd=stg_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED [{workload_name}] NPU={npu}: {result.stderr[:200]}")
        return None
    return workload_dir


def _worker(args):
    config, experiment_dir = args
    return generate_single_workload(config, experiment_dir)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 8: Multi-scale workload generation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=300,
        help="Number of random configurations to generate (default: 300)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--only_configs", action="store_true",
        help="Only generate configuration files, skip workloads"
    )
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="Number of parallel workers (default: 95%% of CPUs)"
    )
    args = parser.parse_args()

    experiment_dir = os.path.dirname(os.path.abspath(__file__))

    # Source configs from experiment 5
    exp5_dir = os.path.join(experiment_dir, "..", "experiment5")
    source_ns3_config = os.path.join(
        exp5_dir, "configuration", "ns3", "configs", "FoldedClos_128_config3.txt"
    )
    source_ns3_topology = os.path.join(
        exp5_dir, "configuration", "ns3", "topologies", "FoldedClosECMP128"
    )

    # 1. Generate configuration files
    print("=" * 60)
    print("Step 1: Generating configuration files")
    print("=" * 60)
    generate_all_configs(experiment_dir, source_ns3_config, source_ns3_topology)

    if args.only_configs:
        print("Done (configs only).")
        sys.exit(0)

    # 2. Sample random workload configurations
    print("\n" + "=" * 60)
    print(f"Step 2: Sampling {args.num_samples} configurations (seed={args.seed})")
    print("=" * 60)
    configs = sample_search_space(args.num_samples, seed=args.seed)

    # Print distribution summary
    from collections import Counter
    npu_dist = Counter(c['npu_count'] for c in configs)
    print(f"NPU distribution: {dict(sorted(npu_dist.items()))}")
    dmodel_dist = Counter(c['d_model'] for c in configs)
    print(f"d_model distribution: {dict(sorted(dmodel_dist.items()))}")

    # Save config manifest (merge with existing to preserve prior runs)
    manifest_path = os.path.join(experiment_dir, "workload_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            existing_configs = json.load(f)
        existing_keys = {
            (c['npu_count'], c['d_model'], c['num_stacks'], c['seq_len'],
             c['batch'], c['micro_batch'], c['dp'], c['tp'], c.get('sp', 1), c['pp'],
             1 if c['weight_sharded'] else 0)
            for c in existing_configs
        }
        new_only = [
            c for c in configs
            if (c['npu_count'], c['d_model'], c['num_stacks'], c['seq_len'],
                c['batch'], c['micro_batch'], c['dp'], c['tp'], c['sp'], c['pp'],
                1 if c['weight_sharded'] else 0) not in existing_keys
        ]
        merged = existing_configs + new_only
        print(f"Manifest: {len(existing_configs)} existing + {len(new_only)} new = {len(merged)} total")
    else:
        merged = configs
    with open(manifest_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    # 3. Generate workloads in parallel
    print("\n" + "=" * 60)
    print("Step 3: Generating workloads")
    print("=" * 60)
    n_workers = args.parallel or max(1, int(multiprocessing.cpu_count() * 0.95))
    work_items = [(c, experiment_dir) for c in configs]

    with multiprocessing.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_worker, work_items),
            total=len(work_items),
            desc="Generating workloads",
        ))

    success = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)

    # 4. Summary
    print("\n" + "=" * 60)
    print(f"Experiment 8 generation complete: {success} OK, {failed} failed")
    print("=" * 60)
    for npu_count in NPU_COUNTS:
        wdir = os.path.join(experiment_dir, "workload", f"npu_{npu_count}")
        if os.path.isdir(wdir):
            n = len([d for d in os.listdir(wdir) if os.path.isdir(os.path.join(wdir, d))])
            print(f"  NPU {npu_count:>3}: {n} workloads")
