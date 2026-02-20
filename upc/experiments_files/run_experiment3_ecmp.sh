#!/bin/bash
# Quick launcher for Experiment 3 - ECMP Mode
# Template for a new experiment

export EXPERIMENT_NUM=3
export MODE="ecmp"
export NPUS_COUNT=16
export TIMEOUT="1500m"
export MAX_PARALLEL_JOBS=24
export NUM_RUNS=1
export BASE_ECMP_SEED=25
export WORKLOAD_DIR_REL="upc/experiments_files/experiment3/workload/Llama8B_ecmp"

export SYSTEM_CONFIG_NAMES="Ring"
export TOPOLOGY_NAMES="FoldedClos_16_ECMP_ECMP Dragonfly_16_ECMP_ECMP Jellyfish_16_ECMP_ECMP"
export NS3_CONFIG_INDICES="5"
export SIM_TYPES="analytical"


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
