#!/bin/bash
# Quick launcher for Experiment 4 - ECMP Mode
# Template for a new experiment

export EXPERIMENT_NUM=4
export MODE="ecmp"
export NPUS_COUNT=128
export TIMEOUT="10000m"
export MAX_PARALLEL_JOBS=6
export NUM_RUNS=2
export BASE_ECMP_SEED=25
export WORKLOAD_DIR_REL="upc/experiments_files/experiment4/workload/GPT13B"

export SYSTEM_CONFIG_NAMES="Ring"
export TOPOLOGY_NAMES="FoldedClos_128_ECMP_sw"
export NS3_CONFIG_INDICES="5"
export SIM_TYPES="ns3"


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
