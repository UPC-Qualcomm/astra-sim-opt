#!/bin/bash
# Quick launcher for Experiment 4 - ECMP Mode
# Template for a new experiment

export EXPERIMENT_NUM=4
export MODE="ecmp"
export NPUS_COUNT=16
export TIMEOUT="1000m"
export MAX_PARALLEL_JOBS=20
export NUM_RUNS=1    # only 1 possible path
export BASE_ECMP_SEED=25
export WORKLOAD_DIR_REL="upc/experiments_files/experiment4/workload/Llama8B"

export SYSTEM_CONFIG_NAMES="FoldedClos"
export TOPOLOGY_NAMES="FoldedClosECMP128"
export NS3_CONFIG_INDICES="3 5"
export SIM_TYPES="all"


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
