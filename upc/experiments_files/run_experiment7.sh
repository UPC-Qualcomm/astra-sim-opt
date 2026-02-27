#!/bin/bash
# Quick launcher for Experiment 7 - ECMP Mode
# Template for a new experiment

export EXPERIMENT_NUM=7
export MODE="ecmp"
export NPUS_COUNT=1024
export TIMEOUT="10000m"
export MAX_PARALLEL_JOBS=5
export NUM_RUNS=1
export BASE_ECMP_SEED=25
export WORKLOAD_DIR_REL="upc/experiments_files/experiment7/workload/GPT13B"

export SYSTEM_CONFIG_NAMES="FoldedClos1024"
export TOPOLOGY_NAMES="FoldedClos1024"
export NS3_CONFIG_INDICES="3"
export SIM_TYPES="all"


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
