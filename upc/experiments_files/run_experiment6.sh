#!/bin/bash
# Quick launcher for Experiment 6 - ECMP Mode
# Template for a new experiment

export EXPERIMENT_NUM=6
export MODE="ecmp"
export NPUS_COUNT=128
export TIMEOUT="10000m"
export MAX_PARALLEL_JOBS=5
export NUM_RUNS=1
export BASE_ECMP_SEED=25
export WORKLOAD_DIR_REL="upc/experiments_files/experiment6/workload/GPT40B"

export SYSTEM_CONFIG_NAMES="FoldedClos128"
export TOPOLOGY_NAMES="FoldedClosECMP128"
export NS3_CONFIG_INDICES="3"
export SIM_TYPES="all"


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
