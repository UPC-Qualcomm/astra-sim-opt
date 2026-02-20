#!/bin/bash
# Quick launcher for Experiment 2 - ECMP Mode
# Template for a new experiment

export EXPERIMENT_NUM=2
export MODE="ecmp"
export NPUS_COUNT=16
export TIMEOUT="300m"
export MAX_PARALLEL_JOBS=16
export NUM_RUNS=25
export BASE_ECMP_SEED=25
export WORKLOAD_DIR_REL="upc/experiments_files/experiment2/workload/multiple_collectives_ecmp"

export SYSTEM_CONFIG_NAMES="Ring"
export TOPOLOGY_NAMES="Dragonfly_16_ECMP_ECMP"
export NS3_CONFIG_INDICES="5"
export SIM_TYPES="all"


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
