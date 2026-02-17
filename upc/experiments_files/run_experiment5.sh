#!/bin/bash
# Quick launcher for Experiment 5 - Deterministic Mode
# Template for a new experiment

export EXPERIMENT_NUM=5
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="1500m"
export MAX_PARALLEL_JOBS=16
export WORKLOAD_DIR_REL="upc/experiments_files/experiment5/workload/Llama8B"

export SYSTEM_CONFIG_NAMES="Ring"
export TOPOLOGY_NAMES="FoldedClos_16_v1_Random"
export NS3_CONFIG_INDICES="5 1 64"
export SIM_TYPES="all"


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
