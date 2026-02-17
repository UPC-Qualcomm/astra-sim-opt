#!/bin/bash
# Quick launcher for Experiment 2 - Deterministic Mode
# Template for a new experiment

export EXPERIMENT_NUM=2
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="100m"
export MAX_PARALLEL_JOBS=10
export WORKLOAD_DIR_REL="upc/experiments_files/experiment2/workload/six_consecutive_collectives"

export SYSTEM_CONFIG_NAMES="Ring"
export TOPOLOGY_NAMES="FoldedClos_16_v1_Random Dragonfly_16_v1_Random"
export NS3_CONFIG_INDICES="5"
export SIM_TYPES="analytical"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
