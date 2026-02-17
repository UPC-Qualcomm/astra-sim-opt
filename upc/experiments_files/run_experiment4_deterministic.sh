#!/bin/bash
# Quick launcher for Experiment 4 - Deterministic Mode
# Template for a new experiment

export EXPERIMENT_NUM=4
export MODE="deterministic"
export NPUS_COUNT=128
export TIMEOUT="1500m"
export MAX_PARALLEL_JOBS=16
export WORKLOAD_DIR_REL="upc/experiments_files/experiment4/workload/Llama8B"

export SYSTEM_CONFIG_NAMES="Ring"
export TOPOLOGY_NAMES="FoldedClos_128_v2_Random"
export NS3_CONFIG_INDICES="5"
export SIM_TYPES="ns3"


# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main