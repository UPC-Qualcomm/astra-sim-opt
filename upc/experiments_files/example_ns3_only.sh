#!/bin/bash
# Example: Run only NS3 simulations (skip analytical and g2)
# This is useful when you want to focus on a specific simulator

export EXPERIMENT_NUM=1
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="300m"
export MAX_PARALLEL_JOBS=8

export SYSTEM_CONFIG_NAMES="FullyConnected"
export TOPOLOGY_NAMES="FoldedClos_16_v1_Random"
export NS3_CONFIG_INDICES="1 2 3 4"

# Only run NS3 simulations
export SIM_TYPES="ns3"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
