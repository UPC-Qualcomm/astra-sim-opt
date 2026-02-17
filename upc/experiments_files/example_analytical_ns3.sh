#!/bin/bash
# Example: Compare analytical vs NS3 (skip g2)
# Useful for quick analytical baseline comparison

export EXPERIMENT_NUM=1
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="300m"
export MAX_PARALLEL_JOBS=8

export SYSTEM_CONFIG_NAMES="FullyConnected"
export TOPOLOGY_NAMES="FoldedClos_16_v1_Random FoldedClos_16_v2_Random"
export NS3_CONFIG_INDICES="2"

# Run analytical and NS3 only
export SIM_TYPES="analytical ns3"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
