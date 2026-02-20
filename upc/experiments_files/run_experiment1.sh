#!/bin/bash
# Quick launcher for Experiment 1 - Deterministic Mode
# Compares network performance with deterministic routing

export EXPERIMENT_NUM=1
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="100m"
export MAX_PARALLEL_JOBS=16

export SYSTEM_CONFIG_NAMES="FullyConnected"
export TOPOLOGY_NAMES="FoldedClos_16_v1_Random Dragonfly_16_v1_Random"
export NS3_CONFIG_INDICES="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
export SIM_TYPES="analytical"  # Options: "all", "analytical", "g2", "ns3", or combinations like "analytical ns3"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
