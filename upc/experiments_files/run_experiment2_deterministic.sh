#!/bin/bash
# Quick launcher for Experiment 2 - Deterministic Mode
# Template for a new experiment

export EXPERIMENT_NUM=2
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="100m"
export MAX_PARALLEL_JOBS=10
export WORKLOAD_DIR_REL="upc/experiments_files/experiment2/workload/multiple_collectives_deterministic"

export SYSTEM_CONFIG_NAMES="FoldedClos"
export TOPOLOGY_NAMES="FoldedClosDet_v1"
export NS3_CONFIG_INDICES="3"
export SIM_TYPES="all"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main

export EXPERIMENT_NUM=2
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="100m"
export MAX_PARALLEL_JOBS=10
export WORKLOAD_DIR_REL="upc/experiments_files/experiment2/workload/six_consecutive_collectives"
<
export SYSTEM_CONFIG_NAMES="FoldedClos"
export TOPOLOGY_NAMES="FoldedClosDet_v1"
export NS3_CONFIG_INDICES="3"
export SIM_TYPES="all"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main

export EXPERIMENT_NUM=2
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="100m"
export MAX_PARALLEL_JOBS=1
export WORKLOAD_DIR_REL="upc/experiments_files/experiment2/workload/multiple_collectives_deterministic"

export SYSTEM_CONFIG_NAMES="Dragonfly"
export TOPOLOGY_NAMES="DragonflyDet_v1"
export NS3_CONFIG_INDICES="3"
export SIM_TYPES="all"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main


export EXPERIMENT_NUM=2
export MODE="deterministic"
export NPUS_COUNT=16
export TIMEOUT="100m"
export MAX_PARALLEL_JOBS=1
export WORKLOAD_DIR_REL="upc/experiments_files/experiment2/workload/six_consecutive_collectives"

export SYSTEM_CONFIG_NAMES="Dragonfly"
export TOPOLOGY_NAMES="DragonflyDet_v1"
export NS3_CONFIG_INDICES="3"
export SIM_TYPES="all"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main

