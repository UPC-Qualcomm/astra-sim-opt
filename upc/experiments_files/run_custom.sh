#!/bin/bash
# =================================================================================
# CUSTOM EXPERIMENT - Run with fully custom parameters
# =================================================================================
# This script allows you to specify all parameters without modifying the template.
# Set your parameters below and run.
# =================================================================================

# --- EXPERIMENT CONFIGURATION ---
export EXPERIMENT_NUM=1
export MODE="deterministic"  # "deterministic" or "ecmp"

# --- SCALE CONFIGURATION ---
export NPUS_COUNT=16

# --- EXECUTION CONFIGURATION ---
export TIMEOUT="300m"
export MAX_PARALLEL_JOBS=8

# --- FOR ECMP MODE ONLY ---
export NUM_RUNS=1        # Number of runs with different ECMP seeds
export BASE_ECMP_SEED=25 # Starting seed value

# --- SYSTEM & TOPOLOGY CONFIGURATION ---
# Space-separated list of system configurations
export SYSTEM_CONFIG_NAMES="FullyConnected"

# Space-separated list of topology names
# For deterministic: use *_Deterministic suffix
# For ECMP: use *_Random suffix
export TOPOLOGY_NAMES="FoldedClos_16_v1_Deterministic FoldedClos_16_v2_Deterministic"

# Space-separated list of NS3 config file indices
export NS3_CONFIG_INDICES="2"

# --- SIMULATOR SELECTION ---
# Which simulators to run: "all" or space-separated list like "analytical ns3"
# Options: analytical, g2, ns3
export SIM_TYPES="all"

# --- WORKLOAD CONFIGURATION ---
# Relative to ASTRA_SIM_ROOT
export WORKLOAD_DIR_REL="upc/experiments_files/experiment${EXPERIMENT_NUM}/workload/multiple_collectives"

# =================================================================================
# --- EXECUTION ---
# =================================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Display configuration
echo "========================================================================="
echo "CUSTOM EXPERIMENT CONFIGURATION"
echo "========================================================================="
echo "Experiment        : experiment${EXPERIMENT_NUM}"
echo "Mode              : $MODE"
echo "NPUs              : $NPUS_COUNT"
echo "Max Parallel Jobs : $MAX_PARALLEL_JOBS"
echo "Timeout           : $TIMEOUT"
if [ "$MODE" == "ecmp" ]; then
echo "Runs per config   : $NUM_RUNS"
echo "Base ECMP Seed    : $BASE_ECMP_SEED"
fi
echo "System Configs    : $SYSTEM_CONFIG_NAMES"
echo "Topologies        : $TOPOLOGY_NAMES"
echo "NS3 Config Index  : $NS3_CONFIG_INDICES"
echo "Workload          : $WORKLOAD_DIR_REL"
echo "========================================================================="
echo ""

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
