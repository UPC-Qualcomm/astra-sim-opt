#!/bin/bash
# =================================================================================
# QUICK RUN - Flexible experiment launcher with command-line options
# =================================================================================
# Usage examples:
#   ./quick_run.sh 1 deterministic    # Run experiment 1 in deterministic mode
#   ./quick_run.sh 1 ecmp 10          # Run experiment 1 with 10 ECMP runs
#   ./quick_run.sh 2 deterministic    # Run experiment 2 in deterministic mode
# =================================================================================

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_number> <mode> [num_runs] [max_jobs]"
    echo ""
    echo "Arguments:"
    echo "  experiment_number : Experiment number (1, 2, 3, etc.)"
    echo "  mode             : 'deterministic' or 'ecmp'"
    echo "  num_runs         : Number of runs for ECMP mode (default: 10)"
    echo "  max_jobs         : Maximum parallel jobs (default: 8 for deterministic, 20 for ecmp)"
    echo ""
    echo "Examples:"
    echo "  $0 1 deterministic"
    echo "  $0 1 ecmp 20"
    echo "  $0 2 deterministic 1 4"
    exit 1
fi

export EXPERIMENT_NUM=$1
export MODE=$2
export NUM_RUNS=${3:-10}

# Set default max jobs based on mode
if [ "$MODE" == "ecmp" ]; then
    export MAX_PARALLEL_JOBS=${4:-20}
else
    export MAX_PARALLEL_JOBS=${4:-8}
fi

# Common configuration for 16-node experiments
export NPUS_COUNT=16

# Timeout based on mode
if [ "$MODE" == "ecmp" ]; then
    export TIMEOUT="1500m"
else
    export TIMEOUT="300m"
fi

# Default configurations (can be overridden)
export SYSTEM_CONFIG_NAMES="${SYSTEM_CONFIG_NAMES:-FullyConnected}"
export NS3_CONFIG_INDICES="${NS3_CONFIG_INDICES:-2}"
export BASE_ECMP_SEED="${BASE_ECMP_SEED:-25}"
export SIM_TYPES="${SIM_TYPES:-all}"  # Can be overridden: "all", "analytical", "g2", "ns3", or combinations

# Set topology names based on mode
if [ "$MODE" == "ecmp" ]; then
    export TOPOLOGY_NAMES="${TOPOLOGY_NAMES:-FoldedClos_16_v1_Random FoldedClos_16_v2_Random}"
else
    export TOPOLOGY_NAMES="${TOPOLOGY_NAMES:-FoldedClos_16_v1_Deterministic FoldedClos_16_v2_Deterministic}"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================="
echo "QUICK RUN CONFIGURATION"
echo "========================================================================="
echo "Experiment Number  : $EXPERIMENT_NUM"
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
echo "Simulators        : $SIM_TYPES"
echo "========================================================================="
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."
echo ""

# Source and run the template
source "$SCRIPT_DIR/compare_networks_template.sh"
main
