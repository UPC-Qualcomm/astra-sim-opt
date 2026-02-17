#!/bin/bash
# =================================================================================
# NEW EXPERIMENT GENERATOR
# =================================================================================
# This script helps you quickly create a new experiment launcher script
# =================================================================================

echo "========================================================================="
echo "NEW EXPERIMENT SCRIPT GENERATOR"
echo "========================================================================="
echo ""

# Get experiment number
read -p "Enter experiment number (e.g., 3): " EXP_NUM
if [ -z "$EXP_NUM" ]; then
    echo "Error: Experiment number is required"
    exit 1
fi

# Get mode
read -p "Enter mode [deterministic/ecmp] (default: deterministic): " MODE
MODE=${MODE:-deterministic}

if [ "$MODE" != "deterministic" ] && [ "$MODE" != "ecmp" ]; then
    echo "Error: Mode must be 'deterministic' or 'ecmp'"
    exit 1
fi

# Get NPU count
read -p "Enter NPU count (default: 16): " NPUS
NPUS=${NPUS:-16}

# Get system configs
read -p "Enter system configs space-separated (default: FullyConnected): " SYSTEMS
SYSTEMS=${SYSTEMS:-FullyConnected}

# Get topologies (suggest based on mode)
if [ "$MODE" == "deterministic" ]; then
    DEFAULT_TOPOS="FoldedClos_${NPUS}_v1_Deterministic"
else
    DEFAULT_TOPOS="FoldedClos_${NPUS}_v1_Random"
fi
read -p "Enter topology names space-separated (default: $DEFAULT_TOPOS): " TOPOS
TOPOS=${TOPOS:-$DEFAULT_TOPOS}

# Get additional parameters for ECMP
if [ "$MODE" == "ecmp" ]; then
    read -p "Enter number of runs per config (default: 10): " NUM_RUNS
    NUM_RUNS=${NUM_RUNS:-10}
    
    read -p "Enter base ECMP seed (default: 25): " ECMP_SEED
    ECMP_SEED=${ECMP_SEED:-25}
    
    read -p "Enter max parallel jobs (default: 20): " MAX_JOBS
    MAX_JOBS=${MAX_JOBS:-20}
    
    read -p "Enter timeout (default: 1500m): " TIMEOUT
    TIMEOUT=${TIMEOUT:-1500m}
else
    read -p "Enter max parallel jobs (default: 8): " MAX_JOBS
    MAX_JOBS=${MAX_JOBS:-8}
    
    read -p "Enter timeout (default: 300m): " TIMEOUT
    TIMEOUT=${TIMEOUT:-300m}
fi

# Get NS3 config indices
read -p "Enter NS3 config indices space-separated (default: 2): " NS3_INDICES
NS3_INDICES=${NS3_INDICES:-2}

# Get simulator types
read -p "Enter simulators to run [all/analytical/g2/ns3/combinations] (default: all): " SIM_TYPES
SIM_TYPES=${SIM_TYPES:-all}

# Generate filename
OUTPUT_FILE="run_experiment${EXP_NUM}_${MODE}.sh"

# Check if file exists
if [ -f "$OUTPUT_FILE" ]; then
    read -p "File $OUTPUT_FILE already exists. Overwrite? [y/N]: " OVERWRITE
    if [ "$OVERWRITE" != "y" ] && [ "$OVERWRITE" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Generate script content
cat > "$OUTPUT_FILE" << EOF
#!/bin/bash
# Quick launcher for Experiment $EXP_NUM - ${MODE^} Mode
# Generated on $(date)

export EXPERIMENT_NUM=$EXP_NUM
export MODE="$MODE"
export NPUS_COUNT=$NPUS
export TIMEOUT="$TIMEOUT"
export MAX_PARALLEL_JOBS=$MAX_JOBS
EOF

if [ "$MODE" == "ecmp" ]; then
cat >> "$OUTPUT_FILE" << EOF
export NUM_RUNS=$NUM_RUNS
export BASE_ECMP_SEED=$ECMP_SEED
EOF
fi

cat >> "$OUTPUT_FILE" << EOF

export SYSTEM_CONFIG_NAMES="$SYSTEMS"
export TOPOLOGY_NAMES="$TOPOS"
export NS3_CONFIG_INDICES="$NS3_INDICES"
export SIM_TYPES="$SIM_TYPES"

# Get script directory
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

# Source and run the template
source "\$SCRIPT_DIR/compare_networks_template.sh"
main
EOF

# Make executable
chmod +x "$OUTPUT_FILE"

echo ""
echo "========================================================================="
echo "SUCCESS! Generated: $OUTPUT_FILE"
echo "========================================================================="
echo ""
echo "Configuration:"
echo "  Experiment Number : $EXP_NUM"
echo "  Mode             : $MODE"
echo "  NPUs             : $NPUS"
echo "  Max Jobs         : $MAX_JOBS"
echo "  Timeout          : $TIMEOUT"
if [ "$MODE" == "ecmp" ]; then
echo "  Runs per config  : $NUM_RUNS"
echo "  Base ECMP seed   : $ECMP_SEED"
fi
echo "  System Configs   : $SYSTEMS"
echo "  Topologies       : $TOPOS"
echo "  NS3 Indices      : $NS3_INDICES"
echo "  Simulators       : $SIM_TYPES"
echo ""
echo "To run your experiment:"
echo "  ./$OUTPUT_FILE"
echo ""
echo "To edit configuration:"
echo "  nano $OUTPUT_FILE"
echo "========================================================================="
