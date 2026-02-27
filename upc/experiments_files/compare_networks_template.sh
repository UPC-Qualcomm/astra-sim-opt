#!/bin/bash
# =================================================================================
# COMPARE NETWORKS TEMPLATE - Configurable Simulation Runner
# =================================================================================
# This is a template script that can be easily configured for different experiments.
# Copy or call this script with appropriate parameters.
#
# Usage: 
#   ./compare_networks_template.sh [options]
#   or source this file after setting the configuration variables
# =================================================================================

# --- DEFAULT CONFIGURATION (Override these before sourcing or via command line) ---

# Experiment identification
EXPERIMENT_NUM="${EXPERIMENT_NUM:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-experiment${EXPERIMENT_NUM}}"

# Execution mode: "deterministic" or "ecmp"
MODE="${MODE:-deterministic}"

# General configuration
NPUS_COUNT="${NPUS_COUNT:-16}"
TIMEOUT="${TIMEOUT:-300m}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-8}"

# For ECMP mode only
NUM_RUNS="${NUM_RUNS:-1}"  # Number of runs per configuration (for ECMP)
BASE_ECMP_SEED="${BASE_ECMP_SEED:-25}"

# System configurations to test
SYSTEM_CONFIG_NAMES="${SYSTEM_CONFIG_NAMES:-FullyConnected}"

# Network topologies to test
TOPOLOGY_NAMES="${TOPOLOGY_NAMES:-FoldedClos_16_v1_Random}"

# NS3 config indices (space-separated)
NS3_CONFIG_INDICES="${NS3_CONFIG_INDICES:-2}"

# Simulation types to run (space-separated: analytical g2 ns3)
# Set to "all" to run all simulators, or specify subset like "analytical ns3"
SIM_TYPES="${SIM_TYPES:-all}"

# Workload directory (relative to ASTRA_SIM_ROOT or absolute)
WORKLOAD_DIR_REL="${WORKLOAD_DIR_REL:-upc/experiments_files/${EXPERIMENT_NAME}/workload/multiple_collectives}"

# =================================================================================
# --- SCRIPT START ---
# =================================================================================

# Ensure ASTRA_SIM_ROOT is set
if [ -z "$ASTRA_SIM_ROOT" ]; then
    export ASTRA_SIM_ROOT=$(git rev-parse --show-toplevel)
    echo "ASTRA_SIM_ROOT was not set. Defaulting to: $ASTRA_SIM_ROOT"
fi

# Set paths based on ASTRA_SIM_ROOT
PYTHON_EXEC="${PYTHON_EXEC:-$ASTRA_SIM_ROOT/../../../opt/venv/astra-sim/bin/python}"
BASE_OUTPUT_DIR="$ASTRA_SIM_ROOT/upc/output/comparison_run/${EXPERIMENT_NAME}"
BASE_WORKLOAD_DIR="$ASTRA_SIM_ROOT/$WORKLOAD_DIR_REL"

G2_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/g2/topologies/"
NS3_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/ns3/topologies/"

# Convert space-separated strings to arrays
IFS=' ' read -r -a SYSTEM_CONFIG_NAMES_ARRAY <<< "$SYSTEM_CONFIG_NAMES"
IFS=' ' read -r -a TOPOLOGY_NAMES_ARRAY <<< "$TOPOLOGY_NAMES"
IFS=' ' read -r -a NS3_CONFIG_INDICES_ARRAY <<< "$NS3_CONFIG_INDICES"

# Parse SIM_TYPES ("all" means all simulators)
if [ "$SIM_TYPES" == "all" ]; then
    SIM_TYPES="analytical g2 ns3"
fi
IFS=' ' read -r -a SIM_TYPES_ARRAY <<< "$SIM_TYPES"

# =================================================================================
# --- SIMULATION FUNCTION ---
# =================================================================================
run_single_simulation() {
    local sim_type="$1"
    local workload_dir="$2"
    local sys_name="$3"
    local topo_name="$4"
    local run_num="$5"
    local ns3_conf_idx="${6:-}"

    workload_name=$(basename "$workload_dir")
    TOPOLOGY_SHORT="${topo_name%%_*}"

    # Define config paths
    LOGICAL_CONFIG="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/${sys_name}_logical_dims.json"
    ANALYTICAL_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/${sys_name}_sys.json"
    G2_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/${sys_name}_sys.json"
    NS3_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/${sys_name}_sys.json"
    ANALYTICAL_NET_CONFIG="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/${sys_name}.yml"
    G2_NET_CONFIG="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/${sys_name}.yml"

    if [ "$sim_type" == "analytical" ]; then
        if [ "$MODE" == "ecmp" ]; then
            echo ">>> [ANALYTICAL] Run $run_num/$NUM_RUNS | Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT"
        else
            echo ">>> [ANALYTICAL - DETERMINISTIC] Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT"
        fi
        
        timeout "$TIMEOUT" "$PYTHON_EXEC" $ASTRA_SIM_ROOT/upc/compare_networks.py \
            --workload-dir "$workload_dir" \
            --npus-count "$NPUS_COUNT" \
            --logical-topology-config "$LOGICAL_CONFIG" \
            --analytical-system-config "$ANALYTICAL_SYS_CONFIG" \
            --analytical-network-config "$ANALYTICAL_NET_CONFIG" \
            --topology-name "$TOPOLOGY_SHORT" \
            --run-number "$run_num" \
            --base-output-dir "$BASE_OUTPUT_DIR" \
            --python-exec "$PYTHON_EXEC"

    elif [ "$sim_type" == "g2" ]; then
        # Try to find G2 topology file with different extensions
        # G2_TOPOLOGY_FILE=""
        # for ext in json txt; do
        #     candidate="${G2_TOPOLOGY_BASE}/G2_${topo_name}.${ext}"
        #     if [ -f "$candidate" ]; then
        #         G2_TOPOLOGY_FILE="$candidate"
        #         break
        #     fi
        # done
        G2_TOPOLOGY_FILE="${NS3_TOPOLOGY_BASE}/${topo_name}"
        if [ -z "$G2_TOPOLOGY_FILE" ]; then
            echo ">>> [G2] SKIPPED - Topology file not found: ${G2_TOPOLOGY_BASE}/G2_${topo_name}.{json,txt}"
            return 0
        fi

        if [ "$MODE" == "ecmp" ]; then
            echo ">>> [G2] Run $run_num/$NUM_RUNS | Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT"
        else
            echo ">>> [G2 - DETERMINISTIC] Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT"
        fi

        timeout "$TIMEOUT" "$PYTHON_EXEC" $ASTRA_SIM_ROOT/upc/compare_networks.py \
            --workload-dir "$workload_dir" \
            --npus-count "$NPUS_COUNT" \
            --logical-topology-config "$LOGICAL_CONFIG" \
            --g2-system-config "$G2_SYS_CONFIG" \
            --g2-network-config "$G2_NET_CONFIG" \
            --g2-topology-file "$G2_TOPOLOGY_FILE" \
            --topology-name "$TOPOLOGY_SHORT" \
            --run-number "$run_num" \
            --base-output-dir "$BASE_OUTPUT_DIR" \
            --python-exec "$PYTHON_EXEC"

    elif [ "$sim_type" == "ns3" ]; then
        NS3_TOPOLOGY_FILE="${NS3_TOPOLOGY_BASE}/${topo_name}"
        NS3_CONFIG_FILE="$ASTRA_SIM_ROOT/upc/experiments_files/${EXPERIMENT_NAME}/configuration/ns3/configs/FoldedClos_${NPUS_COUNT}_config${ns3_conf_idx}.txt"

        if [ ! -f "$NS3_CONFIG_FILE" ]; then
            echo ">>> [NS3] SKIPPED - Config file not found: $NS3_CONFIG_FILE"
            return 0
        fi

        if [ "$MODE" == "ecmp" ]; then
            ECMP_SEED=$((BASE_ECMP_SEED + run_num - 1))
            echo ">>> [NS3] Run $run_num/$NUM_RUNS | Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT | ECMP: $ECMP_SEED | Conf: $ns3_conf_idx"
            
            timeout "$TIMEOUT" "$PYTHON_EXEC" $ASTRA_SIM_ROOT/upc/compare_networks.py \
                --workload-dir "$workload_dir" \
                --npus-count "$NPUS_COUNT" \
                --logical-topology-config "$LOGICAL_CONFIG" \
                --ns3-system-config "$NS3_SYS_CONFIG" \
                --ns3-network-config "$NS3_CONFIG_FILE" \
                --ns3-topology-file "$NS3_TOPOLOGY_FILE" \
                --ns3-precomputed-paths 0 \
                --topology-name "$TOPOLOGY_SHORT" \
                --run-number "$run_num" \
                --ns3-ecmp-seed "$ECMP_SEED" \
                --base-output-dir "$BASE_OUTPUT_DIR" \
                --python-exec "$PYTHON_EXEC"
        else
            echo ">>> [NS3 - DETERMINISTIC] Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT | Conf: $ns3_conf_idx"
            
            timeout "$TIMEOUT" "$PYTHON_EXEC" $ASTRA_SIM_ROOT/upc/compare_networks.py \
                --workload-dir "$workload_dir" \
                --npus-count "$NPUS_COUNT" \
                --logical-topology-config "$LOGICAL_CONFIG" \
                --ns3-system-config "$NS3_SYS_CONFIG" \
                --ns3-network-config "$NS3_CONFIG_FILE" \
                --ns3-topology-file "$NS3_TOPOLOGY_FILE" \
                --ns3-precomputed-paths 1 \
                --topology-name "$TOPOLOGY_SHORT" \
                --base-output-dir "$BASE_OUTPUT_DIR" \
                --python-exec "$PYTHON_EXEC"
        fi
    fi
}

# =================================================================================
# --- JOB GENERATION ---
# =================================================================================

# Helper function to check if a simulator type is enabled
is_sim_enabled() {
    local sim_type="$1"
    for enabled_sim in "${SIM_TYPES_ARRAY[@]}"; do
        if [ "$enabled_sim" == "$sim_type" ]; then
            return 0
        fi
    done
    return 1
}

generate_job_list() {
    local job_list_file="$1"
    > "$job_list_file"

    while IFS= read -r workload_dir; do
        for sys_name in "${SYSTEM_CONFIG_NAMES_ARRAY[@]}"; do
            for topo_name in "${TOPOLOGY_NAMES_ARRAY[@]}"; do
                if [ "$MODE" == "ecmp" ]; then
                    for run_num in $(seq 1 $NUM_RUNS); do
                        # Add jobs only for enabled simulators
                        if is_sim_enabled "analytical"; then
                            echo "analytical|$workload_dir|$sys_name|$topo_name|$run_num|" >> "$job_list_file"
                        fi
                        if is_sim_enabled "g2"; then
                            echo "g2|$workload_dir|$sys_name|$topo_name|$run_num|" >> "$job_list_file"
                        fi
                        if is_sim_enabled "ns3"; then
                            for ns3_conf_idx in "${NS3_CONFIG_INDICES_ARRAY[@]}"; do
                                echo "ns3|$workload_dir|$sys_name|$topo_name|$run_num|$ns3_conf_idx" >> "$job_list_file"
                            done
                        fi
                    done
                else
                    # Add jobs only for enabled simulators
                    if is_sim_enabled "analytical"; then
                        echo "analytical|$workload_dir|$sys_name|$topo_name|1|" >> "$job_list_file"
                    fi
                    if is_sim_enabled "g2"; then
                        echo "g2|$workload_dir|$sys_name|$topo_name|1|" >> "$job_list_file"
                    fi
                    if is_sim_enabled "ns3"; then
                        for ns3_conf_idx in "${NS3_CONFIG_INDICES_ARRAY[@]}"; do
                            echo "ns3|$workload_dir|$sys_name|$topo_name|1|$ns3_conf_idx" >> "$job_list_file"
                        done
                    fi
                fi
            done
        done
    done < <(find "$BASE_WORKLOAD_DIR" -mindepth 1 -maxdepth 1 -type d)
    
    total_jobs=$(wc -l < "$job_list_file")
    echo "Generated $total_jobs jobs for parallel execution"
}

execute_job() {
    local job_string="$1"
    IFS='|' read -r sim_type workload_dir sys_name topo_name run_num ns3_conf_idx <<< "$job_string"
    run_single_simulation "$sim_type" "$workload_dir" "$sys_name" "$topo_name" "$run_num" "$ns3_conf_idx"
}

# Export functions and variables for xargs
export -f run_single_simulation execute_job is_sim_enabled
export ASTRA_SIM_ROOT NPUS_COUNT PYTHON_EXEC TIMEOUT BASE_OUTPUT_DIR
export G2_TOPOLOGY_BASE NS3_TOPOLOGY_BASE NUM_RUNS BASE_ECMP_SEED
export MODE EXPERIMENT_NAME
export SYSTEM_CONFIG_NAMES_ARRAY TOPOLOGY_NAMES_ARRAY NS3_CONFIG_INDICES_ARRAY SIM_TYPES_ARRAY

# =================================================================================
# --- MAIN EXECUTION ---
# =================================================================================
main() {
    echo "========================================================================="
    echo "--- EXPERIMENT: $EXPERIMENT_NAME (Mode: $MODE) ---"
    echo "--- NPUs: $NPUS_COUNT | Max Parallel Jobs: $MAX_PARALLEL_JOBS ---"
    echo "--- System Configs: $SYSTEM_CONFIG_NAMES ---"
    echo "--- Topologies: $TOPOLOGY_NAMES ---"
    echo "--- Simulators: $SIM_TYPES ---"
    echo "--- Workload: $BASE_WORKLOAD_DIR ---"
    echo "--- Output: $BASE_OUTPUT_DIR ---"
    if [ "$MODE" == "ecmp" ]; then
        echo "--- Runs per config: $NUM_RUNS | Base ECMP seed: $BASE_ECMP_SEED ---"
    fi
    echo "========================================================================="

    # Verify workload directory exists
    if [ ! -d "$BASE_WORKLOAD_DIR" ]; then
        echo "ERROR: Workload directory not found: $BASE_WORKLOAD_DIR"
        exit 1
    fi

    # Create temporary file for job list
    JOB_LIST_FILE=$(mktemp)
    trap "rm -f $JOB_LIST_FILE" EXIT

    # Generate and execute jobs
    generate_job_list "$JOB_LIST_FILE"
    cat "$JOB_LIST_FILE" | xargs -P "$MAX_PARALLEL_JOBS" -I {} bash -c 'execute_job "{}"'

    echo ""
    echo "#################################################################"
    echo "--- ALL SIMULATIONS FINISHED: $EXPERIMENT_NAME ---"
    echo "#################################################################"
}

# Only run main if script is executed directly (not sourced)
if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
    main
fi
