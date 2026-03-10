#!/bin/bash

# Ensure ASTRA_SIM_ROOT is set. If not, try to get it with git.
if [ -z "$ASTRA_SIM_ROOT" ]; then
    export ASTRA_SIM_ROOT=$(git rev-parse --show-toplevel)
    echo "ASTRA_SIM_ROOT was not set. Defaulting to: $ASTRA_SIM_ROOT"
fi

# --- EXPERIMENT CONFIGURATION ---
EXPERIMENT_NUM=0  # Change this for different experiments (0 for legacy location)
if [ "$EXPERIMENT_NUM" -eq 0 ]; then
    BASE_OUTPUT_DIR="$ASTRA_SIM_ROOT/upc/output/comparison_run"
else
    BASE_OUTPUT_DIR="$ASTRA_SIM_ROOT/upc/output/comparison_run/experiment${EXPERIMENT_NUM}"
fi

# --- GENERAL CONFIGURATION ---
NPUS_COUNT=16
# NPUS_COUNT=128
LOGICAL_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/16_nodes_logical.json"
# LOGICAL_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/128_nodes_logical.json"
PYTHON_EXEC="../../../opt/venv/astra-sim/bin/python"
BASE_WORKLOAD_DIR="$ASTRA_SIM_ROOT/upc/comparing_networks/workload/T5_Small_grouped_ecmp_tp"
BASE_WORKLOAD_DIR="$ASTRA_SIM_ROOT/upc/comparing_networks/workload/multiple_collectives_tp"
# BASE_WORKLOAD_DIR="$ASTRA_SIM_ROOT/upc/comparing_networks/workload/T5_Small_grouped_128"
TIMEOUT="1500m" # Timeout for each individual collective simulation
MAX_PARALLEL_JOBS=20 # Number of parallel jobs, defaults to number of CPU cores
NUM_RUNS=20 # Number of runs for each simulation (10 runs with different ECMP seeds for NS3)
BASE_ECMP_SEED=50 # Starting ECMP seed value for NS3

# --- SYSTEM CONFIGURATIONS ---
# An array of system configuration names to loop through.
# The script will look for files like "upc/configuration/{NAME}_sys.json"
# and "upc/configuration/g2/{NAME}_sys.json".
SYSTEM_CONFIG_NAMES=("FullyConnected" "Ring")
SYSTEM_CONFIG_NAMES=("Ring")

# --- NETWORK & TOPOLOGY CONFIGURATIONS ---
# These will be dynamically set in the loop based on SYSTEM_CONFIG_NAMES
# ANALYTICAL_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/FullyConnected.yml"
G2_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FoldedClos_16_config.yml"
# G2_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FoldedClos_128_config.yml"
# topology bases should only point to the directories for G2 and NS3
G2_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/configuration/g2/topologies/"
NS3_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/configuration/ns3/topologies/"

# --- LISTS FOR LOOPS ---
# NS3_CONFIG_INDICES=(5 7 13 6 8 14) # Indices for NS3 config files to use
NS3_CONFIG_INDICES=(2)
# Use topology names instead of numeric indices. Names are the suffix part (without G2_ / ns3_ prefixes).
# TOPOLOGY_NAMES=("FoldedClos_16_ECMP_ECMP" "Dragonfly_16_ECMP_ECMP" "Jellyfish_16_ECMP_ECMP" )
TOPOLOGY_NAMES=("FoldedClos128_ECMP_ECMP")
TOPOLOGY_NAMES=("Dragonfly_16_ECMP_ECMP")

# =================================================================================
# --- SIMULATION FUNCTION ---
# Runs a single simulation with specific parameters
# =================================================================================
run_single_simulation() {
    local sim_type="$1"
    local workload_dir="$2"
    local sys_name="$3"
    local topo_name="$4"
    local run_num="$5"
    local ns3_conf_idx="${6:-}"  # Optional, only for NS3

    workload_name=$(basename "$workload_dir")
    TOPOLOGY_SHORT="${topo_name%%_*}"  # first word before first underscore

    # --- Define system and network config paths based on the current system name ---
    ANALYTICAL_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/${sys_name}_sys.json"
    G2_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/${sys_name}_sys.json"
    NS3_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/${sys_name}_sys.json"
    ANALYTICAL_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/${sys_name}.yml"

    if [ "$sim_type" == "analytical" ]; then
        echo ">>> [ANALYTICAL] Run $run_num/$NUM_RUNS | Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT"
        timeout "$TIMEOUT" "$PYTHON_EXEC" compare_networks.py \
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
        G2_TOPOLOGY_FILE="${G2_TOPOLOGY_BASE}/G2_${topo_name}.json"
        
        if [ ! -f "$G2_TOPOLOGY_FILE" ]; then
            echo ">>> [G2] SKIPPED - Topology file not found: $G2_TOPOLOGY_FILE"
            return 0
        fi

        echo ">>> [G2] Run $run_num/$NUM_RUNS | Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT"
        timeout "$TIMEOUT" "$PYTHON_EXEC" compare_networks.py \
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
        NS3_TOPOLOGY_FILE="${NS3_TOPOLOGY_BASE}/ns3_${topo_name}"
        NS3_CONFIG_FILE="$ASTRA_SIM_ROOT/upc/configuration/ns3/configs/old/FoldedClos_16_config${ns3_conf_idx}.txt"

        if [ ! -f "$NS3_CONFIG_FILE" ]; then
            echo ">>> [NS3] SKIPPED - Config file not found: $NS3_CONFIG_FILE"
            return 0
        fi

        # Calculate ECMP seed for this run
        ECMP_SEED=$((BASE_ECMP_SEED + run_num - 1))
        
        echo ">>> [NS3] Run $run_num/$NUM_RUNS | Workload: $workload_name | Sys: $sys_name | Topo: $TOPOLOGY_SHORT | ECMP: $ECMP_SEED | Conf: $ns3_conf_idx"
        timeout "$TIMEOUT" "$PYTHON_EXEC" compare_networks.py \
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
    fi
}

# Generate all parameter combinations for parallelization
generate_job_list() {
    local job_list_file="$1"
    > "$job_list_file"  # Clear the file
    
    # Find all workload directories
    while IFS= read -r workload_dir; do
        for sys_name in "${SYSTEM_CONFIG_NAMES[@]}"; do
            for topo_name in "${TOPOLOGY_NAMES[@]}"; do
                for run_num in $(seq 1 $NUM_RUNS); do
                    # Add Analytical job
                    echo "analytical|$workload_dir|$sys_name|$topo_name|$run_num|" >> "$job_list_file"
                    
                    # Add G2 job
                    echo "g2|$workload_dir|$sys_name|$topo_name|$run_num|" >> "$job_list_file"
                    
                    # Add NS3 jobs (one per config index)
                    for ns3_conf_idx in "${NS3_CONFIG_INDICES[@]}"; do
                        echo "ns3|$workload_dir|$sys_name|$topo_name|$run_num|$ns3_conf_idx" >> "$job_list_file"
                    done
                done
            done
        done
    done < <(find "$BASE_WORKLOAD_DIR" -mindepth 1 -maxdepth 1 -type d)
    
    total_jobs=$(wc -l < "$job_list_file")
    echo "Generated $total_jobs jobs for parallel execution"
}

# Wrapper function to parse the job string and call run_single_simulation
execute_job() {
    local job_string="$1"
    IFS='|' read -r sim_type workload_dir sys_name topo_name run_num ns3_conf_idx <<< "$job_string"
    run_single_simulation "$sim_type" "$workload_dir" "$sys_name" "$topo_name" "$run_num" "$ns3_conf_idx"
}

# Export the function and variables to be available in sub-shells spawned by xargs
export -f run_single_simulation execute_job
export ASTRA_SIM_ROOT NPUS_COUNT LOGICAL_CONFIG PYTHON_EXEC TIMEOUT BASE_OUTPUT_DIR
export G2_NET_CONFIG G2_TOPOLOGY_BASE NS3_TOPOLOGY_BASE NUM_RUNS BASE_ECMP_SEED
export SYSTEM_CONFIG_NAMES TOPOLOGY_NAMES NS3_CONFIG_INDICES

# =================================================================================
# --- MAIN EXECUTION ---
# Generates all job combinations and runs them in parallel using xargs.
# =================================================================================
echo "========================================================================="
echo "--- STARTING FULLY PARALLELIZED SIMULATIONS (Max jobs: $MAX_PARALLEL_JOBS) ---"
echo "--- Output directory: $BASE_OUTPUT_DIR ---"
echo "========================================================================="

# Create temporary file for job list
JOB_LIST_FILE=$(mktemp)
trap "rm -f $JOB_LIST_FILE" EXIT

# Generate all job combinations
generate_job_list "$JOB_LIST_FILE"

# Execute all jobs in parallel
cat "$JOB_LIST_FILE" | xargs -P "$MAX_PARALLEL_JOBS" -I {} bash -c 'execute_job "{}"'

echo ""
echo "#################################################################"
echo "--- ALL PARALLEL SIMULATIONS HAVE FINISHED ---"
echo "#################################################################"
echo "#################################################################"