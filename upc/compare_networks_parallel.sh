#!/bin/bash

# Ensure ASTRA_SIM_ROOT is set. If not, try to get it with git.
if [ -z "$ASTRA_SIM_ROOT" ]; then
    export ASTRA_SIM_ROOT=$(git rev-parse --show-toplevel)
    echo "ASTRA_SIM_ROOT was not set. Defaulting to: $ASTRA_SIM_ROOT"
fi

# --- GENERAL CONFIGURATION ---
NPUS_COUNT=16
LOGICAL_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/16_nodes_logical.json"
PYTHON_EXEC="../../../opt/venv/astra-sim/bin/python"
BASE_WORKLOAD_DIR="$ASTRA_SIM_ROOT/upc/comparing_networks/workload/multiple_collectives"
TIMEOUT="300m" # Timeout for each individual collective simulation
MAX_PARALLEL_JOBS=15 # Number of parallel jobs, defaults to number of CPU cores

# --- SYSTEM CONFIGURATIONS ---
ANALYTICAL_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/FullyConnected_sys.json"
G2_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FullyConnected_sys.json"
NS3_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FullyConnected_sys.json" # Using G2 sys for NS3 as in the other script

# --- NETWORK & TOPOLOGY CONFIGURATIONS ---
ANALYTICAL_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/FullyConnected.yml"
G2_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FoldedClos_16_config.yml"
G2_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/configuration/g2/topologies/G2_FoldedClos_16"

# --- LISTS FOR LOOPS ---
# NS3_CONFIG_INDICES=(5 7 13 6 8 14) # Indices for NS3 config files to use
NS3_CONFIG_INDICES=(3 4 7)
TOPOLOGY_INDICES=(1)
NS3_TOPOLOGY_BASE="$ASTRA_SIM_ROOT/upc/configuration/ns3/topologies/ns3_FoldedClos_16"

# =================================================================================
# --- SIMULATION FUNCTION ---
# Defines the sequence of simulations for a single workload directory.
# =================================================================================
run_simulations() {
    workload_dir="$1"
    # Re-create the array from the passed string
    read -r -a NS3_CONFIG_INDICES <<< "$2"
    read -r -a TOPOLOGY_INDICES <<< "$3"

    workload_name=$(basename "$workload_dir")

    echo "#################################################################"
    echo "--- PROCESSING COLLECTIVE: $workload_name (Timeout: $TIMEOUT) ---"
    echo "#################################################################"

    # --- 1. ANALYTICAL MODEL EXECUTION ---
    echo "  --- [1/3] Running Analytical Model for $workload_name ---"
    timeout "$TIMEOUT" "$PYTHON_EXEC" compare_networks.py \
        --workload-dir "$workload_dir" \
        --npus-count "$NPUS_COUNT" \
        --logical-topology-config "$LOGICAL_CONFIG" \
        --analytical-system-config "$ANALYTICAL_SYS_CONFIG" \
        --analytical-network-config "$ANALYTICAL_NET_CONFIG" \
        --g2-network-config "$G2_NET_CONFIG" \
        --python-exec "$PYTHON_EXEC"
    echo "  --- Analytical Model Finished for $workload_name ---"


    # --- 2. G2 MODEL EXECUTION ---
    echo "  --- [2/3] Running G2 Model for $workload_name ---"
    for topo_idx in "${TOPOLOGY_INDICES[@]}"; do
        G2_TOPOLOGY_FILE="${G2_TOPOLOGY_BASE}_v${topo_idx}_Random.json"

        if [ ! -f "$G2_TOPOLOGY_FILE" ]; then
            echo "    -> WARNING: G2 topology file not found, skipping: $G2_TOPOLOGY_FILE"
            continue
        fi

        echo "    -> Using G2 Topology: $topo_idx for $workload_name"
        timeout "$TIMEOUT" "$PYTHON_EXEC" compare_networks.py \
            --workload-dir "$workload_dir" \
            --npus-count "$NPUS_COUNT" \
            --logical-topology-config "$LOGICAL_CONFIG" \
            --g2-system-config "$G2_SYS_CONFIG" \
            --g2-network-config "$G2_NET_CONFIG" \
            --g2-topology-file "$G2_TOPOLOGY_FILE" \
            --python-exec "$PYTHON_EXEC"
    done
    echo "  --- G2 Model Finished for $workload_name ---"


    # --- 3. NS3 MODEL EXECUTION ---
    echo "  --- [3/3] Running NS3 Model for $workload_name ---"
    for topo_idx in "${TOPOLOGY_INDICES[@]}"; do
        NS3_TOPOLOGY_FILE="${NS3_TOPOLOGY_BASE}_v${topo_idx}_Random"
        echo "    -> Using NS3 Topology: $topo_idx for $workload_name"

        for ns3_conf_idx in "${NS3_CONFIG_INDICES[@]}"; do
            NS3_CONFIG_FILE="$ASTRA_SIM_ROOT/upc/configuration/ns3/configs/FoldedClos_16_config${ns3_conf_idx}.txt"

            if [ ! -f "$NS3_CONFIG_FILE" ]; then
                echo "    -> WARNING: NS3 config file not found, skipping: $NS3_CONFIG_FILE"
                continue
            fi

            echo "    -> Using NS3 Config Index: $ns3_conf_idx for $workload_name"
            timeout "$TIMEOUT" "$PYTHON_EXEC" compare_networks.py \
                --workload-dir "$workload_dir" \
                --npus-count "$NPUS_COUNT" \
                --logical-topology-config "$LOGICAL_CONFIG" \
                --ns3-system-config "$NS3_SYS_CONFIG" \
                --ns3-network-config "$NS3_CONFIG_FILE" \
                --ns3-topology-file "$NS3_TOPOLOGY_FILE" \
                --ns3-precomputed-paths 1 \
                --python-exec "$PYTHON_EXEC"
        done
    done
    echo "  --- NS3 Model Finished for $workload_name ---"
}

# Export the function and variables to be available in sub-shells spawned by xargs
export -f run_simulations
export ASTRA_SIM_ROOT NPUS_COUNT LOGICAL_CONFIG PYTHON_EXEC TIMEOUT
export ANALYTICAL_SYS_CONFIG ANALYTICAL_NET_CONFIG G2_NET_CONFIG
export G2_SYS_CONFIG G2_TOPOLOGY_BASE NS3_SYS_CONFIG NS3_TOPOLOGY_BASE

# =================================================================================
# --- MAIN EXECUTION ---
# Finds all workload directories and runs simulations in parallel using xargs.
# =================================================================================
echo "--- STARTING PARALLEL SIMULATIONS (Max jobs: $MAX_PARALLEL_JOBS) ---"

# Convert arrays to a space-separated string for passing
ns3_indices_str="${NS3_CONFIG_INDICES[*]}"
topology_indices_str="${TOPOLOGY_INDICES[*]}"

find "$BASE_WORKLOAD_DIR" -mindepth 1 -maxdepth 1 -type d | \
    xargs -P "$MAX_PARALLEL_JOBS" -I {} bash -c 'run_simulations "{}" "$1" "$2"' _ "$ns3_indices_str" "$topology_indices_str"

echo "#################################################################"
echo "--- ALL SPLIT SIMULATIONS HAVE FINISHED ---"
echo "#################################################################"