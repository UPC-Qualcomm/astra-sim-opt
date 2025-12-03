#!/bin/bash

# Ensure ASTRA_SIM_ROOT is set. If not, try to get it with git.
if [ -z "$ASTRA_SIM_ROOT" ]; then
    export ASTRA_SIM_ROOT=$(git rev-parse --show-toplevel)
    echo "ASTRA_SIM_ROOT was not set. Defaulting to: $ASTRA_SIM_ROOT"
fi

# --- GENERAL CONFIGURATION ---
NPUS_COUNT=16
LOGICAL_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/ns3/16_nodes_logical.json"
COMM_SIZE=150000
PYTHON_EXEC="../../../opt/venv/astra-sim/bin/python"

# --- SYSTEM CONFIGURATIONS ---
G2_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FullyConnected_sys.json"
NS3_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FullyConnected_sys.json"
ANALYTICAL_SYS_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/FullyConnected_sys.json"

# --- NETWORK CONFIGURATIONS ---
G2_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/g2/FoldedClos_16_config.yml"
ANALYTICAL_NET_CONFIG="$ASTRA_SIM_ROOT/upc/configuration/FullyConnected.yml"

# --- WORKLOAD DIRECTORIES ---
WORKLOAD_DIRS=(
    "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/T5_Base_split"
    "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/T5_Base_split_2"
    "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/T5_Base_split_3"
    "$ASTRA_SIM_ROOT/upc/comparing_networks/workload/T5_Base_split_4"
)

TIMEOUT_DURATION="60m" # Set timeout as a unique variable

# --- TOPOLOGY AND CONFIG INDICES ---
NS3_CONFIG_INDICES=(5 7 13)
TOPOLOGY_INDICES=(1 2 3 4)

# =================================================================================
# --- MAIN LOOP ---
# =================================================================================

for workload_dir in "${WORKLOAD_DIRS[@]}"; do
    if [ ! -d "$workload_dir" ]; then
        echo "WARNING: Workload directory not found: $workload_dir"
        continue
    fi

    echo "#################################################################"
    echo "--- PROCESSING WORKLOAD: $(basename "$workload_dir") ---"
    echo "#################################################################"

    # --- 1. ANALYTICAL MODEL EXECUTION ---
    echo "--- Starting Analytical Model Execution ---"
    timeout "$TIMEOUT_DURATION" "$PYTHON_EXEC" compare_networks.py \
        --workload-dir "$workload_dir" \
        --npus-count $NPUS_COUNT \
        --comm-size $COMM_SIZE \
        --logical-topology-config "$LOGICAL_CONFIG" \
        --analytical-system-config "$ANALYTICAL_SYS_CONFIG" \
        --analytical-network-config "$ANALYTICAL_NET_CONFIG" \
        --g2-network-config "$G2_NET_CONFIG" \
        --python-exec "$PYTHON_EXEC"
    echo "--- Finished Analytical Model Execution ---"

    # --- 2. G2 MODEL EXECUTION ---
    echo "--- Starting G2 executions with different topologies ---"
    for topo_idx in "${TOPOLOGY_INDICES[@]}"; do
        echo "  --- Using G2 Topology #$topo_idx ---"

        TOPOLOGY_FILE="$ASTRA_SIM_ROOT/upc/configuration/g2/topologies/G2_FoldedClos_16.0_topology${topo_idx}.json"
        timeout "$TIMEOUT_DURATION" "$PYTHON_EXEC" compare_networks.py \
            --workload-dir "$workload_dir" \
            --npus-count $NPUS_COUNT \
            --comm-size $COMM_SIZE \
            --logical-topology-config "$LOGICAL_CONFIG" \
            --g2-system-config "$G2_SYS_CONFIG" \
            --g2-network-config "$G2_NET_CONFIG" \
            --g2-topology-file "$TOPOLOGY_FILE" \
            --python-exec "$PYTHON_EXEC"
    done
    echo "--- Finished G2 executions ---"

    # --- 3. NS3 MODEL EXECUTION ---
    echo "--- Starting NS3 executions with different topologies ---"
    for ns3_conf_idx in "${NS3_CONFIG_INDICES[@]}"; do
        NS3_CONFIG_FILE="$ASTRA_SIM_ROOT/upc/configuration/ns3/configs/FoldedClos_16_config${ns3_conf_idx}.txt"

        if [ ! -f "$NS3_CONFIG_FILE" ]; then
            echo "  --- WARNING: NS3 config file not found: $NS3_CONFIG_FILE. Skipping... ---"
            continue
        fi

        echo "  --- Using NS3 Config #$ns3_conf_idx ---"

        for topo_idx in "${TOPOLOGY_INDICES[@]}"; do
            echo "    --- Using NS3 Topology #$topo_idx ---"

            TOPOLOGY_FILE="$ASTRA_SIM_ROOT/upc/configuration/ns3/topologies/ns3_FoldedClos_16.0_topology${topo_idx}"

            timeout "$TIMEOUT_DURATION" "$PYTHON_EXEC" compare_networks.py \
                --workload-dir "$workload_dir" \
                --npus-count $NPUS_COUNT \
                --comm-size $COMM_SIZE \
                --logical-topology-config "$LOGICAL_CONFIG" \
                --ns3-system-config "$NS3_SYS_CONFIG" \
                --ns3-network-config "$NS3_CONFIG_FILE" \
                --ns3-topology-file "$TOPOLOGY_FILE" \
                --ns3-precomputed-paths 1 \
                --python-exec "$PYTHON_EXEC"
        done
    done
    echo "--- Finished NS3 executions ---"
done

echo "#################################################################"
echo "--- ALL SIMULATIONS HAVE FINISHED ---"
echo "#################################################################"