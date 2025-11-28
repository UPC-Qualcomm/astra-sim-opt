#!/bin/bash

# This script runs AstraSim with different network backends for comparison.

set -e # Exit on any error

export ASTRA_SIM_ROOT=$(git rev-parse --show-toplevel)

# Define a folder name for organizing outputs.
# This can be changed to reflect the specific experiment.
folder_name="comparison_run"

# Common paths
workload_dir="$ASTRA_SIM_ROOT/upc/comparing_networks/workload/toy_reduce_scatter_one_collective"
memory_config="$ASTRA_SIM_ROOT/upc/configuration/RemoteMemory.json"
base_output_dir="$ASTRA_SIM_ROOT/upc/output/${folder_name}/"
base_network_log_dir="$ASTRA_SIM_ROOT/upc/network_log/${folder_name}/"
base_result_dir="$ASTRA_SIM_ROOT/upc/results/${folder_name}/"

# Python executable path (using the one from the ns3 script for consistency)
PYTHON_EXEC="$ASTRA_SIM_ROOT/astraenv39/bin/python3.9" # Or specify a path like "../../astraenv39/bin/python"

# Set up environment for G2 model
export PYTHONPATH="$ASTRA_SIM_ROOT/extern/network_backend/g2:$PYTHONPATH"

# Clean up previous runs for this folder
rm -rf $base_output_dir
rm -rf $base_network_log_dir
rm -rf $base_result_dir

# Create directories
mkdir -p $base_output_dir
mkdir -p $base_network_log_dir
mkdir -p $base_result_dir

# Array of simulation types to run
sim_types=("analytical_unaware" "analytical_aware" "g2" "ns3")
sim_types=("ns3")

echo "=== Starting AstraSim Network Comparison ==="
echo "Workload Directory: $workload_dir"
echo "Running for models: ${sim_types[@]}"
echo "=========================================="

# Loop over each simulation type
for sim_type in "${sim_types[@]}"; do
    echo ""
    echo "--- Running simulation for: ${sim_type} ---"

    # Per-simulation type output and log directories
    output_dir="${base_output_dir}${sim_type}"
    network_log="${base_network_log_dir}${sim_type}"

    if [ "$sim_type" == "ns3" ]; then
        # NS3 specific configuration
        echo "Using NS3 network model."
        time $PYTHON_EXEC $ASTRA_SIM_ROOT/upc/run_astrasim_ns3.py \
            --workload_dir "$workload_dir" \
            --system "$ASTRA_SIM_ROOT/upc/configuration/ns3/8_nodes_sys.json" \
            --network_config "$ASTRA_SIM_ROOT/extern/network_backend/ns-3/scratch/config/config_8_ring.txt" \
            --logical_topology "$ASTRA_SIM_ROOT/upc/configuration/ns3/8_nodes_logical.json" \
            --memory "$memory_config" \
            --output_dir "$output_dir" \
            --network_log "$network_log"


    elif [ "$sim_type" == "g2" ] ||[ "$sim_type" == "analytical_unaware" ] || [ "$sim_type" == "analytical_aware" ]; then
        # Other models configuration
        echo "Using ${sim_type} network model."
        time $PYTHON_EXEC $ASTRA_SIM_ROOT/upc/run_astrasim.py \
            --workload_dir "$workload_dir" \
            --system "$ASTRA_SIM_ROOT/upc/configuration/Ring_sys.json" \
            --network "$ASTRA_SIM_ROOT/upc/configuration/Ring.yml" \
            --memory "$memory_config" \
            --output_dir "$output_dir" \
            --network_log "$network_log" \
            --sim_type "$sim_type"
    else
        echo "Unknown simulation type: $sim_type"
    fi

    echo "--- Finished simulation for: ${sim_type} ---"
done

echo ""
echo "=== All simulations completed. ==="