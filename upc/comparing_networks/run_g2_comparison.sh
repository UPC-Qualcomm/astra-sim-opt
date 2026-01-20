#!/bin/bash

# Install bc if not present
if ! command -v bc &> /dev/null
then
    echo "bc could not be found, installing..."
    apt-get update && apt-get install -y bc
fi

# This script runs the Astra-G2 simulation four times, each with a different network implementation.
export PYTHONPATH=/app/astra-sim/extern/network_backend/g2:$PYTHONPATH

# Base directory for the network implementations
G2_BACKEND_DIR="/app/astra-sim/extern/network_backend/g2"

# Original network.py file
ORIGINAL_NETWORK_PY="$G2_BACKEND_DIR/network.py"

# Array of network implementations to test
NETWORK_IMPLS=("network.py" "network2.py" "network3.py" "network4.py")

# Simulation arguments
ARGS=(
    "--system-configuration=/app/astra-sim/upc/output/comparison_run/FoldedClos/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/run_20260106_105722_595ms/configs/g2_Ring_sys.json"
    "--workload-configuration=/app/astra-sim/upc/comparing_networks/workload/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024"
    "--comm-group-configuration=/app/astra-sim/upc/comparing_networks/workload/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024.json"
    "--network-configuration=/app/astra-sim/upc/output/comparison_run/FoldedClos/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/run_20260106_105722_595ms/configs/g2_FoldedClos_16_config.yml"
    "--remote-memory-configuration=/app/astra-sim/upc/output/comparison_run/FoldedClos/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/run_20260106_105722_595ms/configs/RemoteMemory.json"
    "--logical-topology=/app/astra-sim/upc/output/comparison_run/FoldedClos/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/run_20260106_105722_595ms/configs/16_nodes_logical.json"
    "--network-log=/app/astra-sim/upc/output/comparison_run/FoldedClos/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/run_20260106_105722_595ms/g2/",
    "--logging-folder=/app/astra-sim/upc/output/comparison_run/FoldedClos/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/run_20260106_105722_595ms/g2/"
)

ARGS=(
    "--system-configuration=/app/astra-sim/upc/output/comparison_run/FoldedClos128/T5_Small_grouped_128/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024/run_20260110_110640_081ms/configs/g2_Ring_sys.json"
    "--workload-configuration=/app/astra-sim/upc/comparing_networks/workload/T5_Small_grouped_128/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024"
    "--comm-group-configuration=/app/astra-sim/upc/comparing_networks/workload/T5_Small_grouped_128/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024.json"
    "--network-configuration=/app/astra-sim/upc/output/comparison_run/FoldedClos128/T5_Small_grouped_128/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024/run_20260110_110640_081ms/configs/g2_FoldedClos_128_config.yml"
    "--remote-memory-configuration=/app/astra-sim/upc/output/comparison_run/FoldedClos128/T5_Small_grouped_128/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024/run_20260110_110640_081ms/configs/RemoteMemory.json"
    "--logical-topology=/app/astra-sim/upc/output/comparison_run/FoldedClos128/T5_Small_grouped_128/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024/run_20260110_110640_081ms/configs/128_nodes_logical.json"
    "--network-log=/app/astra-sim/upc/output/comparison_run/FoldedClos128/T5_Small_grouped_128/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024/run_20260110_110640_081ms/g2/",
    "--logging-folder=/app/astra-sim/upc/output/comparison_run/FoldedClos128/T5_Small_grouped_128/T5_Small_multiple_8_4_4_1_0.seq_2048.batch_1024/run_20260110_110640_081ms/g2/"
)

# Create the output directory if it doesn't exist
OUTPUT_DIR="/app/astra-sim/upc/output/comparison_run/FoldedClos/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/run_20260106_105722_595ms/g2/"
OUTPUT_DIR="/app/astra-sim/upc/output/comparison_run/FoldedClos/T5_Small_grouped_ecmp/T5_Small_multiple_4_2_2_1_0.seq_2048.batch_1024/run_20260106_105722_595ms/g2/"
mkdir -p "$OUTPUT_DIR"

# Save the original network.py if it's not a symlink
if [ -f "$ORIGINAL_NETWORK_PY" ] && [ ! -L "$ORIGINAL_NETWORK_PY" ]; then
    mv "$ORIGINAL_NETWORK_PY" "$ORIGINAL_NETWORK_PY.bak"
fi

baseline_time=0
num_runs=1

# Loop through each network implementation
for i in "${!NETWORK_IMPLS[@]}"; do
    impl=${NETWORK_IMPLS[$i]}
    echo "Running simulation with $impl"

    # If it's the first run, use the backup file. Otherwise, copy the implementation.
    if [ $i -eq 0 ]; then
        if [ -f "$ORIGINAL_NETWORK_PY.bak" ]; then
            cp "$ORIGINAL_NETWORK_PY.bak" "$ORIGINAL_NETWORK_PY"
        else
            echo "Error: $ORIGINAL_NETWORK_PY.bak not found!"
            exit 1
        fi
    else
        cp "$G2_BACKEND_DIR/$impl" "$ORIGINAL_NETWORK_PY"
    fi

    total_time=0
    for j in $(seq 1 $num_runs); do
        echo "Run $j/$num_runs for $impl"
        # Run the simulation and capture time
        exec 3>&1 4>&2
        real_time=$( { time /app/astra-sim/build/astra_g2/build/bin/AstraSim_G2_congestion "${ARGS[@]}"; } 2>&1 )
        exec 3>&- 4>&-

        # Extract real time in seconds
        time_val=$(echo "$real_time" | grep real | awk '{print $2}' | sed 's/m/ /g' | sed 's/s//g' | awk '{print $1*60 + $2}')
        total_time=$(echo "$total_time + $time_val" | bc)
    done

    avg_time=$(echo "scale=2; $total_time / $num_runs" | bc)

    echo "Finished simulation with $impl"
    echo "Average execution time over $num_runs runs: ${avg_time}s"

    if [ $i -eq 0 ]; then
        baseline_time=$avg_time
    else
        if (( $(echo "$avg_time > 0" | bc -l) )); then
            speedup=$(echo "scale=2; $baseline_time / $avg_time" | bc)
            echo "Speedup vs ${NETWORK_IMPLS[0]}: ${speedup}x"
        else
            echo "Cannot calculate speedup, average execution time is zero."
        fi
    fi
    echo "---------------------------------"
done

# Restore the original network.py
if [ -f "$ORIGINAL_NETWORK_PY.bak" ]; then
    mv "$ORIGINAL_NETWORK_PY.bak" "$ORIGINAL_NETWORK_PY"
fi

echo "All simulations finished."
