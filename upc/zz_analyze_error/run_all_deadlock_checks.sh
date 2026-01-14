#!/bin/bash

# This script automates the process of running deadlock detection on a list of trace files.
# For each trace file, it first extracts the necessary workload information,
# then runs the run_temporal.sh script to prepare the JSON data,
# and finally runs the detect_deadlock.py script to analyze for deadlocks.

# Redirect all output to a log file
exec &> deadlock_analysis_results.txt

# Exit immediately if a command exits with a non-zero status.
set -e

# Base directory for workloads
# Note: You might need to adjust this path based on your directory structure
WORKLOAD_BASE_DIR="/app/astra-sim/upc/comparing_networks/workload/T5_Small_grouped_128"
TRACE_BASE_DIR="/app/astra-sim/upc/output/comparison_run/FoldedClos128/T5_Small_grouped_128"

# Get the list of trace files to be analyzed
TRACE_FILES=$(python3 count_trace_files.py "$TRACE_BASE_DIR")

# Change to the script's directory to ensure relative paths are correct
cd "$(dirname "$0")"

for trace_file_path in $TRACE_FILES; do
    echo "========================================================================"
    echo "Processing: ${trace_file_path}"
    echo "========================================================================"

    # Extract the workload name from the trace file path itself.
    # This is the directory name that contains the .et files.
    # e.g., T5_Small_multiple_4_2_16_1_0.seq_2048.batch_1024
    WORKLOAD_NAME=$(echo "$trace_file_path" | cut -d'/' -f1)

    # The workload group directory (e.g., T5_Small_grouped_128) is the second part of the path
    WORKLOAD_GROUP="T5_Small_grouped_128"

    # Construct the full path to the workload directory where the .et files are located
    WORKLOAD_PATH="${WORKLOAD_BASE_DIR}/${WORKLOAD_NAME}"
    
    # Construct the full path to the trace file
    TRACE_FILE_FULL_PATH="${TRACE_BASE_DIR}/${trace_file_path}"

    if [ ! -f "$TRACE_FILE_FULL_PATH" ]; then
        echo "Error: Trace file not found at $TRACE_FILE_FULL_PATH"
        continue
    fi

    echo "WORKLOAD_PATH: $WORKLOAD_PATH"
    echo "WORKLOAD_NAME: $WORKLOAD_NAME"
    echo "TRACE_FILE: $TRACE_FILE_FULL_PATH"

    # Run the temporal script to extract jsons
    echo "-> Running run_temporal.sh"
    ./run_temporal.sh "$WORKLOAD_PATH" "$WORKLOAD_NAME"
    
    # Run the deadlock detection script
    echo "-> Running detect_deadlock.py"
    python3 detect_deadlock.py --trace "$TRACE_FILE_FULL_PATH"
    
    echo "-> Finished processing ${trace_file_path}"
    echo ""
done

echo "========================================================================"
echo "All trace files processed."
echo "========================================================================"
