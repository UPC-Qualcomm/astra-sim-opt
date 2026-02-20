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
ANALYSIS_BASE_DIR="/app/astra-sim/upc/zz_analyze_error/analysis"

# Get the list of trace files to be analyzed
TRACE_FILES=$(python3 count_trace_files.py "$TRACE_BASE_DIR")

# Change to the script's directory to ensure relative paths are correct
cd "$(dirname "$0")"

# Clean up previous analysis directories
rm -rf "$ANALYSIS_BASE_DIR"
mkdir -p "$ANALYSIS_BASE_DIR"

for trace_file_path in $TRACE_FILES; do
    echo "========================================================================"
    echo "Processing: ${trace_file_path}"
    echo "========================================================================"

    # Extract the workload name from the trace file path itself.
    WORKLOAD_DIR_NAME=$(echo "$trace_file_path" | cut -d'/' -f1)
    # Extract the base name of the trace file, which corresponds to the workload name for .et files
    WORKLOAD_NAME=$(basename "$trace_file_path" "_trace.csv")

    # Construct the full path to the workload directory where the .et files are located
    WORKLOAD_PATH="${WORKLOAD_BASE_DIR}/${WORKLOAD_DIR_NAME}"
    
    # Construct the full path to the trace file
    TRACE_FILE_FULL_PATH="${TRACE_BASE_DIR}/${trace_file_path}"

    if [ ! -f "$TRACE_FILE_FULL_PATH" ]; then
        echo "Error: Trace file not found at $TRACE_FILE_FULL_PATH"
        continue
    fi

    # Create a dedicated directory for this trace's analysis
    TRACE_ANALYSIS_DIR_NAME=$(basename "$trace_file_path" .csv)
    TRACE_ANALYSIS_DIR="${ANALYSIS_BASE_DIR}/${TRACE_ANALYSIS_DIR_NAME}"
    mkdir -p "$TRACE_ANALYSIS_DIR"

    echo "WORKLOAD_PATH: $WORKLOAD_PATH"
    echo "WORKLOAD_NAME: $WORKLOAD_NAME"
    echo "TRACE_FILE: $TRACE_FILE_FULL_PATH"
    echo "ANALYSIS_DIR: $TRACE_ANALYSIS_DIR"

    # Run the temporal script to extract jsons into the dedicated directory
    echo "-> Running run_temporal.sh"
    ./run_temporal.sh "$WORKLOAD_PATH" "$WORKLOAD_NAME" "$TRACE_ANALYSIS_DIR"
    
    # Copy the trace file into the analysis directory
    cp "$TRACE_FILE_FULL_PATH" "$TRACE_ANALYSIS_DIR/"

    # Define the output file for the deadlock analysis
    DEADLOCK_OUTPUT_FILE="${TRACE_ANALYSIS_DIR}/deadlock_analysis.txt"
    
    # Find the JSON file in the analysis directory
    JSON_FILE=$(find "$TRACE_ANALYSIS_DIR" -maxdepth 1 -name "*.json" -print -quit)

    # Run the deadlock detection script and redirect its output
    echo "-> Running detect_deadlock.py"
    python3 detect_deadlock.py \
        --trace "$TRACE_ANALYSIS_DIR/$(basename "$TRACE_FILE_FULL_PATH")" \
        --et-dir "$TRACE_ANALYSIS_DIR/et_txts" \
        --comm-group "$JSON_FILE" > "$DEADLOCK_OUTPUT_FILE"
    
    echo "-> Finished processing ${trace_file_path}"
    echo "   Deadlock analysis saved to: $DEADLOCK_OUTPUT_FILE"
    echo ""
done

echo "========================================================================"
echo "All trace files processed."
echo "========================================================================"
