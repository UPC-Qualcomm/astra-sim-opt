#!/bin/bash

# This script removes run directories that contain an 'analytical_unaware' subfolder
# from experiment directories (experiment1 to experiment5).

BASE_DIR="/app/astra-sim/upc/output/comparison_run"

# Check if the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory '$BASE_DIR' not found."
    exit 1
fi

echo "Starting cleanup of analytical_unaware runs..."

# Find and delete the target run directories
find "$BASE_DIR/experiment"{1,2,3,4,5} -type d -name "run_*" | while read run_dir; do
    if [ -d "$run_dir/analytical_unaware" ]; then
        echo "Removing: $run_dir"
        rm -rf "$run_dir"
    fi
done

echo "Cleanup complete."
