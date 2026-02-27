#!/bin/bash

# This script copies run folders created before 13:00 from a source directory
# to a destination directory, recreating the same structure.

# Get the absolute path of the script's directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Define source and destination directories relative to the script's location
SOURCE_DIR="${SCRIPT_DIR}/output/comparison_run/experiment3/FoldedClosECMP/Llama8B_ecmp"
DEST_DIR="${SCRIPT_DIR}/output/comparison_run/experiment3/FoldedClosECMP/Llama8B_ecmp_2"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory not found: $SOURCE_DIR"
    exit 1
fi

echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"

# Find and copy the old run folders
find "$SOURCE_DIR" -mindepth 2 -maxdepth 2 -type d -name 'run_*' | while read -r rundir; do
    # Extract the timestamp from the directory name
    timestamp=$(basename "$rundir" | cut -d'_' -f 4)

    # Check if the time is before 13:00 (130000)
    if [ "$timestamp" -lt "130000" ]; then
        # Get the parent directory name (e.g., Llama8B_last_1_2_2_4_0.seq_2048.batch_64)
        parentdir_name=$(basename "$(dirname "$rundir")")

        # Create the corresponding parent directory in the destination
        mkdir -p "$DEST_DIR/$parentdir_name"

        # Move the run directory to the destination
        echo "Moving $rundir to $DEST_DIR/$parentdir_name/"
        mv "$rundir" "$DEST_DIR/$parentdir_name/"
    fi
done

echo "Script finished."
