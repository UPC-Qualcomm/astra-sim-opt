#!/bin/bash

# This script converts all Chakra trace files (e.g., .et) in an input directory
# to JSON format in an output directory using the chakra_jsonizer tool.

# --- Configuration ---
# Assuming this script is run from the root of the astra-sim workspace.

# --- Script ---



INPUT_DIR="/app/astra-sim/upc/comparing_networks/workload/GPT_3_1300M"
OUTPUT_DIR="/app/astra-sim/upc/comparing_networks/workload/GPT_3_1300M_json"


# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output will be saved in '$OUTPUT_DIR'"

# Process each file in the input directory
for input_file in "$INPUT_DIR"/*; do
    if [ -f "$input_file" ]; then
        filename=$(basename -- "$input_file")
        # Change the file extension to .json
        output_filename="${filename%.*}.json"
        output_file="$OUTPUT_DIR/$output_filename"

        echo "Converting '$filename'..."

        # Execute the jsonizer
        chakra_jsonizer \
            --input_filename "$input_file" \
            --output_filename "$output_file"

        if [ $? -ne 0 ]; then
            echo "Error processing '$filename'. Aborting."
            exit 1
        fi
    fi
done

echo "Conversion complete."