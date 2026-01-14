#!/bin/bash

WORKLOAD_PATH=$1
WORKLOAD_NAME=$2

for i in $(seq 0 127)
do
  INPUT_FILE="${WORKLOAD_PATH}/${WORKLOAD_NAME}.${i}.et"
  OUTPUT_FILE="/app/astra-sim/upc/zz_analyze_error/jsons/${i}.et.txt"
  
  if [ -f "$INPUT_FILE" ]; then
    chakra_jsonizer --input_filename="$INPUT_FILE" --output_filename="$OUTPUT_FILE"
  else
    echo "Warning: Input file not found: $INPUT_FILE"
  fi
done

# Find and copy the JSON file
JSON_FILE=$(find "$WORKLOAD_PATH" -maxdepth 1 -name "*.json" -print -quit)
if [ -n "$JSON_FILE" ]; then
  cp "$JSON_FILE" "/app/astra-sim/upc/zz_analyze_error/jsons/"
  echo "Copied $JSON_FILE to jsons directory."
else
  echo "Warning: No JSON file found in $WORKLOAD_PATH"
fi
