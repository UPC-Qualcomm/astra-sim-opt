#!/bin/bash

WORKLOAD_PATH=$1
WORKLOAD_NAME=$2
ANALYSIS_DIR=$3

echo "--- Inside run_temporal.sh ---"
echo "WORKLOAD_PATH: ${WORKLOAD_PATH}"
echo "WORKLOAD_NAME: ${WORKLOAD_NAME}"
echo "ANALYSIS_DIR: ${ANALYSIS_DIR}"

ET_TXTS_DIR="${ANALYSIS_DIR}/et_txts"
mkdir -p "$ET_TXTS_DIR"
echo "Created et_txts directory: ${ET_TXTS_DIR}"

for i in $(seq 0 127)
do
  INPUT_FILE="${WORKLOAD_PATH}/${WORKLOAD_NAME}.${i}.et"
  OUTPUT_FILE="${ET_TXTS_DIR}/${i}.et.txt"
  
  echo "Loop ${i}: Checking for ${INPUT_FILE}"

  if [ -f "$INPUT_FILE" ]; then
    echo "  -> File found. Processing..."
    # Extract string data and check if it's empty before piping to chakra_jsonizer
    COMM_DATA=$(strings "$INPUT_FILE" | grep "COMM")
    if [ -n "$COMM_DATA" ]; then
      echo "    --> COMM data found. Running chakra_jsonizer."
      echo "$COMM_DATA" | chakra_jsonizer --input_filename="$INPUT_FILE" --output_filename="$OUTPUT_FILE"
    else
      # Create an empty file to signify that no COMM nodes were found
      echo "    --> No COMM data found. Creating empty file."
      touch "$OUTPUT_FILE"
    fi
  else
    # This is likely where the problem is. This message will now appear in the log.
    echo "  -> Warning: Input file not found: $INPUT_FILE"
  fi
done

echo "--- Finished processing .et files ---"

# Find and copy the JSON file to the root of the analysis directory
JSON_FILE=$(find "$WORKLOAD_PATH" -maxdepth 1 -name "*.json" -print -quit)
if [ -n "$JSON_FILE" ]; then
  cp "$JSON_FILE" "$ANALYSIS_DIR/"
  echo "Copied $JSON_FILE to $ANALYSIS_DIR."
else
  echo "Warning: No JSON file found in $WORKLOAD_PATH"
fi

echo "--- Exiting run_temporal.sh ---"

