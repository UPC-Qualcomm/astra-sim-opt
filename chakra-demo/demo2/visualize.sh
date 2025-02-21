#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
TARGET_WORKLOAD="MLP_HybridParallel_Model_Data"

# Run visualizer
(
chakra_visualizer \
    --input_filename=${SCRIPT_DIR}/workload/"${TARGET_WORKLOAD:?}.31.et" \
    --output_filename=${SCRIPT_DIR}/${TARGET_WORKLOAD:?}.0.pdf
)
