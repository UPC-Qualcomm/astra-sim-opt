#!/bin/bash
# Train and test range-based memory prediction models.

set -e

echo "========================================"
echo "AstraSim Peak Memory Range Pipeline"
echo "========================================"

cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-/media/mohammad/extension/experiments/311_astraenv/bin/python}
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN=${PYTHON_BIN_FALLBACK:-python3}
fi

OUTPUT_DIR=${OUTPUT_DIR:-ml_model/trained_range_models}
INPUT_CSV=${INPUT_CSV:-ml_model/data}
LOWER_QUANTILE=${LOWER_QUANTILE:-0.10}
UPPER_QUANTILE=${UPPER_QUANTILE:-0.90}
TEST_SIZE=${TEST_SIZE:-0.2}
MIN_KEEP_RATIO=${MIN_KEEP_RATIO:-0.85}

TUNE_PEAK_PER_NPU_THRESHOLD=${TUNE_PEAK_PER_NPU_THRESHOLD:-true}
PEAK_PER_NPU_THRESHOLD=${PEAK_PER_NPU_THRESHOLD:-}

if [ "$TUNE_PEAK_PER_NPU_THRESHOLD" = "true" ]; then
    FILTER_FLAGS="--tune_peak_per_npu_threshold --min_keep_ratio $MIN_KEEP_RATIO"
    echo "Outlier filter threshold: AUTO-TUNED (min_keep_ratio=$MIN_KEEP_RATIO)"
elif [ -n "$PEAK_PER_NPU_THRESHOLD" ]; then
    FILTER_FLAGS="--peak_per_npu_threshold $PEAK_PER_NPU_THRESHOLD"
    echo "Outlier filter threshold: FIXED ($PEAK_PER_NPU_THRESHOLD)"
else
    FILTER_FLAGS=""
    echo "Outlier filter threshold: DISABLED"
fi

echo ""
echo "Step 1: Training range models..."
"$PYTHON_BIN" ml_model/train_range_model.py \
    --input_csv "$INPUT_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --test_size "$TEST_SIZE" \
    --lower_quantile "$LOWER_QUANTILE" \
    --upper_quantile "$UPPER_QUANTILE" \
    $FILTER_FLAGS

if [ ! -f "$OUTPUT_DIR/range_lower_model.pkl" ] || [ ! -f "$OUTPUT_DIR/range_upper_model.pkl" ]; then
    echo "Error: Range model training failed!"
    exit 1
fi

echo ""
echo "Range model training completed successfully!"

echo ""
echo "Step 2: Testing range prediction..."

# Example config. Override with env vars as needed.
DIN=${DIN:-50000}
DMODEL=${DMODEL:-4096}
DFF=${DFF:-16384}
BATCH=${BATCH:-512}
MICRO_BATCH=${MICRO_BATCH:-32}
SEQ=${SEQ:-2048}
HEAD=${HEAD:-32}
NUM_STACKS=${NUM_STACKS:-32}
DP=${DP:-2}
MP=${MP:-8}
SP=${SP:-1}
PP=${PP:-4}
FSDP=${FSDP:-1}
NUM_NPUS=${NUM_NPUS:-64}
OOM_CAPACITY_GB=${OOM_CAPACITY_GB:-96}

"$PYTHON_BIN" ml_model/predict_range.py \
    --model_dir "$OUTPUT_DIR" \
    --din "$DIN" \
    --dmodel "$DMODEL" \
    --dff "$DFF" \
    --batch "$BATCH" \
    --micro_batch "$MICRO_BATCH" \
    --seq "$SEQ" \
    --head "$HEAD" \
    --num_stacks "$NUM_STACKS" \
    --dp "$DP" \
    --mp "$MP" \
    --sp "$SP" \
    --pp "$PP" \
    --fsdp "$FSDP" \
    --num_npus "$NUM_NPUS" \
    --oom_capacity_gb "$OOM_CAPACITY_GB"

echo ""
echo "========================================"
echo "Range pipeline completed successfully!"
echo "========================================"
echo ""
echo "Generated files in: $OUTPUT_DIR"
echo "  - range_lower_model.pkl"
echo "  - range_upper_model.pkl"
echo "  - range_model_metadata.pkl"
echo "  - range_model_summary.csv"
echo ""
echo "To run only prediction, use:"
echo "  $PYTHON_BIN ml_model/predict_range.py --model_dir $OUTPUT_DIR [OPTIONS]"
