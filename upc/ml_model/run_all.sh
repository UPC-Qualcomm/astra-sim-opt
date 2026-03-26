#!/bin/bash
# Complete pipeline to generate training data, train model, and make predictions

set -e  # Exit on error

echo "========================================"
echo "AstraSim Peak Memory Prediction Pipeline"
echo "========================================"

# Configuration
TEST_MODE=${TEST_MODE:-false}
PARALLEL_JOBS=${PARALLEL_JOBS:-1}
NUM_MODELS=${NUM_MODELS:-30}
STRATEGY_SAMPLE_RATE=${STRATEGY_SAMPLE_RATE:-0.2}
BATCH_SIZE=${BATCH_SIZE:-1}
NPU_COUNTS=${NPU_COUNTS:-"16,32,64,128,256,512,1024"}
SIM_TYPE=${SIM_TYPE:-"analytical_unaware"}
APPEND=${APPEND:-true}
ENFORCE_MICRO_BATCH_RATIO=${ENFORCE_MICRO_BATCH_RATIO:-true}
if [ "$TEST_MODE" = "true" ]; then
    echo "Running in TEST MODE"
    TEST_FLAG="--test_mode"
    NUM_MODELS=2
else
    TEST_FLAG=""
fi

if [ "$APPEND" = "false" ]; then
    echo "Running in OVERWRITE MODE"
    APPEND_FLAG=""
else
    echo "Running in APPEND MODE"
    APPEND_FLAG="--append"
fi

if [ "$ENFORCE_MICRO_BATCH_RATIO" = "false" ]; then
    echo "Micro-batch ratio enforcement: DISABLED"
    MICRO_BATCH_FLAG="--no_enforce_micro_batch_ratio"
else
    echo "Micro-batch ratio enforcement: ENABLED (micro_batch >= batch/4)"
    MICRO_BATCH_FLAG="--enforce_micro_batch_ratio"
fi

# Step 1: Generate training data
echo ""
echo "Step 1: Generating training data..."
echo "  Number of models: $NUM_MODELS"
echo "  NPU counts: $NPU_COUNTS"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Strategy sample rate: $STRATEGY_SAMPLE_RATE (50% for NPUs <= 128)"
echo "  Batch size: $BATCH_SIZE"
echo "  Simulation type: $SIM_TYPE"
echo "  Micro-batch ratio enforcement: $ENFORCE_MICRO_BATCH_RATIO"
echo ""

cd "$(dirname "$0")/.."

python ml_model/generate_training_data.py \
    --num_models "$NUM_MODELS" \
    --npu_counts "$NPU_COUNTS" \
    --parallel_jobs "$PARALLEL_JOBS" \
    --batch_size "$BATCH_SIZE" \
    --strategy_sample_rate "$STRATEGY_SAMPLE_RATE" \
    --sim_type "$SIM_TYPE" \
    $APPEND_FLAG \
    $MICRO_BATCH_FLAG \
    $TEST_FLAG

if [ ! -f ml_model/data/training_data.csv ]; then
    echo "Error: Training data not generated!"
    exit 1
fi

echo ""
echo "Training data generated successfully!"

# Step 2: Train model
echo ""
echo "Step 2: Training ML models..."
echo ""

python ml_model/train_model.py \
    --input_csv ml_model/data \
    --output_dir ml_model/trained_models

if [ ! -f ml_model/trained_models/best_model.pkl ]; then
    echo "Error: Model training failed!"
    exit 1
fi

echo ""
echo "Model training completed successfully!"

# Step 3: Test prediction with an example
echo ""
echo "Step 3: Testing prediction..."
echo ""

python ml_model/predict.py \
    --model_dir ml_model/trained_models \
    --din 50000 \
    --dmodel 4096 \
    --dff 16384 \
    --batch 512 \
    --micro_batch 32 \
    --seq 2048 \
    --head 32 \
    --num_stacks 32 \
    --dp 2 \
    --mp 8 \
    --sp 1 \
    --pp 4 \
    --fsdp 1 \
    --num_npus 64

echo ""
echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - Training data: ml_model/data/training_data.csv"
echo "  - Trained model: ml_model/trained_models/best_model.pkl"
echo "  - Model results: ml_model/trained_models/model_results.csv"
echo "  - Visualizations: ml_model/trained_models/*.png"
echo ""
echo "To make predictions, use:"
echo "  python ml_model/predict.py --model_dir ml_model/trained_models [OPTIONS]"
echo ""
