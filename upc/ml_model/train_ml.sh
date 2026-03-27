#!/bin/bash
# Complete pipeline to generate training data, train model, and make predictions

set -e  # Exit on error

echo "========================================"
echo "AstraSim Peak Memory Prediction: Train model"
echo "========================================"



cd "$(dirname "$0")/.."

TUNE_PEAK_PER_NPU_THRESHOLD=${TUNE_PEAK_PER_NPU_THRESHOLD:-true}
MIN_KEEP_RATIO=${MIN_KEEP_RATIO:-0.85}

if [ "$TUNE_PEAK_PER_NPU_THRESHOLD" = "true" ]; then
    OUTLIER_FLAGS="--tune_peak_per_npu_threshold --min_keep_ratio $MIN_KEEP_RATIO"
    echo "Outlier filter threshold: AUTO-TUNED (min_keep_ratio=$MIN_KEEP_RATIO)"
else
    OUTLIER_FLAGS=""
    echo "Outlier filter threshold: DISABLED"
fi

#  Train model
echo ""
echo " Training ML models..."
echo ""

python ml_model/train_model.py \
    --input_csv ml_model/data \
    --output_dir ml_model/trained_models_big_data \
    $OUTLIER_FLAGS

if [ ! -f ml_model/trained_models_big_data/best_model.pkl ]; then
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
    --model_dir ml_model/trained_models_big_data \
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
echo "  - Trained model: ml_model/trained_models_big_data/best_model.pkl"
echo "  - Model results: ml_model/trained_models_big_data/model_results.csv"
echo "  - Visualizations: ml_model/trained_models_big_data/*.png"
echo ""
echo "To make predictions, use:"
echo "  python ml_model/predict.py --model_dir ml_model/trained_models_big_data [OPTIONS]"
echo ""
