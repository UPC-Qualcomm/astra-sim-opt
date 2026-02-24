#!/bin/bash
# Complete pipeline to generate training data, train model, and make predictions

set -e  # Exit on error

echo "========================================"
echo "AstraSim Peak Memory Prediction: Train model"
echo "========================================"



cd "$(dirname "$0")/.."

#  Train model
echo ""
echo " Training ML models..."
echo ""

python ml_model/train_model.py \
    --input_csv ml_model/training_data.csv \
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
echo "  - Training data: ml_model/training_data.csv"
echo "  - Trained model: ml_model/trained_models/best_model.pkl"
echo "  - Model results: ml_model/trained_models/model_results.csv"
echo "  - Visualizations: ml_model/trained_models/*.png"
echo ""
echo "To make predictions, use:"
echo "  python ml_model/predict.py --model_dir ml_model/trained_models [OPTIONS]"
echo ""
