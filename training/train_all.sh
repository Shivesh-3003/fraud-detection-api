#!/bin/bash
# ==============================================================================
# Train all pipeline steps for a given dataset.
#
# Usage:
#   bash train_all.sh ulb      # European credit card (anonymous V1-V28)
#   bash train_all.sh sparkov  # Sparkov synthetic (named features, SHAP demo)
#
# The 4-step pipeline:
#   1. Preprocessing  — feature engineering, scaling, saves feature_config.json
#   2. Autoencoder    — trains on normal-only data, learns reconstruction
#   3. AE Features    — appends Reconstruction_Error to feature matrices
#   4. Classifier     — trains MLP, finds optimal threshold
# ==============================================================================

DATASET=${1:-ulb}

if [[ "$DATASET" != "ulb" && "$DATASET" != "sparkov" ]]; then
    echo "❌ Invalid dataset: '$DATASET'. Use 'ulb' or 'sparkov'."
    exit 1
fi

echo "========================================"
echo "Training pipeline for dataset: $DATASET"
echo "========================================"

echo ""
echo "[1/4] Preprocessing..."
python credit_card_eda_preprocessing.py --dataset "$DATASET" || exit 1

echo ""
echo "[2/4] Training Autoencoder..."
python train_autoencoder.py --dataset "$DATASET" || exit 1

echo ""
echo "[3/4] Generating Reconstruction Error Features..."
python generate_ae_features.py --dataset "$DATASET" || exit 1

echo ""
echo "[4/4] Training MLP Classifier..."
python train_classifier.py --dataset "$DATASET" || exit 1

echo ""
echo "========================================"
echo "✅ Training complete for $DATASET"
echo "   Model artifacts saved to: models/$DATASET/"
echo ""
echo "Next steps:"
echo "  1. Copy models/$DATASET/ to python-ml-service/models/$DATASET/"
echo "  2. Start services: DATASET=$DATASET docker-compose up"
echo "========================================"
