# ML Training Pipeline

This directory contains the complete machine learning pipeline for training the fraud detection models.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: credit_card_eda_preprocessing.py                                   │
│                                                                             │
│  Input:  creditcard.csv (Kaggle dataset)                                    │
│  Output: X_train_AE.csv      → Normal transactions only (for Autoencoder)   │
│          X_train_MLP.csv     → All training data (for later classifier)     │
│          y_train_MLP.csv     → Training labels                              │
│          X_test.csv          → Test features                                │
│          y_test.csv          → Test labels                                  │
│          scaler.pkl          → Fitted StandardScaler                        │
│                                                                             │
│  Key Operations:                                                            │
│    - Log transform Amount → Amount_Log                                      │
│    - Cyclic encoding Time → Hour_sin, Hour_cos                              │
│    - Stratified train/test split                                            │
│    - Extract NORMAL-ONLY subset for autoencoder training                    │
│    - Fit scaler on normal data only                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: train_autoencoder.py                                               │
│                                                                             │
│  Input:  X_train_AE.csv (normal transactions only)                          │
│  Output: autoencoder_model.pth                                              │
│                                                                             │
│  Architecture: 31 → 20 → 14 → 8 → 14 → 20 → 31                              │
│  Activation:   Tanh                                                         │
│  Loss:         MSE (reconstruction error)                                   │
│  Optimizer:    Adam (lr=0.001)                                              │
│  Epochs:       50                                                           │
│                                                                             │
│  Purpose: Learn to reconstruct NORMAL transactions. Fraudulent              │
│           transactions will have higher reconstruction error.               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: generate_ae_features.py                                            │
│                                                                             │
│  Input:  X_train_MLP.csv, X_test.csv, autoencoder_model.pth                 │
│  Output: X_train_final.csv (31 features + Reconstruction_Error)             │
│          X_test_final.csv  (31 features + Reconstruction_Error)             │
│                                                                             │
│  Purpose: Generate the "anomaly signal" - reconstruction error for          │
│           each transaction. This becomes an additional feature for          │
│           the classifier.                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: train_classifier.py                                                │
│                                                                             │
│  Input:  X_train_final.csv (32 features), y_train_MLP.csv                   │
│  Output: mlp_classifier.pth                                                 │
│                                                                             │
│  Architecture: 32 → 16 → 1                                                  │
│  Activation:   ReLU + Dropout(0.3)                                          │
│  Loss:         BCEWithLogitsLoss (weighted for imbalance)                   │
│  Optimizer:    Adam (lr=0.001)                                              │
│  Epochs:       30                                                           │
│                                                                             │
│  Purpose: Final fraud classification using original features PLUS           │
│           the reconstruction error from the autoencoder.                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Execution Order

Run the scripts in this exact order:

```bash
# 1. Download creditcard.csv from Kaggle and place in this directory
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 2. Preprocess and split data
python credit_card_eda_preprocessing.py

# 3. Train the autoencoder on normal transactions
python train_autoencoder.py

# 4. Generate reconstruction error features
python generate_ae_features.py

# 5. Train the final classifier
python train_classifier.py
```

## Output Files for Inference

After training, copy these files to `../python-ml-service/models/`:

| File | Purpose |
|------|---------|
| `scaler.pkl` | StandardScaler for feature normalization |
| `autoencoder_model.pth` | Autoencoder weights for reconstruction error |
| `mlp_classifier.pth` | Final classifier weights |

```bash
# Copy to inference service
cp scaler.pkl ../python-ml-service/models/
cp autoencoder_model.pth ../python-ml-service/models/
cp mlp_classifier.pth ../python-ml-service/models/
```

## Feature Engineering Details

### Original Features (from Kaggle dataset)
- `V1` - `V28`: PCA-transformed features (anonymized)
- `Amount`: Transaction amount
- `Time`: Seconds since first transaction
- `Class`: 0 = Normal, 1 = Fraud

### Engineered Features
- `Amount_Log`: `log1p(Amount)` - handles skewed distribution
- `Hour_sin`: `sin(2π × Hour / 24)` - cyclic time encoding
- `Hour_cos`: `cos(2π × Hour / 24)` - cyclic time encoding
- `Reconstruction_Error`: MSE from autoencoder - anomaly signal

### Final Feature Set (32 features)
```
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28,
Amount_Log, Hour_sin, Hour_cos,
Reconstruction_Error
```

## Model Architecture Summary

### Autoencoder
```
Input Layer:  31 neurons (scaled features)
Encoder:      31 → 20 (Tanh) → 14 (Tanh) → 8 (Tanh)
Decoder:      8 → 14 (Tanh) → 20 (Tanh) → 31 (Linear)
Output Layer: 31 neurons (reconstructed features)
```

### MLP Classifier
```
Input Layer:   32 neurons (features + reconstruction error)
Hidden Layer:  16 neurons (ReLU + Dropout 0.3)
Output Layer:  1 neuron (Logits → Sigmoid for probability)
```

## Performance Metrics

The classifier is evaluated on:
- **Precision**: TP / (TP + FP) - How many flagged transactions are actually fraud
- **Recall**: TP / (TP + FN) - How many actual frauds are caught
- **F1-Score**: Harmonic mean of Precision and Recall
- **AUC-ROC**: Area under the ROC curve
- **G-mean**: √(Recall × Specificity) - Balance metric for imbalanced data

## Notes

- The autoencoder is trained on **NORMAL transactions only** so it learns the "normal" pattern
- Fraudulent transactions produce higher reconstruction error (anomaly signal)
- Class weights are applied during classifier training to handle the extreme imbalance (~0.17% fraud)
- Threshold optimization is performed to find the best F1-score cutoff
