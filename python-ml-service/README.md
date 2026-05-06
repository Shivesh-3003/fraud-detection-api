# Fraud Detection ML Service

Python-based ML inference service for real-time fraud detection.

## Architecture

```
Request (28 V-features + Amount + Time)
    │
    ▼
┌─────────────────────────────────────────┐
│  1. PREPROCESS                          │
│     - Amount → log1p(Amount)            │
│     - Time → Hour_sin, Hour_cos         │
│     → 31 features                       │
├─────────────────────────────────────────┤
│  2. SCALE                               │
│     - Apply fitted StandardScaler       │
│     → 31 scaled features                │
├─────────────────────────────────────────┤
│  3. AUTOENCODER                         │
│     - Forward pass                      │
│     - MSE(input, reconstruction)        │
│     → reconstruction_error              │
├─────────────────────────────────────────┤
│  4. CONCATENATE                         │
│     - 31 features + reconstruction_error│
│     → 32 features                       │
├─────────────────────────────────────────┤
│  5. CLASSIFIER                          │
│     - MLP forward pass                  │
│     - sigmoid(logits)                   │
│     → fraud_probability                 │
├─────────────────────────────────────────┤
│  6. SHAP (optional)                     │
│     - KernelExplainer on classifier     │
│     → feature contributions             │
└─────────────────────────────────────────┘
    │
    ▼
Response (fraud_probability + reconstruction_error + shap_values)
```

## Files Required

Trained artefacts live in **per-dataset subdirectories** under `models/`. The
service mounts a single subdirectory at runtime via the `DATASET` env var
(default: `ulb`).

```
models/
├── ulb/                       # default — European credit card
│   ├── scaler.pkl
│   ├── autoencoder_model.pth
│   ├── mlp_classifier.pth
│   ├── feature_config.json
│   └── optimal_threshold.json
└── sparkov/                   # synthetic dataset (named features, SHAP demo)
    ├── scaler.pkl
    ├── onehot_encoder.pkl
    ├── autoencoder_model.pth
    ├── mlp_classifier.pth
    ├── feature_config.json
    └── optimal_threshold.json
```

| File | Description |
|------|-------------|
| `scaler.pkl` | Fitted StandardScaler (from preprocessing) |
| `onehot_encoder.pkl` | OneHotEncoder for categorical features (sparkov only) |
| `autoencoder_model.pth` | Trained Autoencoder weights |
| `mlp_classifier.pth` | Trained MLP Classifier weights |
| `feature_config.json` | Feature names, dims, and dataset metadata |
| `optimal_threshold.json` | F1-optimal classification threshold |

These files are produced by the training pipeline (`../training/`) and are not
in git. See `models/README.md` for details on generating and staging them.

## API Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "autoencoder_input_dim": 31,
  "classifier_input_dim": 32
}
```

### POST /predict

Run fraud detection on a single transaction.

**Request:**
```json
{
  "features": [-1.359807, -0.072781, ...],  // 28 V-features
  "amount": 149.62,
  "time": 0.0
}
```

**Response:**
```json
{
  "fraud_probability": 0.0234,
  "reconstruction_error": 0.0012
}
```

### POST /predict?explain=true

Prediction with SHAP explanation.

**Response:**
```json
{
  "fraud_probability": 0.9456,
  "reconstruction_error": 15.234,
  "shap_values": {
    "V14": -2.341,
    "V12": -1.892,
    "Reconstruction_Error": 0.543,
    ...
  },
  "base_value": 0.00172
}
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set model path to the chosen dataset subdir
export MODEL_PATH=./models/ulb        # or ./models/sparkov

# Run service
uvicorn app.main:app --reload --port 8000
```

## Docker

```bash
# Build
docker build -t fraud-ml-service .

# Run (mount the chosen dataset subdir as /app/models)
docker run -p 8000:8000 -v "$(pwd)/models/ulb:/app/models:ro" fraud-ml-service
```

In normal use you should launch via the project-root `docker-compose.yml`,
which selects the dataset subdirectory automatically based on the `DATASET`
env var.

## Feature Order

The 31 features (after preprocessing) are in this order:
```
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28,
Amount_Log, Hour_sin, Hour_cos
```

The classifier receives 32 features (31 + Reconstruction_Error).
