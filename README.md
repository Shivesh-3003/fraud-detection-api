# Fraud Detection System

A real-time fraud detection system with explainability, built as a microservices architecture.

## Architecture

```
┌─────────────────┐         ┌──────────────────────────────────────────┐
│                 │         │           Docker Network                  │
│    Client       │         │                                          │
│   (curl/app)    │   ──►   │  ┌────────────┐      ┌────────────────┐  │
│                 │         │  │  Go API    │      │  Python ML     │  │
└─────────────────┘         │  │  :8080     │ ──►  │  Service :8000 │  │
                            │  │            │      │                │  │
                            │  │ • Routing  │      │ • Autoencoder  │  │
                            │  │ • Alerts   │      │ • MLP          │  │
                            │  │ • Logging  │      │ • SHAP         │  │
                            │  └────────────┘      └────────────────┘  │
                            └──────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- Your trained model files:
  - `autoencoder_model.pth`
  - `mlp_classifier.pth`
  - `scaler.pkl`

### 2. Setup Model Files

Copy your trained models to the Python service's models directory:

```bash
# Create the models directory
mkdir -p python-ml-service/models

# Copy your trained models
cp /path/to/autoencoder_model.pth python-ml-service/models/
cp /path/to/mlp_classifier.pth python-ml-service/models/
cp /path/to/scaler.pkl python-ml-service/models/
```

### 3. Start the System

```bash
# Build and start both services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### 4. Verify Services

```bash
# Check Go API health
curl http://localhost:8080/health

# Check ML Service health
curl http://localhost:8000/health
```

## API Usage

### Predict a Transaction

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "amount": 149.62,
    "time": 0,
    "features": {
      "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
      "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
      "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
      "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
      "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
      "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
      "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
      "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
      "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
      "V28": -0.021053
    }
  }'
```

### Test Python ML Service Directly

```bash
curl -X POST http://localhost:8000/predict?explain=true \
  -H "Content-Type: application/json" \
  -d '{
    "features": [-1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
                 0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
                 -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
                 -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
                 -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
                 -0.189115, 0.133558, -0.021053],
    "amount": 149.62,
    "time": 0
  }'
```

## Project Structure

```
fraud-detection-api/
├── docker-compose.yml          # Orchestrates both services
├── test_system.py              # End-to-end test script
│
├── training/                   # ML Training Pipeline
│   ├── README.md               # Pipeline documentation
│   ├── credit_card_eda_preprocessing.py  # Step 1: Data prep
│   ├── train_autoencoder.py    # Step 2: Train autoencoder
│   ├── generate_ae_features.py # Step 3: Generate error features
│   ├── train_classifier.py     # Step 4: Train classifier
│   └── analysis/               # Post-training evaluation scripts
│       ├── 01_threshold_extended.py    # F1/Precision/Recall vs threshold
│       ├── 02_density_separation.py    # AE error density (normal vs fraud)
│       ├── 03_roc_curve.py             # ROC curve + zoom
│       ├── 04_benchmarks.py            # LR / RF / XGBoost vs AE+MLP
│       ├── 05_inference_timing.py      # Per-stage latency benchmark
│       └── outputs/                    # Generated PNG / CSV / JSON
│
├── go-api/                     # Go API Orchestrator
│   ├── Dockerfile
│   ├── go.mod
│   ├── cmd/api/main.go         # Entry point
│   ├── config/config.go        # Configuration
│   └── internal/
│       ├── handlers/           # HTTP handlers
│       ├── models/             # Data structures
│       ├── services/           # ML client, alerting
│       └── middleware/         # Logging, CORS, recovery
│
└── python-ml-service/          # Python ML Service
    ├── Dockerfile
    ├── requirements.txt
    ├── test_local.py           # Local testing script
    ├── app/
    │   ├── main.py             # FastAPI application
    │   ├── models.py           # Pydantic schemas
    │   ├── ml_models.py        # PyTorch model definitions
    │   ├── inference.py        # Prediction pipeline
    │   └── explainer.py        # SHAP integration
    └── models/                 # Your trained models go here
        ├── autoencoder_model.pth
        ├── mlp_classifier.pth
        └── scaler.pkl
```

## Training the Models

If you need to retrain the models, see the `training/` directory. The pipeline is:

```bash
cd training

# 1. Download creditcard.csv from Kaggle first
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 2. Run the pipeline in order
python credit_card_eda_preprocessing.py   # Data preparation
python train_autoencoder.py               # Train autoencoder
python generate_ae_features.py            # Generate error features
python train_classifier.py                # Train classifier

# 3. Copy models to inference service
cp scaler.pkl ../python-ml-service/models/
cp autoencoder_model.pth ../python-ml-service/models/
cp mlp_classifier.pth ../python-ml-service/models/
```

See `training/README.md` for detailed documentation.

## Configuration

### Go API Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `ENV` | `development` | Environment |
| `ML_SERVICE_URL` | `http://localhost:8000` | Python ML service URL |
| `ML_SERVICE_TIMEOUT` | `5s` | Request timeout |
| `FRAUD_THRESHOLD` | `0.5` | Classification threshold |
| `SLACK_WEBHOOK_URL` | - | Slack alerts webhook |

### Python ML Service Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models` | Path to model files |

## Testing

### Run End-to-End Tests

```bash
# Install test dependencies
pip install requests

# Test ML service directly
python test_system.py --ml-only

# Test full system
python test_system.py
```

### Manual Testing

```bash
# Health check
curl http://localhost:8080/health

# Predict (normal transaction - should return low probability)
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @test_normal.json

# Predict (suspicious transaction - should return high probability)
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @test_fraud.json
```

## Logs

```bash
# View all logs
docker-compose logs -f

# View only Go API logs
docker-compose logs -f go-api

# View only ML service logs
docker-compose logs -f ml-service
```

## Stopping the System

```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Troubleshooting

### Models not loading

Check that model files exist and have correct names:
```bash
ls -la python-ml-service/models/
# Should show:
# - autoencoder_model.pth
# - mlp_classifier.pth  
# - scaler.pkl
```

### ML Service unhealthy

Check logs for errors:
```bash
docker-compose logs ml-service
```

### Connection refused

Ensure services are running:
```bash
docker-compose ps
```

## Performance

Per-transaction latency, measured locally (Apple Silicon, CPU-only PyTorch,
n=5,000, 100-iter warm-up — see `training/analysis/05_inference_timing.py`):

| Stage | Mean | p95 | p99 |
|---|---|---|---|
| Preprocess | 0.002 ms | 0.002 ms | 0.010 ms |
| Scale | 0.023 ms | 0.028 ms | 0.035 ms |
| Autoencoder forward | 0.044 ms | 0.051 ms | 0.059 ms |
| Classifier forward | 0.021 ms | 0.025 ms | 0.029 ms |
| **ML inference (AE + MLP)** | **0.065 ms** | 0.076 ms | 0.086 ms |
| **End-to-end (single txn)** | **0.090 ms** | 0.105 ms | 0.118 ms |

SHAP explanation adds ~hundreds of ms (depends on `nsamples`); the Go API
only invokes it when the fast path flags fraud.

### Model performance vs. baselines

Isolated baselines on the same 31 raw engineered features (V1–V28,
Amount_Log, Hour_sin, Hour_cos); AE+MLP uses its full 32-dim pipeline.
Threshold for each model picked by max-F1 on the test PR curve.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8387 | 0.7959 | 0.8168 | 0.7460 | 0.9723 |
| Random Forest | 0.9318 | 0.8367 | 0.8817 | 0.8604 | 0.9523 |
| XGBoost | 0.9756 | 0.8163 | 0.8889 | **0.8777** | 0.9727 |
| **AE+MLP (ours)** | 0.8438 | **0.8265** | 0.8351 | 0.7521 | **0.9783** |

Reproduce with `python3 training/analysis/04_benchmarks.py`.
