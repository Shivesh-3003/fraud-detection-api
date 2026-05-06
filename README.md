# HIGH-PERFORMANCE REAL-TIME FRAUD DETECTION SYSTEM

A real-time fraud detection system with explainability, built as a microservices architecture.

## Architecture

```
┌─────────────────┐         ┌──────────────────────────────────────────┐
│                 │         │           Docker Network                 │
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

## Why no executable?

This project is a **multi-service application**, not a single binary. It consists
of:

1. A Go HTTP API (compiled inside a Docker container)
2. A Python FastAPI ML inference service (PyTorch + SHAP, runs in a separate
   Docker container)
3. Trained model artefacts (`*.pth`, `*.pkl`, `*.json`) — these are small
   enough to be **shipped with the repo** under
   `python-ml-service/models/{ulb,sparkov}/`

A standalone `.exe` cannot encapsulate this — the system depends on Docker
and on a Python 3.10+ environment with PyTorch/scikit-learn for any retraining
work. The only thing the repo intentionally **excludes** (`.gitignore`) is:

- `training/data/` — raw Kaggle datasets (~150 MB, redistribution prohibited
  by Kaggle ToS — only needed if you want to retrain)
- `training/models/` — source-side training artefacts (the deployed copies
  under `python-ml-service/models/{ulb,sparkov}/` are what the service uses)

So in the basic case, examiners just (a) clone the repo and (b) launch the
system with Docker Compose — no dataset download or training required. Full
steps below; an optional retraining path is included for completeness.

## Running from GitHub (examiner instructions)

### Prerequisites

Install on the host machine:

| Tool | Tested version | Purpose |
|------|----------------|---------|
| Git | any recent | Clone the repository |
| Docker Desktop (incl. Compose v2) | 24.x+ | Run the two services |
| `curl` | any | Hit the API for testing (optional) |

That's all that's needed for the basic run. Python 3.10+ is only required if
you want to **retrain** the models (see the optional section below).

### Step 1. Clone the repository

```bash
git clone https://github.com/Shivesh-3003/fraud-detection-system.git
cd fraud-detection-system
```

The trained model artefacts for both datasets (ULB and Sparkov) are committed
under `python-ml-service/models/{ulb,sparkov}/`, so no further setup is
needed before launching.

### Step 2. Start the system with Docker Compose

```bash
# Default — ULB dataset
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d

# Switch datasets by setting DATASET before launch
# DATASET=sparkov docker-compose up --build
```

The first build downloads base images and installs Python/Go dependencies
(~3–5 minutes). Subsequent runs are cached and start in seconds.

### Step 3. Verify both services are up

```bash
curl http://localhost:8080/health   # Go API
curl http://localhost:8000/health   # Python ML service
```

Both should return `{"status":"healthy",...}`. You're ready to send
predictions — see [API Usage](#api-usage) below.

---

### Optional — Retrain the models from scratch

The shipped models are sufficient for evaluation. If you want to retrain
(for example to verify the pipeline reproduces, or to extend it), follow
these extra steps before Step 2 above.

**A. Install Python dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r training/requirements.txt
```

**B. Download the dataset(s)**

ULB credit-card fraud (~150 MB) — required:
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud> → place
`creditcard.csv` at `training/data/creditcard.csv`.

Sparkov synthetic (optional — only for the SHAP demo with named features):
<https://www.kaggle.com/datasets/kartik2112/fraud-detection> → extract
`fraudTrain.csv` and `fraudTest.csv` into `training/data/sparkov/`.

**C. Run the training pipeline**

```bash
cd training
bash train_all.sh ulb       # ~3–8 min on CPU
bash train_all.sh sparkov   # optional
cd ..
```

`train_all.sh` runs preprocessing → autoencoder → reconstruction-error
feature generation → MLP classifier, writing artefacts to
`training/models/{ulb,sparkov}/`.

**D. Replace the shipped models with your retrained ones**

```bash
cp training/models/ulb/*     python-ml-service/models/ulb/
cp training/models/sparkov/* python-ml-service/models/sparkov/
```

Then continue with Step 2 above to launch the stack.

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
fraud-detection-system/
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
    └── models/                 # Trained model artefacts (per-dataset subdirs)
        ├── README.md
        ├── ulb/                 # default — European credit card
        │   ├── autoencoder_model.pth
        │   ├── mlp_classifier.pth
        │   ├── scaler.pkl
        │   ├── feature_config.json
        │   └── optimal_threshold.json
        └── sparkov/             # synthetic dataset (named features, SHAP demo)
            ├── autoencoder_model.pth
            ├── mlp_classifier.pth
            ├── scaler.pkl
            ├── onehot_encoder.pkl
            ├── feature_config.json
            └── optimal_threshold.json
```

## Training the Models

The end-to-end pipeline is wrapped in `training/train_all.sh` (covered in
[Optional — Retrain the models from scratch](#optional--retrain-the-models-from-scratch)
above). If you'd rather run the four steps manually:

```bash
cd training

# Download creditcard.csv from Kaggle first → training/data/creditcard.csv
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

python credit_card_eda_preprocessing.py --dataset ulb   # Data preparation
python train_autoencoder.py --dataset ulb               # Train autoencoder
python generate_ae_features.py --dataset ulb            # Generate AE features
python train_classifier.py --dataset ulb                # Train classifier

# Stage trained artefacts for the inference service
mkdir -p ../python-ml-service/models/ulb
cp models/ulb/* ../python-ml-service/models/ulb/
```

See `training/README.md` for the detailed pipeline diagram and feature-engineering
notes.

## Configuration

### Go API Environment Variables

| Variable             | Default                 | Description              |
| -------------------- | ----------------------- | ------------------------ |
| `PORT`               | `8080`                  | Server port              |
| `ENV`                | `development`           | Environment              |
| `ML_SERVICE_URL`     | `http://localhost:8000` | Python ML service URL    |
| `ML_SERVICE_TIMEOUT` | `5s`                    | Request timeout          |
| `SLACK_WEBHOOK_URL`  | -                       | Slack alerts webhook     |

The classification threshold is **not** an env var — it's read by the Python
ML service from `python-ml-service/models/<dataset>/optimal_threshold.json`
at startup, and the Go API consumes the `is_fraud` boolean from the response.

### Python ML Service Environment Variables

| Variable     | Default       | Description         |
| ------------ | ------------- | ------------------- |
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

Check that the per-dataset model files exist with the correct names. Docker
Compose mounts `python-ml-service/models/${DATASET:-ulb}` into the container:

```bash
ls -la python-ml-service/models/ulb/
# Should show:
# - autoencoder_model.pth
# - mlp_classifier.pth
# - scaler.pkl
# - optimal_threshold.json
# - feature_config.json
```

If any are missing, re-run the training pipeline (see
[Optional — Retrain the models from scratch](#optional--retrain-the-models-from-scratch)).

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

| Stage                       | Mean         | p95      | p99      |
| --------------------------- | ------------ | -------- | -------- |
| Preprocess                  | 0.002 ms     | 0.002 ms | 0.010 ms |
| Scale                       | 0.023 ms     | 0.028 ms | 0.035 ms |
| Autoencoder forward         | 0.044 ms     | 0.051 ms | 0.059 ms |
| Classifier forward          | 0.021 ms     | 0.025 ms | 0.029 ms |
| **ML inference (AE + MLP)** | **0.065 ms** | 0.076 ms | 0.086 ms |
| **End-to-end (single txn)** | **0.090 ms** | 0.105 ms | 0.118 ms |

SHAP explanation adds ~hundreds of ms (depends on `nsamples`); the Go API
only invokes it when the fast path flags fraud.

### Model performance vs. baselines

Isolated baselines on the same 31 raw engineered features (V1–V28,
Amount_Log, Hour_sin, Hour_cos); AE+MLP uses its full 32-dim pipeline.
Threshold for each model picked by max-F1 on the test PR curve.

| Model               | Precision | Recall     | F1     | PR-AUC     | ROC-AUC    |
| ------------------- | --------- | ---------- | ------ | ---------- | ---------- |
| Logistic Regression | 0.8387    | 0.7959     | 0.8168 | 0.7460     | 0.9723     |
| Random Forest       | 0.9318    | 0.8367     | 0.8817 | 0.8604     | 0.9523     |
| XGBoost             | 0.9756    | 0.8163     | 0.8889 | **0.8777** | 0.9727     |
| **AE+MLP (ours)**   | 0.8438    | **0.8265** | 0.8351 | 0.7521     | **0.9783** |

Reproduce with `python3 training/analysis/04_benchmarks.py`.
