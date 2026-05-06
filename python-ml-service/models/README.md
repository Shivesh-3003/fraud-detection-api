# Model Files Directory

Trained model artefacts live in **per-dataset subdirectories**, one per dataset
the system supports. The Python service mounts a single subdirectory at runtime
based on the `DATASET` env var passed to Docker Compose.

## Layout

```
models/
├── ulb/        # European credit card (anonymous V1–V28)  — default
│   ├── scaler.pkl
│   ├── autoencoder_model.pth
│   ├── mlp_classifier.pth
│   ├── feature_config.json
│   └── optimal_threshold.json
└── sparkov/    # Sparkov synthetic (named features, SHAP demo)
    ├── scaler.pkl
    ├── onehot_encoder.pkl
    ├── autoencoder_model.pth
    ├── mlp_classifier.pth
    └── feature_config.json
```

These files are produced by the training pipeline and are **not committed to
git** (see `.gitignore`). Generate them with:

```bash
cd ../../training
bash train_all.sh ulb       # → ../python-ml-service/models/ulb/ via copy step
bash train_all.sh sparkov   # → ../python-ml-service/models/sparkov/
```

Then stage them for the inference service:

```bash
mkdir -p ulb sparkov
cp ../../training/models/ulb/*     ulb/
cp ../../training/models/sparkov/* sparkov/
```

## Selecting a dataset at runtime

`docker-compose.yml` mounts `./python-ml-service/models/${DATASET:-ulb}` into
the container at `/app/models`. Switch datasets with the `DATASET` env var:

```bash
docker-compose up                       # ulb (default)
DATASET=sparkov docker-compose up       # sparkov
```

For local (non-Docker) runs, point `MODEL_PATH` directly at the chosen subdir:

```bash
export MODEL_PATH=$(pwd)/ulb
uvicorn app.main:app --port 8000
```

## File reference

| File | Purpose | Created by |
|------|---------|------------|
| `scaler.pkl` | StandardScaler fitted on normal data | `credit_card_eda_preprocessing.py` |
| `onehot_encoder.pkl` | OneHotEncoder for categorical features (sparkov only) | `credit_card_eda_preprocessing.py` |
| `autoencoder_model.pth` | Autoencoder weights (state dict) | `train_autoencoder.py` |
| `mlp_classifier.pth` | MLP classifier weights (state dict) | `train_classifier.py` |
| `feature_config.json` | Feature names, dimensions, and dataset metadata | `credit_card_eda_preprocessing.py` |
| `optimal_threshold.json` | F1-optimal classification threshold (ulb only) | `train_classifier.py` |

## Verification

```bash
ls -la ulb/
# autoencoder_model.pth  feature_config.json  mlp_classifier.pth
# optimal_threshold.json scaler.pkl
```

## Troubleshooting

**"Model not found" / "feature_config.json missing"**
1. Confirm files are inside the per-dataset subdirectory (not at the top level
   of `models/`)
2. Run the training pipeline if any are absent
3. Check that `DATASET` env var matches an existing subdirectory

**"state_dict" / shape errors**
- Models must be saved with `torch.save(model.state_dict(), path)`
- Architecture in `app/ml_models.py` must match the dimensions in
  `feature_config.json` for that dataset
