# Model Files Directory

Place your trained model files here before running the service.

## Required Files

| File | Description | Created By |
|------|-------------|------------|
| `autoencoder_model.pth` | Trained autoencoder weights | `train_autoencoder.py` |
| `mlp_classifier.pth` | Trained MLP classifier weights | `train_classifier.py` |
| `scaler.pkl` | StandardScaler fitted on normal data | `credit_card_eda_preprocessing.py` |

## Verification

After copying your files, verify they exist:

```bash
ls -la python-ml-service/models/
# Should show:
# -rw-r--r--  autoencoder_model.pth
# -rw-r--r--  mlp_classifier.pth
# -rw-r--r--  scaler.pkl
```

## Model Specifications

### Autoencoder (`autoencoder_model.pth`)
- **Input**: 31 features (V1-V28, Amount_Log, Hour_sin, Hour_cos)
- **Architecture**: 31 → 20 → 14 → 8 → 14 → 20 → 31
- **Activation**: Tanh
- **Purpose**: Generates reconstruction error as anomaly signal

### MLP Classifier (`mlp_classifier.pth`)
- **Input**: 32 features (31 original + Reconstruction_Error)
- **Architecture**: 32 → 16 → 1
- **Activation**: ReLU + Dropout(0.3)
- **Output**: Logits (apply sigmoid for probability)

### Scaler (`scaler.pkl`)
- **Type**: sklearn.preprocessing.StandardScaler
- **Fitted on**: Normal transactions only (Class 0)
- **Features**: 31 (V1-V28, Amount_Log, Hour_sin, Hour_cos)

## Troubleshooting

If you see "Model not found" errors:
1. Check file names match exactly (case-sensitive)
2. Ensure files are in this directory, not a subdirectory
3. Verify file permissions (readable)

If you see "state_dict" errors:
1. Models must be saved with `torch.save(model.state_dict(), path)`
2. Architecture must match exactly what's defined in `app/ml_models.py`
