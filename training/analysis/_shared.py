"""Shared loaders for the analysis scripts.

Reuses already-trained ULB artifacts:
  - data/X_test_final.csv  (32 features incl. Reconstruction_Error)
  - data/y_test.csv
  - models/mlp_classifier.pth
  - models/autoencoder_model.pth
  - models/scaler.pkl
"""
from __future__ import annotations

import pathlib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "analysis" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Architectures must match the originals exactly so state_dicts load.
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.Tanh(),
            nn.Linear(20, 14), nn.Tanh(),
            nn.Linear(14, 8), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 14), nn.Tanh(),
            nn.Linear(14, 20), nn.Tanh(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class FraudClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.layer2(x)


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    X = pd.read_csv(DATA_DIR / "X_test_final.csv").values.astype(np.float32)
    y = pd.read_csv(DATA_DIR / "y_test.csv").values.astype(np.float32).flatten()
    return X, y


def load_train_set() -> tuple[np.ndarray, np.ndarray]:
    X = pd.read_csv(DATA_DIR / "X_train_final.csv").values.astype(np.float32)
    y = pd.read_csv(DATA_DIR / "y_train_MLP.csv").values.astype(np.float32).flatten()
    return X, y


def load_mlp(input_dim: int) -> nn.Module:
    dev = device()
    model = FraudClassifier(input_dim).to(dev)
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_classifier.pth", map_location=dev))
    model.eval()
    return model


def predict_proba_mlp(X: np.ndarray) -> np.ndarray:
    dev = device()
    model = load_mlp(X.shape[1])
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=dev)
        logits = model(t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs
