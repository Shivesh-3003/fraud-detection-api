"""Shared loaders for the analysis scripts.

Loads the trained ULB or Sparkov artifacts from training/data/ and
training/models/ so each script can run independently without re-training.

ULB artifacts live at the legacy flat paths (data/X_*.csv,
models/*.pth); Sparkov lives under data/sparkov/ and models/sparkov/.
"""
from __future__ import annotations

import json
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


def dataset_paths(dataset: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Return (data_dir, models_dir) for the requested dataset."""
    if dataset == "ulb":
        return DATA_DIR, MODELS_DIR
    if dataset == "sparkov":
        return DATA_DIR / "sparkov", MODELS_DIR / "sparkov"
    raise ValueError(f"unknown dataset: {dataset!r}")


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Architectures duplicated from the training scripts so state_dicts load
# without importing the training modules (which run argparse on import).
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


def load_test_set(dataset: str = "ulb") -> tuple[np.ndarray, np.ndarray]:
    data_dir, _ = dataset_paths(dataset)
    X = pd.read_csv(data_dir / "X_test_final.csv").values.astype(np.float32)
    y = pd.read_csv(data_dir / "y_test.csv").values.astype(np.float32).flatten()
    return X, y


def load_train_set(dataset: str = "ulb") -> tuple[np.ndarray, np.ndarray]:
    data_dir, _ = dataset_paths(dataset)
    X = pd.read_csv(data_dir / "X_train_final.csv").values.astype(np.float32)
    y = pd.read_csv(data_dir / "y_train_MLP.csv").values.astype(np.float32).flatten()
    return X, y


def load_mlp(input_dim: int, dataset: str = "ulb") -> nn.Module:
    _, models_dir = dataset_paths(dataset)
    dev = device()
    model = FraudClassifier(input_dim).to(dev)
    model.load_state_dict(torch.load(models_dir / "mlp_classifier.pth", map_location=dev))
    model.eval()
    return model


def predict_proba_mlp(X: np.ndarray, dataset: str = "ulb") -> np.ndarray:
    dev = device()
    model = load_mlp(X.shape[1], dataset=dataset)
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=dev)
        logits = model(t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs


def load_optimal_threshold(dataset: str = "ulb") -> float:
    _, models_dir = dataset_paths(dataset)
    with open(models_dir / "optimal_threshold.json") as f:
        return float(json.load(f)["optimal_threshold"])
