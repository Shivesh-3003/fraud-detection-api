"""
Inference Pipeline for Fraud Detection

This module handles:
1. Loading trained model artifacts (scaler, autoencoder, classifier, feature_config)
2. Preprocessing raw transaction data (dataset-aware: ULB or Sparkov)
3. Running the two-stage prediction pipeline
4. Returning fraud probability and reconstruction error

Dataset-aware design:
    Each model directory contains a feature_config.json that records the
    feature names, input dimensions, and preprocessing parameters.
    The pipeline reads this file on startup and self-configures — no
    hardcoded feature counts or names survive in inference code.

ULB feature order (31):
    V1, V2, ..., V28, Amount_Log, Hour_sin, Hour_cos
    + Reconstruction_Error = 32 classifier inputs

Sparkov feature order (20):
    amt_log, hour_sin, hour_cos, age, gender_M, city_pop_log, distance,
    category_<name>×13
    + Reconstruction_Error = 21 classifier inputs
"""

import os
import json
import logging
import time
import math
import numpy as np
import torch
import joblib
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional
from pathlib import Path

from .ml_models import Autoencoder, FraudClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ULB fallback constants — used when feature_config.json is absent (backward-compat)
_ULB_FEATURE_NAMES_31 = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount_Log', 'Hour_sin', 'Hour_cos'
]

# Legacy constants kept for backward-compatibility with any external imports
AUTOENCODER_INPUT_DIM = 31
CLASSIFIER_INPUT_DIM = 32
FEATURE_NAMES_31 = _ULB_FEATURE_NAMES_31
FEATURE_NAMES_32 = _ULB_FEATURE_NAMES_31 + ['Reconstruction_Error']


@dataclass
class FeatureConfig:
    """Metadata loaded from feature_config.json — single source of truth for dims/names."""
    dataset_type: str       # "ulb" | "sparkov"
    feature_names: list     # ordered feature names, length = ae_input_dim
    ae_input_dim: int
    clf_input_dim: int      # ae_input_dim + 1 (Reconstruction_Error)
    has_ohe: bool           # True for Sparkov (category one-hot encoder)
    label_column: str       # "Class" for ULB, "is_fraud" for Sparkov


def load_feature_config(models_dir: Path) -> FeatureConfig:
    """
    Load feature configuration from models_dir/feature_config.json.
    Falls back to ULB hardcoded defaults for backward compatibility when
    the file doesn't exist (e.g. existing flat model directories).
    """
    config_path = models_dir / "feature_config.json"
    if not config_path.exists():
        logger.warning(
            "feature_config.json not found in %s — assuming ULB defaults. "
            "Re-run preprocessing with --dataset ulb to generate this file.",
            models_dir
        )
        return FeatureConfig(
            dataset_type="ulb",
            feature_names=_ULB_FEATURE_NAMES_31,
            ae_input_dim=31,
            clf_input_dim=32,
            has_ohe=False,
            label_column="Class"
        )

    with open(config_path) as f:
        raw = json.load(f)

    return FeatureConfig(
        dataset_type=raw["dataset_type"],
        feature_names=raw["feature_names"],
        ae_input_dim=raw["ae_input_dim"],
        clf_input_dim=raw["clf_input_dim"],
        has_ohe=raw.get("has_ohe", False),
        label_column=raw.get("label_column", "Class")
    )


class FraudDetectionPipeline:
    """
    Complete fraud detection pipeline.

    Handles the full flow from raw transaction data to fraud prediction:
    1. Load feature_config.json → configure dimensions and preprocessing
    2. Preprocess: Transform raw fields to engineered features (ULB or Sparkov)
    3. Scale: Apply StandardScaler fitted on normal training data
    4. Autoencoder: Get reconstruction error (anomaly signal)
    5. Classifier: Get final fraud probability
    """

    def __init__(self, models_dir: str = "/app/models"):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cpu")

        self.feature_config: Optional[FeatureConfig] = None
        self.scaler = None
        self.ohe = None          # OneHotEncoder — only used for Sparkov
        self.autoencoder = None
        self.classifier = None
        self.threshold = 0.5
        self._models_loaded = False

        logger.info(f"FraudDetectionPipeline initialized (models_dir: {models_dir})")

    def load_models(self) -> None:
        """
        Load all model artifacts from disk.

        Expected files in models_dir:
            - feature_config.json   (new — written by preprocessing script)
            - scaler.pkl            Fitted StandardScaler
            - onehot_encoder.pkl    Fitted OneHotEncoder (Sparkov only)
            - autoencoder_model.pth Trained Autoencoder weights
            - mlp_classifier.pth    Trained FraudClassifier weights
            - optimal_threshold.json

        Raises:
            FileNotFoundError: If any required file is missing
            RuntimeError: If model loading fails
        """
        # 1. Load feature config (must come first — determines everything else)
        self.feature_config = load_feature_config(self.models_dir)
        ae_dim  = self.feature_config.ae_input_dim
        clf_dim = self.feature_config.clf_input_dim
        logger.info(
            f"Dataset: {self.feature_config.dataset_type} | "
            f"AE input dim: {ae_dim} | CLF input dim: {clf_dim}"
        )

        # 2. Load scaler
        scaler_path = self.models_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        if self.scaler.n_features_in_ != ae_dim:
            raise RuntimeError(
                f"Scaler expects {self.scaler.n_features_in_} features, "
                f"feature_config says {ae_dim}"
            )
        logger.info(f"  ✓ Scaler loaded ({self.scaler.n_features_in_} features)")

        # 3. Load OneHotEncoder (Sparkov only)
        if self.feature_config.has_ohe:
            ohe_path = self.models_dir / "onehot_encoder.pkl"
            if not ohe_path.exists():
                raise FileNotFoundError(
                    f"Missing: {ohe_path} (required for Sparkov dataset)"
                )
            self.ohe = joblib.load(ohe_path)
            logger.info(f"  ✓ OneHotEncoder loaded ({len(self.ohe.categories_[0])} categories)")

        # 4. Load Autoencoder
        ae_path = self.models_dir / "autoencoder_model.pth"
        if not ae_path.exists():
            raise FileNotFoundError(f"Missing: {ae_path}")
        self.autoencoder = Autoencoder(input_dim=ae_dim)
        self.autoencoder.load_state_dict(
            torch.load(ae_path, map_location=self.device)
        )
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        logger.info("  ✓ Autoencoder loaded")

        # 5. Load Classifier
        clf_path = self.models_dir / "mlp_classifier.pth"
        if not clf_path.exists():
            raise FileNotFoundError(f"Missing: {clf_path}")
        self.classifier = FraudClassifier(input_dim=clf_dim)
        self.classifier.load_state_dict(
            torch.load(clf_path, map_location=self.device)
        )
        self.classifier.to(self.device)
        self.classifier.eval()
        logger.info("  ✓ Classifier loaded")

        # 6. Load threshold
        threshold_path = self.models_dir / "optimal_threshold.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                self.threshold = float(json.load(f)["optimal_threshold"])
            logger.info(f"  ✓ Threshold: {self.threshold:.4f}")
        else:
            self.threshold = 0.5
            logger.warning("  ⚠ optimal_threshold.json not found, defaulting to 0.5")

        self._models_loaded = True
        logger.info("All models loaded successfully!")

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    def _preprocess_ulb(
        self,
        v_features: list,
        amount: float,
        time_val: float
    ) -> np.ndarray:
        """
        ULB preprocessing: same transformations as credit_card_eda_preprocessing.py.
        Applies Amount_Log and cyclic Hour encoding, returns 31-float array.
        """
        if len(v_features) != 28:
            raise ValueError(f"Expected 28 V-features, got {len(v_features)}")

        amount_log = np.log1p(amount)
        hour       = np.floor(time_val / 3600) % 24
        hour_sin   = np.sin(2 * np.pi * hour / 24)
        hour_cos   = np.cos(2 * np.pi * hour / 24)

        return np.array(v_features + [amount_log, hour_sin, hour_cos], dtype=np.float32)

    def _preprocess_sparkov(self, raw: dict) -> np.ndarray:
        """
        Sparkov preprocessing: same transformations as credit_card_eda_preprocessing.py.

        Expected raw dict keys:
            amt (float), trans_datetime (ISO str), dob (ISO date str),
            gender ('M'/'F'), city_pop (int), lat (float), long (float),
            merch_lat (float), merch_long (float), category (str)

        Returns a float32 array matching the feature order in feature_config.feature_names.
        """
        trans_dt = datetime.fromisoformat(raw["trans_datetime"])
        dob      = datetime.fromisoformat(raw["dob"])

        amt_log      = math.log1p(raw["amt"])
        hour         = trans_dt.hour
        hour_sin     = math.sin(2 * math.pi * hour / 24)
        hour_cos     = math.cos(2 * math.pi * hour / 24)
        age          = (trans_dt - dob).days / 365.25
        gender_M     = 1.0 if raw["gender"] == "M" else 0.0
        city_pop_log = math.log1p(raw["city_pop"])
        distance     = math.sqrt(
            (raw["lat"]  - raw["merch_lat"])**2 +
            (raw["long"] - raw["merch_long"])**2
        )

        numeric = [amt_log, hour_sin, hour_cos, age, gender_M, city_pop_log, distance]

        # One-hot encode category — handle_unknown='ignore' returns zeros for unseen categories
        cat_array = self.ohe.transform([[raw["category"]]])  # shape (1, n_categories-1)
        cat_flat  = cat_array.flatten().tolist()

        return np.array(numeric + cat_flat, dtype=np.float32)

    def preprocess(self, raw_input: dict) -> np.ndarray:
        """
        Dispatch to dataset-specific preprocessing.

        For ULB:     raw_input must contain 'v_features', 'amount', 'time'
        For Sparkov: raw_input must contain 'amt', 'trans_datetime', 'dob',
                     'gender', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'category'
        """
        if self.feature_config.dataset_type == "ulb":
            return self._preprocess_ulb(
                v_features=raw_input["v_features"],
                amount=raw_input["amount"],
                time_val=raw_input["time"]
            )
        else:
            return self._preprocess_sparkov(raw_input)

    def scale(self, features: np.ndarray) -> np.ndarray:
        """Apply StandardScaler to features (1D or 2D input)."""
        if features.ndim == 1:
            return self.scaler.transform(features.reshape(1, -1)).flatten()
        return self.scaler.transform(features)

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(self, raw_input: dict) -> Tuple[float, float, np.ndarray, float]:
        """
        Run full prediction pipeline on a single transaction.

        Args:
            raw_input: Dict with dataset-appropriate fields (see preprocess docstring)

        Returns:
            Tuple of:
                - fraud_probability (float): 0.0 to 1.0
                - reconstruction_error (float): MSE from autoencoder
                - classifier_input (np.ndarray): features fed to classifier (ae_dim+1)
                - inference_time_ms (float): Isolated PyTorch forward-pass time
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Step 1: Preprocess raw fields → feature array (not timed — data prep, not inference)
        features = self.preprocess(raw_input)

        # Step 2: Scale features (not timed — sklearn, not neural network)
        scaled_features = self.scale(features)

        # ISOLATED ML INFERENCE TIMING
        inference_start = time.perf_counter()

        # Step 3: Autoencoder forward pass → reconstruction error
        with torch.no_grad():
            tensor_ae = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            reconstruction_error = self.autoencoder.get_reconstruction_error(tensor_ae).item()

        # Step 4: Concatenate features + reconstruction error for classifier input
        classifier_input = np.append(scaled_features, reconstruction_error)

        # Step 5: Classifier forward pass → fraud probability
        with torch.no_grad():
            tensor_clf = torch.tensor(classifier_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            fraud_probability = self.classifier.predict_proba(tensor_clf).item()

        inference_end = time.perf_counter()
        inference_time_ms = (inference_end - inference_start) * 1000

        return fraud_probability, reconstruction_error, classifier_input, inference_time_ms

    def predict_batch(self, transactions: list) -> list:
        """
        Run prediction on multiple transactions.

        Args:
            transactions: List of raw_input dicts (same format as predict())

        Returns:
            List of (fraud_probability, reconstruction_error, inference_ms) tuples
        """
        results = []
        for raw_input in transactions:
            prob, error, _, inference_ms = self.predict(raw_input)
            results.append((prob, error, inference_ms))
        return results

    def is_ready(self) -> bool:
        """Check if pipeline is ready for predictions."""
        return self._models_loaded

    def get_feature_names(self, include_reconstruction_error: bool = True) -> list:
        """
        Get ordered feature names from feature_config.

        Args:
            include_reconstruction_error: If True, append 'Reconstruction_Error'

        Returns:
            List of feature name strings (length ae_dim or clf_dim)
        """
        names = self.feature_config.feature_names.copy()
        if include_reconstruction_error:
            names = names + ['Reconstruction_Error']
        return names


# Global pipeline instance (singleton)
_pipeline: Optional[FraudDetectionPipeline] = None


def get_pipeline() -> FraudDetectionPipeline:
    """Get (or create) the global pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        models_dir = os.environ.get("MODEL_PATH", "/app/models")
        _pipeline = FraudDetectionPipeline(models_dir=models_dir)
        _pipeline.load_models()
    return _pipeline


def reset_pipeline() -> None:
    """Reset the global pipeline (useful for testing)."""
    global _pipeline
    _pipeline = None
