"""
Inference Pipeline for Fraud Detection

This module handles:
1. Loading trained model artifacts (scaler, autoencoder, classifier)
2. Preprocessing raw transaction data
3. Running the two-stage prediction pipeline
4. Returning fraud probability and reconstruction error

FEATURE ORDER (Critical - must match training):
    Scaled features (31): V1, V2, ..., V28, Amount_Log, Hour_sin, Hour_cos
    Classifier input (32): V1, V2, ..., V28, Amount_Log, Hour_sin, Hour_cos, Reconstruction_Error
"""

import os
import json
import logging
import time
import numpy as np
import torch
import joblib
from typing import Tuple, Optional
from pathlib import Path

from .ml_models import Autoencoder, FraudClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
AUTOENCODER_INPUT_DIM = 31  # V1-V28 + Amount_Log + Hour_sin + Hour_cos
CLASSIFIER_INPUT_DIM = 32   # Above + Reconstruction_Error

# Feature names in order (for SHAP explanations)
FEATURE_NAMES_31 = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount_Log', 'Hour_sin', 'Hour_cos'
]

FEATURE_NAMES_32 = FEATURE_NAMES_31 + ['Reconstruction_Error']


class FraudDetectionPipeline:
    """
    Complete fraud detection pipeline.
    
    Handles the full flow from raw transaction data to fraud prediction:
    1. Preprocess: Transform raw Amount/Time to engineered features
    2. Scale: Apply StandardScaler fitted on normal training data
    3. Autoencoder: Get reconstruction error (anomaly signal)
    4. Classifier: Get final fraud probability
    """
    
    def __init__(self, models_dir: str = "/app/models"):
        """
        Initialize pipeline by loading all model artifacts.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device("cpu")
        
        self.scaler = None
        self.autoencoder = None
        self.classifier = None
        self.threshold = 0.5
        self._models_loaded = False
        
        logger.info(f"FraudDetectionPipeline initialized (models_dir: {models_dir})")
    
    def load_models(self) -> None:
        """
        Load all model artifacts from disk.
        
        Expected files in models_dir:
            - scaler.pkl: Fitted StandardScaler
            - autoencoder_model.pth: Trained Autoencoder weights
            - mlp_classifier.pth: Trained FraudClassifier weights
            
        Raises:
            FileNotFoundError: If any required file is missing
            RuntimeError: If model loading fails
        """
        
        scaler_path = self.models_dir / "scaler.pkl"
        autoencoder_path = self.models_dir / "autoencoder_model.pth"
        classifier_path = self.models_dir / "mlp_classifier.pth"
        
        missing_files = []
        for path in [scaler_path, autoencoder_path, classifier_path]:
            if not path.exists():
                missing_files.append(str(path))
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required model files: {missing_files}"
            )
        
        logger.info(f"  Loading scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        logger.info(f"  ✓ Scaler loaded (expects {self.scaler.n_features_in_} features)")
        
        if self.scaler.n_features_in_ != AUTOENCODER_INPUT_DIM:
            raise RuntimeError(
                f"Scaler expects {self.scaler.n_features_in_} features, "
                f"but pipeline expects {AUTOENCODER_INPUT_DIM}"
            )
        
        logger.info(f"  Loading autoencoder: {autoencoder_path}")
        self.autoencoder = Autoencoder(input_dim=AUTOENCODER_INPUT_DIM)
        self.autoencoder.load_state_dict(
            torch.load(autoencoder_path, map_location=self.device)
        )
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        logger.info("  ✓ Autoencoder loaded")
        
        logger.info(f"  Loading classifier: {classifier_path}")
        self.classifier = FraudClassifier(input_dim=CLASSIFIER_INPUT_DIM)
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=self.device)
        )
        self.classifier.to(self.device)
        self.classifier.eval()
        logger.info("  ✓ Classifier loaded")
        
        threshold_path = self.models_dir / "optimal_threshold.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                self.threshold = float(json.load(f)["optimal_threshold"])
            logger.info(f"  ✓ Optimal threshold loaded: {self.threshold:.4f}")
        else:
            self.threshold = 0.5
            logger.warning("  ⚠ optimal_threshold.json not found, defaulting to 0.5")

        self._models_loaded = True
        logger.info("All models loaded successfully!")
    
    def preprocess(
        self,
        v_features: list[float],
        amount: float,
        time_val: float
    ) -> np.ndarray:
        """
        Transform raw transaction data to model-ready features.
        
        Applies the same transformations as credit_card_eda_preprocessing.py:
        1. Amount → Amount_Log = log1p(Amount)
        2. Time → Hour = floor(Time / 3600) % 24
        3. Hour → Hour_sin = sin(2π * Hour / 24)
        4. Hour → Hour_cos = cos(2π * Hour / 24)
        
        Args:
            v_features: List of 28 V-features [V1, V2, ..., V28]
            amount: Raw transaction amount
            time: Raw time in seconds since first transaction
            
        Returns:
            numpy array of shape (31,) with features in correct order
            
        Raises:
            ValueError: If v_features doesn't have exactly 28 elements
        """
        if len(v_features) != 28:
            raise ValueError(
                f"Expected 28 V-features, got {len(v_features)}"
            )
        
        amount_log = np.log1p(amount)
        
        hour = np.floor(time_val / 3600) % 24
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        features = np.array(
            v_features + [amount_log, hour_sin, hour_cos],
            dtype=np.float32
        )
        
        return features
    
    def scale(self, features: np.ndarray) -> np.ndarray:
        """
        Apply StandardScaler to features.
        
        Args:
            features: Array of shape (31,) or (batch_size, 31)
            
        Returns:
            Scaled features with same shape
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
            return self.scaler.transform(features).flatten()
        else:
            return self.scaler.transform(features)
    
    def predict(
        self,
        v_features: list[float],
        amount: float,
        time_val: float
    ) -> Tuple[float, float, np.ndarray, float]:
        """
        Run full prediction pipeline on a single transaction.
        Args:
            v_features: List of 28 V-features [V1, V2, ..., V28]
            amount: Raw transaction amount
            time: Raw time in seconds
        
        Returns:
            Tuple of:
                - fraud_probability (float): 0.0 to 1.0
                - reconstruction_error (float): MSE from autoencoder
                - classifier_input (np.ndarray): The 32 features fed to classifier
                - inference_time_ms (float): Isolated PyTorch inference time
        Raises:
            RuntimeError: If models not loaded
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Step 1: Preprocess raw features (not timed — this is data prep, not inference)
        features_31 = self.preprocess(v_features, amount, time_val)
        
        # Step 2: Scale features (not timed — this is sklearn, not neural network)
        scaled_features = self.scale(features_31)
        
        # ISOLATED ML INFERENCE TIMING
        # Only measures the actual PyTorch forward passes:
        #   1. Autoencoder reconstruction error
        #   2. MLP classifier probability
        # Excludes: HTTP, JSON, preprocessing, scaling, numpy ops
        inference_start = time.perf_counter()
        
        # Step 3: Autoencoder forward pass → reconstruction error
        with torch.no_grad():
            tensor_31 = torch.tensor(
                scaled_features, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            reconstruction_error = self.autoencoder.get_reconstruction_error(
                tensor_31
            ).item()
        
        # Step 4: Concatenate for classifier input
        classifier_input = np.append(scaled_features, reconstruction_error)
        
        # Step 5: Classifier forward pass → fraud probability
        with torch.no_grad():
            tensor_32 = torch.tensor(
                classifier_input, dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            fraud_probability = self.classifier.predict_proba(tensor_32).item()
        
        inference_end = time.perf_counter()
        inference_time_ms = (inference_end - inference_start) * 1000
        
        return fraud_probability, reconstruction_error, classifier_input, inference_time_ms
    
    def predict_batch(
        self,
        transactions: list[dict]
    ) -> list[Tuple[float, float, float]]:
        """
        Run prediction pipeline on multiple transactions.
        
        Args:
            transactions: List of dicts with keys 'v_features', 'amount', 'time'
            
        Returns:
            List of (fraud_probability, reconstruction_error) tuples
        """
        results = []
        for txn in transactions:
            prob, error, _, inference_ms = self.predict(
                v_features=txn['v_features'],
                amount=txn['amount'],
                time=txn['time']
            )
            results.append((prob, error, inference_ms))
        return results
    
    def get_classifier_input_for_shap(
        self,
        v_features: list[float],
        amount: float,
        time_val: float
    ) -> np.ndarray:
        """Get the 32-feature vector for SHAP explanations."""
        _, _, classifier_input, _ = self.predict(v_features, amount, time_val)
        return classifier_input
    
    def is_ready(self) -> bool:
        """Check if pipeline is ready for predictions."""
        return self._models_loaded
    
    def get_feature_names(self, include_reconstruction_error: bool = True) -> list[str]:
        """
        Get feature names in order.
        
        Args:
            include_reconstruction_error: If True, return 32 names (classifier input)
                                          If False, return 31 names (autoencoder input)
        """
        if include_reconstruction_error:
            return FEATURE_NAMES_32.copy()
        return FEATURE_NAMES_31.copy()


# Global pipeline instance (singleton pattern)
_pipeline: Optional[FraudDetectionPipeline] = None


def get_pipeline() -> FraudDetectionPipeline:
    """
    Get the global pipeline instance.
    
    Creates and loads models on first call.
    """
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