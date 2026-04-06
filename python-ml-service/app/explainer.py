"""
SHAP Explainer for Fraud Detection MLP Classifier

This module provides SHAP (SHapley Additive exPlanations) integration
for explaining individual fraud predictions.

We explain the MLP CLASSIFIER's decisions, not the autoencoder.
The classifier takes 32 features:
    - 31 scaled transaction features (V1-V28, Amount_Log, Hour_sin, Hour_cos)
    - 1 reconstruction error from the autoencoder

SHAP tells us which of these 32 features contributed most to the
fraud probability, enabling interpretable fraud alerts.
"""

import logging
import numpy as np
import torch
from typing import Dict, Optional, Tuple
import shap

from .ml_models import FraudClassifier
from .inference import FEATURE_NAMES_32, CLASSIFIER_INPUT_DIM  # kept for backward-compat imports

logger = logging.getLogger(__name__)


class FraudExplainer:
    """
    SHAP-based explainer for fraud predictions.
    
    Uses KernelExplainer which is model-agnostic and works reliably
    with any model type. Slower than DeepExplainer but more robust.
    """
    
    def __init__(
        self,
        classifier: FraudClassifier,
        feature_names: Optional[list] = None,
        background_data: Optional[np.ndarray] = None,
        n_background_samples: int = 100
    ):
        """
        Initialize the explainer.

        Args:
            classifier: Trained FraudClassifier model
            feature_names: Ordered list of feature names for SHAP output keys.
                           Should include 'Reconstruction_Error' as last element.
                           Defaults to ULB FEATURE_NAMES_32 for backward compatibility.
            background_data: Background samples for SHAP (shape: n_samples, clf_input_dim)
                             If None, uses zeros as background
            n_background_samples: Number of background samples to use
        """
        self.classifier = classifier
        self.classifier.eval()
        self.feature_names = feature_names if feature_names is not None else FEATURE_NAMES_32
        
        # Create background data if not provided
        if background_data is None:
            logger.warning(
                "No background data provided. Using zeros. "
                "SHAP values may be less accurate."
            )
            # Use zeros as a simple baseline; size derived from feature_names
            clf_input_dim = len(self.feature_names)
            self.background = np.zeros((1, clf_input_dim), dtype=np.float32)
        else:
            # Sample if we have too many
            if len(background_data) > n_background_samples:
                indices = np.random.choice(
                    len(background_data),
                    n_background_samples,
                    replace=False
                )
                self.background = background_data[indices].astype(np.float32)
            else:
                self.background = background_data.astype(np.float32)
        
        logger.info(f"Background data shape: {self.background.shape}")
        
        # Create the SHAP explainer
        # We wrap the classifier's predict_proba for SHAP
        self.explainer = shap.KernelExplainer(
            model=self._predict_fn,
            data=self.background,
            link="identity"  # We're explaining probabilities directly
        )
        
        logger.info("SHAP KernelExplainer initialized")
    
    def _predict_fn(self, x: np.ndarray) -> np.ndarray:
        """
        Prediction function for SHAP.
        
        Args:
            x: Input array of shape (n_samples, 32)
            
        Returns:
            Fraud probabilities of shape (n_samples,)
        """
        self.classifier.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            probs = self.classifier.predict_proba(tensor).numpy()
        return probs
    
    def explain(
        self,
        classifier_input: np.ndarray,
        nsamples: int = 100
    ) -> Tuple[Dict[str, float], float]:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            classifier_input: The 32-feature array fed to the classifier
            nsamples: Number of samples for SHAP approximation (more = slower but more accurate)
            
        Returns:
            Tuple of:
                - shap_values: Dict mapping feature name to SHAP value
                - base_value: The expected value (average prediction on background)
        """
        # Ensure correct shape
        if classifier_input.ndim == 1:
            classifier_input = classifier_input.reshape(1, -1)
        
        # Compute SHAP values
        # KernelExplainer returns shape (n_samples, n_features)
        shap_values = self.explainer.shap_values(
            classifier_input,
            nsamples=nsamples,
            silent=True
        )
        
        # Get values for the single sample
        if isinstance(shap_values, list):
            # For binary classification, might return list
            shap_values = shap_values[0] if len(shap_values) > 1 else shap_values
        
        sample_shap = shap_values[0] if shap_values.ndim > 1 else shap_values
        
        # Create feature name -> SHAP value mapping
        shap_dict = {
            name: float(value)
            for name, value in zip(self.feature_names, sample_shap)
        }
        
        # Base value is the expected prediction
        base_value = float(self.explainer.expected_value)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])
        
        return shap_dict, base_value
    

# Global explainer instance
_explainer: Optional[FraudExplainer] = None


def get_explainer(classifier: FraudClassifier) -> FraudExplainer:
    """
    Get or create the global explainer instance.
    
    Args:
        classifier: The trained classifier model
        
    Returns:
        Initialized FraudExplainer
    """
    global _explainer
    
    if _explainer is None:
        logger.info("Initializing SHAP explainer...")
        _explainer = FraudExplainer(classifier=classifier)
        logger.info("SHAP explainer ready")
    
    return _explainer


def reset_explainer() -> None:
    """Reset the global explainer (useful for testing)."""
    global _explainer
    _explainer = None
