"""
Pydantic Models for Fraud Detection API

These schemas define the request/response format for the Python ML service.
They must be compatible with what the Go API sends/expects.

Go API Contract:
    Request:
        - features: [float] (28 V-features as array)
        - amount: float
        - time: float
        
    Response:
        - fraud_probability: float
        - reconstruction_error: float
        - shap_values: {feature_name: float} (optional)
        - base_value: float (optional)
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """
    Request body for /predict endpoint.
    
    Matches the MLPredictRequest struct from Go:
        type MLPredictRequest struct {
            Features []float64 `json:"features"`
            Amount   float64   `json:"amount"`
            Time     float64   `json:"time"`
        }
    """
    features: List[float] = Field(
        ...,
        description="28 PCA features (V1-V28) as an ordered array",
        min_length=28,
        max_length=28
    )
    amount: float = Field(
        ...,
        description="Transaction amount (raw, not log-transformed)",
        ge=0
    )
    time: float = Field(
        ...,
        description="Time in seconds since first transaction in dataset",
        ge=0
    )
    
    @field_validator('features')
    @classmethod
    def validate_features_length(cls, v):
        if len(v) != 28:
            raise ValueError(f"Expected exactly 28 features, got {len(v)}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [
                        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
                        0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
                        -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
                        -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
                        -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
                        -0.189115, 0.133558, -0.021053
                    ],
                    "amount": 149.62,
                    "time": 0.0
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """
    Response body for /predict endpoint.
    
    Matches the MLPredictResponse struct from Go:
        type MLPredictResponse struct {
            FraudProbability    float64            `json:"fraud_probability"`
            ReconstructionError float64            `json:"reconstruction_error"`
            ShapValues          map[string]float64 `json:"shap_values,omitempty"`
            BaseValue           float64            `json:"base_value,omitempty"`
        }
    """
    fraud_probability: float = Field(
        ...,
        description="Probability of fraud (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    reconstruction_error: float = Field(
        ...,
        description="MSE reconstruction error from autoencoder"
    )
    shap_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="SHAP values for each feature (only if explain=true)"
    )
    base_value: Optional[float] = Field(
        default=None,
        description="SHAP base value (expected prediction)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fraud_probability": 0.0234,
                    "reconstruction_error": 0.0012,
                    "shap_values": None,
                    "base_value": None
                },
                {
                    "fraud_probability": 0.9456,
                    "reconstruction_error": 15.234,
                    "shap_values": {
                        "V14": -2.341,
                        "V12": -1.892,
                        "V10": 1.234,
                        "Reconstruction_Error": -0.987
                    },
                    "base_value": 0.00172
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str = Field(
        ...,
        description="Service health status"
    )
    models_loaded: bool = Field(
        ...,
        description="Whether ML models are loaded and ready"
    )
    autoencoder_input_dim: Optional[int] = Field(
        default=None,
        description="Expected input dimension for autoencoder"
    )
    classifier_input_dim: Optional[int] = Field(
        default=None,
        description="Expected input dimension for classifier"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "models_loaded": True,
                    "autoencoder_input_dim": 31,
                    "classifier_input_dim": 32
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional details")
