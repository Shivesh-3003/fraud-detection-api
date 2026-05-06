"""
Pydantic Models for Fraud Detection API

These schemas define the request/response format for the Python ML service.
They must be compatible with what the Go API sends/expects.

Dataset-aware design:
    PredictRequest holds optional fields for both ULB and Sparkov.
    The /predict handler validates completeness based on the loaded dataset type
    (read from pipeline.feature_config.dataset_type) and builds the raw_input
    dict for the preprocessing pipeline.

ULB contract (dataset_type="ulb"):
    features: [float×28], amount: float, time: float

Sparkov contract (dataset_type="sparkov"):
    amt, trans_datetime, dob, gender, city_pop, lat, long, merch_lat, merch_long, category
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """
    Unified request body for /predict endpoint.

    Fields are optional to support both ULB and Sparkov datasets.
    The active dataset type is determined by the loaded model's feature_config.json.
    The handler validates that the required fields for the active dataset are present.

    ULB fields (dataset_type="ulb"):
        features: 28 PCA-anonymous V-features
        amount:   raw transaction amount
        time:     seconds since first transaction in dataset

    Sparkov fields (dataset_type="sparkov"):
        amt:            transaction amount
        trans_datetime: ISO datetime string, e.g. "2019-06-15T14:30:00"
        dob:            date of birth ISO string, e.g. "1985-03-12"
        gender:         "M" or "F"
        city_pop:       city population (integer)
        lat, long:      cardholder location coordinates
        merch_lat, merch_long: merchant location coordinates
        category:       transaction category string, e.g. "shopping_net"
    """

    # ULB fields
    features: Optional[List[float]] = Field(
        default=None,
        description="28 PCA features (V1-V28) as ordered array — ULB dataset only"
    )
    amount: Optional[float] = Field(
        default=None,
        ge=0,
        description="Transaction amount (raw) — ULB dataset only"
    )
    time: Optional[float] = Field(
        default=None,
        ge=0,
        description="Seconds since first transaction in dataset — ULB dataset only"
    )

    # Sparkov fields
    amt: Optional[float] = Field(
        default=None,
        ge=0,
        description="Transaction amount — Sparkov dataset only"
    )
    trans_datetime: Optional[str] = Field(
        default=None,
        description="Transaction datetime ISO string, e.g. '2019-06-15T14:30:00' — Sparkov only"
    )
    dob: Optional[str] = Field(
        default=None,
        description="Cardholder date of birth ISO string, e.g. '1985-03-12' — Sparkov only"
    )
    gender: Optional[str] = Field(
        default=None,
        description="Cardholder gender: 'M' or 'F' — Sparkov only"
    )
    city_pop: Optional[int] = Field(
        default=None,
        ge=0,
        description="City population — Sparkov only"
    )
    lat: Optional[float] = Field(default=None, description="Cardholder latitude — Sparkov only")
    long: Optional[float] = Field(default=None, description="Cardholder longitude — Sparkov only")
    merch_lat: Optional[float] = Field(default=None, description="Merchant latitude — Sparkov only")
    merch_long: Optional[float] = Field(default=None, description="Merchant longitude — Sparkov only")
    category: Optional[str] = Field(
        default=None,
        description="Transaction category, e.g. 'shopping_net' — Sparkov only"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "ULB example",
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
                },
                {
                    "description": "Sparkov example",
                    "amt": 120.50,
                    "trans_datetime": "2019-06-15T14:30:00",
                    "dob": "1985-03-12",
                    "gender": "F",
                    "city_pop": 5000,
                    "lat": 36.07,
                    "long": -81.17,
                    "merch_lat": 36.01,
                    "merch_long": -82.04,
                    "category": "shopping_net"
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """
    Response body for /predict endpoint.

    Matches the MLPredictResponse struct from Go:
        type MLPredictResponse struct {
            IsFraud             bool               `json:"is_fraud"`
            FraudProbability    float64            `json:"fraud_probability"`
            ReconstructionError float64            `json:"reconstruction_error"`
            ShapValues          map[string]float64 `json:"shap_values,omitempty"`
            BaseValue           float64            `json:"base_value,omitempty"`
        }
    """
    is_fraud: bool = Field(
        ...,
        description="True if fraud_probability >= the model's trained optimal threshold"
    )
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
    inference_time_ms: Optional[float] = Field(
        default=None,
        description="Isolated PyTorch inference time in milliseconds"
    )
    shap_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="SHAP values per feature (only if explain=true). "
                    "Keys are human-readable feature names for Sparkov, "
                    "V1-V28 style names for ULB."
    )
    base_value: Optional[float] = Field(
        default=None,
        description="SHAP base value (expected prediction on background data)"
    )


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str = Field(..., description="Service health status")
    models_loaded: bool = Field(..., description="Whether ML models are loaded and ready")
    autoencoder_input_dim: Optional[int] = Field(default=None)
    classifier_input_dim: Optional[int] = Field(default=None)


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional details")
