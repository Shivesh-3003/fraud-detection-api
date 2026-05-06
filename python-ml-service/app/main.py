"""
FastAPI Application for Fraud Detection ML Service

This service provides the ML inference layer for the fraud detection system.
It's called by the Go API orchestrator.

Endpoints:
    GET  /health           - Health check
    POST /predict          - Single transaction prediction
    POST /predict?explain=true - Prediction with SHAP explanation

Architecture:
    Go API (port 8080) --HTTP--> Python ML Service (port 8000)
"""

import logging
import time
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .models import PredictRequest, PredictResponse, HealthResponse, ErrorResponse
from .inference import get_pipeline, FraudDetectionPipeline
from .explainer import FraudExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global references (populated on startup)
pipeline: Optional[FraudDetectionPipeline] = None
explainer: Optional[FraudExplainer] = None

def _build_background_data(pipe: FraudDetectionPipeline, n_samples: int = 100) -> np.ndarray:
    """
    Generate synthetic background data for SHAP from the fitted scaler.
    Instead of using zeros, we sample from N(0,1) per feature (what
    StandardScaler produces for normal data), then run through the
    autoencoder to get realistic reconstruction errors.
    """
    import torch

    ae_dim = pipe.feature_config.ae_input_dim

    np.random.seed(42)
    scaled_samples = np.random.randn(n_samples, ae_dim).astype(np.float32)

    pipe.autoencoder.eval()
    with torch.no_grad():
        tensor = torch.tensor(scaled_samples, dtype=torch.float32)
        errors = pipe.autoencoder.get_reconstruction_error(tensor).numpy()

    background = np.column_stack([scaled_samples, errors])

    logger.info(f"Generated SHAP background data: {background.shape} "
                f"(mean recon error: {errors.mean():.4f})")
    return background

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI.
    
    Loads models on startup, cleans up on shutdown.
    """
    global pipeline, explainer
    
    logger.info("="*60)
    logger.info("STARTING FRAUD DETECTION ML SERVICE")
    logger.info("="*60)
    
    try:
        logger.info("[1/2] Loading inference pipeline...")
        pipeline = get_pipeline()
        logger.info("✓ Inference pipeline loaded")
        
        logger.info("[2/2] Initializing SHAP explainer...")
        background = _build_background_data(pipeline, n_samples=100)
        explainer = FraudExplainer(
            classifier=pipeline.classifier,
            feature_names=pipeline.get_feature_names(include_reconstruction_error=True),
            background_data=background,
            n_background_samples=100
        )
        logger.info(
            f"✓ SHAP explainer initialized (dataset: {pipeline.feature_config.dataset_type}, "
            f"{len(pipeline.get_feature_names())} features)"
        )
        
        logger.info("="*60)
        logger.info("ML SERVICE READY")
        logger.info("="*60)
        
    except FileNotFoundError as e:
        logger.error(f"❌ Model files not found: {e}")
        logger.error("Ensure model files are mounted at /app/models/")
        logger.error("Required files: scaler.pkl, autoencoder_model.pth, mlp_classifier.pth")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise
    
    yield 
    
    # Shutdown
    logger.info("Shutting down ML service...")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection ML Service",
    description="Python ML inference service for real-time fraud detection",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Add CORS middleware (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for Docker and Go API.
    
    Returns service status and model information.
    """
    if pipeline is None or not pipeline.is_ready():
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            autoencoder_input_dim=None,
            classifier_input_dim=None
        )

    return HealthResponse(
        status="healthy",
        models_loaded=True,
        autoencoder_input_dim=pipeline.feature_config.ae_input_dim,
        classifier_input_dim=pipeline.feature_config.clf_input_dim
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    explain: bool = Query(default=False)
):
    """
    Main prediction endpoint.
    
    Flow:
        1. Preprocess raw features (transform Amount, Time)
        2. Scale features using fitted StandardScaler
        3. Pass through Autoencoder → get reconstruction error
        4. Concatenate features + reconstruction error
        5. Pass through MLP Classifier → get fraud probability
        6. (Optional) Compute SHAP values if explain=true
    
    Args:
        request: Transaction data with 28 V-features, Amount, Time
        explain: Whether to compute SHAP explanation
        
    Returns:
        PredictResponse with fraud_probability, reconstruction_error,
        and optionally shap_values + base_value
    """
    # Validate pipeline is ready
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Service is starting up."
        )
    
    try:
        # Build raw_input dict for dataset-aware preprocessing
        dataset_type = pipeline.feature_config.dataset_type
        if dataset_type == "ulb":
            if request.features is None or request.amount is None or request.time is None:
                raise HTTPException(
                    status_code=400,
                    detail="ULB dataset requires: features (28 floats), amount, time"
                )
            raw_input = {
                "v_features": request.features,
                "amount": request.amount,
                "time": request.time
            }
        else:  # sparkov
            required = ["amt", "trans_datetime", "dob", "gender", "city_pop",
                        "lat", "long", "merch_lat", "merch_long", "category"]
            missing = [f for f in required if getattr(request, f) is None]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sparkov dataset requires: {missing}"
                )
            raw_input = {
                "amt":           request.amt,
                "trans_datetime": request.trans_datetime,
                "dob":           request.dob,
                "gender":        request.gender,
                "city_pop":      request.city_pop,
                "lat":           request.lat,
                "long":          request.long,
                "merch_lat":     request.merch_lat,
                "merch_long":    request.merch_long,
                "category":      request.category
            }

        # Run prediction pipeline — returns isolated inference time
        fraud_probability, reconstruction_error, classifier_input, inference_time_ms = pipeline.predict(
            raw_input
        )

        # Build response — apply the trained per-dataset threshold here so the
        # Go API consumes a single authoritative is_fraud boolean.
        response = PredictResponse(
            is_fraud=fraud_probability >= pipeline.threshold,
            fraud_probability=fraud_probability,
            reconstruction_error=reconstruction_error,
            inference_time_ms=round(inference_time_ms, 4)
        )
        
        # Add SHAP explanation if requested
        if explain and explainer is not None:
            try:
                shap_values, base_value = explainer.explain(
                    classifier_input=classifier_input,
                    nsamples=100 # Balance between speed and accuracy
                )
                response.shap_values = shap_values
                response.base_value = base_value
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        logger.info(
            f"Prediction: fraud_prob={fraud_probability:.4f}, "
            f"recon_error={reconstruction_error:.4f}, "
            f"inference={inference_time_ms:.3f}ms, "
            f"explain={explain}"
        )
        
        return response
        
    except ValueError as e:
        # Input validation errors
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with basic service info."""
    return {
        "service": "Fraud Detection ML Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "predict_with_explanation": "POST /predict?explain=true"
        }
    }


# For running with uvicorn directly (development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")