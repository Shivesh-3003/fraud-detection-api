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
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .models import PredictRequest, PredictResponse, HealthResponse, ErrorResponse
from .inference import get_pipeline, FraudDetectionPipeline, AUTOENCODER_INPUT_DIM, CLASSIFIER_INPUT_DIM
from .explainer import get_explainer, FraudExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global references (populated on startup)
pipeline: Optional[FraudDetectionPipeline] = None
explainer: Optional[FraudExplainer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI.
    
    Loads models on startup, cleans up on shutdown.
    """
    global pipeline, explainer
    
    # Startup
    logger.info("="*60)
    logger.info("STARTING FRAUD DETECTION ML SERVICE")
    logger.info("="*60)
    
    try:
        # Load the inference pipeline (scaler + autoencoder + classifier)
        logger.info("[1/2] Loading inference pipeline...")
        pipeline = get_pipeline()
        logger.info("✓ Inference pipeline loaded")
        
        # Initialize SHAP explainer
        logger.info("[2/2] Initializing SHAP explainer...")
        explainer = get_explainer(pipeline.classifier)
        logger.info("✓ SHAP explainer initialized")
        
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
    
    yield  # Application runs here
    
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


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the ML service is healthy and models are loaded"
)
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
        autoencoder_input_dim=AUTOENCODER_INPUT_DIM,
        classifier_input_dim=CLASSIFIER_INPUT_DIM
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Fraud",
    description="Run fraud detection on a single transaction"
)
async def predict(
    request: PredictRequest,
    explain: bool = Query(
        default=False,
        description="If true, include SHAP explanation in response"
    )
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
    start_time = time.time()
    
    # Validate pipeline is ready
    if pipeline is None or not pipeline.is_ready():
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Service is starting up."
        )
    
    try:
        # Run prediction pipeline
        fraud_probability, reconstruction_error, classifier_input = pipeline.predict(
            v_features=request.features,
            amount=request.amount,
            time=request.time
        )
        
        # Build response
        response = PredictResponse(
            fraud_probability=fraud_probability,
            reconstruction_error=reconstruction_error
        )
        
        # Add SHAP explanation if requested
        if explain and explainer is not None:
            try:
                shap_values, base_value = explainer.explain(
                    classifier_input=classifier_input,
                    nsamples=100  # Balance between speed and accuracy
                )
                response.shap_values = shap_values
                response.base_value = base_value
            except Exception as e:
                # Don't fail the whole request if SHAP fails
                logger.warning(f"SHAP explanation failed: {e}")
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Prediction complete: fraud_prob={fraud_probability:.4f}, "
            f"recon_error={reconstruction_error:.4f}, "
            f"explain={explain}, "
            f"time={elapsed_ms:.2f}ms"
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
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
