#!/usr/bin/env python3
"""
Test Script for Fraud Detection ML Service

This script tests the ML pipeline components locally without starting the HTTP server.
Run this to verify your models are correctly loaded before deploying.

Usage:
    python test_local.py            # default: ulb dataset
    DATASET=sparkov python test_local.py

Requirements:
    - Model files in ./models/<dataset>/ directory
    - Python dependencies installed
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set model path for local testing — defaults to ulb subdir
_dataset = os.environ.get("DATASET", "ulb")
os.environ["MODEL_PATH"] = os.path.join(os.path.dirname(__file__), "models", _dataset)


def test_pipeline():
    """Test the full prediction pipeline."""
    print("=" * 60)
    print("TESTING FRAUD DETECTION ML PIPELINE")
    print("=" * 60)
    
    # Check model files exist
    model_path = os.environ["MODEL_PATH"]
    required_files = ["scaler.pkl", "autoencoder_model.pth", "mlp_classifier.pth"]
    
    print(f"\n[1] Checking model files in: {model_path}")
    missing = []
    for f in required_files:
        path = os.path.join(model_path, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    ✓ {f} ({size_mb:.2f} MB)")
        else:
            print(f"    ✗ {f} NOT FOUND")
            missing.append(f)
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        print("   Please copy your trained model files to the models/ directory.")
        return False
    
    # Load pipeline
    print("\n[2] Loading inference pipeline...")
    try:
        from app.inference import get_pipeline, reset_pipeline
        reset_pipeline()  # Clear any cached state
        pipeline = get_pipeline()
        print("    ✓ Pipeline loaded successfully")
    except Exception as e:
        print(f"    ✗ Failed to load pipeline: {e}")
        return False
    
    # Test with sample data (normal transaction)
    print("\n[3] Testing with sample NORMAL transaction...")
    normal_features = [
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
        0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
        -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
        -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
        -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
        -0.189115, 0.133558, -0.021053
    ]
    normal_amount = 149.62
    normal_time = 0.0
    
    try:
        prob, recon_error, classifier_input, inference_ms = pipeline.predict({
            "v_features": normal_features,
            "amount": normal_amount,
            "time": normal_time,
        })
        print(f"    Fraud Probability:    {prob:.6f}")
        print(f"    Reconstruction Error: {recon_error:.6f}")
        print(f"    Inference Time:       {inference_ms:.3f} ms")
        print(f"    Classifier Input Shape: {classifier_input.shape}")

        if prob < 0.5:
            print("    ✓ Correctly classified as NORMAL (prob < 0.5)")
        else:
            print("    ⚠ Unexpectedly classified as FRAUD")
    except Exception as e:
        print(f"    ✗ Prediction failed: {e}")
        return False
    
    # Test with anomalous data (simulated fraud-like pattern)
    print("\n[4] Testing with anomalous transaction...")
    # Extreme values that might trigger fraud detection
    anomalous_features = [
        -5.0, 10.0, -8.0, 5.0, -3.0,
        2.0, -7.0, 8.0, -4.0, 6.0,
        -9.0, 7.0, -6.0, 10.0, -5.0,
        4.0, -8.0, 6.0, -3.0, 5.0,
        -7.0, 8.0, -4.0, 6.0, -5.0,
        7.0, -8.0, 9.0
    ]
    anomalous_amount = 9999.99
    anomalous_time = 80000.0
    
    try:
        prob, recon_error, _, inference_ms = pipeline.predict({
            "v_features": anomalous_features,
            "amount": anomalous_amount,
            "time": anomalous_time,
        })
        print(f"    Fraud Probability:    {prob:.6f}")
        print(f"    Reconstruction Error: {recon_error:.6f}")
        print(f"    Inference Time:       {inference_ms:.3f} ms")
        print(f"    Note: Higher recon_error indicates anomaly detected")
    except Exception as e:
        print(f"    ✗ Prediction failed: {e}")
        return False
    
    # Test SHAP explainer
    print("\n[5] Testing SHAP explainer...")
    try:
        from app.explainer import get_explainer, reset_explainer
        reset_explainer()
        explainer = get_explainer(pipeline.classifier)
        
        # Get classifier input for explanation
        _, _, classifier_input, _ = pipeline.predict({
            "v_features": anomalous_features,
            "amount": anomalous_amount,
            "time": anomalous_time,
        })
        
        shap_values, base_value = explainer.explain(
            classifier_input=classifier_input,
            nsamples=50  # Fewer samples for faster test
        )
        
        print(f"    Base value: {base_value:.6f}")
        print(f"    Number of SHAP values: {len(shap_values)}")

        # Top-N ranking lives in the Go API now (handlers.buildExplanation),
        # so reproduce it inline here for visibility.
        top_features = sorted(
            shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:5]
        print("    Top contributing features:")
        for name, value in top_features:
            arrow = "↑" if value > 0 else "↓"
            print(f"      {arrow} {name}: {value:.4f}")

        print("    ✓ SHAP explanation generated")
    except Exception as e:
        print(f"    ✗ SHAP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nYou can now:")
    print("  1. Run: uvicorn app.main:app --port 8000")
    print("  2. Or build Docker: docker build -t fraud-ml-service .")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
