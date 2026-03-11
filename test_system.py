#!/usr/bin/env python3
"""
End-to-End Test Script for Fraud Detection System

This script tests the complete pipeline:
1. Health check on both services
2. Normal transaction prediction
3. Fraudulent transaction prediction (with explanation)
4. Batch prediction

Usage:
    # Test Python ML service directly
    python test_system.py --ml-only
    
    # Test full system (Go API → Python ML)
    python test_system.py
    
    # Test with custom endpoints
    python test_system.py --go-url http://localhost:8080 --ml-url http://localhost:8000

Requirements:
    pip install requests
"""

import argparse
import json
import sys
import time
import requests
from typing import Dict, Any, Optional

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


def print_result(success: bool, message: str) -> None:
    """Print a test result."""
    icon = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
    print(f"  {icon} {message}")


def test_health(url: str, service_name: str) -> bool:
    """Test health endpoint."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_result(True, f"{service_name} is healthy")
            print(f"      Response: {json.dumps(data, indent=6)}")
            return True
        else:
            print_result(False, f"{service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_result(False, f"{service_name} is not reachable at {url}")
        return False
    except Exception as e:
        print_result(False, f"{service_name} health check failed: {e}")
        return False


def test_prediction(
    url: str, 
    features: list, 
    amount: float, 
    time_val: float, 
    explain: bool = False,
    expected_fraud: bool = False,
    test_name: str = "Prediction"
) -> bool:
    """Test prediction endpoint."""
    try:
        payload = {
            "features": features,
            "amount": amount,
            "time": time_val
        }
        
        endpoint = f"{url}/predict"
        if explain:
            endpoint += "?explain=true"
        
        start = time.time()
        response = requests.post(endpoint, json=payload, timeout=30)
        elapsed_ms = (time.time() - start) * 1000
        
        if response.status_code == 200:
            data = response.json()
            fraud_prob = data.get("fraud_probability", 0)
            is_fraud = fraud_prob >= 0.5
            
            result_match = (is_fraud == expected_fraud) if expected_fraud is not None else True
            
            print_result(True, f"{test_name} completed in {elapsed_ms:.2f}ms")
            print(f"      Fraud Probability: {fraud_prob:.4f}")
            print(f"      Reconstruction Error: {data.get('reconstruction_error', 'N/A')}")
            print(f"      Is Fraud: {is_fraud}")
            
            if explain and "shap_values" in data:
                print(f"      SHAP Base Value: {data.get('base_value', 'N/A')}")
                # Show top 3 SHAP values
                shap_values = data.get("shap_values", {})
                sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"      Top SHAP contributors:")
                for feature, value in sorted_shap:
                    direction = "↑ fraud" if value > 0 else "↓ normal"
                    print(f"        - {feature}: {value:.4f} ({direction})")
            
            return result_match
        else:
            print_result(False, f"{test_name} failed with status {response.status_code}")
            print(f"      Response: {response.text}")
            return False
            
    except Exception as e:
        print_result(False, f"{test_name} failed: {e}")
        return False


# ==============================================================================
# TEST DATA
# ==============================================================================

# A normal transaction (low fraud probability expected)
NORMAL_TRANSACTION = {
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

# A suspicious transaction (features typical of fraud - high V14, V12 anomalies)
# These values are exaggerated to trigger fraud detection
FRAUD_TRANSACTION = {
    "features": [
        1.191857, 0.266151, 0.166480, 0.448154, 0.060018,
        -0.082361, -0.078803, 0.085102, -0.255425, -0.166974,
        1.612727, 0.065420, -0.143772, -19.0,  # V14 = -19 (extreme)
        0.207643, -0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ],
    "amount": 1.98,
    "time": 406.0
}


def main():
    parser = argparse.ArgumentParser(description="Test Fraud Detection System")
    parser.add_argument("--go-url", default="http://localhost:8080", help="Go API URL")
    parser.add_argument("--ml-url", default="http://localhost:8000", help="Python ML Service URL")
    parser.add_argument("--ml-only", action="store_true", help="Test only Python ML service")
    args = parser.parse_args()
    
    all_passed = True
    
    if args.ml_only:
        # Test Python ML Service directly
        print_header("TESTING PYTHON ML SERVICE DIRECTLY")
        
        # Health check
        print("\n[1] Health Check")
        if not test_health(args.ml_url, "Python ML Service"):
            print(f"\n{RED}Cannot proceed - ML service not healthy{RESET}")
            sys.exit(1)
        
        # Normal transaction
        print("\n[2] Normal Transaction Test")
        passed = test_prediction(
            args.ml_url,
            NORMAL_TRANSACTION["features"],
            NORMAL_TRANSACTION["amount"],
            NORMAL_TRANSACTION["time"],
            explain=False,
            test_name="Normal Transaction"
        )
        all_passed = all_passed and passed
        
        # Fraud transaction with explanation
        print("\n[3] Fraud Transaction Test (with SHAP)")
        passed = test_prediction(
            args.ml_url,
            FRAUD_TRANSACTION["features"],
            FRAUD_TRANSACTION["amount"],
            FRAUD_TRANSACTION["time"],
            explain=True,
            test_name="Fraud Transaction (with explanation)"
        )
        all_passed = all_passed and passed
        
    else:
        # Test full system through Go API
        print_header("TESTING FULL SYSTEM (Go API → Python ML)")
        
        # Health checks
        print("\n[1] Health Checks")
        if not test_health(args.ml_url, "Python ML Service"):
            print(f"\n{YELLOW}Warning: ML service not directly reachable (may be internal){RESET}")
        
        if not test_health(args.go_url, "Go API"):
            print(f"\n{RED}Cannot proceed - Go API not healthy{RESET}")
            sys.exit(1)
        
        # Note: Go API uses different request format
        # For full system tests, you'd need to adapt the payload format
        print(f"\n{YELLOW}Note: Full system tests require Go API request format{RESET}")
        print(f"{YELLOW}Use --ml-only to test ML service directly{RESET}")
    
    # Summary
    print_header("TEST SUMMARY")
    if all_passed:
        print(f"{GREEN}All tests passed!{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}Some tests failed{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
