#!/bin/bash
# ==============================================================================
# FRAUD DETECTION API - TEST COMMANDS
# ==============================================================================
# Run these commands to test the API after starting with docker-compose up
#
# Usage:
#   ./test_curl.sh          # Run all tests
#   ./test_curl.sh health   # Run only health check
#   ./test_curl.sh predict  # Run only prediction test
# ==============================================================================

set -e

GO_API_URL="${GO_API_URL:-http://localhost:8080}"
ML_SERVICE_URL="${ML_SERVICE_URL:-http://localhost:8000}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}FRAUD DETECTION API - TEST SUITE${NC}"
echo -e "${BOLD}============================================${NC}"
echo ""

# ------------------------------------------------------------------------------
# TEST 1: Health Checks
# ------------------------------------------------------------------------------
test_health() {
    echo -e "${BOLD}[TEST 1] Health Checks${NC}"
    echo "----------------------------------------"
    
    echo -e "\n${YELLOW}Testing ML Service health...${NC}"
    curl -s "${ML_SERVICE_URL}/health" | python3 -m json.tool || echo -e "${RED}ML Service not reachable${NC}"
    
    echo -e "\n${YELLOW}Testing Go API health...${NC}"
    curl -s "${GO_API_URL}/health" | python3 -m json.tool || echo -e "${RED}Go API not reachable${NC}"
    
    echo ""
}

# ------------------------------------------------------------------------------
# TEST 2: Normal Transaction (should be low fraud probability)
# ------------------------------------------------------------------------------
test_normal_transaction() {
    echo -e "${BOLD}[TEST 2] Normal Transaction Prediction${NC}"
    echo "----------------------------------------"
    echo "Expected: Low fraud probability (< 0.5)"
    echo ""
    
    curl -s -X POST "${GO_API_URL}/api/v1/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "transaction_id": "test_normal_001",
            "amount": 149.62,
            "time": 0,
            "features": {
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
                "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
                "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
                "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
                "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
                "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
                "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
                "V28": -0.021053
            }
        }' | python3 -m json.tool
    
    echo ""
}

# ------------------------------------------------------------------------------
# TEST 3: Suspicious Transaction (should be higher fraud probability)
# ------------------------------------------------------------------------------
test_suspicious_transaction() {
    echo -e "${BOLD}[TEST 3] Suspicious Transaction Prediction${NC}"
    echo "----------------------------------------"
    echo "Expected: Higher fraud probability (extreme V14, V12 values)"
    echo ""
    
    curl -s -X POST "${GO_API_URL}/api/v1/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "transaction_id": "test_suspicious_001",
            "amount": 1.00,
            "time": 3600,
            "features": {
                "V1": 1.191857, "V2": 0.266151, "V3": 0.166480,
                "V4": 0.448154, "V5": 0.060018, "V6": -0.082361,
                "V7": -0.078803, "V8": 0.085102, "V9": -0.255425,
                "V10": -5.0, "V11": 1.612727, "V12": -10.0,
                "V13": -0.143772, "V14": -15.0, "V15": 0.207643,
                "V16": 0.0, "V17": -5.0, "V18": 0.0,
                "V19": 0.0, "V20": 0.0, "V21": 0.0,
                "V22": 0.0, "V23": 0.0, "V24": 0.0,
                "V25": 0.0, "V26": 0.0, "V27": 0.0,
                "V28": 0.0
            }
        }' | python3 -m json.tool
    
    echo ""
}

# ------------------------------------------------------------------------------
# TEST 4: ML Service Direct Test
# ------------------------------------------------------------------------------
test_ml_service_direct() {
    echo -e "${BOLD}[TEST 4] ML Service Direct Prediction${NC}"
    echo "----------------------------------------"
    echo "Testing Python ML service directly (with SHAP explanation)"
    echo ""
    
    curl -s -X POST "${ML_SERVICE_URL}/predict?explain=true" \
        -H "Content-Type: application/json" \
        -d '{
            "features": [
                -1.359807, -0.072781, 2.536347, 1.378155, -0.338321,
                0.462388, 0.239599, 0.098698, 0.363787, 0.090794,
                -0.551600, -0.617801, -0.991390, -0.311169, 1.468177,
                -0.470401, 0.207971, 0.025791, 0.403993, 0.251412,
                -0.018307, 0.277838, -0.110474, 0.066928, 0.128539,
                -0.189115, 0.133558, -0.021053
            ],
            "amount": 149.62,
            "time": 0
        }' | python3 -m json.tool
    
    echo ""
}

# ------------------------------------------------------------------------------
# TEST 5: Batch Prediction
# ------------------------------------------------------------------------------
test_batch_prediction() {
    echo -e "${BOLD}[TEST 5] Batch Prediction${NC}"
    echo "----------------------------------------"
    echo "Testing batch of 2 transactions"
    echo ""
    
    curl -s -X POST "${GO_API_URL}/api/v1/batch" \
        -H "Content-Type: application/json" \
        -d '{
            "transactions": [
                {
                    "transaction_id": "batch_001",
                    "amount": 149.62,
                    "time": 0,
                    "features": {
                        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
                        "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
                        "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
                        "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
                        "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
                        "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
                        "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
                        "V28": -0.021053
                    }
                },
                {
                    "transaction_id": "batch_002",
                    "amount": 1.00,
                    "time": 3600,
                    "features": {
                        "V1": 1.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
                        "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": -5.0,
                        "V11": 0.0, "V12": -10.0, "V13": 0.0, "V14": -15.0, "V15": 0.0,
                        "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
                        "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
                        "V26": 0.0, "V27": 0.0, "V28": 0.0
                    }
                }
            ]
        }' | python3 -m json.tool
    
    echo ""
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
case "${1:-all}" in
    health)
        test_health
        ;;
    predict)
        test_normal_transaction
        ;;
    suspicious)
        test_suspicious_transaction
        ;;
    ml)
        test_ml_service_direct
        ;;
    batch)
        test_batch_prediction
        ;;
    all|*)
        test_health
        test_ml_service_direct
        test_normal_transaction
        test_suspicious_transaction
        test_batch_prediction
        echo -e "${GREEN}${BOLD}All tests completed!${NC}"
        ;;
esac
