package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"fraud-detection-api/config"
	"fraud-detection-api/internal/models"
	"fraud-detection-api/internal/services"
)

// noAlertCfg returns a Config with no alert destinations (Slack/email disabled).
func noAlertCfg() *config.Config {
	return &config.Config{}
}

// VALIDATION TESTS
func TestValidatePredictRequest_MissingID(t *testing.T) {
	req := &models.PredictRequest{
		TransactionID: "",
		Amount:        100.0,
	}
	errors := validatePredictRequest(req)
	if len(errors) == 0 {
		t.Fatal("expected validation error for empty transaction_id")
	}
	if errors[0].Field != "transaction_id" {
		t.Errorf("expected error on field 'transaction_id', got '%s'", errors[0].Field)
	}
}

func TestValidatePredictRequest_NegativeAmount(t *testing.T) {
	req := &models.PredictRequest{
		TransactionID: "txn_001",
		Amount:        -50.0,
	}
	errors := validatePredictRequest(req)
	if len(errors) == 0 {
		t.Fatal("expected validation error for negative amount")
	}
	if errors[0].Field != "amount" {
		t.Errorf("expected error on field 'amount', got '%s'", errors[0].Field)
	}
}

func TestValidatePredictRequest_Valid(t *testing.T) {
	req := &models.PredictRequest{
		TransactionID: "txn_001",
		Amount:        149.62,
		Time:          0,
	}
	errors := validatePredictRequest(req)
	if len(errors) != 0 {
		t.Errorf("expected no errors, got %d: %+v", len(errors), errors)
	}
}

func TestValidatePredictRequest_ZeroAmount(t *testing.T) {
	req := &models.PredictRequest{
		TransactionID: "txn_001",
		Amount:        0.0,
	}
	errors := validatePredictRequest(req)
	if len(errors) != 0 {
		t.Errorf("zero amount should be valid, got %d errors", len(errors))
	}
}

func TestValidatePredictRequest_LargeAmount(t *testing.T) {
	req := &models.PredictRequest{
		TransactionID: "txn_001",
		Amount:        1_000_000.0,
	}
	errors := validatePredictRequest(req)
	if len(errors) != 0 {
		t.Errorf("very large amount should be valid, got %d errors", len(errors))
	}
}

// CONFIDENCE LEVEL TESTS
func TestGetConfidenceLevel(t *testing.T) {
	tests := []struct {
		name        string
		probability float64
		want        string
	}{
		{"very high fraud", 0.95, "high"},
		{"very low fraud", 0.05, "high"},
		{"moderate fraud", 0.75, "medium"},
		{"moderate normal", 0.25, "medium"},
		{"uncertain", 0.50, "low"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getConfidenceLevel(tt.probability)
			if got != tt.want {
				t.Errorf("getConfidenceLevel(%.2f) = %q, want %q", tt.probability, got, tt.want)
			}
		})
	}
}

// HEALTH CHECK HANDLER TEST
// mockMLServer spins up a fake Python ML service that returns 200 on /health
func mockMLServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
}

// mockMLDegraded spins up a fake ML service that always returns 500
func mockMLDegraded(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
}

func TestHealthCheck_Healthy(t *testing.T) {
	ml := mockMLServer(t)
	defer ml.Close()

	cfg := noAlertCfg()
	mlClient := services.NewMLClient(ml.URL, 5e9) // 5 second timeout
	h := NewHandler(cfg, mlClient, services.NewAlertService(cfg))

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	h.HealthCheck(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp models.HealthResponse
	json.NewDecoder(rec.Body).Decode(&resp)

	if resp.Status != "healthy" {
		t.Errorf("expected status 'healthy', got %q", resp.Status)
	}
	if resp.Services.MLService != "up" {
		t.Errorf("expected ml_service 'up', got %q", resp.Services.MLService)
	}
}

func TestHealthCheck_Degraded(t *testing.T) {
	ml := mockMLDegraded(t)
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	h.HealthCheck(rec, req)

	// health endpoint always returns 200; status field indicates degradation
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 even when degraded, got %d", rec.Code)
	}

	var resp models.HealthResponse
	json.NewDecoder(rec.Body).Decode(&resp)

	if resp.Status != "degraded" {
		t.Errorf("expected status 'degraded', got %q", resp.Status)
	}
	if resp.Services.MLService != "down" {
		t.Errorf("expected ml_service 'down', got %q", resp.Services.MLService)
	}
}

// PREDICT HANDLER TESTS
// mockMLPredict spins up a fake ML service that returns a canned prediction.
// The Python service applies its trained threshold and returns is_fraud; the
// mock mirrors that with a 0.5 cut so existing test inputs (0.02 vs 0.92)
// continue to land on the right side.
func mockMLPredict(t *testing.T, probability float64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := models.MLPredictResponse{
			IsFraud:             probability >= 0.5,
			FraudProbability:    probability,
			ReconstructionError: 0.0042,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
}

// mockMLPredictWithSHAP returns a canned prediction that includes SHAP values.
func mockMLPredictWithSHAP(t *testing.T, probability float64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := models.MLPredictResponse{
			IsFraud:             probability >= 0.5,
			FraudProbability:    probability,
			ReconstructionError: 0.0042,
			ShapValues:          map[string]float64{"V1": 0.12, "V14": -0.08, "Amount_Log": 0.05},
			BaseValue:           0.03,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
}

func TestPredict_InvalidJSON(t *testing.T) {
	ml := mockMLPredict(t, 0.1)
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/predict", strings.NewReader("not json"))
	rec := httptest.NewRecorder()

	h.Predict(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad JSON, got %d", rec.Code)
	}
}

func TestPredict_LegitimateTransaction(t *testing.T) {
	ml := mockMLPredict(t, 0.02) // low probability → not fraud
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	body := `{
		"transaction_id": "txn_001",
		"amount": 149.62,
		"time": 0,
		"features": {
			"V1":-1.35,"V2":-0.07,"V3":2.53,"V4":1.37,"V5":-0.33,
			"V6":0.46,"V7":0.23,"V8":0.09,"V9":0.36,"V10":0.09,
			"V11":-0.55,"V12":-0.61,"V13":-0.99,"V14":-0.31,"V15":1.46,
			"V16":-0.47,"V17":0.20,"V18":0.02,"V19":0.40,"V20":0.25,
			"V21":-0.01,"V22":0.27,"V23":-0.11,"V24":0.06,"V25":0.12,
			"V26":-0.18,"V27":0.13,"V28":-0.02
		}
	}`

	req := httptest.NewRequest(http.MethodPost, "/api/v1/predict", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.Predict(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp models.PredictResponse
	json.NewDecoder(rec.Body).Decode(&resp)

	if resp.Prediction.IsFraud {
		t.Error("expected is_fraud=false for probability 0.02")
	}
	if resp.Explanation != nil {
		t.Error("expected no explanation for legitimate transaction")
	}
}

func TestPredict_FraudulentTransaction(t *testing.T) {
	ml := mockMLPredict(t, 0.92) // high probability → fraud
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	body := `{
		"transaction_id": "txn_fraud",
		"amount": 1.00,
		"time": 3600,
		"features": {
			"V1":1.0,"V2":0.0,"V3":0.0,"V4":0.0,"V5":0.0,
			"V6":0.0,"V7":0.0,"V8":0.0,"V9":0.0,"V10":-5.0,
			"V11":0.0,"V12":-10.0,"V13":0.0,"V14":-15.0,"V15":0.0,
			"V16":0.0,"V17":0.0,"V18":0.0,"V19":0.0,"V20":0.0,
			"V21":0.0,"V22":0.0,"V23":0.0,"V24":0.0,"V25":0.0,
			"V26":0.0,"V27":0.0,"V28":0.0
		}
	}`

	req := httptest.NewRequest(http.MethodPost, "/api/v1/predict", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.Predict(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp models.PredictResponse
	json.NewDecoder(rec.Body).Decode(&resp)

	if !resp.Prediction.IsFraud {
		t.Error("expected is_fraud=true for probability 0.92")
	}
}

func TestPredict_MLUnavailable(t *testing.T) {
	ml := mockMLPredict(t, 0.02)
	ml.Close() // close immediately so all requests fail

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	body := `{"transaction_id":"txn_001","amount":100.0,"time":0}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/predict", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.Predict(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 when ML service unreachable, got %d", rec.Code)
	}
}

// helper that builds a minimal transaction JSON object
func minimalTxnJSON(id string) string {
	return `{"transaction_id":"` + id + `","amount":1.0,"time":0,` +
		`"features":{"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,` +
		`"V8":0,"V9":0,"V10":0,"V11":0,"V12":0,"V13":0,"V14":0,"V15":0,` +
		`"V16":0,"V17":0,"V18":0,"V19":0,"V20":0,"V21":0,"V22":0,"V23":0,` +
		`"V24":0,"V25":0,"V26":0,"V27":0,"V28":0}}`
}

func TestBatchPredict_MultipleTransactions(t *testing.T) {
	ml := mockMLPredict(t, 0.02) // all legitimate
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	body := `{"transactions":[` + minimalTxnJSON("txn_001") + `,` + minimalTxnJSON("txn_002") + `]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/batch", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.BatchPredict(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp models.BatchPredictResponse
	json.NewDecoder(rec.Body).Decode(&resp)

	if resp.Summary.Total != 2 {
		t.Errorf("expected total=2, got %d", resp.Summary.Total)
	}
	if resp.Summary.Legitimate != 2 {
		t.Errorf("expected legitimate=2, got %d", resp.Summary.Legitimate)
	}
	if resp.Summary.Fraudulent != 0 {
		t.Errorf("expected fraudulent=0, got %d", resp.Summary.Fraudulent)
	}
}

func TestBatchPredict_MixedTransactions(t *testing.T) {
	ml := mockMLPredict(t, 0.92) // all flagged as fraud
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	body := `{"transactions":[` + minimalTxnJSON("txn_a") + `,` + minimalTxnJSON("txn_b") + `]}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/batch", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.BatchPredict(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp models.BatchPredictResponse
	json.NewDecoder(rec.Body).Decode(&resp)

	if resp.Summary.Total != 2 {
		t.Errorf("expected total=2, got %d", resp.Summary.Total)
	}
	if resp.Summary.Fraudulent != 2 {
		t.Errorf("expected fraudulent=2, got %d", resp.Summary.Fraudulent)
	}
	if resp.Summary.Legitimate != 0 {
		t.Errorf("expected legitimate=0, got %d", resp.Summary.Legitimate)
	}
}

func TestBatchPredict_TooMany(t *testing.T) {
	ml := mockMLPredict(t, 0.02)
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	// build 101 transaction objects
	txns := make([]string, 101)
	for i := range txns {
		txns[i] = minimalTxnJSON("txn")
	}
	body := `{"transactions":[` + strings.Join(txns, ",") + `]}`

	req := httptest.NewRequest(http.MethodPost, "/api/v1/batch", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.BatchPredict(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for >100 transactions, got %d", rec.Code)
	}
}

// EXPLAIN HANDLER TESTS

func TestExplain_InvalidJSON(t *testing.T) {
	ml := mockMLPredictWithSHAP(t, 0.85)
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/explain", strings.NewReader("not json"))
	rec := httptest.NewRecorder()

	h.Explain(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for bad JSON, got %d", rec.Code)
	}
}

func TestExplain_MissingTransactionID(t *testing.T) {
	ml := mockMLPredictWithSHAP(t, 0.85)
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	body := `{"transaction_id":"","amount":100.0,"time":0,"features":{}}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/explain", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.Explain(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for missing transaction_id, got %d", rec.Code)
	}
}

func TestExplain_ReturnsShapValues(t *testing.T) {
	ml := mockMLPredictWithSHAP(t, 0.85)
	defer ml.Close()

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	body := `{
		"transaction_id": "txn_explain",
		"amount": 149.62,
		"time": 3600,
		"features": {
			"V1":-1.35,"V2":-0.07,"V3":2.53,"V4":1.37,"V5":-0.33,
			"V6":0.46,"V7":0.23,"V8":0.09,"V9":0.36,"V10":0.09,
			"V11":-0.55,"V12":-0.61,"V13":-0.99,"V14":-0.31,"V15":1.46,
			"V16":-0.47,"V17":0.20,"V18":0.02,"V19":0.40,"V20":0.25,
			"V21":-0.01,"V22":0.27,"V23":-0.11,"V24":0.06,"V25":0.12,
			"V26":-0.18,"V27":0.13,"V28":-0.02
		}
	}`

	req := httptest.NewRequest(http.MethodPost, "/api/v1/explain", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.Explain(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var resp models.ExplainResponse
	json.NewDecoder(rec.Body).Decode(&resp)

	if resp.TransactionID != "txn_explain" {
		t.Errorf("expected transaction_id 'txn_explain', got %q", resp.TransactionID)
	}
	if len(resp.Explanation.ShapValues) == 0 {
		t.Error("expected non-empty shap_values in explanation")
	}
	if len(resp.Explanation.TopFeatures) == 0 {
		t.Error("expected non-empty top_features in explanation")
	}
}

func TestExplain_MLUnavailable(t *testing.T) {
	ml := mockMLPredictWithSHAP(t, 0.85)
	ml.Close() // close immediately

	cfg := noAlertCfg()
	h := NewHandler(cfg, services.NewMLClient(ml.URL, 5e9), services.NewAlertService(cfg))

	body := `{"transaction_id":"txn_001","amount":100.0,"time":0}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/explain", strings.NewReader(body))
	rec := httptest.NewRecorder()

	h.Explain(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 when ML service unreachable, got %d", rec.Code)
	}
}
