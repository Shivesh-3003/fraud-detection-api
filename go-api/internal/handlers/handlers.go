package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"sort"
	"time"

	"fraud-detection-api/config"
	"fraud-detection-api/internal/models"
	"fraud-detection-api/internal/services"
)

type Handler struct {
	config       *config.Config
	mlClient     *services.MLClient
	alertService *services.AlertService
}

func NewHandler(cfg *config.Config, mlClient *services.MLClient, alertService *services.AlertService) *Handler {
	return &Handler{config: cfg, mlClient: mlClient, alertService: alertService}
}

func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	mlStatus := "up"
	if err := h.mlClient.HealthCheck(ctx); err != nil {
		mlStatus = "down"
		log.Printf("ML service health check failed: %v", err)
	}
	status := "healthy"
	if mlStatus == "down" {
		status = "degraded"
	}
	resp := models.HealthResponse{
		Status:    status,
		Timestamp: time.Now().UTC(),
		Services:  models.ServiceStatus{GoAPI: "up", MLService: mlStatus},
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) Predict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	var req models.PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Invalid request body", nil)
		return
	}

	if errors := validatePredictRequest(&req); len(errors) > 0 {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Request validation failed", errors)
		return
	}

	mlReq := &models.MLPredictRequest{
		Features: req.Features.ToSlice(),
		Amount:   req.Amount,
		Time:     req.Time,
	}

	// Step 1: Fast prediction WITHOUT SHAP
	mlResp, err := h.mlClient.Predict(ctx, mlReq)
	if err != nil {
		log.Printf("ML service error: %v", err)
		writeError(w, http.StatusServiceUnavailable, "ML_SERVICE_UNAVAILABLE", "ML service is unavailable", nil)
		return
	}

	isFraud := mlResp.FraudProbability >= h.config.FraudThreshold

	// Step 2: Only compute SHAP if fraud detected
	if isFraud {
		mlRespExplained, err := h.mlClient.PredictWithExplanation(ctx, mlReq)
		if err != nil {
			log.Printf("SHAP explanation failed, using prediction without explanation: %v", err)
		} else {
			mlResp = mlRespExplained
		}
	}

	prediction := models.Prediction{
		IsFraud:             isFraud,
		FraudProbability:    mlResp.FraudProbability,
		Confidence:          getConfidenceLevel(mlResp.FraudProbability),
		ReconstructionError: mlResp.ReconstructionError,
	}

	resp := models.PredictResponse{
		TransactionID:    req.TransactionID,
		Prediction:       prediction,
		InferenceTimeMs:  mlResp.InferenceTimeMs,
		ProcessingTimeMs: float64(time.Since(start).Microseconds()) / 1000.0,
		Timestamp:        time.Now().UTC(),
	}

	if isFraud {
		explanation := buildExplanation(mlResp)
		resp.Explanation = explanation
		if h.alertService.IsConfigured() {
			go func() {
				if err := h.alertService.SendFraudAlert(ctx, req.TransactionID, prediction, explanation); err != nil {
					log.Printf("Failed to send fraud alert: %v", err)
				}
			}()
			resp.AlertSent = true
		}
	}

	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) BatchPredict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	var req models.BatchPredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Invalid request body", nil)
		return
	}

	if len(req.Transactions) == 0 {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "No transactions provided", nil)
		return
	}
	if len(req.Transactions) > 100 {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Maximum 100 transactions per batch", nil)
		return
	}

	results := make([]models.PredictResponse, 0, len(req.Transactions))
	var fraudResults []models.PredictResponse
	fraudCount := 0

	for _, txn := range req.Transactions {
		txnStart := time.Now()

		mlReq := &models.MLPredictRequest{Features: txn.Features.ToSlice(), Amount: txn.Amount, Time: txn.Time}

		mlResp, err := h.mlClient.Predict(ctx, mlReq)
		if err != nil {
			log.Printf("ML service error for txn %s: %v", txn.TransactionID, err)
			continue
		}

		isFraud := mlResp.FraudProbability >= h.config.FraudThreshold

		if isFraud {
			mlRespExplained, err := h.mlClient.PredictWithExplanation(ctx, mlReq)
			if err == nil {
				mlResp = mlRespExplained
			}
		}

		prediction := models.Prediction{
			IsFraud:             isFraud,
			FraudProbability:    mlResp.FraudProbability,
			Confidence:          getConfidenceLevel(mlResp.FraudProbability),
			ReconstructionError: mlResp.ReconstructionError,
		}
		result := models.PredictResponse{
			TransactionID:    txn.TransactionID,
			Prediction:       prediction,
			InferenceTimeMs:  mlResp.InferenceTimeMs,
			ProcessingTimeMs: float64(time.Since(txnStart).Microseconds()) / 1000.0,
			Timestamp:        time.Now().UTC(),
		}
		if isFraud {
			fraudCount++
			result.Explanation = buildExplanation(mlResp)
			fraudResults = append(fraudResults, result)
		}
		results = append(results, result)
	}

	alertSent := false
	if fraudCount > 0 && h.alertService.IsConfigured() {
		alertSent = true
		go func() {
			if err := h.alertService.SendBatchFraudAlert(ctx, len(req.Transactions), fraudResults); err != nil {
				log.Printf("Failed to send batch fraud alert: %v", err)
			}
		}()
	}

	resp := models.BatchPredictResponse{
		Results:          results,
		Summary:          models.BatchSummary{Total: len(results), Fraudulent: fraudCount, Legitimate: len(results) - fraudCount},
		AlertSent:        alertSent,
		ProcessingTimeMs: float64(time.Since(start).Microseconds()) / 1000.0,
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) PredictSparkov(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	var req models.SparkovPredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Invalid request body", nil)
		return
	}
	if req.TransactionID == "" {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Request validation failed",
			[]models.ErrorDetail{{Field: "transaction_id", Issue: "required"}})
		return
	}

	mlReq := &models.MLSparkovPredictRequest{
		Amt:           req.Features.Amt,
		TransDatetime: req.Features.TransDatetime,
		Dob:           req.Features.Dob,
		Gender:        req.Features.Gender,
		CityPop:       req.Features.CityPop,
		Lat:           req.Features.Lat,
		Long:          req.Features.Long,
		MerchLat:      req.Features.MerchLat,
		MerchLong:     req.Features.MerchLong,
		Category:      req.Features.Category,
	}

	// Step 1: Fast prediction WITHOUT SHAP
	mlResp, err := h.mlClient.PredictSparkov(ctx, mlReq)
	if err != nil {
		log.Printf("ML service error: %v", err)
		writeError(w, http.StatusServiceUnavailable, "ML_SERVICE_UNAVAILABLE", "ML service is unavailable", nil)
		return
	}

	isFraud := mlResp.FraudProbability >= h.config.FraudThreshold

	// Step 2: Only compute SHAP if fraud detected
	if isFraud {
		mlRespExplained, err := h.mlClient.PredictSparkovWithExplanation(ctx, mlReq)
		if err != nil {
			log.Printf("SHAP explanation failed, using prediction without explanation: %v", err)
		} else {
			mlResp = mlRespExplained
		}
	}

	prediction := models.Prediction{
		IsFraud:             isFraud,
		FraudProbability:    mlResp.FraudProbability,
		Confidence:          getConfidenceLevel(mlResp.FraudProbability),
		ReconstructionError: mlResp.ReconstructionError,
	}

	resp := models.PredictResponse{
		TransactionID:    req.TransactionID,
		Prediction:       prediction,
		InferenceTimeMs:  mlResp.InferenceTimeMs,
		ProcessingTimeMs: float64(time.Since(start).Microseconds()) / 1000.0,
		Timestamp:        time.Now().UTC(),
	}

	if isFraud {
		// For Sparkov, feature names are already human-readable — getFeatureLabel
		// falls through to returning the raw name, so no mapping changes needed.
		explanation := buildExplanation(mlResp)
		resp.Explanation = explanation
		if h.alertService.IsConfigured() {
			go func() {
				if err := h.alertService.SendFraudAlert(ctx, req.TransactionID, prediction, explanation); err != nil {
					log.Printf("Failed to send fraud alert: %v", err)
				}
			}()
			resp.AlertSent = true
		}
	}

	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) Explain(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	var req models.ExplainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Invalid request body", nil)
		return
	}
	if req.TransactionID == "" {
		writeError(w, http.StatusBadRequest, "VALIDATION_ERROR", "Request validation failed",
			[]models.ErrorDetail{{Field: "transaction_id", Issue: "required"}})
		return
	}
	mlReq := &models.MLPredictRequest{Features: req.Features.ToSlice(), Amount: req.Amount, Time: req.Time}
	mlResp, err := h.mlClient.PredictWithExplanation(ctx, mlReq)
	if err != nil {
		log.Printf("ML service error: %v", err)
		writeError(w, http.StatusServiceUnavailable, "ML_SERVICE_UNAVAILABLE", "ML service is unavailable", nil)
		return
	}
	explanation := buildExplanation(mlResp)
	resp := models.ExplainResponse{TransactionID: req.TransactionID, Explanation: *explanation}
	writeJSON(w, http.StatusOK, resp)
}

func validatePredictRequest(req *models.PredictRequest) []models.ErrorDetail {
	var errors []models.ErrorDetail
	if req.TransactionID == "" {
		errors = append(errors, models.ErrorDetail{Field: "transaction_id", Issue: "required"})
	}
	if req.Amount < 0 {
		errors = append(errors, models.ErrorDetail{Field: "amount", Issue: "must be non-negative"})
	}
	return errors
}

func getConfidenceLevel(probability float64) string {
	if probability > 0.9 || probability < 0.1 {
		return "high"
	} else if probability > 0.7 || probability < 0.3 {
		return "medium"
	}
	return "low"
}

func buildExplanation(mlResp *models.MLPredictResponse) *models.Explanation {
	if mlResp.ShapValues == nil {
		return nil
	}
	var contributions []models.FeatureContribution
	for feature, value := range mlResp.ShapValues {
		direction := "decreases_fraud"
		if value > 0 {
			direction = "increases_fraud"
		}
		contributions = append(contributions, models.FeatureContribution{
			Feature:      feature,
			Contribution: value,
			Direction:    direction,
		})
	}
	sort.Slice(contributions, func(i, j int) bool {
		absI, absJ := contributions[i].Contribution, contributions[j].Contribution
		if absI < 0 {
			absI = -absI
		}
		if absJ < 0 {
			absJ = -absJ
		}
		return absI > absJ
	})
	topN := 5
	if len(contributions) < topN {
		topN = len(contributions)
	}
	topFeatures := contributions[:topN]

	var topIncreasing []string
	for _, f := range topFeatures {
		if f.Direction == "increases_fraud" {
			topIncreasing = append(topIncreasing, f.Feature)
			if len(topIncreasing) >= 2 {
				break
			}
		}
	}
	summary := "High fraud probability"
	if len(topIncreasing) > 0 {
		summary += " driven primarily by " + topIncreasing[0]
		if len(topIncreasing) > 1 {
			summary += " and " + topIncreasing[1]
		}
	}
	return &models.Explanation{TopFeatures: topFeatures, ShapValues: mlResp.ShapValues, BaseValue: mlResp.BaseValue, Summary: summary}
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, code, message string, details []models.ErrorDetail) {
	resp := models.ErrorResponse{
		Error:     models.APIError{Code: code, Message: message, Details: details},
		Timestamp: time.Now().UTC(),
	}
	writeJSON(w, status, resp)
}
