package models

import "time"

type TransactionFeatures struct {
	V1  float64 `json:"V1"`
	V2  float64 `json:"V2"`
	V3  float64 `json:"V3"`
	V4  float64 `json:"V4"`
	V5  float64 `json:"V5"`
	V6  float64 `json:"V6"`
	V7  float64 `json:"V7"`
	V8  float64 `json:"V8"`
	V9  float64 `json:"V9"`
	V10 float64 `json:"V10"`
	V11 float64 `json:"V11"`
	V12 float64 `json:"V12"`
	V13 float64 `json:"V13"`
	V14 float64 `json:"V14"`
	V15 float64 `json:"V15"`
	V16 float64 `json:"V16"`
	V17 float64 `json:"V17"`
	V18 float64 `json:"V18"`
	V19 float64 `json:"V19"`
	V20 float64 `json:"V20"`
	V21 float64 `json:"V21"`
	V22 float64 `json:"V22"`
	V23 float64 `json:"V23"`
	V24 float64 `json:"V24"`
	V25 float64 `json:"V25"`
	V26 float64 `json:"V26"`
	V27 float64 `json:"V27"`
	V28 float64 `json:"V28"`
}

func (f *TransactionFeatures) ToSlice() []float64 {
	return []float64{
		f.V1, f.V2, f.V3, f.V4, f.V5, f.V6, f.V7,
		f.V8, f.V9, f.V10, f.V11, f.V12, f.V13, f.V14,
		f.V15, f.V16, f.V17, f.V18, f.V19, f.V20, f.V21,
		f.V22, f.V23, f.V24, f.V25, f.V26, f.V27, f.V28,
	}
}

type PredictRequest struct {
	TransactionID string              `json:"transaction_id"`
	Amount        float64             `json:"amount"`
	Time          float64             `json:"time"`
	Features      TransactionFeatures `json:"features"`
}

type BatchPredictRequest struct {
	Transactions []PredictRequest `json:"transactions"`
}

type ExplainRequest struct {
	TransactionID string              `json:"transaction_id"`
	Amount        float64             `json:"amount"`
	Time          float64             `json:"time"`
	Features      TransactionFeatures `json:"features"`
}

type Prediction struct {
	IsFraud             bool    `json:"is_fraud"`
	FraudProbability    float64 `json:"fraud_probability"`
	Confidence          string  `json:"confidence"`
	ReconstructionError float64 `json:"reconstruction_error"`
}

type FeatureContribution struct {
	Feature      string  `json:"feature"`
	Contribution float64 `json:"contribution"`
	Direction    string  `json:"direction"`
}

type Explanation struct {
	TopFeatures []FeatureContribution `json:"top_features"`
	ShapValues  map[string]float64    `json:"shap_values,omitempty"`
	BaseValue   float64               `json:"base_value,omitempty"`
	Summary     string                `json:"summary"`
}

type PredictResponse struct {
	TransactionID    string       `json:"transaction_id"`
	Prediction       Prediction   `json:"prediction"`
	Explanation      *Explanation `json:"explanation,omitempty"`
	AlertSent        bool         `json:"alert_sent,omitempty"`
	InferenceTimeMs  float64      `json:"inference_time_ms"`
	ProcessingTimeMs float64      `json:"processing_time_ms"`
	Timestamp        time.Time    `json:"timestamp"`
}

type BatchSummary struct {
	Total      int `json:"total"`
	Fraudulent int `json:"fraudulent"`
	Legitimate int `json:"legitimate"`
}

type BatchPredictResponse struct {
	Results          []PredictResponse `json:"results"`
	Summary          BatchSummary      `json:"summary"`
	AlertSent        bool              `json:"alert_sent,omitempty"`
	ProcessingTimeMs float64           `json:"processing_time_ms"`
}

type ExplainResponse struct {
	TransactionID string      `json:"transaction_id"`
	Explanation   Explanation `json:"explanation"`
}

type ServiceStatus struct {
	GoAPI     string `json:"go_api"`
	MLService string `json:"ml_service"`
}

type HealthResponse struct {
	Status    string        `json:"status"`
	Timestamp time.Time     `json:"timestamp"`
	Services  ServiceStatus `json:"services"`
}

type ErrorDetail struct {
	Field string `json:"field"`
	Issue string `json:"issue"`
}

type APIError struct {
	Code    string        `json:"code"`
	Message string        `json:"message"`
	Details []ErrorDetail `json:"details,omitempty"`
}

type ErrorResponse struct {
	Error     APIError  `json:"error"`
	Timestamp time.Time `json:"timestamp"`
}

type MLPredictRequest struct {
	Features []float64 `json:"features"`
	Amount   float64   `json:"amount"`
	Time     float64   `json:"time"`
}

// SparkovTransactionFeatures holds the raw named fields from the Sparkov dataset.
// Used for the /api/v1/predict/sparkov endpoint (explainability demo).
type SparkovTransactionFeatures struct {
	Amt           float64 `json:"amt"`
	TransDatetime string  `json:"trans_datetime"`
	Dob           string  `json:"dob"`
	Gender        string  `json:"gender"`
	CityPop       int     `json:"city_pop"`
	Lat           float64 `json:"lat"`
	Long          float64 `json:"long"`
	MerchLat      float64 `json:"merch_lat"`
	MerchLong     float64 `json:"merch_long"`
	Category      string  `json:"category"`
}

type SparkovPredictRequest struct {
	TransactionID string                     `json:"transaction_id"`
	Features      SparkovTransactionFeatures `json:"features"`
}

// MLSparkovPredictRequest is what the Go API sends to the Python ML service
// when operating in Sparkov (named-feature) mode.
type MLSparkovPredictRequest struct {
	Amt           float64 `json:"amt"`
	TransDatetime string  `json:"trans_datetime"`
	Dob           string  `json:"dob"`
	Gender        string  `json:"gender"`
	CityPop       int     `json:"city_pop"`
	Lat           float64 `json:"lat"`
	Long          float64 `json:"long"`
	MerchLat      float64 `json:"merch_lat"`
	MerchLong     float64 `json:"merch_long"`
	Category      string  `json:"category"`
}

type MLPredictResponse struct {
	IsFraud             bool               `json:"is_fraud"`
	FraudProbability    float64            `json:"fraud_probability"`
	ReconstructionError float64            `json:"reconstruction_error"`
	InferenceTimeMs     float64            `json:"inference_time_ms"`
	ShapValues          map[string]float64 `json:"shap_values,omitempty"`
	BaseValue           float64            `json:"base_value,omitempty"`
}
