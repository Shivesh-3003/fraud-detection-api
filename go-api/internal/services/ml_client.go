package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"fraud-detection-api/internal/models"
)

type MLClient struct {
	baseURL    string
	httpClient *http.Client
}

func NewMLClient(baseURL string, timeout time.Duration) *MLClient {
	return &MLClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

func (c *MLClient) Predict(ctx context.Context, req *models.MLPredictRequest) (*models.MLPredictResponse, error) {
	return c.post(ctx, "/predict", req)
}

func (c *MLClient) PredictWithExplanation(ctx context.Context, req *models.MLPredictRequest) (*models.MLPredictResponse, error) {
	return c.post(ctx, "/predict?explain=true", req)
}

func (c *MLClient) HealthCheck(ctx context.Context) error {
	url := c.baseURL + "/health"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("creating health check request: %w", err)
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("ML service unreachable: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ML service unhealthy: status %d", resp.StatusCode)
	}
	return nil
}

func (c *MLClient) post(ctx context.Context, path string, reqBody interface{}) (*models.MLPredictResponse, error) {
	url := c.baseURL + path
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling ML service: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}
	var mlResp models.MLPredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&mlResp); err != nil {
		return nil, fmt.Errorf("decoding ML response: %w", err)
	}
	return &mlResp, nil
}
