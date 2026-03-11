package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"fraud-detection-api/internal/models"
)

type AlertService struct {
	slackWebhookURL string
	httpClient      *http.Client
}

func NewAlertService(slackWebhookURL string) *AlertService {
	return &AlertService{
		slackWebhookURL: slackWebhookURL,
		httpClient:      &http.Client{Timeout: 10 * time.Second},
	}
}

type SlackMessage struct {
	Text        string       `json:"text"`
	Attachments []Attachment `json:"attachments,omitempty"`
}

type Attachment struct {
	Color  string  `json:"color"`
	Title  string  `json:"title"`
	Text   string  `json:"text"`
	Fields []Field `json:"fields,omitempty"`
}

type Field struct {
	Title string `json:"title"`
	Value string `json:"value"`
	Short bool   `json:"short"`
}

func (s *AlertService) SendFraudAlert(ctx context.Context, txnID string, prediction models.Prediction, explanation *models.Explanation) error {
	if s.slackWebhookURL == "" {
		return nil
	}
	var explanationText string
	if explanation != nil && len(explanation.TopFeatures) > 0 {
		var features []string
		limit := 3
		if len(explanation.TopFeatures) < limit {
			limit = len(explanation.TopFeatures)
		}
		for _, f := range explanation.TopFeatures[:limit] {
			direction := "↑"
			if f.Direction == "decreases_fraud" {
				direction = "↓"
			}
			features = append(features, fmt.Sprintf("%s %s (%.3f)", direction, f.Feature, f.Contribution))
		}
		explanationText = strings.Join(features, "\n")
	}
	msg := SlackMessage{
		Text: "🚨 *Fraud Alert* - Suspicious Transaction Detected",
		Attachments: []Attachment{{
			Color: "#FF0000",
			Title: fmt.Sprintf("Transaction: %s", txnID),
			Fields: []Field{
				{Title: "Fraud Probability", Value: fmt.Sprintf("%.2f%%", prediction.FraudProbability*100), Short: true},
				{Title: "Confidence", Value: prediction.Confidence, Short: true},
				{Title: "Reconstruction Error", Value: fmt.Sprintf("%.4f", prediction.ReconstructionError), Short: true},
				{Title: "Top Contributing Features", Value: explanationText, Short: false},
			},
		}},
	}
	return s.sendSlackMessage(ctx, msg)
}

func (s *AlertService) sendSlackMessage(ctx context.Context, msg SlackMessage) error {
	jsonBody, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshaling slack message: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.slackWebhookURL, bytes.NewBuffer(jsonBody))
	if err != nil {
		return fmt.Errorf("creating slack request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("sending slack message: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("slack returned status %d", resp.StatusCode)
	}
	return nil
}

func (s *AlertService) IsConfigured() bool {
	return s.slackWebhookURL != ""
}
