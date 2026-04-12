package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/smtp"
	"strings"
	"time"

	"fraud-detection-api/config"
	"fraud-detection-api/internal/models"
)

type AlertService struct {
	slackWebhookURL string
	alertEmail      string
	smtpHost        string
	smtpPort        string
	smtpUsername    string
	smtpPassword    string
	httpClient      *http.Client
}

func NewAlertService(cfg *config.Config) *AlertService {
	return &AlertService{
		slackWebhookURL: cfg.SlackWebhookURL,
		alertEmail:      cfg.AlertEmail,
		smtpHost:        cfg.SMTPHost,
		smtpPort:        cfg.SMTPPort,
		smtpUsername:    cfg.SMTPUsername,
		smtpPassword:    cfg.SMTPPassword,
		httpClient:      &http.Client{Timeout: 10 * time.Second},
	}
}

func (s *AlertService) IsConfigured() bool {
	return s.slackWebhookURL != "" || (s.alertEmail != "" && s.smtpHost != "")
}

// Slack types

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

// Single-transaction fraud alert

func (s *AlertService) SendFraudAlert(ctx context.Context, txnID string, prediction models.Prediction, explanation *models.Explanation) error {
	body := s.buildSingleAlertBody(txnID, prediction, explanation)

	var errs []string

	if s.slackWebhookURL != "" {
		msg := SlackMessage{
			Text: "🚨 *Fraud Alert* — Suspicious Transaction Detected",
			Attachments: []Attachment{{
				Color: "#FF0000",
				Title: fmt.Sprintf("Transaction: %s", txnID),
				Fields: []Field{
					{Title: "Fraud Probability", Value: fmt.Sprintf("%.2f%%", prediction.FraudProbability*100), Short: true},
					{Title: "Confidence", Value: prediction.Confidence, Short: true},
					{Title: "Anomaly Score", Value: fmt.Sprintf("%.4f", prediction.ReconstructionError), Short: true},
					{Title: "Why This Was Flagged", Value: s.buildExplanationText(explanation), Short: false},
				},
			}},
		}
		if err := s.sendSlackMessage(context.Background(), msg); err != nil {
			log.Printf("Slack alert failed for %s: %v", txnID, err)
			errs = append(errs, "slack: "+err.Error())
		}
	}

	if s.alertEmail != "" && s.smtpHost != "" {
		subject := fmt.Sprintf("Fraud Alert: Transaction %s flagged (%.1f%%)", txnID, prediction.FraudProbability*100)
		if err := s.sendEmail(subject, body); err != nil {
			log.Printf("Email alert failed for %s: %v", txnID, err)
			errs = append(errs, "email: "+err.Error())
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("alert delivery errors: %s", strings.Join(errs, "; "))
	}
	return nil
}

// Batch summary alert

func (s *AlertService) SendBatchFraudAlert(ctx context.Context, total int, fraudResults []models.PredictResponse) error {
	if len(fraudResults) == 0 {
		return nil
	}

	var lines []string
	var attachments []Attachment
	for _, r := range fraudResults {
		lines = append(lines, fmt.Sprintf("• %s — %.1f%% probability", r.TransactionID, r.Prediction.FraudProbability*100))

		fields := []Field{
			{Title: "Fraud Probability", Value: fmt.Sprintf("%.2f%%", r.Prediction.FraudProbability*100), Short: true},
			{Title: "Confidence", Value: r.Prediction.Confidence, Short: true},
			{Title: "Anomaly Score", Value: fmt.Sprintf("%.4f", r.Prediction.ReconstructionError), Short: true},
		}
		if r.Explanation != nil {
			fields = append(fields, Field{Title: "Why This Was Flagged", Value: s.buildExplanationText(r.Explanation), Short: false})
		}
		attachments = append(attachments, Attachment{
			Color:  "#FF0000",
			Title:  fmt.Sprintf("Transaction: %s", r.TransactionID),
			Fields: fields,
		})
	}
	summary := fmt.Sprintf("🚨 Batch Fraud Alert — %d of %d transactions flagged\n%s",
		len(fraudResults), total, strings.Join(lines, "\n"))

	var errs []string

	if s.slackWebhookURL != "" {
		msg := SlackMessage{
			Text:        summary,
			Attachments: attachments,
		}
		if err := s.sendSlackMessage(context.Background(), msg); err != nil {
			log.Printf("Slack batch alert failed: %v", err)
			errs = append(errs, "slack: "+err.Error())
		}
	}

	if s.alertEmail != "" && s.smtpHost != "" {
		subject := fmt.Sprintf("Batch Fraud Alert: %d of %d transactions flagged", len(fraudResults), total)
		if err := s.sendEmail(subject, summary); err != nil {
			log.Printf("Email batch alert failed: %v", err)
			errs = append(errs, "email: "+err.Error())
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("batch alert delivery errors: %s", strings.Join(errs, "; "))
	}
	return nil
}

// Internal helpers

func (s *AlertService) buildExplanationText(explanation *models.Explanation) string {
	if explanation == nil || len(explanation.TopFeatures) == 0 {
		return ""
	}
	limit := 3
	if len(explanation.TopFeatures) < limit {
		limit = len(explanation.TopFeatures)
	}
	var parts []string
	for _, f := range explanation.TopFeatures[:limit] {
		direction := "↑"
		if f.Direction == "decreases_fraud" {
			direction = "↓"
		}
		parts = append(parts, fmt.Sprintf("%s %s (%.3f)", direction, f.Feature, f.Contribution))
	}
	return strings.Join(parts, "\n")
}

func (s *AlertService) buildSingleAlertBody(txnID string, prediction models.Prediction, explanation *models.Explanation) string {
	body := fmt.Sprintf(
		"Fraud Alert\n\nTransaction: %s\nFraud Probability: %.2f%%\nConfidence: %s\nAnomaly Score: %.4f",
		txnID,
		prediction.FraudProbability*100,
		prediction.Confidence,
		prediction.ReconstructionError,
	)
	if txt := s.buildExplanationText(explanation); txt != "" {
		body += "\n\nWhy This Was Flagged:\n" + txt
	}
	return body
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

func (s *AlertService) sendEmail(subject, body string) error {
	addr := s.smtpHost + ":" + s.smtpPort

	var auth smtp.Auth
	if s.smtpUsername != "" {
		auth = smtp.PlainAuth("", s.smtpUsername, s.smtpPassword, s.smtpHost)
	}

	from := s.smtpUsername
	if from == "" {
		from = s.alertEmail
	}

	msg := []byte(
		"From: " + from + "\r\n" +
			"To: " + s.alertEmail + "\r\n" +
			"Subject: " + subject + "\r\n" +
			"Content-Type: text/plain; charset=UTF-8\r\n" +
			"\r\n" +
			body + "\r\n",
	)

	if err := smtp.SendMail(addr, auth, from, []string{s.alertEmail}, msg); err != nil {
		return fmt.Errorf("smtp.SendMail: %w", err)
	}
	return nil
}
