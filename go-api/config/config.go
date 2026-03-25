package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	Port             string
	Env              string
	MLServiceURL     string
	MLServiceTimeout time.Duration
	SlackWebhookURL  string
	AlertEmail       string
	SMTPHost         string
	SMTPPort         string
	SMTPUsername     string
	SMTPPassword     string
	AlertOnFraud     bool
	FraudThreshold   float64
}

func Load() *Config {
	return &Config{
		Port:             getEnv("PORT", "8080"),
		Env:              getEnv("ENV", "development"),
		MLServiceURL:     getEnv("ML_SERVICE_URL", "http://localhost:8000"),
		MLServiceTimeout: getDurationEnv("ML_SERVICE_TIMEOUT", 5*time.Second),
		SlackWebhookURL:  getEnv("SLACK_WEBHOOK_URL", ""),
		AlertEmail:       getEnv("ALERT_EMAIL", ""),
		SMTPHost:         getEnv("SMTP_HOST", ""),
		SMTPPort:         getEnv("SMTP_PORT", "587"),
		SMTPUsername:     getEnv("SMTP_USERNAME", ""),
		SMTPPassword:     getEnv("SMTP_PASSWORD", ""),
		AlertOnFraud:     getBoolEnv("ALERT_ON_FRAUD", true),
		FraudThreshold:   getFloatEnv("FRAUD_THRESHOLD", 0.5),
	}
}

func (c *Config) IsDevelopment() bool { return c.Env == "development" }

func (c *Config) IsAlertingEnabled() bool {
	return c.AlertOnFraud && (c.SlackWebhookURL != "" || c.AlertEmail != "")
}

func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}

func getBoolEnv(key string, defaultValue bool) bool {
	if value, exists := os.LookupEnv(key); exists {
		if parsed, err := strconv.ParseBool(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getFloatEnv(key string, defaultValue float64) float64 {
	if value, exists := os.LookupEnv(key); exists {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if value, exists := os.LookupEnv(key); exists {
		if parsed, err := time.ParseDuration(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}
