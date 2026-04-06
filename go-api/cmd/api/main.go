package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"fraud-detection-api/config"
	"fraud-detection-api/internal/handlers"
	"fraud-detection-api/internal/middleware"
	"fraud-detection-api/internal/services"
)

func main() {
	// 1. Load config
	cfg := config.Load()
	log.Printf("Starting Fraud Detection API (env: %s)", cfg.Env)

	// 2. Create service dependencies
	mlClient := services.NewMLClient(cfg.MLServiceURL, cfg.MLServiceTimeout)
	alertService := services.NewAlertService(cfg)

	// 3. Create handler (injecting dependencies)
	h := handlers.NewHandler(cfg, mlClient, alertService)

	// 4. Register routes
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", h.HealthCheck)
	mux.HandleFunc("POST /api/v1/predict", h.Predict)
	mux.HandleFunc("POST /api/v1/batch", h.BatchPredict)
	mux.HandleFunc("POST /api/v1/explain", h.Explain)
	mux.HandleFunc("POST /api/v1/predict/sparkov", h.PredictSparkov)

	// 5. Wrap with middleware (applied in reverse order)
	var handler http.Handler = mux
	handler = middleware.Logging(handler)
	handler = middleware.CORS(handler)
	handler = middleware.Recovery(handler)

	// 6. Create and start the server
	server := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      handler,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		log.Printf("Server listening on :%s", cfg.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// 7. Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}
	log.Println("Server stopped")
}
