.PHONY: build up down logs test test-go test-python

# ── Docker ────────────────────────────────────────────────────────────────────

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# ── Tests ─────────────────────────────────────────────────────────────────────

test-go:
	cd go-api && go test ./... -v

test-python:
	cd python-ml-service && python -m pytest test_ml_service.py -v

test: test-go test-python

# ── Training pipeline (run in order) ─────────────────────────────────────────

train-preprocess:
	cd training && python credit_card_eda_preprocessing.py

train-autoencoder:
	cd training && python train_autoencoder.py

train-generate-features:
	cd training && python generate_ae_features.py

train-classifier:
	cd training && python train_classifier.py

train-all: train-preprocess train-autoencoder train-generate-features train-classifier
