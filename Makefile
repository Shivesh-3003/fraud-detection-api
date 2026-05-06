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

# Training pipeline: see training/train_all.sh (supports ulb|sparkov dataset).
