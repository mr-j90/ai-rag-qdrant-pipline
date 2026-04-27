.DEFAULT_GOAL := help

UV     := uv run
COMPOSE := docker compose

.PHONY: help install qdrant qdrant-down api ui dagster dev down ingest test lint clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Sync deps (dev + dagster groups)
	uv sync --group dev --group dagster

qdrant: ## Start Qdrant via docker compose (idempotent)
	$(COMPOSE) up -d qdrant
	@echo "Qdrant dashboard: http://localhost:6333/dashboard"

qdrant-down: ## Stop Qdrant
	$(COMPOSE) down

api: qdrant ## Run FastAPI on :8000 (foreground)
	$(UV) uvicorn src.api.main:app --reload --port 8000

ui: ## Run Streamlit UI on :8501 (foreground)
	$(UV) streamlit run scripts/ui.py

dagster: qdrant ## Run Dagster dev server on :3000 (foreground)
	$(UV) dagster dev

dev: qdrant ## Run API + UI + Dagster together; Ctrl-C stops all
	@echo "API: http://localhost:8000  |  UI: http://localhost:8501  |  Dagster: http://localhost:3000"
	@echo "Ctrl-C to stop all three."
	@trap 'kill 0' INT TERM; \
	  ( $(UV) uvicorn src.api.main:app --reload --port 8000 2>&1 | awk '{print "[api]     " $$0; fflush()}' ) & \
	  ( $(UV) streamlit run scripts/ui.py 2>&1            | awk '{print "[ui]      " $$0; fflush()}' ) & \
	  ( $(UV) dagster dev 2>&1                            | awk '{print "[dagster] " $$0; fflush()}' ) & \
	  wait

down: qdrant-down ## Stop Qdrant (foreground apps are stopped via Ctrl-C)

ingest: ## Ingest PDFs from ./data/pdfs into Qdrant
	$(UV) python -m scripts.ingest_cli ./data/pdfs

test: ## Run pytest
	$(UV) pytest

lint: ## Lint with ruff
	$(UV) ruff check .

clean: ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
