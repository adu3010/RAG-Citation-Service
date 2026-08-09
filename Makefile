.PHONY: help install dev lint fmt test eval index serve docker docker-run clean
.DEFAULT_GOAL := help

PY ?= python3
export PYTHONPATH := src

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

dev:  ## Install runtime + development dependencies
	$(PY) -m pip install -r requirements-dev.txt

lint:  ## Run ruff checks
	ruff check src tests
	ruff format --check src tests

fmt:  ## Auto-format with ruff
	ruff format src tests
	ruff check --fix src tests

test:  ## Run the unit and API test suite
	$(PY) -m pytest

eval:  ## Run the offline eval harness with regression gates
	$(PY) -m rag.eval.runner --dataset evals/qa_dataset.jsonl \
		--fail-under-recall 0.70 --fail-under-groundedness 0.40

index:  ## Build and persist the index from data/corpus
	$(PY) -m rag --corpus data/corpus --out data/index

serve:  ## Run the API locally with hot reload
	uvicorn rag.api:app --reload --host 0.0.0.0 --port 8000 --app-dir src

docker:  ## Build the container image
	docker build -t rag-citations-service:latest .

docker-run:  ## Run the container image on port 8000
	docker run --rm -p 8000:8000 rag-citations-service:latest

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache data/index evals/results
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
