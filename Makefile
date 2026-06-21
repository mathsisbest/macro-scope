.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help install install-dev seed ingest dbt-build ml ai dashboard demo test lint format typecheck all clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install core package (editable)
	$(PY) -m pip install -e .

install-dev: ## Install with ml + dashboard + dev extras
	$(PY) -m pip install -e ".[ml,dashboard,dev]"

seed: ## Create a small synthetic sample dataset in DuckDB (no network)
	$(PY) -m mmi.cli seed

ingest: ## Pull live data from free APIs into DuckDB raw schema
	$(PY) -m mmi.cli ingest

dbt-build: ## Run dbt build (staging -> marts) against the DuckDB file
	cd transform && dbt build --profiles-dir .

ml: ## Train + score forecast and regime models
	$(PY) -m mmi.cli ml

ai: ## Generate the GenAI market brief (uses LLM_PROVIDER)
	$(PY) -m mmi.cli ai

dashboard: ## Launch the Streamlit dashboard
	streamlit run dashboard/app.py

demo: seed ## Seed sample data, build marts (if dbt installed), launch dashboard
	-cd transform && dbt build --profiles-dir . 2>/dev/null || echo "dbt not installed; dashboard will read sample marts"
	streamlit run dashboard/app.py

test: ## Run the test suite
	pytest

lint: ## Lint with ruff
	ruff check .

format: ## Auto-format with ruff
	ruff format .

typecheck: ## Type-check with mypy
	mypy

all: seed dbt-build ml ai ## Run the whole offline pipeline end-to-end

clean: ## Remove generated data + caches
	rm -rf data/*.duckdb data/briefs/*.md transform/target transform/dbt_packages \
		.pytest_cache .ruff_cache .mypy_cache **/__pycache__ 2>/dev/null || true
