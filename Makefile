.DEFAULT_GOAL := help
PY ?= python3

# Prefer the project virtualenv if present, so `make ci` works without activating it.
VENV := $(CURDIR)/.venv
ifneq ($(wildcard $(VENV)/bin/python),)
  PY := $(VENV)/bin/python
  BIN := $(VENV)/bin/
endif

.PHONY: help setup install install-dev seed ingest dbt-build ml ai dashboard demo test lint format typecheck ci all clean

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
	$(BIN)pytest

lint: ## Lint with ruff
	$(BIN)ruff check .

format: ## Auto-format with ruff
	$(BIN)ruff format .

typecheck: ## Type-check with mypy
	$(BIN)mypy

setup: ## One-time local setup: create .venv (needs `brew install python@3.11`) + install all extras
	"$$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[all]"
	@echo "Setup complete. Run: make ci"

ci: ## Full local gate — run before every PR; the reviewer runs this too (no GitHub Actions)
	$(BIN)ruff check .
	$(BIN)ruff format --check .
	$(BIN)mypy
	MMI_DUCKDB_PATH=$(CURDIR)/data/ci.duckdb $(PY) -m mmi.cli seed
	MMI_DUCKDB_PATH=$(CURDIR)/data/ci.duckdb $(BIN)dbt build --project-dir transform --profiles-dir transform --target dev
	MMI_DUCKDB_PATH=$(CURDIR)/data/ci.duckdb PYTHONPATH=. $(PY) -c "from dashboard import data; assert not data.assets().empty, 'dashboard cannot read marts'; print('dashboard read-path OK')"
	$(BIN)pytest
	@echo "make ci: PASS"

all: seed dbt-build ml ai ## Run the whole offline pipeline end-to-end

clean: ## Remove generated data + caches
	rm -rf data/*.duckdb data/briefs/*.md transform/target transform/dbt_packages \
		.pytest_cache .ruff_cache .mypy_cache **/__pycache__ 2>/dev/null || true
