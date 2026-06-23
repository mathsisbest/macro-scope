.DEFAULT_GOAL := help
PY ?= python3

# Prefer the project virtualenv if present, so `make ci` works without activating it.
VENV := $(CURDIR)/.venv
ifneq ($(wildcard $(VENV)/bin/python),)
  PY := $(VENV)/bin/python
  BIN := $(VENV)/bin/
endif

# The local gate runs strictly against a local DuckDB file — never MotherDuck — even if the
# developer's .env enables MotherDuck. Empty MotherDuck vars force use_motherduck = False.
CI_DB := $(CURDIR)/data/ci.duckdb
CI_ENV := MMI_MOTHERDUCK_DATABASE= MOTHERDUCK_TOKEN= MMI_DUCKDB_PATH=$(CI_DB)

.PHONY: help setup install install-dev seed ingest dbt-build ml ai dashboard demo test lint format typecheck ci all clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install core package (editable)
	$(PY) -m pip install -e .

install-dev: ## Install with ml + dashboard + transform(dbt) + dev extras
	$(PY) -m pip install -e ".[ml,dashboard,transform,dev]"

seed: ## Create a small synthetic sample dataset in DuckDB (no network)
	$(PY) -m mmi.cli seed

ingest: ## Pull live data from free APIs into DuckDB raw schema
	$(PY) -m mmi.cli ingest

dbt-build: ## Run dbt build (staging -> marts) against the local DuckDB file
	$(BIN)dbt build --project-dir transform --profiles-dir transform --target dev

ml: ## Train + score forecast and regime models
	$(PY) -m mmi.cli ml

ai: ## Generate the GenAI market brief (uses LLM_PROVIDER)
	$(PY) -m mmi.cli ai

dashboard: ## Launch the Streamlit dashboard
	PYTHONPATH=$(CURDIR) $(BIN)streamlit run dashboard/app.py

demo: seed ## Seed sample data, build marts (if dbt installed), launch dashboard
	-cd transform && dbt build --profiles-dir . 2>/dev/null || echo "dbt not installed; dashboard will read sample marts"
	PYTHONPATH=$(CURDIR) $(BIN)streamlit run dashboard/app.py

test: ## Run the test suite
	$(BIN)pytest

lint: ## Lint with ruff
	$(BIN)ruff check .

format: ## Auto-format with ruff
	$(BIN)ruff format .

typecheck: ## Type-check with mypy
	$(BIN)mypy

setup: ## One-time local setup: create .venv + install all extras (needs Homebrew python@3.11)
	@command -v brew >/dev/null 2>&1 || { echo "Homebrew not found — install from https://brew.sh, then: brew install python@3.11"; exit 1; }
	@brew --prefix python@3.11 >/dev/null 2>&1 || { echo "python@3.11 not installed — run: brew install python@3.11"; exit 1; }
	"$$(brew --prefix python@3.11)/bin/python3.11" -m venv .venv
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[all]"
	@echo "Setup complete. Run: make ci"

ci: ## Full local gate — run before every PR; the reviewer runs this too (no GitHub Actions)
	$(BIN)ruff check .
	$(BIN)ruff format --check .
	$(BIN)mypy
	$(CI_ENV) $(PY) -m mmi.cli seed
	$(CI_ENV) $(PY) -m mmi.cli portfolio
	$(CI_ENV) $(PY) -c "import duckdb, os; c = duckdb.connect(os.environ['MMI_DUCKDB_PATH']); c.execute('drop schema if exists marts cascade'); c.execute('drop schema if exists staging cascade'); c.close()"
	$(CI_ENV) $(BIN)dbt build --project-dir transform --profiles-dir transform --target dev
	$(CI_ENV) PYTHONPATH=. $(PY) scripts/dashboard_smoke.py
	$(BIN)pytest
	@echo "make ci: PASS"

all: seed dbt-build ml ai ## Run the whole offline pipeline end-to-end

clean: ## Remove generated data + caches
	rm -rf data/*.duckdb data/briefs/*.md transform/target transform/dbt_packages \
		.pytest_cache .ruff_cache .mypy_cache **/__pycache__ 2>/dev/null || true
