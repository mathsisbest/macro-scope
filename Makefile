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
# It also clears the LLM provider keys so `mmi ai` always takes the deterministic offline
# template (no network, no metered API call) — the gate stays hermetic and £0 regardless of .env.
CI_DB := $(CURDIR)/data/ci.duckdb
CI_ENV := MMI_MOTHERDUCK_DATABASE= MOTHERDUCK_TOKEN= MMI_DUCKDB_PATH=$(CI_DB) GEMINI_API_KEY= GROQ_API_KEY= ANTHROPIC_API_KEY=
# The dev DuckDB (matches settings.duckdb_path = REPO_ROOT/data/mmi.duckdb). Absolute, so dbt
# resolves it correctly when run with --project-dir from the repo root (the profile default
# `../data/mmi.duckdb` is relative to transform/ and otherwise resolves one dir too high).
DEV_DB := $(CURDIR)/data/mmi.duckdb

.PHONY: help setup install install-dev seed ingest healthcheck dbt-build ml ai snapshot dashboard demo test lint format typecheck ci all clean app-smoke import-smoke

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

healthcheck: ## Probe every data source for connectivity + key presence
	$(PY) -m mmi.cli healthcheck

dbt-build: ## Run dbt build (staging -> marts) against the local DuckDB file
	MMI_DUCKDB_PATH=$(DEV_DB) $(BIN)dbt build --project-dir transform --profiles-dir transform --target dev

ml: ## Train + score forecast and regime models
	$(PY) -m mmi.cli ml

ai: ## Generate the GenAI market brief (uses LLM_PROVIDER)
	$(PY) -m mmi.cli ai

snapshot: ## Export marts.* to data/public/*.parquet (the public demo's data source)
	$(PY) -m mmi.cli snapshot

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

app-smoke: ## Render smoke: AppTest in populated-DB + empty-snapshot modes (run after `make ci` seed steps)
	$(CI_ENV) PYTHONPATH=. $(PY) scripts/dashboard_app_smoke.py

import-smoke: ## Import guard: fail if sklearn/scipy/dbt reachable at module-scope from dashboard imports
	PYTHONPATH=. $(PY) scripts/public_import_smoke.py

ci: ## Full local gate — run before every PR; the reviewer runs this too (no GitHub Actions)
	$(BIN)ruff check .
	$(BIN)ruff format --check .
	$(BIN)mypy
	$(CI_ENV) $(PY) -m mmi.cli seed
	$(CI_ENV) $(PY) -m mmi.cli portfolio
	$(CI_ENV) $(PY) -c "import duckdb, os; c = duckdb.connect(os.environ['MMI_DUCKDB_PATH']); c.execute('drop schema if exists marts cascade'); c.execute('drop schema if exists staging cascade'); c.close()"
	$(CI_ENV) $(BIN)dbt build --project-dir transform --profiles-dir transform --target dev
	$(CI_ENV) $(PY) -m mmi.cli ml
	$(CI_ENV) $(PY) -m mmi.cli ai
	$(CI_ENV) PYTHONPATH=. $(PY) scripts/dashboard_smoke.py
	$(CI_ENV) PYTHONPATH=. $(PY) scripts/dashboard_app_smoke.py
	$(BIN)pytest
	@echo "make ci: PASS"

all: seed dbt-build ml ai ## Run the whole offline pipeline end-to-end

clean: ## Remove generated data + caches
	rm -rf data/*.duckdb data/briefs/*.md transform/target transform/dbt_packages \
		.pytest_cache .ruff_cache .mypy_cache **/__pycache__ 2>/dev/null || true
