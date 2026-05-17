# F1 RAG developer shortcuts
.DEFAULT_GOAL := help

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Local dev
# ---------------------------------------------------------------------------
.PHONY: venv
venv: ## Create venv + install dev deps
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: test
test: ## Run pytest
	$(PY) -m pytest -q

.PHONY: lint
lint: ## Ruff lint
	$(VENV)/bin/ruff check .

.PHONY: fmt
fmt: ## Auto-fix lint
	$(VENV)/bin/ruff check . --fix

# ---------------------------------------------------------------------------
# Compose lifecycle
# ---------------------------------------------------------------------------
.PHONY: up
up: ## docker compose up --build (foreground)
	docker compose up --build

.PHONY: up-d
up-d: ## docker compose up --build (detached)
	docker compose up --build -d

.PHONY: down
down: ## Stop and remove containers
	docker compose down

.PHONY: logs
logs: ## Tail compose logs
	docker compose logs -f --tail=200

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
.PHONY: eval
eval: ## Run the eval harness inside the running api container
	docker compose exec api python -m eval.runner

.PHONY: eval-local
eval-local: ## Run the eval harness locally (needs Qdrant + ANTHROPIC_API_KEY)
	$(PY) -m eval.runner

# ---------------------------------------------------------------------------
# Ingestion (local; usually only needed when iterating outside docker)
# ---------------------------------------------------------------------------
.PHONY: stats
stats: ## Run the stats ETL locally → data/f1_stats.sqlite
	$(PY) -m ingestion.stats_etl

.PHONY: wiki
wiki: ## Run the Wikipedia ETL locally → data/wikipedia_races.jsonl
	$(PY) -m ingestion.wikipedia_etl

.PHONY: regs
regs: ## Run the FIA regs ETL locally → data/fia_regulations.jsonl
	$(PY) -m ingestion.regulations_etl

.PHONY: index
index: ## Build Qdrant indexes locally (needs Qdrant reachable)
	$(PY) -m ingestion.build_indexes
