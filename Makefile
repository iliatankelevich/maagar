.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync dependencies (incl. dev) via uv
	uv sync

fmt: ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint with ruff (no changes)
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## Type-check with pyright
	uv run pyright

# No database needed. Everything here is about the eviction rule and the tenant type; the
# integration coverage — provisioning, RLS, both placements — lives in the consumer's isolation
# suite, which is the only place it can be exercised against real entities.
test: ## Run the suite
	uv run pytest -q

check: lint typecheck test ## Every quality gate
