UV := $(shell if [ -x ./.venv/bin/uv ]; then echo ./.venv/bin/uv; \
	elif command -v uv >/dev/null 2>&1; then echo uv; \
	else echo "python3 -m uv"; fi)
.DEFAULT_GOAL := help
.PHONY: help install install-dev install-full lint lint-unsafe format mypy test docs docs-serve docs-clean lock clean clean-all check coverage-open

help: ## Show available make targets
	@printf "Available targets:\n"
	@printf "  install       \tInstall production deps (uv sync --no-dev)\n"
	@printf "  install-dev   \tInstall dev dependencies (uv sync)\n"
	@printf "  install-full  \tInstall all dependencies including all extras\n"
	@printf "  lint          \tRun ruff on source\n"
	@printf "  lint-unsafe   \tRun ruff with unsafe fixes\n"
	@printf "  format        \tFormat source code via ruff\n"
	@printf "  test          \tRun pytest with coverage\n"
	@printf "  test-light    \tRun unit tests only (no integration/stress)\n"
	@printf "  test-integration \tRun integration tests\n"
	@printf "  test-stress   \tRun stress tests (slow, resource-intensive)\n"
	@printf "  test-all      \tRun all tests including integration and stress\n"
	@printf "  docs          \tBuild docs (requires docs/ tree)\n"
	@printf "  docs-serve    \tLive-reload docs and open browser\n"
	@printf "  docs-clean    \tRemove docs build artifacts\n"
	@printf "  lock          \tRegenerate uv.lock\n"
	@printf "  check         \tLint + mypy\n"
	@printf "  coverage-open \tOpen coverage HTML report in browser\n"
	@printf "  clean         \tRemove caches, build artifacts, and generated files\n"
	@printf "  clean-all     \tRemove everything including virtual environment\n"

install: ## Install production dependencies
	$(UV) sync --no-dev

install-dev: ## Install all dependencies (including dev)
	$(UV) sync --extra dev

install-full: ## Install all dependencies including all extras
	$(UV) sync --extra common --extra dev --extra torch --extra redis --extra viz --extra pandas --extra polars --extra zarr --extra notebooks --extra taichi --extra docs

lint: ## Run ruff on the source tree
	$(UV) run --extra dev ruff check src/diffract --fix

lint-unsafe: ## Run ruff on the source tree
	$(UV) run --extra dev ruff check src/diffract --unsafe-fixes

mypy: ## Run mypy type checks
	@# Best-effort developer aid: show mypy output, but do not fail the build.
	-$(UV) run --extra dev mypy src/diffract

format: ## Format source via ruff
	$(UV) run --extra dev ruff format src/diffract

test: ## Run pytest with coverage
	PYTHONPATH=src $(UV) run --python 3.12.12 --extra dev --extra torch --extra redis --extra frameworks --extra viz --extra pandas --extra polars --extra zarr pytest tests --cov=src/diffract --cov-report=term-missing --cov-report=html

test-light: ## Run unit tests only (no integration/stress)
	PYTHONPATH=src $(UV) run --python 3.12.12 --extra dev --extra viz --extra torch --extra zarr pytest -m "not integration and not stress" tests --cov=src/diffract --cov-report=term-missing

test-integration: ## Run integration tests
	PYTHONPATH=src $(UV) run --python 3.12.12 --extra dev --extra torch --extra redis --extra frameworks --extra viz --extra pandas --extra polars pytest -m "integration and not stress" tests

test-stress: ## Run stress tests (slow, resource-intensive)
	PYTHONPATH=src $(UV) run --python 3.12.12 --extra dev --extra torch --extra redis --extra frameworks --extra viz --extra pandas --extra polars pytest -m "stress" tests

check: ## Run lint and mypy
	$(MAKE) lint
	$(MAKE) mypy

docs: ## Build Sphinx docs (expects docs/ sources)
	$(UV) run --extra docs sphinx-build -b html docs docs/_build/html

docs-serve: docs-clean ## Watch docs and open local server (requires docs extra)
	$(UV) run --extra docs python -m sphinx_autobuild docs docs/_build/html --open-browser -aE

docs-clean: ## Remove docs build artifacts
	rm -rf docs/_build

lock: ## Regenerate uv.lock
	$(UV) lock

coverage-open: ## Open coverage HTML report in browser
	@if [ -f htmlcov/index.html ]; then \
		open htmlcov/index.html; \
	else \
		echo "Coverage report not found. Run 'make test' first to generate it."; \
		exit 1; \
	fi

clean: ## Remove build artifacts, caches, and generated files
	# Cache directories
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov || true
	# Build artifacts
	@rm -rf dist build *.egg-info docs/_build .sphinx || true
	# Python cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	# Jupyter notebooks checkpoints
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	# Log files
	@rm -rf *.log examples/*.log notebooks/*.log || true
	# Generated images in examples
	@rm -rf examples/images || true
	@rm -rf notebooks/images || true

clean-all: ## Remove everything including virtual environment
	$(MAKE) clean
	@rm -rf .venv .env
