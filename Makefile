.PHONY: help install install-dev clean clean-all build test test-cov lint format check docs serve-docs run examples
.DEFAULT_GOAL := help

# Project configuration
PROJECT_NAME := codin
PYTHON_VERSION := 3.13
SRC_DIR := src
TESTS_DIR := tests
DOCS_DIR := docs
EXAMPLES_DIR := examples

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)$(PROJECT_NAME) - Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# Environment Management
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	uv sync --no-dev

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	uv sync

setup: install-dev ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	@echo "$(GREEN)✓ Virtual environment created and dependencies installed$(RESET)"
	@echo "$(YELLOW)Run 'source .venv/bin/activate' (Linux/Mac) or '.venv\\Scripts\\activate' (Windows) to activate the environment$(RESET)"

# Build and Distribution
build: clean ## Build the package
	@echo "$(BLUE)Building package...$(RESET)"
	uv build

build-wheel: clean ## Build wheel distribution only
	@echo "$(BLUE)Building wheel...$(RESET)"
	uv build --wheel

build-sdist: clean ## Build source distribution only
	@echo "$(BLUE)Building source distribution...$(RESET)"
	uv build --sdist

publish: build ## Publish to PyPI (requires proper credentials)
	@echo "$(BLUE)Publishing to PyPI...$(RESET)"
	uv publish

publish-test: build ## Publish to Test PyPI
	@echo "$(BLUE)Publishing to Test PyPI...$(RESET)"
	uv publish --index-url https://test.pypi.org/simple/

# Testing
test: ## Run tests
	@echo "$(BLUE)Running tests...$(RESET)"
	uv run pytest $(TESTS_DIR)

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	uv run pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-fast: ## Run tests in parallel (if pytest-xdist is available)
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	uv run pytest $(TESTS_DIR) -n auto 2>/dev/null || uv run pytest $(TESTS_DIR)

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	uv run pytest-watch $(TESTS_DIR)

# Code Quality
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	uv run ruff check $(SRC_DIR) $(TESTS_DIR)

lint-fix: ## Run linting checks and fix issues
	@echo "$(BLUE)Running linting checks and fixing issues...$(RESET)"
	uv run ruff check --fix $(SRC_DIR) $(TESTS_DIR)

format: ## Format code
	@echo "$(BLUE)Formatting code...$(RESET)"
	uv run ruff format $(SRC_DIR) $(TESTS_DIR)

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	uv run ruff format --check $(SRC_DIR) $(TESTS_DIR)

type-check: ## Run type checking
	@echo "$(BLUE)Running type checks...$(RESET)"
	uv run mypy $(SRC_DIR)

check: lint format-check type-check ## Run all code quality checks
	@echo "$(GREEN)✓ All code quality checks passed$(RESET)"

fix: lint-fix format ## Fix linting issues and format code
	@echo "$(GREEN)✓ Code fixed and formatted$(RESET)"

# Documentation
docs: ## Build documentation (if using Sphinx or similar)
	@echo "$(BLUE)Building documentation...$(RESET)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		cd $(DOCS_DIR) && uv run make html; \
	else \
		echo "$(YELLOW)No docs directory found. Skipping documentation build.$(RESET)"; \
	fi

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation...$(RESET)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		cd $(DOCS_DIR) && uv run make clean; \
	else \
		echo "$(YELLOW)No docs directory found.$(RESET)"; \
	fi

serve-docs: docs ## Build and serve documentation locally
	@echo "$(BLUE)Serving documentation...$(RESET)"
	@if [ -d "$(DOCS_DIR)/_build/html" ]; then \
		cd $(DOCS_DIR)/_build/html && python -m http.server 8000; \
	else \
		echo "$(RED)Documentation not found. Run 'make docs' first.$(RESET)"; \
	fi

# Cleaning
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-all: clean ## Clean all artifacts including virtual environment
	@echo "$(BLUE)Cleaning all artifacts...$(RESET)"
	rm -rf .venv/

# Development
run: ## Run the CLI application
	@echo "$(BLUE)Running $(PROJECT_NAME)...$(RESET)"
	uv run codin

run-dev: ## Run the application in development mode
	@echo "$(BLUE)Running $(PROJECT_NAME) in development mode...$(RESET)"
	uv run python -m codin.cli.commands

examples: ## Run example scripts
	@echo "$(BLUE)Running examples...$(RESET)"
	@if [ -d "$(EXAMPLES_DIR)" ]; then \
		for example in $(EXAMPLES_DIR)/*.py; do \
			echo "$(YELLOW)Running $$example...$(RESET)"; \
			uv run python "$$example" || true; \
		done; \
	else \
		echo "$(YELLOW)No examples directory found.$(RESET)"; \
	fi

# Utilities
upgrade: ## Upgrade all dependencies
	@echo "$(BLUE)Upgrading dependencies...$(RESET)"
	uv sync --upgrade

lock: ## Update lock file
	@echo "$(BLUE)Updating lock file...$(RESET)"
	uv lock

tree: ## Show dependency tree
	@echo "$(BLUE)Dependency tree:$(RESET)"
	uv tree

info: ## Show project information
	@echo "$(BLUE)Project Information:$(RESET)"
	@echo "Name: $(PROJECT_NAME)"
	@echo "Python Version: $(PYTHON_VERSION)"
	@echo "Source Directory: $(SRC_DIR)"
	@echo "Tests Directory: $(TESTS_DIR)"
	@echo ""
	@echo "$(BLUE)UV Information:$(RESET)"
	@uv --version
	@echo ""
	@echo "$(BLUE)Virtual Environment:$(RESET)"
	@uv python list

env: ## Show environment information
	@echo "$(BLUE)Environment Information:$(RESET)"
	uv run python --version
	uv run python -c "import sys; print('Python executable:', sys.executable)"
	uv run python -c "import sys; print('Python path:', sys.path)"

# Security
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	@echo "$(YELLOW)Install bandit with: uv add --group dev bandit$(RESET)"
	@uv run bandit -r $(SRC_DIR) 2>/dev/null || echo "$(YELLOW)Bandit not installed. Run: uv add --group dev bandit$(RESET)"

audit: ## Audit dependencies for known vulnerabilities
	@echo "$(BLUE)Auditing dependencies...$(RESET)"
	@echo "$(YELLOW)Install safety with: uv add --group dev safety$(RESET)"
	@uv run safety check 2>/dev/null || echo "$(YELLOW)Safety not installed. Run: uv add --group dev safety$(RESET)"

# Git hooks (optional)
pre-commit: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(RESET)"
	@echo "$(YELLOW)Install pre-commit with: uv add --group dev pre-commit$(RESET)"
	@uv run pre-commit install 2>/dev/null || echo "$(YELLOW)Pre-commit not installed. Run: uv add --group dev pre-commit$(RESET)"

# Release helpers
version: ## Show current version
	@echo "$(BLUE)Current version:$(RESET)"
	@uv run python -c "from importlib.metadata import version; print(version('$(PROJECT_NAME)'))" 2>/dev/null || echo "Package not installed"

bump-patch: ## Bump patch version (requires bump2version)
	@echo "$(BLUE)Bumping patch version...$(RESET)"
	@uv run bump2version patch 2>/dev/null || echo "$(YELLOW)bump2version not installed. Run: uv add --group dev bump2version$(RESET)"

bump-minor: ## Bump minor version (requires bump2version)
	@echo "$(BLUE)Bumping minor version...$(RESET)"
	@uv run bump2version minor 2>/dev/null || echo "$(YELLOW)bump2version not installed. Run: uv add --group dev bump2version$(RESET)"

bump-major: ## Bump major version (requires bump2version)
	@echo "$(BLUE)Bumping major version...$(RESET)"
	@uv run bump2version major 2>/dev/null || echo "$(YELLOW)bump2version not installed. Run: uv add --group dev bump2version$(RESET)"

# Complete workflow targets
ci: clean install-dev check test ## Run CI pipeline (install, check, test)
	@echo "$(GREEN)✓ CI pipeline completed successfully$(RESET)"

release: clean check test build ## Prepare for release (clean, check, test, build)
	@echo "$(GREEN)✓ Release preparation completed$(RESET)"
	@echo "$(YELLOW)Ready to publish with 'make publish' or 'make publish-test'$(RESET)" 