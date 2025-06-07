# Build and Development Guide

This project provides multiple ways to run common development tasks using `uv` package manager.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager installed

## Platform-Specific Usage

### Linux/macOS (using Makefile)

If you have `make` installed:

```bash
# Show all available targets
make help

# Setup development environment
make setup

# Run tests
make test

# Run linting and formatting
make check

# Build the package
make build

# Full CI pipeline
make ci
```

### Windows (using PowerShell script)

Use the provided PowerShell script:

```powershell
# Show all available targets
.\make.ps1 help

# Setup development environment
.\make.ps1 setup

# Run tests
.\make.ps1 test

# Run linting and formatting
.\make.ps1 check

# Build the package
.\make.ps1 build

# Full CI pipeline
.\make.ps1 ci
```

### Universal (using uv directly)

You can also run commands directly with `uv`:

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests

# Run linting
uv run ruff check src tests

# Format code
uv run ruff format src tests

# Build package
uv build
```

## Available Targets

### Environment Management
- **install** - Install production dependencies only
- **install-dev** - Install development dependencies
- **setup** - Setup complete development environment

### Build and Distribution
- **build** - Build both wheel and source distribution
- **build-wheel** - Build wheel distribution only
- **build-sdist** - Build source distribution only
- **publish** - Publish to PyPI (requires credentials)
- **publish-test** - Publish to Test PyPI

### Testing
- **test** - Run tests with pytest
- **test-cov** - Run tests with coverage reporting
- **test-fast** - Run tests in parallel (if pytest-xdist available)

### Code Quality
- **lint** - Run ruff linting checks
- **lint-fix** - Run linting and auto-fix issues
- **format** - Format code with ruff
- **format-check** - Check if code is properly formatted
- **type-check** - Run mypy type checking
- **check** - Run all quality checks (lint + format + type)
- **fix** - Auto-fix linting issues and format code

### Cleaning
- **clean** - Remove build artifacts, cache files
- **clean-all** - Remove everything including virtual environment

### Development
- **run** - Run the CLI application (`codin`)
- **run-dev** - Run application in development mode
- **examples** - Run all example scripts

### Utilities
- **upgrade** - Upgrade all dependencies to latest versions
- **lock** - Update the lock file (`uv.lock`)
- **tree** - Show dependency tree
- **info** - Show project and environment information
- **env** - Show detailed environment information

### Complete Workflows
- **ci** - Complete CI pipeline: clean → install → check → test
- **release** - Release preparation: clean → check → test → build

## Development Workflow

### Quick Start
```bash
# Clone and setup
git clone <repository>
cd codin

# Setup development environment
make setup          # Linux/macOS
.\make.ps1 setup     # Windows

# Alternatively using pip
python -m pip install -e .[dev]

# Run tests to verify everything works
make test           # Linux/macOS
.\make.ps1 test     # Windows
```

### Daily Development
```bash
# Before committing changes
make check          # Run all quality checks
make test-cov       # Run tests with coverage

# Fix any issues
make fix            # Auto-fix formatting and linting
```

### Release Process
```bash
# Prepare release
make release        # Runs complete validation and builds

# Publish to Test PyPI first
make publish-test

# After testing, publish to PyPI
make publish
```

## IDE Integration

Both the Makefile and PowerShell script are designed to be IDE-friendly. You can:

1. **VS Code**: Add tasks in `.vscode/tasks.json` that call these targets
2. **PyCharm**: Create run configurations that execute the targets
3. **Terminal**: Use directly from integrated terminals

## Troubleshooting

### Virtual Environment Issues
```bash
# Clean and recreate environment
make clean-all
make setup
```

### Dependency Conflicts
```bash
# Update lock file
make lock

# Upgrade all dependencies
make upgrade
```

### Build Issues
```bash
# Clean build artifacts
make clean

# Rebuild
make build
```

## Adding New Dependencies

```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --group dev package-name

# Update lock file
make lock
``` 