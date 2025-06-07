# PowerShell equivalent of Makefile for Windows users
# Usage: .\make.ps1 <target>

param(
    [Parameter(Position=0)]
    [string]$Target = "help"
)

# Project configuration
$PROJECT_NAME = "codin"
$PYTHON_VERSION = "3.13"
$SRC_DIR = "src"
$TESTS_DIR = "tests"
$DOCS_DIR = "docs"
$EXAMPLES_DIR = "examples"

# Colors for output
$Blue = "`e[34m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$Reset = "`e[0m"

function Write-ColorOutput {
    param($Color, $Message)
    Write-Host "$Color$Message$Reset"
}

function Show-Help {
    Write-ColorOutput $Blue "$PROJECT_NAME - Development Commands"
    Write-Host ""
    Write-ColorOutput $Green "Environment Management:"
    Write-Host "  install          Install production dependencies"
    Write-Host "  install-dev      Install development dependencies"
    Write-Host "  setup            Setup development environment"
    Write-Host ""
    Write-ColorOutput $Green "Build and Distribution:"
    Write-Host "  build            Build the package"
    Write-Host "  build-wheel      Build wheel distribution only"
    Write-Host "  build-sdist      Build source distribution only"
    Write-Host "  publish          Publish to PyPI"
    Write-Host "  publish-test     Publish to Test PyPI"
    Write-Host ""
    Write-ColorOutput $Green "Testing:"
    Write-Host "  test             Run tests"
    Write-Host "  test-cov         Run tests with coverage"
    Write-Host "  test-fast        Run tests in parallel"
    Write-Host ""
    Write-ColorOutput $Green "Code Quality:"
    Write-Host "  lint             Run linting checks"
    Write-Host "  lint-fix         Run linting checks and fix issues"
    Write-Host "  format           Format code"
    Write-Host "  format-check     Check code formatting"
    Write-Host "  type-check       Run type checking"
    Write-Host "  check            Run all code quality checks"
    Write-Host "  fix              Fix linting issues and format code"
    Write-Host ""
    Write-ColorOutput $Green "Cleaning:"
    Write-Host "  clean            Clean build artifacts"
    Write-Host "  clean-all        Clean all artifacts including venv"
    Write-Host ""
    Write-ColorOutput $Green "Development:"
    Write-Host "  run              Run the CLI application"
    Write-Host "  run-dev          Run in development mode"
    Write-Host "  examples         Run example scripts"
    Write-Host ""
    Write-ColorOutput $Green "Utilities:"
    Write-Host "  upgrade          Upgrade all dependencies"
    Write-Host "  lock             Update lock file"
    Write-Host "  tree             Show dependency tree"
    Write-Host "  info             Show project information"
    Write-Host "  env              Show environment information"
    Write-Host ""
    Write-ColorOutput $Green "Complete Workflows:"
    Write-Host "  ci               Run CI pipeline"
    Write-Host "  release          Prepare for release"
    Write-Host ""
}

function Invoke-Install {
    Write-ColorOutput $Blue "Installing production dependencies..."
    uv sync --no-dev
}

function Invoke-InstallDev {
    Write-ColorOutput $Blue "Installing development dependencies..."
    uv sync
}

function Invoke-Setup {
    Write-ColorOutput $Blue "Setting up development environment..."
    uv sync
    Write-ColorOutput $Green "✓ Virtual environment created and dependencies installed"
    Write-ColorOutput $Yellow "Run '.venv\Scripts\activate' to activate the environment"
}

function Invoke-Build {
    Invoke-Clean
    Write-ColorOutput $Blue "Building package..."
    uv build
}

function Invoke-BuildWheel {
    Invoke-Clean
    Write-ColorOutput $Blue "Building wheel..."
    uv build --wheel
}

function Invoke-BuildSdist {
    Invoke-Clean
    Write-ColorOutput $Blue "Building source distribution..."
    uv build --sdist
}

function Invoke-Publish {
    Invoke-Build
    Write-ColorOutput $Blue "Publishing to PyPI..."
    uv publish
}

function Invoke-PublishTest {
    Invoke-Build
    Write-ColorOutput $Blue "Publishing to Test PyPI..."
    uv publish --index-url https://test.pypi.org/simple/
}

function Invoke-Test {
    Write-ColorOutput $Blue "Running tests..."
    uv run pytest $TESTS_DIR
}

function Invoke-TestCov {
    Write-ColorOutput $Blue "Running tests with coverage..."
    uv run pytest $TESTS_DIR --cov=$SRC_DIR --cov-report=html --cov-report=term-missing
}

function Invoke-TestFast {
    Write-ColorOutput $Blue "Running tests in parallel..."
    try {
        uv run pytest $TESTS_DIR -n auto
    } catch {
        uv run pytest $TESTS_DIR
    }
}

function Invoke-Lint {
    Write-ColorOutput $Blue "Running linting checks..."
    uv run ruff check $SRC_DIR $TESTS_DIR
}

function Invoke-LintFix {
    Write-ColorOutput $Blue "Running linting checks and fixing issues..."
    uv run ruff check --fix $SRC_DIR $TESTS_DIR
}

function Invoke-Format {
    Write-ColorOutput $Blue "Formatting code..."
    uv run ruff format $SRC_DIR $TESTS_DIR
}

function Invoke-FormatCheck {
    Write-ColorOutput $Blue "Checking code formatting..."
    uv run ruff format --check $SRC_DIR $TESTS_DIR
}

function Invoke-TypeCheck {
    Write-ColorOutput $Blue "Running type checks..."
    uv run mypy $SRC_DIR
}

function Invoke-Check {
    Invoke-Lint
    Invoke-FormatCheck
    Invoke-TypeCheck
    Write-ColorOutput $Green "✓ All code quality checks passed"
}

function Invoke-Fix {
    Invoke-LintFix
    Invoke-Format
    Write-ColorOutput $Green "✓ Code fixed and formatted"
}

function Invoke-Clean {
    Write-ColorOutput $Blue "Cleaning build artifacts..."
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    Get-ChildItem -Path "." -Filter "*.egg-info" -Directory | Remove-Item -Recurse -Force
    if (Test-Path ".pytest_cache") { Remove-Item -Recurse -Force ".pytest_cache" }
    if (Test-Path ".coverage") { Remove-Item -Force ".coverage" }
    if (Test-Path "htmlcov") { Remove-Item -Recurse -Force "htmlcov" }
    if (Test-Path ".mypy_cache") { Remove-Item -Recurse -Force ".mypy_cache" }
    if (Test-Path ".ruff_cache") { Remove-Item -Recurse -Force ".ruff_cache" }
    Get-ChildItem -Path "." -Recurse -Name "*.pyc" | Remove-Item -Force
    Get-ChildItem -Path "." -Recurse -Name "__pycache__" -Directory | Remove-Item -Recurse -Force
}

function Invoke-CleanAll {
    Invoke-Clean
    Write-ColorOutput $Blue "Cleaning all artifacts..."
    if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
}

function Invoke-Run {
    Write-ColorOutput $Blue "Running $PROJECT_NAME..."
    uv run codin
}

function Invoke-RunDev {
    Write-ColorOutput $Blue "Running $PROJECT_NAME in development mode..."
    uv run python -m codin.cli.commands
}

function Invoke-Examples {
    Write-ColorOutput $Blue "Running examples..."
    if (Test-Path $EXAMPLES_DIR) {
        Get-ChildItem -Path $EXAMPLES_DIR -Filter "*.py" | ForEach-Object {
            Write-ColorOutput $Yellow "Running $($_.FullName)..."
            try {
                uv run python $_.FullName
            } catch {
                Write-Host "Example failed, continuing..."
            }
        }
    } else {
        Write-ColorOutput $Yellow "No examples directory found."
    }
}

function Invoke-Upgrade {
    Write-ColorOutput $Blue "Upgrading dependencies..."
    uv sync --upgrade
}

function Invoke-Lock {
    Write-ColorOutput $Blue "Updating lock file..."
    uv lock
}

function Invoke-Tree {
    Write-ColorOutput $Blue "Dependency tree:"
    uv tree
}

function Invoke-Info {
    Write-ColorOutput $Blue "Project Information:"
    Write-Host "Name: $PROJECT_NAME"
    Write-Host "Python Version: $PYTHON_VERSION"
    Write-Host "Source Directory: $SRC_DIR"
    Write-Host "Tests Directory: $TESTS_DIR"
    Write-Host ""
    Write-ColorOutput $Blue "UV Information:"
    uv --version
    Write-Host ""
    Write-ColorOutput $Blue "Virtual Environment:"
    uv python list
}

function Invoke-Env {
    Write-ColorOutput $Blue "Environment Information:"
    uv run python --version
    uv run python -c "import sys; print('Python executable:', sys.executable)"
    uv run python -c "import sys; print('Python path:', sys.path)"
}

function Invoke-CI {
    Invoke-Clean
    Invoke-InstallDev
    Invoke-Check
    Invoke-Test
    Write-ColorOutput $Green "✓ CI pipeline completed successfully"
}

function Invoke-Release {
    Invoke-Clean
    Invoke-Check
    Invoke-Test
    Invoke-Build
    Write-ColorOutput $Green "✓ Release preparation completed"
    Write-ColorOutput $Yellow "Ready to publish with '.\make.ps1 publish' or '.\make.ps1 publish-test'"
}

# Main switch
switch ($Target.ToLower()) {
    "help" { Show-Help }
    "install" { Invoke-Install }
    "install-dev" { Invoke-InstallDev }
    "setup" { Invoke-Setup }
    "build" { Invoke-Build }
    "build-wheel" { Invoke-BuildWheel }
    "build-sdist" { Invoke-BuildSdist }
    "publish" { Invoke-Publish }
    "publish-test" { Invoke-PublishTest }
    "test" { Invoke-Test }
    "test-cov" { Invoke-TestCov }
    "test-fast" { Invoke-TestFast }
    "lint" { Invoke-Lint }
    "lint-fix" { Invoke-LintFix }
    "format" { Invoke-Format }
    "format-check" { Invoke-FormatCheck }
    "type-check" { Invoke-TypeCheck }
    "check" { Invoke-Check }
    "fix" { Invoke-Fix }
    "clean" { Invoke-Clean }
    "clean-all" { Invoke-CleanAll }
    "run" { Invoke-Run }
    "run-dev" { Invoke-RunDev }
    "examples" { Invoke-Examples }
    "upgrade" { Invoke-Upgrade }
    "lock" { Invoke-Lock }
    "tree" { Invoke-Tree }
    "info" { Invoke-Info }
    "env" { Invoke-Env }
    "ci" { Invoke-CI }
    "release" { Invoke-Release }
    default {
        Write-ColorOutput $Red "Unknown target: $Target"
        Write-Host ""
        Show-Help
        exit 1
    }
} 