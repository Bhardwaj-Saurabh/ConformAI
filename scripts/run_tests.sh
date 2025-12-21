#!/bin/bash

# Test Runner Script for ConformAI
# Runs different test suites with appropriate markers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  $1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Parse arguments
TEST_SUITE="all"
COVERAGE=false
VERBOSE=false
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_SUITE="unit"
            MARKERS="-m unit"
            shift
            ;;
        --integration)
            TEST_SUITE="integration"
            MARKERS="-m integration"
            shift
            ;;
        --e2e)
            TEST_SUITE="e2e"
            MARKERS="-m e2e"
            shift
            ;;
        --rag)
            MARKERS="-m rag"
            shift
            ;;
        --data-pipeline)
            MARKERS="-m data_pipeline"
            shift
            ;;
        --api)
            MARKERS="-m api"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./scripts/run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit              Run unit tests only"
            echo "  --integration       Run integration tests only"
            echo "  --e2e               Run end-to-end tests only"
            echo "  --rag               Run RAG pipeline tests"
            echo "  --data-pipeline     Run data pipeline tests"
            echo "  --api               Run API tests"
            echo "  --coverage          Generate coverage report"
            echo "  --verbose, -v       Verbose output"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./scripts/run_tests.sh --unit --coverage"
            echo "  ./scripts/run_tests.sh --e2e --verbose"
            echo "  ./scripts/run_tests.sh --rag"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Display test configuration
print_header "ConformAI Test Runner"
echo "Test Suite: $TEST_SUITE"
echo "Coverage: $COVERAGE"
echo "Verbose: $VERBOSE"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed"
    echo "Install with: pip install pytest pytest-cov pytest-asyncio"
    exit 1
fi

print_success "pytest found"

# Build pytest command
PYTEST_CMD="pytest"

# Add markers
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=services --cov=shared --cov-report=html --cov-report=term-missing"
fi

# Run tests
print_header "Running Tests"
echo "Command: $PYTEST_CMD"
echo ""

if eval $PYTEST_CMD; then
    echo ""
    print_success "All tests passed!"

    # Display coverage report location if generated
    if [ "$COVERAGE" = true ]; then
        echo ""
        print_info "Coverage report generated at: htmlcov/index.html"
        echo "  View with: open htmlcov/index.html"
    fi

    exit 0
else
    echo ""
    print_error "Tests failed!"
    exit 1
fi
