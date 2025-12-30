#!/bin/bash
#=============================================================================
# Comprehensive RAG Experiment Runner
# Runs experiments for multiple models with both split strategies (recent & modN)
# Supports resume from checkpoint after interruption
#=============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters
MODELS=("bge-small" "bge-large" "gte-large")
BACKEND="postgres"
RESUME=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 ]] && [[ ! $1 == --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --no-resume)
            RESUME=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODEL1 MODEL2 ...    Models to test (default: bge-small bge-large gte-large)"
            echo "                                Available: bge-small, bge-large, bge-m3, gte-qwen2,"
            echo "                                          nomic-embed, gte-large, e5-large"
            echo "  --backend BACKEND             Vector backend (default: postgres)"
            echo "                                Options: postgres, qdrant"
            echo "  --no-resume                   Start fresh (ignore checkpoint)"
            echo "  --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 --models bge-small bge-large       # Run specific models"
            echo "  $0 --backend qdrant                   # Use Qdrant instead of PostgreSQL"
            echo "  $0 --no-resume                        # Start fresh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

#=============================================================================
# Pre-flight Checks
#=============================================================================
echo "================================================================================"
echo "COMPREHENSIVE RAG EXPERIMENT"
echo "================================================================================"
echo ""

log "Configuration:"
echo "  Models: ${MODELS[*]}"
echo "  Backend: $BACKEND"
echo "  Resume: $RESUME"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check podman
if ! command -v podman &> /dev/null; then
    log_error "Podman not found. Please install Podman."
    exit 1
fi

#=============================================================================
# Install Dependencies
#=============================================================================
log "Checking Python dependencies..."

if ! python3 -c "import einops" 2>/dev/null; then
    log_warning "einops not found. Installing..."
    pip3 install einops
fi

if ! python3 -c "import psycopg2" 2>/dev/null; then
    log_warning "psycopg2-binary not found. Installing..."
    pip3 install psycopg2-binary
fi

log_success "Dependencies OK"

#=============================================================================
# Start PostgreSQL (if using postgres backend)
#=============================================================================
if [ "$BACKEND" == "postgres" ]; then
    log "Checking PostgreSQL status..."

    if ! podman ps --filter name=semantic_vectors_db --format "{{.Names}}" | grep -q "^semantic_vectors_db$"; then
        log "Starting PostgreSQL..."
        ./start_postgres.sh

        # Wait for PostgreSQL to be ready
        log "Waiting for PostgreSQL to be ready..."
        sleep 10

        # Verify connection
        if ! podman exec semantic_vectors_db pg_isready -U postgres > /dev/null 2>&1; then
            log_error "PostgreSQL failed to start properly"
            exit 1
        fi

        log_success "PostgreSQL is running"
    else
        log_success "PostgreSQL already running"
    fi
fi

#=============================================================================
# Run Comprehensive Experiments
#=============================================================================
log "Starting comprehensive experiments..."
echo ""

# Build command
CMD="python3 run_comprehensive_experiments.py"
CMD="$CMD --models ${MODELS[*]}"
CMD="$CMD --backend $BACKEND"

if [ "$RESUME" == false ]; then
    CMD="$CMD --no-resume"
fi

log "Executing: $CMD"
echo ""

# Run the experiment
if $CMD; then
    log_success "Experiments completed successfully!"
else
    log_error "Experiments failed!"
    exit 1
fi

#=============================================================================
# Backup Results (PostgreSQL only)
#=============================================================================
if [ "$BACKEND" == "postgres" ]; then
    echo ""
    log "Creating PostgreSQL backup..."

    if ./backup_postgres.sh; then
        log_success "Backup completed"
    else
        log_warning "Backup failed (non-critical)"
    fi
fi

#=============================================================================
# Summary
#=============================================================================
echo ""
echo "================================================================================"
echo "EXPERIMENT COMPLETE!"
echo "================================================================================"
echo ""
echo "Results are available in:"
echo "  - experiment_results/comprehensive_results.csv (all results)"
echo "  - experiment_results/results_<model>.csv (per-model results)"
echo ""

if [ "$BACKEND" == "postgres" ]; then
    echo "PostgreSQL backup:"
    echo "  - experiment_results/backups/exp3_vectors_backup_*.sql"
    echo ""
fi

echo "To view results interactively:"
echo "  streamlit run experiment_ui.py"
echo ""
echo "================================================================================"

exit 0
