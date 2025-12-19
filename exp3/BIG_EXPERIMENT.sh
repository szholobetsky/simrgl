#!/bin/bash
# =============================================================================
# BIG EXPERIMENT - Run RAG experiments with multiple embedding models
# =============================================================================
# This script runs the complete experiment workflow for multiple models.
# Each model creates separate Qdrant collections and result files.
#
# Usage:
#   ./BIG_EXPERIMENT.sh              # Run all models
#   ./BIG_EXPERIMENT.sh bge-small bge-large  # Run specific models
#
# Available models:
#   bge-small, bge-large, bge-m3, gte-qwen2, nomic-embed, gte-large, e5-large
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default models to test (can be overridden by command line args)
if [ $# -eq 0 ]; then
    MODELS=("bge-small" "bge-large" "bge-m3" "gte-qwen2" "nomic-embed")
else
    MODELS=("$@")
fi

SPLIT_STRATEGY="recent"
RESULTS_DIR="experiment_results"
COMBINED_RESULTS="all_models_results.csv"
LOG_FILE="big_experiment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

# =============================================================================
# Pre-flight checks
# =============================================================================
echo "================================================================================"
echo "BIG EXPERIMENT - Multi-Model RAG Evaluation"
echo "================================================================================"
echo ""

log "Starting BIG EXPERIMENT"
log "Models to test: ${MODELS[*]}"
log "Split strategy: $SPLIT_STRATEGY"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check docker/podman
if command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
    COMPOSE_CMD="docker-compose"
elif command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
    COMPOSE_CMD="podman-compose"
else
    log_error "Neither Docker nor Podman found. Please install one."
    exit 1
fi

log "Using container runtime: $CONTAINER_CMD"

# Create results directory
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Start Qdrant (only once)
# =============================================================================
log "Starting Qdrant vector database..."

# Stop existing container if running
$COMPOSE_CMD down 2>/dev/null || true

# Clear old storage for fresh start
if [ -d "qdrant_storage" ]; then
    log "Clearing old Qdrant storage..."
    rm -rf qdrant_storage
fi
mkdir -p qdrant_storage

# Start Qdrant
$COMPOSE_CMD up -d
sleep 10

# Verify Qdrant is running
if ! curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    log_warning "Qdrant may not be ready, waiting additional 10 seconds..."
    sleep 10
fi

if ! curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    log_error "Cannot connect to Qdrant. Please check the container."
    exit 1
fi

log_success "Qdrant is running"

# =============================================================================
# Run experiments for each model
# =============================================================================
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0
FAILED_MODELS=()
SUCCESSFUL_MODELS=()

for MODEL in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo ""
    echo "================================================================================"
    echo "MODEL $CURRENT/$TOTAL_MODELS: $MODEL"
    echo "================================================================================"

    MODEL_RESULTS="${RESULTS_DIR}/results_${MODEL}.csv"
    MODEL_LOG="${RESULTS_DIR}/log_${MODEL}.log"

    # Run ETL Pipeline
    log "[$MODEL] Running ETL Pipeline..."
    if python3 etl_pipeline.py \
        --split_strategy "$SPLIT_STRATEGY" \
        --model "$MODEL" \
        > "$MODEL_LOG" 2>&1; then
        log_success "[$MODEL] ETL Pipeline completed"
    else
        log_error "[$MODEL] ETL Pipeline failed. Check $MODEL_LOG for details."
        FAILED_MODELS+=("$MODEL")
        continue
    fi

    # Run Experiments
    log "[$MODEL] Running experiments..."
    if python3 run_experiments.py \
        --split_strategy "$SPLIT_STRATEGY" \
        --model "$MODEL" \
        --output "$MODEL_RESULTS" \
        >> "$MODEL_LOG" 2>&1; then
        log_success "[$MODEL] Experiments completed"
        log_success "[$MODEL] Results saved to: $MODEL_RESULTS"
        SUCCESSFUL_MODELS+=("$MODEL")
    else
        log_error "[$MODEL] Experiments failed. Check $MODEL_LOG for details."
        FAILED_MODELS+=("$MODEL")
        continue
    fi

    # Backup Qdrant for this model
    QDRANT_BACKUP_DIR="qdrant_snapshots_${MODEL}"
    log "[$MODEL] Backing up Qdrant to $QDRANT_BACKUP_DIR..."
    if python3 backup_restore_qdrant.py --action backup --output "$QDRANT_BACKUP_DIR" >> "$MODEL_LOG" 2>&1; then
        log_success "[$MODEL] Qdrant backup saved to: $QDRANT_BACKUP_DIR/"
    else
        log_warning "[$MODEL] Qdrant backup failed (non-critical)"
    fi

    # Show quick summary
    if [ -f "$MODEL_RESULTS" ]; then
        echo ""
        log "[$MODEL] Quick Results Summary:"
        head -5 "$MODEL_RESULTS" | column -t -s',' 2>/dev/null || head -5 "$MODEL_RESULTS"
        echo "..."
    fi
done

# =============================================================================
# Combine all results
# =============================================================================
echo ""
echo "================================================================================"
echo "COMBINING RESULTS"
echo "================================================================================"

if [ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]; then
    log "Combining results from ${#SUCCESSFUL_MODELS[@]} models..."

    # Create combined CSV with model column
    echo "model,experiment_id,source,target,window,split,MAP,MRR,P@1,R@1,P@3,R@3,P@5,R@5,P@10,R@10" > "$COMBINED_RESULTS"

    for MODEL in "${SUCCESSFUL_MODELS[@]}"; do
        MODEL_RESULTS="${RESULTS_DIR}/results_${MODEL}.csv"
        if [ -f "$MODEL_RESULTS" ]; then
            # Skip header and add model column
            tail -n +2 "$MODEL_RESULTS" | while read line; do
                echo "$MODEL,$line" >> "$COMBINED_RESULTS"
            done
        fi
    done

    log_success "Combined results saved to: $COMBINED_RESULTS"
fi

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "================================================================================"
echo "BIG EXPERIMENT COMPLETE"
echo "================================================================================"
echo ""

if [ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]; then
    echo -e "${GREEN}Successful models (${#SUCCESSFUL_MODELS[@]}):${NC}"
    for MODEL in "${SUCCESSFUL_MODELS[@]}"; do
        echo "  - $MODEL"
    done
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed models (${#FAILED_MODELS[@]}):${NC}"
    for MODEL in "${FAILED_MODELS[@]}"; do
        echo "  - $MODEL (see ${RESULTS_DIR}/log_${MODEL}.log)"
    done
fi

echo ""
echo "Output files:"
echo "  - Combined results: $COMBINED_RESULTS"
echo "  - Individual results: ${RESULTS_DIR}/results_*.csv"
echo "  - Logs: ${RESULTS_DIR}/log_*.log"
echo "  - Qdrant backups: qdrant_snapshots_<model>/ (one per model)"
echo ""
echo "To restore a specific model's Qdrant data:"
echo "  python3 backup_restore_qdrant.py --action restore --input qdrant_snapshots_bge-large"
echo ""
echo "To view results in UI:"
echo "  streamlit run experiment_ui.py"
echo ""
echo "================================================================================"

# Exit with error if any models failed
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    exit 1
fi
