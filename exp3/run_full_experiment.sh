#!/bin/bash
#
# Full Comprehensive RAG Experiment
# 6 models × 2 strategies × 3 sources × 2 targets × 3 windows = 216 experiments
#
set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}FULL COMPREHENSIVE RAG EXPERIMENT${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Configuration
MODELS=("bge-small" "bge-large" "gte-large" "bge-m3" "gte-qwen2" "nomic-embed")
STRATEGIES=("recent" "modn")
SOURCES=("title" "desc" "comments")
TARGETS=("file" "module")
WINDOWS=("w100" "w1000" "all")
BACKEND="postgres"

# Calculate total experiments
TOTAL_MODELS=${#MODELS[@]}
TOTAL_STRATEGIES=${#STRATEGIES[@]}
TOTAL_SOURCES=${#SOURCES[@]}
TOTAL_TARGETS=${#TARGETS[@]}
TOTAL_WINDOWS=${#WINDOWS[@]}
TOTAL_EXPERIMENTS=$((TOTAL_MODELS * TOTAL_STRATEGIES * TOTAL_SOURCES * TOTAL_TARGETS * TOTAL_WINDOWS))

echo -e "${GREEN}Configuration:${NC}"
echo "  Models: ${MODELS[*]}"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Sources: ${SOURCES[*]}"
echo "  Targets: ${TARGETS[*]}"
echo "  Windows: ${WINDOWS[*]}"
echo "  Backend: ${BACKEND}"
echo ""
echo -e "${YELLOW}Total experiments: ${TOTAL_EXPERIMENTS}${NC}"
echo ""

# Check if venv exists
if [ ! -d "venv_py312" ]; then
    echo -e "${RED}Error: venv_py312 not found${NC}"
    echo "Please run: python3.12 -m venv venv_py312"
    exit 1
fi

# Activate venv
echo -e "${GREEN}[1/6] Activating Python 3.12 virtual environment...${NC}"
source venv_py312/bin/activate

# Verify GPU
echo -e "${GREEN}[2/6] Verifying GPU support...${NC}"
python -c "import torch; assert torch.cuda.is_available(), 'GPU not available'; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
if [ $? -ne 0 ]; then
    echo -e "${RED}GPU verification failed!${NC}"
    exit 1
fi

# Check PostgreSQL
echo -e "${GREEN}[3/6] Checking PostgreSQL container...${NC}"
if ! podman ps | grep -q postgres; then
    echo -e "${YELLOW}PostgreSQL not running, starting...${NC}"
    podman-compose up -d
    sleep 5
fi

# Verify PostgreSQL connection
podman exec exp3-postgres-1 psql -U postgres -d semantic_vectors -c "SELECT 1;" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}PostgreSQL connection failed!${NC}"
    exit 1
fi
echo "✓ PostgreSQL ready"

# Show disk space
echo ""
echo -e "${GREEN}[4/6] Checking disk space...${NC}"
df -h . | tail -1

# Ask for confirmation
echo ""
echo -e "${YELLOW}This will run ${TOTAL_EXPERIMENTS} experiments and may take several hours.${NC}"
echo -e "${YELLOW}Estimated time: 6-10 hours with GPU${NC}"
echo ""
read -p "Continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create results directory
mkdir -p experiment_results
mkdir -p backups

# Run experiment
echo ""
echo -e "${GREEN}[5/6] Starting comprehensive experiment...${NC}"
echo "Started at: $(date)"
echo ""

python run_comprehensive_experiments.py \
    --models ${MODELS[@]} \
    --strategies ${STRATEGIES[@]} \
    --sources ${SOURCES[@]} \
    --targets ${TARGETS[@]} \
    --windows ${WINDOWS[@]} \
    --backend ${BACKEND} \
    2>&1 | tee experiment_full_run.log

EXIT_CODE=$?

# Check if experiment completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[6/6] Experiment completed successfully!${NC}"
    echo "Completed at: $(date)"

    # Create backup
    echo ""
    echo -e "${GREEN}Creating PostgreSQL backup...${NC}"
    ./backup_postgres.sh

    # Show results summary
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}RESULTS SUMMARY${NC}"
    echo -e "${BLUE}================================================================================${NC}"

    if [ -f "experiment_results/comprehensive_results.csv" ]; then
        echo ""
        echo "Top 10 configurations by MAP:"
        head -1 experiment_results/comprehensive_results.csv
        tail -n +2 experiment_results/comprehensive_results.csv | sort -t',' -k7 -rn | head -10

        echo ""
        echo "Results saved to:"
        ls -lh experiment_results/*.csv
    fi

    echo ""
    echo -e "${GREEN}Backup saved to:${NC}"
    ls -lht backups/*.sql | head -1

else
    echo ""
    echo -e "${RED}Experiment failed with exit code ${EXIT_CODE}${NC}"
    echo "Check experiment_full_run.log for details"
    exit $EXIT_CODE
fi

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}All done!${NC}"
echo -e "${BLUE}================================================================================${NC}"
