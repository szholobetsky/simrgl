#!/bin/bash
#
# Subset Experiment: Test with 3 models including bge-m3 for memory testing
# 3 models × 2 strategies × 3 sources × 2 targets × 3 windows = 108 experiments
#
set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}SUBSET EXPERIMENT (3 models - includes bge-m3 for GPU memory testing)${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Configuration - test small, large, and m3 (m3 tests GPU memory management)
MODELS=("bge-small" "bge-large" "bge-m3")
STRATEGIES=("recent" "modn")
SOURCES=("title" "desc" "comments")
TARGETS=("file" "module")
WINDOWS=("w100" "w1000" "all")
BACKEND="postgres"

TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#STRATEGIES[@]} * ${#SOURCES[@]} * ${#TARGETS[@]} * ${#WINDOWS[@]}))

echo -e "${GREEN}Configuration:${NC}"
echo "  Models: ${MODELS[*]}"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Sources: ${SOURCES[*]}"
echo "  Targets: ${TARGETS[*]}"
echo "  Windows: ${WINDOWS[*]}"
echo ""
echo -e "${YELLOW}Total experiments: ${TOTAL_EXPERIMENTS}${NC}"
echo -e "${YELLOW}Estimated time: 3-4 hours with GPU${NC}"
echo -e "${YELLOW}Note: bge-m3 will test GPU memory management (cache clearing, garbage collection)${NC}"
echo ""

# Activate venv
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv_py312/bin/activate

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(); print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"

# Check PostgreSQL
if ! podman ps | grep -q semantic_vectors_db; then
    echo -e "${YELLOW}Starting PostgreSQL...${NC}"
    podman-compose -f postgres-compose.yml up -d
    sleep 5
fi

# Ask for confirmation
read -p "Continue with subset experiment? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Run experiment
echo ""
echo -e "${GREEN}Starting subset experiment...${NC}"
echo "Started at: $(date)"
echo ""

python run_comprehensive_experiments.py \
    --models ${MODELS[@]} \
    --strategies ${STRATEGIES[@]} \
    --sources ${SOURCES[@]} \
    --targets ${TARGETS[@]} \
    --windows ${WINDOWS[@]} \
    --backend ${BACKEND} \
    2>&1 | tee experiment_subset_run.log

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Subset experiment completed!${NC}"
    echo "Completed at: $(date)"

    # Show results
    if [ -f "experiment_results/comprehensive_results.csv" ]; then
        echo ""
        echo "Results summary:"
        wc -l experiment_results/comprehensive_results.csv
        head -1 experiment_results/comprehensive_results.csv
        tail -5 experiment_results/comprehensive_results.csv
    fi
fi
