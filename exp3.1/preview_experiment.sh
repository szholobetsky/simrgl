#!/bin/bash
#
# Preview the full experiment plan without running it
#

# Configuration
MODELS=("bge-small" "bge-large" "gte-large" "bge-m3" "gte-qwen2" "nomic-embed")
STRATEGIES=("recent" "modn")
SOURCES=("title" "desc" "comments")
TARGETS=("file" "module")
WINDOWS=("w100" "w1000" "all")

# Calculate totals
TOTAL_MODELS=${#MODELS[@]}
TOTAL_STRATEGIES=${#STRATEGIES[@]}
TOTAL_SOURCES=${#SOURCES[@]}
TOTAL_TARGETS=${#TARGETS[@]}
TOTAL_WINDOWS=${#WINDOWS[@]}
TOTAL_EXPERIMENTS=$((TOTAL_MODELS * TOTAL_STRATEGIES * TOTAL_SOURCES * TOTAL_TARGETS * TOTAL_WINDOWS))

echo "================================================================================"
echo "FULL EXPERIMENT PREVIEW"
echo "================================================================================"
echo ""
echo "Models (${TOTAL_MODELS}):"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

echo "Strategies (${TOTAL_STRATEGIES}):"
for strategy in "${STRATEGIES[@]}"; do
    echo "  - $strategy"
done
echo ""

echo "Sources (${TOTAL_SOURCES}):"
for source in "${SOURCES[@]}"; do
    echo "  - $source"
done
echo ""

echo "Targets (${TOTAL_TARGETS}):"
for target in "${TARGETS[@]}"; do
    echo "  - $target"
done
echo ""

echo "Windows (${TOTAL_WINDOWS}):"
for window in "${WINDOWS[@]}"; do
    echo "  - $window"
done
echo ""

echo "================================================================================"
echo "TOTAL: ${TOTAL_EXPERIMENTS} experiments"
echo "================================================================================"
echo ""

# Generate sample experiment IDs
echo "Sample experiment IDs (first 20):"
count=0
for model in "${MODELS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        for source in "${SOURCES[@]}"; do
            for target in "${TARGETS[@]}"; do
                for window in "${WINDOWS[@]}"; do
                    count=$((count + 1))
                    if [ $count -le 20 ]; then
                        echo "  $count. ${model}_${strategy}_${source}_${target}_${window}"
                    fi
                done
            done
        done
    done
done

echo "  ..."
echo "  $TOTAL_EXPERIMENTS. ${MODELS[-1]}_${STRATEGIES[-1]}_${SOURCES[-1]}_${TARGETS[-1]}_${WINDOWS[-1]}"
echo ""

# Estimate time and resources
echo "================================================================================"
echo "RESOURCE ESTIMATES"
echo "================================================================================"
echo ""
echo "Time estimates (with GPU):"
echo "  - bge-small: ~8 min/experiment × 36 experiments = ~4.8 hours"
echo "  - bge-large/gte-large: ~10 min/experiment × 72 experiments = ~12 hours"
echo "  - bge-m3/gte-qwen2/nomic-embed: ~15 min/experiment × 108 experiments = ~27 hours"
echo "  - TOTAL: ~6-10 hours (with parallelization)"
echo ""
echo "Storage estimates:"
echo "  - PostgreSQL vectors: ~216 collections × 50MB = ~10-20GB"
echo "  - Embedding models: ~6-15GB"
echo "  - Results/logs: ~100MB"
echo "  - TOTAL: ~20-35GB"
echo ""

echo "To run the full experiment:"
echo "  ./run_full_experiment.sh"
echo ""
