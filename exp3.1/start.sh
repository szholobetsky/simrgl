#!/bin/bash
# RAG Research Experiment - Linux Startup Script
# This script will run the complete experiment workflow

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "RAG Research Experiment - Complete Workflow"
echo "================================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Check if podman-compose is available
if ! command -v podman-compose &> /dev/null; then
    echo -e "${RED}ERROR: podman-compose not found.${NC}"
    echo "Install with: pip install podman-compose"
    echo "Or use: podman compose (if you have newer podman)"
    exit 1
fi

echo -e "${GREEN}✓ podman-compose found${NC}"
echo ""

# Step 1: Stop existing containers
echo -e "${BLUE}[Step 1/6] Stopping any existing Qdrant containers...${NC}"
podman-compose down || true
echo ""

# Step 2: Clear old storage
echo -e "${BLUE}[Step 2/6] Clearing old Qdrant storage...${NC}"
if [ -d "qdrant_storage" ]; then
    echo "Deleting qdrant_storage contents..."
    rm -rf qdrant_storage/*
else
    mkdir -p qdrant_storage
fi
echo -e "${GREEN}✓ Storage cleared${NC}"
echo ""

# Step 3: Start Qdrant
echo -e "${BLUE}[Step 3/6] Starting Qdrant vector database...${NC}"
podman-compose up -d
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to start Qdrant${NC}"
    exit 1
fi
echo "Waiting for Qdrant to be ready..."
sleep 10
echo -e "${GREEN}✓ Qdrant started${NC}"
echo ""

# Step 4: Check Qdrant connection
echo -e "${BLUE}[Step 4/6] Checking Qdrant connection...${NC}"
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Qdrant is ready!${NC}"
else
    echo -e "${YELLOW}WARNING: Qdrant might not be ready yet. Waiting additional 5 seconds...${NC}"
    sleep 5
    if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Qdrant is ready!${NC}"
    else
        echo -e "${RED}ERROR: Cannot connect to Qdrant${NC}"
        echo "Check logs with: podman logs rag_experiment_qdrant"
        exit 1
    fi
fi
echo ""

# Step 5: Run ETL Pipeline
echo -e "${BLUE}[Step 5/6] Running ETL Pipeline...${NC}"
echo "This will process 9,799 tasks and create 18 experiment collections"
echo -e "${YELLOW}Estimated time: 20-30 minutes${NC}"
echo ""
python3 etl_pipeline.py --split_strategy recent
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: ETL Pipeline failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ ETL Pipeline complete${NC}"
echo ""

# Step 6: Run Experiments
echo -e "${BLUE}[Step 6/6] Running Experiments...${NC}"
echo "This will evaluate all 18 experiment combinations"
echo -e "${YELLOW}Estimated time: 10-15 minutes${NC}"
echo ""
python3 run_experiments.py --split_strategy recent
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Experiments failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Experiments complete${NC}"
echo ""

# Success summary
echo "================================================================================"
echo -e "${GREEN}SUCCESS! Experiment Complete${NC}"
echo "================================================================================"
echo ""
echo "Results saved to: experiment_results.csv"
echo "Test set saved to: test_set.json"
echo "Logs saved to: experiment.log"
echo ""
echo "Starting Streamlit UI..."
echo -e "${YELLOW}Open your browser to: http://localhost:8501${NC}"
echo ""
echo "Press Ctrl+C to stop the UI when done"
echo "================================================================================"
echo ""

# Start Streamlit UI
python3 -m streamlit run experiment_ui.py
