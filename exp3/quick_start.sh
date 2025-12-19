#!/bin/bash
# Quick Start - Just launch the UI (experiments must be completed first)

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "RAG Research Experiment - Quick Start UI"
echo "================================================================================"
echo ""

# Check if results exist
if [ ! -f "experiment_results.csv" ]; then
    echo -e "${RED}ERROR: No results found.${NC}"
    echo "Please run ./start.sh first to complete experiments."
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Results found: experiment_results.csv${NC}"
echo -e "${GREEN}✓ Test set found: test_set.json${NC}"
echo ""
echo "Starting Streamlit UI..."
echo -e "${YELLOW}Open your browser to: http://localhost:8501${NC}"
echo ""
echo "Press Ctrl+C to stop the UI"
echo "================================================================================"
echo ""

# Start Streamlit
python3 -m streamlit run experiment_ui.py
