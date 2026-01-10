#!/bin/bash

echo "============================================================"
echo "LOCAL CODING AGENT (Simple Version)"
echo "============================================================"
echo ""
echo "This is the SIMPLE agent (not two-phase)"
echo ""
echo "Features:"
echo "  - Single-pass RAG pipeline"
echo "  - Faster than two-phase agent"
echo "  - Good for quick lookups"
echo ""
echo "For advanced features, use: two_phase_agent.sh"
echo ""
echo "Prerequisites:"
echo "  [!] PostgreSQL running"
echo "  [!] Ollama running (ollama serve)"
echo ""
echo "============================================================"
echo ""

# Check if Ollama is running
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "[ERROR] Ollama is not running!"
    echo ""
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo ""
    exit 1
fi

echo "[OK] Ollama is running"
echo ""

# Run the agent
python3 local_agent.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Agent failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "Agent session ended"
echo "============================================================"
