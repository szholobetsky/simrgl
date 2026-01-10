#!/bin/bash

echo "============================================================"
echo "TWO-PHASE RAG AGENT - WEB INTERFACE"
echo "============================================================"
echo ""
echo "Starting Gradio web interface for dual indexing two-phase agent"
echo ""
echo "Features:"
echo "  - Phase 1: File Selection (DUAL search: RECENT + ALL)"
echo "  - Phase 2: Deep Analysis (actual file content)"
echo "  - Phase 3: Reflection (confidence scores)"
echo ""
echo "Prerequisites:"
echo "  [!] PostgreSQL running (with dual collections)"
echo "  [!] Ollama running (ollama serve)"
echo "  [!] RAWDATA migrated to PostgreSQL"
echo ""
echo "============================================================"
echo ""

# Check if Ollama is running
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo ""
    echo "[ERROR] Ollama is not running"
    echo ""
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo ""
    exit 1
fi

echo "[OK] Ollama is running"
echo ""

# Launch the web interface
echo "Launching web interface..."
echo ""
echo "The web UI will open at: http://127.0.0.1:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

python3 two_phase_agent_web.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Web interface failed to start"
    echo ""
    echo "Make sure gradio is installed:"
    echo "  pip install gradio"
    echo ""
    read -p "Press Enter to continue..."
    exit 1
fi
