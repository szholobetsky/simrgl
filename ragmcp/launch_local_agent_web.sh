#!/bin/bash

echo "============================================================"
echo "LOCAL AGENT - WEB INTERFACE (Simple Version)"
echo "============================================================"
echo ""
echo "This is the SIMPLE agent (single-pass RAG)"
echo ""
echo "For advanced features, use: ./launch_two_phase_web.sh"
echo ""
echo "Features:"
echo "  - Single-pass RAG pipeline"
echo "  - Faster than two-phase agent"
echo "  - Good for quick lookups"
echo "  - Searches RECENT collections only"
echo ""
echo "Prerequisites:"
echo "  [!] PostgreSQL running"
echo "  [!] Ollama running (ollama serve)"
echo "  [!] RECENT collections created (w100)"
echo ""
echo "============================================================"
echo ""

# Check if Ollama is running
echo "[1/2] Checking Ollama..."
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

# Launch the web interface
echo "[2/2] Launching web interface..."
echo ""
echo "Server will be available at: http://127.0.0.1:7861"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"
echo ""

python3 local_agent_web.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Web interface failed to start"
    echo ""
    echo "Common issues:"
    echo "  - PostgreSQL not running"
    echo "  - Missing Python packages (pip install -r requirements.txt)"
    echo "  - Port 7861 already in use"
    echo ""
    exit 1
fi
