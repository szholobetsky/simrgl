#!/bin/bash

echo "============================================================"
echo "TWO-PHASE REFLECTIVE RAG AGENT"
echo "With DUAL Collection Search"
echo "============================================================"
echo ""
echo "This agent uses:"
echo "  - MCP Server: mcp_server_dual.py"
echo "  - LLM: Ollama (qwen2.5-coder:latest)"
echo "  - Search: DUAL mode (RECENT + ALL collections)"
echo ""
echo "Features:"
echo "  Phase 1: Reasoning & File Selection (searches both collections)"
echo "  Phase 2: Deep Analysis (analyzes actual file content)"
echo "  Phase 3: Final Reflection (self-critique with confidence)"
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
    echo "[ERROR] Ollama is not running!"
    echo ""
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "[OK] Ollama is running"
echo ""

# Run the agent
python3 two_phase_agent.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Agent failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "Agent session ended"
echo "============================================================"
