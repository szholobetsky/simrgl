#!/bin/bash

echo "============================================================"
echo "PULLING LFM2:1.2B MODEL"
echo "============================================================"
echo ""
echo "This is the smallest and fastest model for testing."
echo ""
echo "Model specs:"
echo "  - Size: ~1.2B parameters"
echo "  - Disk: ~700 MB"
echo "  - RAM: ~2-3 GB"
echo "  - Speed: ~1-2 seconds per response (CPU)"
echo ""
echo "============================================================"
echo ""

ollama pull lfm2:1.2b

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to pull model"
    echo ""
    echo "Make sure Ollama is running:"
    echo "  ollama serve"
    echo ""
    exit 1
fi

echo ""
echo "============================================================"
echo "SUCCESS: Model pulled!"
echo "============================================================"
echo ""
echo "You can now use the agents with the lfm2:1.2b model."
echo ""
echo "To test:"
echo "  cd ragmcp"
echo "  ./launch_local_agent_web.sh"
echo ""
echo "============================================================"
echo ""
