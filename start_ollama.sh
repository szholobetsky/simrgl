#!/bin/bash

echo "========================================"
echo "Starting Ollama LLM Server"
echo "========================================"
echo ""

echo "Checking if Ollama is installed..."
if ! command -v ollama &> /dev/null; then
    echo "✗ Ollama is not installed"
    echo ""
    echo "Please install Ollama from: https://ollama.ai"
    echo ""
    exit 1
fi

echo "✓ Ollama is installed"
echo ""

echo "Starting Ollama server..."
echo "(Running in background)"
echo ""
nohup ollama serve > ollama.log 2>&1 &

echo "Waiting for Ollama to be ready..."
sleep 5

echo ""
echo "Checking if server is responding..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama server is RUNNING on port 11434"
    echo ""
else
    echo "! Server may still be starting..."
    echo ""
fi

echo ""
echo "Checking for qwen2.5-coder model..."
if ollama list | grep -q "qwen2.5-coder"; then
    echo "✓ qwen2.5-coder model is available"
    echo ""
else
    echo "! qwen2.5-coder model not found"
    echo ""
    echo "Downloading qwen2.5-coder model..."
    echo "This will take a few minutes (1.9 GB download)"
    echo ""
    ollama pull qwen2.5-coder
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Model downloaded successfully"
    else
        echo ""
        echo "✗ Model download failed"
    fi
fi

echo ""
echo "========================================"
echo "Available Models:"
echo "========================================"
ollama list

echo ""
echo "========================================"
echo "Server running at: http://localhost:11434"
echo "========================================"
echo ""
