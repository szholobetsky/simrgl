#!/bin/bash

echo "========================================"
echo "Starting ALL Services for RAG System"
echo "========================================"
echo ""
echo "This will start:"
echo "1. PostgreSQL (vector database)"
echo "2. Qdrant (vector database)"
echo "3. Ollama (LLM server)"
echo ""
echo "========================================"
echo ""

cd exp3

echo "[1/3] Starting PostgreSQL..."
echo ""
podman-compose -f postgres-compose.yml up -d
if [ $? -eq 0 ]; then
    echo "✓ PostgreSQL started"
else
    echo "✗ PostgreSQL failed to start"
fi
echo ""

echo "[2/3] Starting Qdrant..."
echo ""
podman-compose -f qdrant-compose.yml up -d
if [ $? -eq 0 ]; then
    echo "✓ Qdrant started"
else
    echo "✗ Qdrant failed to start"
fi
echo ""

echo "[3/3] Starting Ollama..."
echo ""
echo "Checking if Ollama is installed..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    echo ""
    echo "Starting Ollama server in background..."
    nohup ollama serve > ollama.log 2>&1 &
    sleep 3
    echo "✓ Ollama server started"
    echo ""
    echo "Checking for qwen2.5-coder model..."
    if ollama list | grep -q "qwen2.5-coder"; then
        echo "✓ qwen2.5-coder model is available"
    else
        echo "! qwen2.5-coder model not found"
        echo ""
        echo "Downloading qwen2.5-coder model..."
        echo "This will take a few minutes (1.9 GB download)"
        ollama pull qwen2.5-coder
        echo "✓ Model downloaded"
    fi
else
    echo "✗ Ollama is not installed"
    echo ""
    echo "Please install Ollama from: https://ollama.ai"
    echo ""
fi
echo ""

echo "========================================"
echo "Service Status Summary"
echo "========================================"
echo ""

echo "Checking PostgreSQL..."
if podman ps | grep -q "semantic_vectors_db"; then
    echo "✓ PostgreSQL: RUNNING on port 5432"
    echo "  Connection: postgresql://postgres:postgres@localhost:5432/semantic_vectors"
else
    echo "✗ PostgreSQL: NOT RUNNING"
fi
echo ""

echo "Checking Qdrant..."
if podman ps | grep -q "qdrant"; then
    echo "✓ Qdrant: RUNNING on port 6333"
    echo "  Dashboard: http://localhost:6333/dashboard"
else
    echo "✗ Qdrant: NOT RUNNING"
fi
echo ""

echo "Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama: RUNNING on port 11434"
    echo "  Available models:"
    ollama list
else
    echo "✗ Ollama: NOT RUNNING"
    echo "  Run: ollama serve"
fi
echo ""

echo "========================================"
echo "Next Steps"
echo "========================================"
echo ""
echo "All services are ready! You can now:"
echo ""
echo "1. Run ETL (choose one):"
echo "   - For quick test:  cd exp3 && ./run_etl_test_postgres.sh"
echo "   - For production:  cd exp3 && ./run_etl_postgres.sh"
echo ""
echo "2. Launch Gradio UI:"
echo "   cd ragmcp && python gradio_ui.py"
echo ""
echo "3. Open in browser:"
echo "   http://localhost:7860"
echo ""
echo "========================================"
echo ""
