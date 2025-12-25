#!/bin/bash

echo "========================================"
echo "Stopping ALL Services for RAG System"
echo "========================================"
echo ""
echo "This will stop:"
echo "1. PostgreSQL (vector database)"
echo "2. Qdrant (vector database)"
echo "3. Ollama (LLM server)"
echo ""
echo "========================================"
echo ""

cd exp3

echo "[1/3] Stopping PostgreSQL..."
echo ""
podman-compose -f postgres-compose.yml down
if [ $? -eq 0 ]; then
    echo "✓ PostgreSQL stopped"
else
    echo "! PostgreSQL may not be running"
fi
echo ""

echo "[2/3] Stopping Qdrant..."
echo ""
podman-compose -f qdrant-compose.yml down
if [ $? -eq 0 ]; then
    echo "✓ Qdrant stopped"
else
    echo "! Qdrant may not be running"
fi
echo ""

echo "[3/3] Stopping Ollama..."
echo ""
echo "Checking for Ollama process..."
if pgrep -x "ollama" > /dev/null; then
    pkill -x "ollama"
    if [ $? -eq 0 ]; then
        echo "✓ Ollama stopped"
    else
        echo "! Could not stop Ollama"
    fi
else
    echo "! Ollama is not running"
fi
echo ""

echo "========================================"
echo "Service Status"
echo "========================================"
echo ""

echo "Checking containers..."
if podman ps --format "{{.Names}}" | grep -qE "postgres|qdrant"; then
    echo ""
    echo "Still running:"
    podman ps --format "table {{.Names}}\t{{.Status}}" | grep -E "NAMES|postgres|qdrant"
    echo ""
else
    echo "✓ All containers stopped"
fi

echo ""
echo "========================================"
echo "All services stopped!"
echo "========================================"
echo ""
