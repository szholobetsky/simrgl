#!/bin/bash

echo "========================================"
echo "Starting Qdrant Vector Database"
echo "========================================"
echo ""

cd exp3
podman-compose -f qdrant-compose.yml up -d

echo ""
echo "Waiting for Qdrant to be ready..."
sleep 3

echo ""
if podman ps | grep -q "qdrant"; then
    echo ""
    echo "✓ Qdrant is RUNNING"
    echo ""
    echo "Dashboard: http://localhost:6333/dashboard"
    echo "API: http://localhost:6333"
    echo ""
else
    echo ""
    echo "✗ Qdrant failed to start"
    echo ""
    echo "Check logs with: podman logs qdrant"
fi

echo ""
