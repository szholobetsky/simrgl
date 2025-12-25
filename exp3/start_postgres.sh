#!/bin/bash

echo "========================================"
echo "Starting PostgreSQL with pgvector"
echo "========================================"
echo ""
echo "This will start PostgreSQL database for vector storage"
echo ""
echo "Database: semantic_vectors"
echo "Port: 5432"
echo "User: postgres"
echo "Password: postgres"
echo ""
echo "Using podman-compose..."
echo "========================================"
echo ""

podman-compose -f postgres-compose.yml up -d

echo ""
echo "========================================"
echo "PostgreSQL started!"
echo ""
echo "Check status: podman ps"
echo "View logs: podman logs semantic_vectors_db"
echo ""
echo "Connection string:"
echo "postgresql://postgres:postgres@localhost:5432/semantic_vectors"
echo "========================================"
