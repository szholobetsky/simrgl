#!/bin/bash
# Start MCP Server (PostgreSQL Edition)
# This is for testing only - Claude Desktop will start it automatically

echo "============================================================"
echo "Starting Semantic Module Search MCP Server (PostgreSQL)"
echo "============================================================"
echo ""
echo "This server provides semantic search tools for:"
echo "  - Module search (folder-level)"
echo "  - File search (file-level)"
echo "  - Similar task search (historical)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Check if PostgreSQL is running
if ! podman ps --filter name=semantic_vectors_db --format "{{.Names}}" | grep -q "^semantic_vectors_db$"; then
    echo "[WARNING] PostgreSQL container is not running!"
    echo "Please start it first: podman start semantic_vectors_db"
    echo ""
    exit 1
fi

echo "[OK] PostgreSQL is running"
echo ""

# Start the MCP server
python3 mcp_server_postgres.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] MCP server failed to start"
    echo ""
    echo "Common issues:"
    echo "  1. Missing dependencies: pip install mcp sentence-transformers psycopg2-binary"
    echo "  2. PostgreSQL not running: podman start semantic_vectors_db"
    echo "  3. Collections not created: run ETL pipeline first"
    echo ""
    exit 1
fi
