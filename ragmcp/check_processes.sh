#!/bin/bash

echo "============================================================"
echo "CHECKING FOR RUNNING PYTHON PROCESSES"
echo "============================================================"
echo ""

echo "[1/4] All Python processes:"
ps aux | grep python | grep -v grep || echo "No Python processes found"

echo ""
echo "[2/4] Checking for MCP server processes:"
ps aux | grep mcp_server | grep -v grep || echo "No MCP server processes found"

echo ""
echo "[3/4] Checking for migration processes:"
ps aux | grep migrate_rawdata | grep -v grep || echo "No migration processes found"

echo ""
echo "[4/4] Checking for two-phase agent processes:"
ps aux | grep two_phase_agent | grep -v grep || echo "No two-phase agent processes found"

echo ""
echo "============================================================"
echo "If you see multiple MCP server processes, kill them with:"
echo "  kill -9 [process_id]"
echo "============================================================"
