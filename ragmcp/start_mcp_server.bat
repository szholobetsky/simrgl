@echo off
REM Start MCP Server (PostgreSQL Edition)
REM This is for testing only - Claude Desktop will start it automatically

echo ============================================================
echo Starting Semantic Module Search MCP Server (PostgreSQL)
echo ============================================================
echo.
echo This server provides semantic search tools for:
echo   - Module search (folder-level)
echo   - File search (file-level)
echo   - Similar task search (historical)
echo.
echo Press Ctrl+C to stop the server
echo.

REM Check if PostgreSQL is running
podman ps --filter name=semantic_vectors_db --format "{{.Names}}" | findstr /C:"semantic_vectors_db" >nul
if errorlevel 1 (
    echo [WARNING] PostgreSQL container is not running!
    echo Please start it first: podman start semantic_vectors_db
    echo.
    pause
    exit /b 1
)

echo [OK] PostgreSQL is running
echo.

REM Start the MCP server
python mcp_server_postgres.py

if errorlevel 1 (
    echo.
    echo [ERROR] MCP server failed to start
    echo.
    echo Common issues:
    echo   1. Missing dependencies: pip install mcp sentence-transformers psycopg2-binary
    echo   2. PostgreSQL not running: podman start semantic_vectors_db
    echo   3. Collections not created: run ETL pipeline first
    echo.
    pause
    exit /b 1
)
