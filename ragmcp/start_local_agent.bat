@echo off
REM Start Local Offline Coding Agent - CLI Mode
REM Works completely offline with MCP + Ollama

echo ============================================================
echo Local Offline Coding Agent - CLI Mode
echo ============================================================
echo.
echo This agent works 100%% OFFLINE using:
echo   - MCP Server (semantic search via PostgreSQL)
echo   - Ollama (local LLM - qwen2.5-coder)
echo   - No cloud services or API keys needed
echo.

REM Check PostgreSQL
podman ps --filter name=semantic_vectors_db --format "{{.Names}}" | findstr /C:"semantic_vectors_db" >nul
if errorlevel 1 (
    echo [ERROR] PostgreSQL is not running!
    echo Please start it: podman start semantic_vectors_db
    echo.
    pause
    exit /b 1
)
echo [OK] PostgreSQL is running

REM Check Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not running!
    echo Please start it: ollama serve
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama is running
echo.

REM Start the agent
python local_agent.py

if errorlevel 1 (
    echo.
    echo [ERROR] Agent failed to start
    pause
    exit /b 1
)
