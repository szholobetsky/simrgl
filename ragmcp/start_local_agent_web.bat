@echo off
REM Start Local Offline Coding Agent - Web Interface
REM Works completely offline with MCP + Ollama

echo ============================================================
echo Local Offline Coding Agent - Web Interface
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
echo Starting web interface on http://127.0.0.1:7861
echo.

REM Start the web agent
python local_agent_web.py

if errorlevel 1 (
    echo.
    echo [ERROR] Web agent failed to start
    pause
    exit /b 1
)
