@echo off

echo ============================================================
echo TWO-PHASE REFLECTIVE RAG AGENT
echo With DUAL Collection Search
echo ============================================================
echo.
echo This agent uses:
echo   - MCP Server: mcp_server_dual.py
echo   - LLM: Ollama (qwen2.5-coder:latest)
echo   - Search: DUAL mode (RECENT + ALL collections)
echo.
echo Features:
echo   Phase 1: Reasoning ^& File Selection (searches both collections)
echo   Phase 2: Deep Analysis (analyzes actual file content)
echo   Phase 3: Final Reflection (self-critique with confidence)
echo.
echo Prerequisites:
echo   [!] PostgreSQL running (with dual collections)
echo   [!] Ollama running (ollama serve)
echo   [!] RAWDATA migrated to PostgreSQL
echo.
echo ============================================================
echo.

REM Check if Ollama is running
echo Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Ollama is not running!
    echo.
    echo Please start Ollama first:
    echo   ollama serve
    echo.
    echo Then run this script again.
    exit /b 1
)

echo [OK] Ollama is running
echo.

REM Run the agent
python two_phase_agent.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Agent failed
    exit /b 1
)

echo.
echo ============================================================
echo Agent session ended
echo ============================================================
