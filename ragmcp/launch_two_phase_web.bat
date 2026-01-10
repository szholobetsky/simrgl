@echo off

echo ============================================================
echo TWO-PHASE RAG AGENT - WEB INTERFACE
echo ============================================================
echo.
echo Starting Gradio web interface for dual indexing two-phase agent
echo.
echo Features:
echo   - Phase 1: File Selection (DUAL search: RECENT + ALL)
echo   - Phase 2: Deep Analysis (actual file content)
echo   - Phase 3: Reflection (confidence scores)
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
    echo [ERROR] Ollama is not running
    echo.
    echo Please start Ollama first:
    echo   ollama serve
    echo.
    exit /b 1
)

echo [OK] Ollama is running
echo.

REM Launch the web interface
echo Launching web interface...
echo.
echo The web UI will open at: http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python two_phase_agent_web.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Web interface failed to start
    echo.
    echo Make sure gradio is installed:
    echo   pip install gradio
    echo.
    pause
    exit /b 1
)
