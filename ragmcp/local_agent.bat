@echo off

echo ============================================================
echo LOCAL CODING AGENT (Simple Version)
echo ============================================================
echo.
echo This is the SIMPLE agent (not two-phase)
echo.
echo Features:
echo   - Single-pass RAG pipeline
echo   - Faster than two-phase agent
echo   - Good for quick lookups
echo.
echo For advanced features, use: two_phase_agent.bat
echo.
echo Prerequisites:
echo   [!] PostgreSQL running
echo   [!] Ollama running (ollama serve)
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
    exit /b 1
)

echo [OK] Ollama is running
echo.

REM Run the agent
python local_agent.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Agent failed
    exit /b 1
)

echo.
echo ============================================================
echo Agent session ended
echo ============================================================
