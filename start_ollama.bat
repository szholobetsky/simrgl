@echo off
echo ========================================
echo Starting Ollama LLM Server
echo ========================================
echo.

echo Checking if Ollama is installed...
ollama --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ✗ Ollama is not installed
    echo.
    echo Please install Ollama from: https://ollama.ai
    echo.
    pause
    exit /b 1
)

echo ✓ Ollama is installed
echo.

echo Starting Ollama server...
echo (This will open in a new window)
echo.
start "Ollama Server" ollama serve

echo Waiting for Ollama to be ready...
timeout /t 5 /nobreak >nul

echo.
echo Checking if server is responding...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Ollama server is RUNNING on port 11434
    echo.
) else (
    echo ! Server may still be starting...
    echo.
)

echo.
echo Checking for qwen2.5-coder model...
ollama list | findstr "qwen2.5-coder" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ qwen2.5-coder model is available
    echo.
) else (
    echo ! qwen2.5-coder model not found
    echo.
    echo Downloading qwen2.5-coder model...
    echo This will take a few minutes (1.9 GB download)
    echo.
    ollama pull qwen2.5-coder
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✓ Model downloaded successfully
    ) else (
        echo.
        echo ✗ Model download failed
    )
)

echo.
echo ========================================
echo Available Models:
echo ========================================
ollama list

echo.
echo ========================================
echo Server running at: http://localhost:11434
echo ========================================
echo.
pause
