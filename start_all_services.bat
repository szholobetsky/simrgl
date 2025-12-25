@echo off
echo ========================================
echo Starting ALL Services for RAG System
echo ========================================
echo.
echo This will start:
echo 1. PostgreSQL (vector database)
echo 2. Qdrant (vector database)
echo 3. Ollama (LLM server)
echo.
echo ========================================
echo.

cd exp3

echo [1/3] Starting PostgreSQL...
echo.
podman-compose -f postgres-compose.yml up -d
if %ERRORLEVEL% EQU 0 (
    echo ✓ PostgreSQL started
) else (
    echo ✗ PostgreSQL failed to start
)
echo.

echo [2/3] Starting Qdrant...
echo.
podman-compose -f qdrant-compose.yml up -d
if %ERRORLEVEL% EQU 0 (
    echo ✓ Qdrant started
) else (
    echo ✗ Qdrant failed to start
)
echo.

echo [3/3] Starting Ollama...
echo.
echo Checking if Ollama is installed...
ollama --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Ollama is installed
    echo.
    echo Starting Ollama server in background...
    start "Ollama Server" ollama serve
    timeout /t 3 /nobreak >nul
    echo ✓ Ollama server started
    echo.
    echo Checking for qwen2.5-coder model...
    ollama list | findstr "qwen2.5-coder" >nul
    if %ERRORLEVEL% EQU 0 (
        echo ✓ qwen2.5-coder model is available
    ) else (
        echo ! qwen2.5-coder model not found
        echo.
        echo Downloading qwen2.5-coder model...
        echo This will take a few minutes (1.9 GB download)
        ollama pull qwen2.5-coder
        echo ✓ Model downloaded
    )
) else (
    echo ✗ Ollama is not installed
    echo.
    echo Please install Ollama from: https://ollama.ai
    echo.
)
echo.

echo ========================================
echo Service Status Summary
echo ========================================
echo.

echo Checking PostgreSQL...
podman ps | findstr "semantic_vectors_db" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ PostgreSQL: RUNNING on port 5432
    echo   Connection: postgresql://postgres:postgres@localhost:5432/semantic_vectors
) else (
    echo ✗ PostgreSQL: NOT RUNNING
)
echo.

echo Checking Qdrant...
podman ps | findstr "qdrant" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Qdrant: RUNNING on port 6333
    echo   Dashboard: http://localhost:6333/dashboard
) else (
    echo ✗ Qdrant: NOT RUNNING
)
echo.

echo Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Ollama: RUNNING on port 11434
    echo   Available models:
    ollama list
) else (
    echo ✗ Ollama: NOT RUNNING
    echo   Run: ollama serve
)
echo.

echo ========================================
echo Next Steps
echo ========================================
echo.
echo All services are ready! You can now:
echo.
echo 1. Run ETL (choose one):
echo    - For quick test:  cd exp3 ^&^& run_etl_test_postgres.bat
echo    - For production:  cd exp3 ^&^& run_etl_postgres.bat
echo.
echo 2. Launch Gradio UI:
echo    cd ragmcp ^&^& python gradio_ui.py
echo.
echo 3. Open in browser:
echo    http://localhost:7860
echo.
echo ========================================
echo.
pause
