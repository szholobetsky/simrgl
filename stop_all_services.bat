@echo off
echo ========================================
echo Stopping ALL Services for RAG System
echo ========================================
echo.
echo This will stop:
echo 1. PostgreSQL (vector database)
echo 2. Qdrant (vector database)
echo 3. Ollama (LLM server)
echo.
echo ========================================
echo.

cd exp3

echo [1/3] Stopping PostgreSQL...
echo.
podman-compose -f postgres-compose.yml down
if %ERRORLEVEL% EQU 0 (
    echo ✓ PostgreSQL stopped
) else (
    echo ! PostgreSQL may not be running
)
echo.

echo [2/3] Stopping Qdrant...
echo.
podman-compose -f qdrant-compose.yml down
if %ERRORLEVEL% EQU 0 (
    echo ✓ Qdrant stopped
) else (
    echo ! Qdrant may not be running
)
echo.

echo [3/3] Stopping Ollama...
echo.
echo Checking for Ollama process...
tasklist | findstr "ollama" >nul
if %ERRORLEVEL% EQU 0 (
    taskkill /F /IM ollama.exe >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✓ Ollama stopped
    ) else (
        echo ! Could not stop Ollama (may need admin rights)
        echo   Please close the "Ollama Server" window manually
    )
) else (
    echo ! Ollama is not running
)
echo.

echo ========================================
echo Service Status
echo ========================================
echo.

echo Checking containers...
podman ps --format "{{.Names}}" | findstr -i "postgres qdrant" >nul
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Still running:
    podman ps --format "table {{.Names}}\t{{.Status}}" | findstr -i "postgres qdrant"
    echo.
) else (
    echo ✓ All containers stopped
)

echo.
echo ========================================
echo All services stopped!
echo ========================================
echo.
pause
