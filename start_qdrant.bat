@echo off
echo ========================================
echo Starting Qdrant Vector Database
echo ========================================
echo.

cd exp3
podman-compose -f qdrant-compose.yml up -d

echo.
echo Waiting for Qdrant to be ready...
timeout /t 3 /nobreak >nul

echo.
podman ps | findstr "qdrant"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Qdrant is RUNNING
    echo.
    echo Dashboard: http://localhost:6333/dashboard
    echo API: http://localhost:6333
    echo.
) else (
    echo.
    echo ✗ Qdrant failed to start
    echo.
    echo Check logs with: podman logs qdrant
)

echo.
pause
