@echo off
echo ========================================
echo Starting PostgreSQL
echo ========================================
echo.

cd exp3
podman-compose -f postgres-compose.yml up -d

echo.
echo Waiting for PostgreSQL to be ready...
timeout /t 3 /nobreak >nul

echo.
podman ps | findstr "semantic_vectors_db"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ PostgreSQL is RUNNING
    echo.
    echo Connection details:
    echo   Host: localhost
    echo   Port: 5432
    echo   Database: semantic_vectors
    echo   User: postgres
    echo   Password: postgres
    echo.
    echo Connection string:
    echo   postgresql://postgres:postgres@localhost:5432/semantic_vectors
    echo.
) else (
    echo.
    echo ✗ PostgreSQL failed to start
    echo.
    echo Check logs with: podman logs semantic_vectors_db
)

echo.
pause
