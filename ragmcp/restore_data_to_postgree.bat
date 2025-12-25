@echo off
REM Restore PostgreSQL Vector Collections
REM This script restores the vectors schema from a backup file

echo ============================================================
echo Restoring PostgreSQL Vector Collections
echo ============================================================

REM Configuration
set CONTAINER_NAME=semantic_vectors_db
set DB_NAME=semantic_vectors
set DB_USER=postgres
set SCHEMA_NAME=vectors

REM Get backup file from command line argument or use latest
if "%~1"=="" (
    echo No backup file specified. Looking for latest backup...
    for /f "delims=" %%i in ('dir /b /o-d ".\backups\vectors_backup_*.sql" 2^>nul') do (
        set BACKUP_FILE=.\backups\%%i
        goto :found
    )
    echo [ERROR] No backup files found in .\backups\
    echo Usage: %~nx0 [backup_file.sql]
    exit /b 1
    :found
    echo Found latest backup: %BACKUP_FILE%
) else (
    set BACKUP_FILE=%~1
    if not exist "%BACKUP_FILE%" (
        echo [ERROR] Backup file not found: %BACKUP_FILE%
        exit /b 1
    )
)

echo.
echo Restore Configuration:
echo   Container: %CONTAINER_NAME%
echo   Database: %DB_NAME%
echo   Schema: %SCHEMA_NAME%
echo   Backup file: %BACKUP_FILE%
echo.

REM Check if container is running
podman ps --filter name=%CONTAINER_NAME% --format "{{.Names}}" | findstr /C:"%CONTAINER_NAME%" >nul
if errorlevel 1 (
    echo [ERROR] Container %CONTAINER_NAME% is not running!
    echo Please start PostgreSQL first: podman start %CONTAINER_NAME%
    exit /b 1
)

echo [WARNING] This will DROP and recreate the '%SCHEMA_NAME%' schema!
echo Press Ctrl+C to cancel, or
pause

echo.
echo [1/2] Restoring from backup...
type "%BACKUP_FILE%" | podman exec -i %CONTAINER_NAME% psql -U %DB_USER% -d %DB_NAME%

if errorlevel 1 (
    echo [ERROR] Restore failed!
    exit /b 1
)

echo [2/2] Verifying restore...
podman exec %CONTAINER_NAME% psql -U %DB_USER% -d %DB_NAME% -c "\dt %SCHEMA_NAME%.*"

echo.
echo ============================================================
echo [SUCCESS] Restore completed successfully!
echo ============================================================
echo   Restored from: %BACKUP_FILE%
echo   Schema: %SCHEMA_NAME%
echo.
echo You can now use the restored collections in your RAG system.
echo ============================================================

exit /b 0
