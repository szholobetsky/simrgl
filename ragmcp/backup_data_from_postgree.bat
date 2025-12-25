@echo off
REM Backup PostgreSQL Vector Collections
REM This script backs up the vectors schema containing all embeddings

echo ============================================================
echo Backing up PostgreSQL Vector Collections
echo ============================================================

REM Configuration
set CONTAINER_NAME=semantic_vectors_db
set DB_NAME=semantic_vectors
set DB_USER=postgres
set SCHEMA_NAME=vectors
set BACKUP_DIR=.\backups
set TIMESTAMP=%date:~-4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set BACKUP_FILE=%BACKUP_DIR%\vectors_backup_%TIMESTAMP%.sql

REM Create backup directory if it doesn't exist
if not exist "%BACKUP_DIR%" (
    echo Creating backup directory: %BACKUP_DIR%
    mkdir "%BACKUP_DIR%"
)

echo.
echo Backup Configuration:
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

echo [1/2] Creating backup...
podman exec %CONTAINER_NAME% pg_dump -U %DB_USER% -d %DB_NAME% -n %SCHEMA_NAME% --clean --if-exists > "%BACKUP_FILE%"

if errorlevel 1 (
    echo [ERROR] Backup failed!
    exit /b 1
)

echo [2/2] Verifying backup...
for %%A in ("%BACKUP_FILE%") do set BACKUP_SIZE=%%~zA

if not defined BACKUP_SIZE (
    echo [ERROR] Could not get backup file size!
    exit /b 1
)

if %BACKUP_SIZE% LSS 1000 (
    echo [ERROR] Backup file is too small ^(%BACKUP_SIZE% bytes^). Something went wrong!
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] Backup completed successfully!
echo ============================================================
echo   File: %BACKUP_FILE%
echo   Size: %BACKUP_SIZE% bytes
echo.
echo To restore this backup, run:
echo   restore_data_to_postgree.bat "%BACKUP_FILE%"
echo ============================================================

exit /b 0
