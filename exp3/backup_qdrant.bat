@echo off
REM Backup Qdrant collections to snapshots
REM Creates snapshot files in qdrant_snapshots folder

echo ================================================================================
echo Qdrant Backup - Creating Snapshots
echo ================================================================================
echo.

REM Create backup directory
if not exist qdrant_snapshots mkdir qdrant_snapshots

REM Check Qdrant is running
curl -s http://localhost:6333/collections >nul 2>&1
if errorlevel 1 (
    echo ERROR: Qdrant is not running. Please start it first.
    pause
    exit /b 1
)

echo Backing up all collections...
echo.

python backup_restore_qdrant.py --action backup --output qdrant_snapshots
if errorlevel 1 (
    echo ERROR: Backup failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Backup Complete
echo ================================================================================
echo Snapshots saved to: qdrant_snapshots/
echo.
echo To restore on another machine:
echo   1. Copy qdrant_snapshots folder to the new machine
echo   2. Start Qdrant
echo   3. Run: restore_qdrant.bat
echo ================================================================================
pause
