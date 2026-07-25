@echo off
REM Restore Qdrant collections from snapshots
REM Reads snapshot files from qdrant_snapshots folder

echo ================================================================================
echo Qdrant Restore - Loading Snapshots
echo ================================================================================
echo.

REM Check backup directory exists
if not exist qdrant_snapshots (
    echo ERROR: qdrant_snapshots folder not found
    echo Please copy the backup folder to this directory first.
    pause
    exit /b 1
)

REM Start Qdrant if not running
podman start rag_experiment_qdrant >nul 2>&1
ping -n 6 127.0.0.1 >nul

REM Check Qdrant is running
curl -s http://localhost:6333/collections >nul 2>&1
if errorlevel 1 (
    echo ERROR: Qdrant is not running. Please start it first.
    pause
    exit /b 1
)

echo Restoring all collections...
echo.

python backup_restore_qdrant.py --action restore --input qdrant_snapshots
if errorlevel 1 (
    echo ERROR: Restore failed
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Restore Complete
echo ================================================================================
echo Collections restored from: qdrant_snapshots/
echo.
echo You can now run experiments: rerun_experiments.bat
echo ================================================================================
pause
