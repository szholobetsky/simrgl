@echo off
REM RAG Research Experiment - Windows Startup Script
REM This script will run the complete experiment workflow

echo ================================================================================
echo RAG Research Experiment - Complete Workflow
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

REM Check if podman-compose is available
podman-compose --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: podman-compose not found. Please install: pip install podman-compose
    pause
    exit /b 1
)

echo [Step 1/6] Stopping any existing Qdrant containers...
podman-compose down
echo.

echo [Step 2/6] Clearing old Qdrant storage...
if exist qdrant_storage (
    echo Deleting qdrant_storage contents...
    rmdir /s /q qdrant_storage 2>nul
    mkdir qdrant_storage
) else (
    mkdir qdrant_storage
)
echo.

echo [Step 3/6] Starting Qdrant vector database...
podman-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start Qdrant
    pause
    exit /b 1
)
echo Waiting for Qdrant to be ready...
timeout /t 10 /nobreak >nul
echo.

echo [Step 4/6] Checking Qdrant connection...
curl -s http://localhost:6333/collections >nul 2>&1
if errorlevel 1 (
    echo WARNING: Qdrant might not be ready yet. Waiting additional 5 seconds...
    timeout /t 5 /nobreak >nul
)
echo Qdrant is ready!
echo.

echo [Step 5/6] Running ETL Pipeline...
echo This will process 9,799 tasks and create 18 experiment collections
echo Estimated time: 20-30 minutes
echo.
python etl_pipeline.py --split_strategy recent
if errorlevel 1 (
    echo ERROR: ETL Pipeline failed
    pause
    exit /b 1
)
echo.

echo [Step 6/7] Running Experiments...
echo This will evaluate all 18 experiment combinations
echo Estimated time: 10-15 minutes
echo.
python run_experiments.py --split_strategy recent
if errorlevel 1 (
    echo ERROR: Experiments failed
    pause
    exit /b 1
)
echo.

echo [Step 7/7] Backing up Qdrant collections...
echo This allows you to restore data on another machine without re-running ETL
echo.
python backup_restore_qdrant.py --action backup --output qdrant_snapshots
if errorlevel 1 (
    echo WARNING: Backup failed, but experiments completed successfully
)
echo.

echo ================================================================================
echo SUCCESS! Experiment Complete
echo ================================================================================
echo.
echo Results saved to: experiment_results.csv
echo Test set saved to: test_set.json
echo Logs saved to: experiment.log
echo Qdrant backup saved to: qdrant_snapshots/
echo.
echo To restore on another machine:
echo   1. Copy qdrant_snapshots folder and test_set.json
echo   2. Run: restore_qdrant.bat
echo   3. Run: rerun_experiments.bat
echo.
echo Starting Streamlit UI...
echo Open your browser to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the UI when done
echo ================================================================================
echo.

REM Start Streamlit UI
streamlit run experiment_ui.py

pause
