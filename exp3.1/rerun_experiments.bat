@echo off
REM RAG Research Experiment - Rerun Experiments Only
REM Use this script when Qdrant already has the collections from a previous ETL run

echo ================================================================================
echo RAG Research Experiment - Rerun Experiments Only
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

echo [Step 1/4] Checking if Qdrant container exists...
podman ps -a --filter "name=rag_experiment_qdrant" --format "{{.Names}}" | findstr "rag_experiment_qdrant" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Qdrant container not found. Please run start.bat first to create collections.
    pause
    exit /b 1
)

echo [Step 2/4] Starting Qdrant if not running...
podman start rag_experiment_qdrant >nul 2>&1
echo Waiting for Qdrant to be ready...
ping -n 6 127.0.0.1 >nul
echo.

echo [Step 3/4] Verifying Qdrant connection and collections...
curl -s http://localhost:6333/collections >nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot connect to Qdrant. Please ensure it's running.
    pause
    exit /b 1
)
echo Qdrant is ready!
echo.

echo [Step 4/4] Running Experiments...
echo This will evaluate all 18 experiment combinations using existing collections
echo Estimated time: 10-15 minutes
echo.
python run_experiments.py --split_strategy recent
if errorlevel 1 (
    echo ERROR: Experiments failed
    pause
    exit /b 1
)
echo.

echo ================================================================================
echo SUCCESS! Experiments Complete
echo ================================================================================
echo.
echo Results saved to: experiment_results.csv
echo.
echo To view results in the UI, run: quick_start.bat
echo ================================================================================
echo.

pause
