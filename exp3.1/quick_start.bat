@echo off
REM Quick Start - Just launch the UI (experiments must be completed first)

echo ================================================================================
echo RAG Research Experiment - Quick Start UI
echo ================================================================================
echo.

REM Check if results exist
if not exist experiment_results.csv (
    echo ERROR: No results found. Please run start.bat first to complete experiments.
    echo.
    pause
    exit /b 1
)

echo Results found: experiment_results.csv
echo Test set found: test_set.json
echo.
echo Starting Streamlit UI...
echo Open your browser to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the UI
echo ================================================================================
echo.

streamlit run experiment_ui.py

pause
