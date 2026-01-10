@echo off
echo ============================================================
echo CREATE MISSING TASK EMBEDDINGS
echo ============================================================
echo.
echo This will create the missing task embedding collections
echo that were not created by the previous run.
echo.
echo Collections to create:
echo   - task_embeddings_all_bge-small (ALL tasks)
echo.
echo Estimated time: 2-5 minutes
echo.
echo ============================================================
echo.
pause
echo.

echo Creating task_embeddings_all_bge-small...
python create_task_collection.py --backend postgres --window all --model bge-small

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create task embeddings
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS: Task embeddings created!
echo ============================================================
echo.
echo Collection created:
echo   - task_embeddings_all_bge-small
echo.
echo You can now use the simple agent and two-phase agent.
echo ============================================================
echo.
pause
