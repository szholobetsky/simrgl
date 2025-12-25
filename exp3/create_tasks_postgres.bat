@echo off
echo ========================================
echo Create Task Embeddings - PostgreSQL
echo ========================================
echo.
echo This will create individual task embeddings
echo in PostgreSQL for task-to-task similarity
echo search and module recreation.
echo.
echo Parameters:
echo - Backend: PostgreSQL (pgvector)
echo - Source: TITLE + DESCRIPTION
echo - Granularity: Individual tasks
echo - Model: bge-small (384 dim)
echo.
echo Expected output:
echo   task_embeddings_all_bge-small (PostgreSQL table)
echo   (~9,799 tasks)
echo.
echo Estimated time: 10-15 minutes (CPU)
echo                 3-5 minutes (GPU)
echo ========================================
echo.

python create_task_collection.py --backend postgres

echo.
echo ========================================
echo Task embeddings created in PostgreSQL!
echo.
echo You can now:
echo - Search for similar historical tasks
echo - Recreate module embeddings from tasks
echo - Find what similar problems were solved
echo ========================================
pause
