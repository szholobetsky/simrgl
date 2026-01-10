@echo off

echo ========================================
echo RAG ETL Pipeline - DUAL INDEXING MODE
echo PostgreSQL Backend
echo ========================================
echo.
echo This creates TWO sets of vector collections:
echo.
echo 1. RECENT collections (last 100 tasks)
echo    - High precision for recent development work
echo    - Fast and accurate for current code changes
echo    - Collections:
echo      * rag_exp_desc_module_w100_modn_bge-small
echo      * rag_exp_desc_file_w100_modn_bge-small
echo      * task_embeddings_w100_bge-small
echo.
echo 2. ALL collections (complete history)
echo    - Comprehensive coverage of entire codebase
echo    - Finds rare/old functionality
echo    - Collections:
echo      * rag_exp_desc_module_all_modn_bge-small
echo      * rag_exp_desc_file_all_modn_bge-small
echo      * task_embeddings_all_bge-small
echo.
echo Parameters:
echo - Backend: PostgreSQL (pgvector)
echo - Split Strategy: modn (ID mod 200)
echo - Source: desc (TITLE + DESCRIPTION)
echo - Targets: module + file
echo - Windows: w100 (recent) + all (complete)
echo - Model: bge-small (384 dim)
echo.
echo Estimated time:
echo   CPU: 35-45 minutes
echo   GPU: 12-18 minutes
echo.
echo ========================================
echo.

REM Phase 1: Create RECENT collections (w100)
echo.
echo ========================================
echo PHASE 1: Creating RECENT collections
echo ========================================
echo.

python etl_pipeline.py --backend postgres --split_strategy modn --sources desc --targets module file --windows w100 --model bge-small

if errorlevel 1 (
    echo.
    echo ERROR: Phase 1 creation error
    pause
    exit /b 1
)

echo.
echo Phase 1 complete: RECENT collections created
echo.

REM Phase 2: Create ALL collections
echo.
echo ========================================
echo PHASE 2: Creating ALL collections
echo ========================================
echo.

python etl_pipeline.py --backend postgres --split_strategy modn --sources desc --targets module file --windows all --model bge-small

if errorlevel 1 (
    echo.
    echo ERROR: Phase 2 creation error
    pause
    exit /b 1
)

echo.
echo Phase 2 complete: ALL collections created
echo.

REM Phase 3: Create task embeddings for both
echo.
echo ========================================
echo PHASE 3: Creating task embeddings
echo ========================================
echo.

echo Creating task embeddings for recent tasks (w100)...
python create_task_collection.py --backend postgres --window w100 --model bge-small

echo.
echo Creating task embeddings for all tasks...
python create_task_collection.py --backend postgres --window all --model bge-small

echo.
echo ========================================
echo DUAL INDEXING COMPLETE!
echo ========================================
echo.
echo Collections created in PostgreSQL:
echo.
echo RECENT (last 100 tasks):
echo   [OK] rag_exp_desc_module_w100_modn_bge-small
echo   [OK] rag_exp_desc_file_w100_modn_bge-small
echo   [OK] task_embeddings_w100_bge-small
echo.
echo ALL (complete history):
echo   [OK] rag_exp_desc_module_all_modn_bge-small
echo   [OK] rag_exp_desc_file_all_modn_bge-small
echo   [OK] task_embeddings_all_bge-small
echo.
echo Next steps:
echo 1. Update ragmcp\config.py to use dual collections
echo 2. Run migrate_rawdata_to_postgres.bat for file access
echo 3. Test with two_phase_agent.bat
echo.
echo The agent will now:
echo - Search RECENT collections for current work (high precision)
echo - Search ALL collections for comprehensive coverage
echo - Merge results intelligently
echo ========================================
