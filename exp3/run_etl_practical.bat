@echo off
echo ========================================
echo ETL Pipeline - Practical Configuration
echo For MCP Server Production Use
echo ========================================
echo.
echo Creating embeddings for ALL tasks with:
echo - Split Strategy: modn (ID mod 200)
echo - Source: desc (TITLE + DESCRIPTION combined)
echo - Targets: module + file
echo - Window: all (complete history)
echo - Model: bge-small
echo.
echo This will create 2 Qdrant collections:
echo   1. rag_exp_desc_module_all_modn  (for semantic module search)
echo   2. rag_exp_desc_file_all_modn    (for file-level precision)
echo.
echo Estimated time: 15-20 minutes
echo ========================================
echo.

python etl_pipeline.py --split_strategy modn --sources desc --targets module file --windows all --model bge-small

echo.
echo ========================================
echo ETL Pipeline completed!
echo.
echo Collections created:
echo - rag_exp_desc_module_all_modn
echo - rag_exp_desc_file_all_modn
echo ========================================
pause
