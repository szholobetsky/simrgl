@echo off

echo ============================================================
echo CLEAR POSTGRESQL VECTOR COLLECTIONS
echo ============================================================
echo.
echo Database: localhost:5432/semantic_vectors
echo Schema: vectors
echo.
echo WARNING: This will delete ALL vector collections and rawdata!
echo ============================================================
echo.

python clear_postgres_vectors.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to clear PostgreSQL vectors
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS: All vector collections cleared
echo ============================================================
echo.
echo You can now run:
echo   run_etl_dual_postgres.bat
echo.
echo to create fresh DUAL collections (RECENT + ALL)
echo ============================================================
