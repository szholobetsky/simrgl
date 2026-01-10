@echo off
echo ============================================================
echo CHECKING POSTGRESQL VECTOR COLLECTIONS
echo ============================================================
echo.
echo This will show which collections exist in PostgreSQL
echo.

psql -h localhost -U postgres -d semantic_vectors -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'vectors' AND (table_name LIKE 'rag_%%' OR table_name LIKE 'task_%%') ORDER BY table_name;"

echo.
echo ============================================================
echo.
pause
