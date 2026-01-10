@echo off
echo ============================================================
echo CHECKING TASK EMBEDDING COLLECTIONS
echo ============================================================
echo.

echo Checking if task embedding collections exist...
echo.

psql -h localhost -U postgres -d semantic_vectors -c "SELECT table_name, pg_size_pretty(pg_total_relation_size('vectors.' || table_name)) as size FROM information_schema.tables WHERE table_schema = 'vectors' AND table_name LIKE 'task_embeddings%%' ORDER BY table_name;"

echo.
echo ============================================================
echo Checking vector counts...
echo ============================================================
echo.

psql -h localhost -U postgres -d semantic_vectors << EOF
-- Check w100 collection
SELECT 'task_embeddings_w100_bge-small' as collection, COUNT(*) as vectors
FROM vectors."task_embeddings_w100_bge-small"
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'vectors'
    AND table_name = 'task_embeddings_w100_bge-small'
);

-- Check all collection
SELECT 'task_embeddings_all_bge-small' as collection, COUNT(*) as vectors
FROM vectors."task_embeddings_all_bge-small"
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'vectors'
    AND table_name = 'task_embeddings_all_bge-small'
);
EOF

echo.
echo ============================================================
echo.
pause
