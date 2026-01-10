-- Check which vector collections exist in PostgreSQL
-- Run this: psql -h localhost -U postgres -d semantic_vectors -f check_collections.sql

\echo '=== CHECKING VECTOR COLLECTIONS IN POSTGRESQL ==='
\echo ''

-- Check all tables in vectors schema
\echo '1. All tables in vectors schema:'
SELECT
    table_name,
    pg_size_pretty(pg_total_relation_size('vectors.' || table_name)) as size
FROM information_schema.tables
WHERE table_schema = 'vectors'
  AND table_name LIKE 'rag_%' OR table_name LIKE 'task_%'
ORDER BY table_name;

\echo ''
\echo '2. Detailed collection info:'

-- RECENT collections (w100)
\echo ''
\echo 'RECENT Collections (last 100 tasks):'
SELECT 'rag_exp_desc_module_w100_modn_bge-small' as collection,
       COUNT(*) as vector_count
FROM vectors."rag_exp_desc_module_w100_modn_bge-small"
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'vectors'
      AND table_name = 'rag_exp_desc_module_w100_modn_bge-small'
);

SELECT 'rag_exp_desc_file_w100_modn_bge-small' as collection,
       COUNT(*) as vector_count
FROM vectors."rag_exp_desc_file_w100_modn_bge-small"
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'vectors'
      AND table_name = 'rag_exp_desc_file_w100_modn_bge-small'
);

SELECT 'task_embeddings_w100_bge-small' as collection,
       COUNT(*) as vector_count
FROM vectors."task_embeddings_w100_bge-small"
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'vectors'
      AND table_name = 'task_embeddings_w100_bge-small'
);

-- ALL collections
\echo ''
\echo 'ALL Collections (complete history):'
SELECT 'rag_exp_desc_module_all_modn_bge-small' as collection,
       COUNT(*) as vector_count
FROM vectors."rag_exp_desc_module_all_modn_bge-small"
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'vectors'
      AND table_name = 'rag_exp_desc_module_all_modn_bge-small'
);

SELECT 'rag_exp_desc_file_all_modn_bge-small' as collection,
       COUNT(*) as vector_count
FROM vectors."rag_exp_desc_file_all_modn_bge-small"
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'vectors'
      AND table_name = 'rag_exp_desc_file_all_modn_bge-small'
);

SELECT 'task_embeddings_all_bge-small' as collection,
       COUNT(*) as vector_count
FROM vectors."task_embeddings_all_bge-small"
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'vectors'
      AND table_name = 'task_embeddings_all_bge-small'
);
