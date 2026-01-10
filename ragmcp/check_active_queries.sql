-- Check what's currently running in PostgreSQL
-- Run this: psql -h localhost -U postgres -d semantic_vectors -f check_active_queries.sql

\echo '=== ACTIVE QUERIES ==='
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    LEFT(query, 100) as query_preview,
    query_start,
    NOW() - query_start as duration
FROM pg_stat_activity
WHERE state != 'idle'
  AND pid != pg_backend_pid()
ORDER BY query_start;

\echo ''
\echo '=== LONG-RUNNING QUERIES (>5 seconds) ==='
SELECT
    pid,
    LEFT(query, 200) as query,
    NOW() - query_start as duration
FROM pg_stat_activity
WHERE state = 'active'
  AND NOW() - query_start > interval '5 seconds'
  AND pid != pg_backend_pid();

\echo ''
\echo '=== CHECK IF MIGRATION IS RUNNING ==='
SELECT
    pid,
    state,
    LEFT(query, 100) as query
FROM pg_stat_activity
WHERE query LIKE '%INSERT INTO%rawdata%'
   OR query LIKE '%migrate%';
