# PostgreSQL + pgvector Setup Guide

## Overview

This guide explains how to use PostgreSQL with pgvector as an alternative to Qdrant for storing and searching vector embeddings.

### Why PostgreSQL?

- **Stability**: PostgreSQL is rock-solid and battle-tested
- **No corruption**: Unlike Qdrant which has had issues with collection corruption
- **Familiar**: Standard SQL database with vector extensions
- **Reliability**: Enterprise-grade database with proven reliability
- **Integration**: Easy to integrate with existing PostgreSQL infrastructure

## Prerequisites

- Python 3.8+
- Podman or Docker
- PostgreSQL client libraries

## Installation

### 1. Install Python Dependencies

```bash
pip install psycopg2-binary
```

Or use the requirements file:

```bash
pip install -r requirements_postgres.txt
```

### 2. Start PostgreSQL Container

**Windows:**
```cmd
start_postgres.bat
```

**Linux:**
```bash
chmod +x start_postgres.sh
./start_postgres.sh
```

This will start PostgreSQL 16 with pgvector extension on port 5432.

### 3. Test the Connection

```bash
python test_postgres.py
```

This script will:
- Test connection to PostgreSQL
- Verify pgvector extension
- Test vector operations
- Create the necessary schema

### 4. Configure the Backend

Edit `config.py`:

```python
# Change this line:
VECTOR_BACKEND = 'postgres'  # Instead of 'qdrant'
```

Or use `--backend postgres` when running scripts.

## Running ETL with PostgreSQL

### Quick Start (Recommended)

**Windows:**
```cmd
run_etl_postgres.bat
```

**Linux:**
```bash
chmod +x run_etl_postgres.sh
./run_etl_postgres.sh
```

### Manual Command

```bash
python etl_pipeline.py \
  --backend postgres \
  --split_strategy modn \
  --sources desc \
  --targets module file \
  --windows all \
  --model bge-small
```

### What Gets Created

The ETL pipeline creates PostgreSQL tables in the `vectors` schema:

- `rag_exp_desc_module_all_modn_bge-small` - Module-level vectors (64 modules)
- `rag_exp_desc_file_all_modn_bge-small` - File-level vectors (63,069 files)

Each table has:
- `id` - Primary key
- `path` - Module/file path
- `type` - 'module' or 'file'
- `vector` - 384-dimensional vector (bge-small)
- HNSW index for fast similarity search

## Using ragmcp with PostgreSQL

### 1. Update ragmcp Configuration

Edit `ragmcp/config.py`:

```python
VECTOR_BACKEND = 'postgres'
```

### 2. Launch Gradio UI

```bash
cd ragmcp
python gradio_ui.py
```

The UI will automatically use PostgreSQL backend for searches.

## Database Administration

### Connect to Database

```bash
podman exec -it semantic_vectors_db psql -U postgres -d semantic_vectors
```

### Useful SQL Commands

**List all tables in vectors schema:**
```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'vectors';
```

**Count vectors in a collection:**
```sql
SELECT COUNT(*) FROM vectors.rag_exp_desc_module_all_modn_bge_small;
```

**Check vector dimensions:**
```sql
SELECT vector_dims(vector) FROM vectors.rag_exp_desc_module_all_modn_bge_small LIMIT 1;
```

**Sample search query:**
```sql
SELECT path, type, 1 - (vector <=> '[0.1,0.2,...]'::vector) as similarity
FROM vectors.rag_exp_desc_module_all_modn_bge_small
ORDER BY vector <=> '[0.1,0.2,...]'::vector
LIMIT 10;
```

**Drop a collection:**
```sql
DROP TABLE IF EXISTS vectors.rag_exp_desc_module_all_modn_bge_small;
```

### View Index Information

```sql
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'vectors';
```

## Performance Tuning

### HNSW Index Parameters

The default HNSW index is created with:
```sql
CREATE INDEX idx_name ON table USING hnsw (vector vector_cosine_ops);
```

For better performance, you can customize:

```sql
-- More accurate but slower
CREATE INDEX idx_name ON table USING hnsw (vector vector_cosine_ops)
WITH (m = 32, ef_construction = 200);

-- Faster but less accurate
CREATE INDEX idx_name ON table USING hnsw (vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Query Performance

Set search parameters:
```sql
-- Higher = more accurate but slower
SET hnsw.ef_search = 100;
```

## Backup and Restore

### Backup Entire Database

```bash
podman exec semantic_vectors_db pg_dump -U postgres semantic_vectors > backup.sql
```

### Restore Database

```bash
podman exec -i semantic_vectors_db psql -U postgres semantic_vectors < backup.sql
```

### Backup Specific Table

```bash
podman exec semantic_vectors_db pg_dump -U postgres -t vectors.rag_exp_desc_module_all_modn_bge_small semantic_vectors > module_backup.sql
```

## Switching from Qdrant to PostgreSQL

If you already have data in Qdrant:

1. **Stop using Qdrant:**
   - Update `config.py`: `VECTOR_BACKEND = 'postgres'`

2. **Start PostgreSQL:**
   - Run `start_postgres.bat` or `start_postgres.sh`

3. **Run ETL:**
   - Run `run_etl_postgres.bat` or `run_etl_postgres.sh`
   - This will recreate all vectors in PostgreSQL

4. **Test ragmcp:**
   - Launch Gradio UI and verify searches work

5. **Optional - Stop Qdrant:**
   - `podman-compose -f qdrant-compose.yml down`

## Troubleshooting

### Connection Refused

**Problem:** Can't connect to PostgreSQL

**Solution:**
```bash
# Check if PostgreSQL is running
podman ps | grep semantic_vectors

# View logs
podman logs semantic_vectors_db

# Restart container
podman-compose -f postgres-compose.yml restart
```

### pgvector Extension Missing

**Problem:** `ERROR: type "vector" does not exist`

**Solution:**
```sql
-- Connect to database and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Slow Searches

**Problem:** Queries take too long

**Solutions:**
1. Check if index exists:
   ```sql
   \d vectors.rag_exp_desc_module_all_modn_bge_small
   ```

2. Rebuild index:
   ```sql
   REINDEX INDEX vectors.rag_exp_desc_module_all_modn_bge_small_vector_idx;
   ```

3. Analyze table:
   ```sql
   ANALYZE vectors.rag_exp_desc_module_all_modn_bge_small;
   ```

### Out of Memory

**Problem:** PostgreSQL runs out of memory during ETL

**Solution:** Increase shared_buffers in postgres-compose.yml:
```yaml
command: postgres -c shared_buffers=512MB -c max_connections=100
```

## Configuration Reference

### Database Settings (config.py)

```python
# PostgreSQL Configuration
POSTGRES_HOST = 'localhost'      # PostgreSQL host
POSTGRES_PORT = 5432             # PostgreSQL port
POSTGRES_DB = 'semantic_vectors' # Database name
POSTGRES_USER = 'postgres'       # Username
POSTGRES_PASSWORD = 'postgres'   # Password (change in production!)
POSTGRES_SCHEMA = 'vectors'      # Schema for vector tables
```

### Security Notes

**Default Credentials:**
- Username: `postgres`
- Password: `postgres`

**⚠️ For Production:**
1. Change the password in `postgres-compose.yml`
2. Update `POSTGRES_PASSWORD` in `config.py`
3. Use environment variables instead of hardcoded credentials
4. Enable SSL/TLS connections
5. Configure firewall rules

## Comparison: PostgreSQL vs Qdrant

| Feature | PostgreSQL+pgvector | Qdrant |
|---------|-------------------|--------|
| **Stability** | ✅ Rock-solid | ⚠️ Can corrupt |
| **Maturity** | ✅ 25+ years | ❌ Young project |
| **Query Language** | ✅ SQL | ❌ Custom API |
| **Admin Tools** | ✅ Many options | ⚠️ Limited |
| **Backup/Restore** | ✅ Built-in | ⚠️ Custom tools |
| **Speed (small)** | ✅ Very fast | ✅ Very fast |
| **Speed (large)** | ✅ Fast with HNSW | ✅ Very fast |
| **Integration** | ✅ Easy | ⚠️ Requires SDK |
| **Resource Usage** | ⚠️ Moderate | ✅ Low |

## Additional Resources

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [HNSW Index Tuning](https://github.com/pgvector/pgvector#hnsw)

## Support

If you encounter issues:

1. Check logs: `podman logs semantic_vectors_db`
2. Test connection: `python test_postgres.py`
3. Verify pgvector: `SELECT * FROM pg_extension WHERE extname = 'vector';`
4. Check table existence: `\dt vectors.*` (in psql)

---

**Last Updated:** 2025-12-22
**Version:** 1.0
