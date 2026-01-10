# Collection Existence Checks

## Overview

Both the simple agent (`local_agent.py`) and the two-phase agent (`two_phase_agent.py`) now verify that all required PostgreSQL vector collections exist before starting. This prevents cryptic runtime errors and provides clear guidance when collections are missing.

## What Gets Checked

### Simple Agent (`local_agent.py` and `local_agent_web.py`)

Checks for **3 required collections**:

| Collection | Type | Description |
|-----------|------|-------------|
| `rag_exp_desc_module_w100_modn_bge-small` | RECENT (w100) | Last 100 modules |
| `rag_exp_desc_file_w100_modn_bge-small` | RECENT (w100) | Last 100 files |
| `task_embeddings_all_bge-small` | ALL | All historical tasks |

### Two-Phase Agent (`two_phase_agent.py` and `two_phase_agent_web.py`)

Checks for **6 required collections**:

| Collection | Type | Description |
|-----------|------|-------------|
| `rag_exp_desc_module_w100_modn_bge-small` | RECENT (w100) | Last 100 modules |
| `rag_exp_desc_file_w100_modn_bge-small` | RECENT (w100) | Last 100 files |
| `task_embeddings_w100_bge-small` | RECENT (w100) | Last 100 tasks |
| `rag_exp_desc_module_all_modn_bge-small` | ALL | All modules |
| `rag_exp_desc_file_all_modn_bge-small` | ALL | All files |
| `task_embeddings_all_bge-small` | ALL | All tasks |

## When Checks Happen

### CLI Mode (`local_agent.py`, `two_phase_agent.py`)

Collection check runs during `agent.initialize()`:

```
[INIT] Starting Local Coding Agent...
[INIT] MCP Server: mcp_server_postgres.py
[INIT] LLM Model: qwen2.5-coder:latest
[INIT] Ollama URL: http://localhost:11434

[INIT] Checking PostgreSQL collections...
  ✓ Found 3/3 required collections
[OK] All required collections exist

[OK] Ollama is running
[OK] MCP Server connected (3 tools available)
```

If a collection is missing:

```
[INIT] Checking PostgreSQL collections...
[ERROR] Missing required collections:
  ✗ Tasks (ALL): task_embeddings_all_bge-small

To fix:
  cd exp3
  create_missing_task_embeddings.bat  # Windows
  ./create_missing_task_embeddings.sh  # Linux/Mac
```

The agent will **refuse to start** and return `False` from `initialize()`.

### Web Mode (`local_agent_web.py`, `two_phase_agent_web.py`)

Collection check runs at **startup** before Gradio launches:

```
============================================================
Starting Local Offline Coding Agent - Web Interface
============================================================

Running pre-flight checks...

  [1/2] Checking PostgreSQL connection...
        ✓ PostgreSQL connected (8 tables in 'vectors' schema)
  [2/2] Checking vector collections...
        ✓ Found 3/3 required collections

============================================================
Initializing Gradio UI...
============================================================

✓ Gradio UI ready
Launching web server...
  Server will be available at: http://127.0.0.1:7861
============================================================
```

If a collection is missing, the web server **exits immediately**:

```
  [2/2] Checking vector collections...
        ✓ Found 2/3 required collections

        ⚠ Missing collections:
          ✗ Tasks (ALL): task_embeddings_all_bge-small

        To fix:
          cd exp3
          create_missing_task_embeddings.bat  # Windows
          ./create_missing_task_embeddings.sh  # Linux/Mac

[Script exits with code 1]
```

## Error Messages

### Missing Task Collection

```
[ERROR] Missing required collections:
  ✗ Tasks (ALL): task_embeddings_all_bge-small

To fix:
  cd exp3
  create_missing_task_embeddings.bat  # Windows
  ./create_missing_task_embeddings.sh  # Linux/Mac
```

**Resolution Time:** 2-5 minutes (creates only task embeddings)

### Missing Module/File Collections

```
[ERROR] Missing required collections:
  ✗ Modules (RECENT w100): rag_exp_desc_module_w100_modn_bge-small
  ✗ Files (RECENT w100): rag_exp_desc_file_w100_modn_bge-small

To fix:
  cd exp3
  run_etl_dual_postgres.bat  # Windows
  ./run_etl_dual_postgres.sh  # Linux/Mac
```

**Resolution Time:** 35-45 minutes on CPU, 12-18 minutes on GPU (full dual indexing)

### PostgreSQL Not Running

```
[ERROR] Cannot connect to PostgreSQL: connection refused

Make sure PostgreSQL is running:
  podman ps  # Check if container is running
```

**Resolution:** Start PostgreSQL container

## Implementation Details

### Code Location

**Simple Agent:**
- `ragmcp/local_agent.py:92-154` - `_check_collections()` method
- `ragmcp/local_agent_web.py:195-288` - Startup checks

**Two-Phase Agent:**
- `ragmcp/two_phase_agent_web.py:604-638` - Startup checks

### How It Works

1. **Connect to PostgreSQL:**
   ```python
   conn = psycopg2.connect(
       host=config.POSTGRES_HOST,
       port=config.POSTGRES_PORT,
       database=config.POSTGRES_DB,
       user=config.POSTGRES_USER,
       password=config.POSTGRES_PASSWORD,
       connect_timeout=3
   )
   ```

2. **Query for each collection:**
   ```python
   cursor.execute("""
       SELECT COUNT(*) FROM information_schema.tables
       WHERE table_schema = %s AND table_name = %s
   """, (config.POSTGRES_SCHEMA, collection_name))
   exists = cursor.fetchone()[0] > 0
   ```

3. **Report missing collections:**
   - List each missing collection with friendly name
   - Provide specific fix command based on what's missing
   - Exit gracefully with helpful error message

## Benefits

### Before Collection Checks

**Problem:** Runtime errors deep in execution:

```
> Fix authentication bug

[1/3] Searching semantic database...
[OK] Retrieved semantic context
[2/3] Building context for LLM...
[3/3] Generating LLM recommendations...

Error executing search_similar_tasks: relation "vectors.task_embeddings_all_bge-small" does not exist
LINE 3: FROM vectors."task_embeddings_all_bge-small"
```

**Issues:**
- Error happens after agent starts
- Error message is cryptic (PostgreSQL line numbers)
- User has to dig through logs to understand
- Time wasted on queries before error

### After Collection Checks

**Solution:** Clear error at startup:

```
[INIT] Checking PostgreSQL collections...
[ERROR] Missing required collections:
  ✗ Tasks (ALL): task_embeddings_all_bge-small

To fix:
  cd exp3
  create_missing_task_embeddings.bat
```

**Benefits:**
- Fail fast - error happens immediately
- Clear error message with collection name
- Actionable fix command provided
- No time wasted on doomed operations
- User knows exactly what to do

## Testing Collection Checks

### Test Missing Collection

1. **Rename a collection to simulate missing:**
   ```sql
   -- In psql
   ALTER TABLE vectors.task_embeddings_all_bge-small
   RENAME TO task_embeddings_all_bge-small_backup;
   ```

2. **Try to start agent:**
   ```bash
   cd ragmcp
   python local_agent.py
   ```

3. **Verify error message appears:**
   ```
   [ERROR] Missing required collections:
     ✗ Tasks (ALL): task_embeddings_all_bge-small

   To fix:
     cd exp3
     create_missing_task_embeddings.bat
   ```

4. **Restore collection:**
   ```sql
   ALTER TABLE vectors.task_embeddings_all_bge-small_backup
   RENAME TO task_embeddings_all_bge-small;
   ```

### Test PostgreSQL Connection Failure

1. **Stop PostgreSQL:**
   ```bash
   podman stop postgres_simrgl
   ```

2. **Try to start agent:**
   ```bash
   python local_agent.py
   ```

3. **Verify error message:**
   ```
   [ERROR] Cannot connect to PostgreSQL: connection refused

   Make sure PostgreSQL is running:
     podman ps  # Check if container is running
   ```

4. **Restart PostgreSQL:**
   ```bash
   podman start postgres_simrgl
   ```

## Configuration

Collections are defined in `ragmcp/config.py`:

```python
# RECENT (w100) collections
COLLECTION_MODULE_RECENT = 'rag_exp_desc_module_w100_modn_bge-small'
COLLECTION_FILE_RECENT = 'rag_exp_desc_file_w100_modn_bge-small'
COLLECTION_TASK_RECENT = 'task_embeddings_w100_bge-small'

# ALL collections
COLLECTION_MODULE_ALL = 'rag_exp_desc_module_all_modn_bge-small'
COLLECTION_FILE_ALL = 'rag_exp_desc_file_all_modn_bge-small'
COLLECTION_TASK_ALL = 'task_embeddings_all_bge-small'

# Simple agent uses these
COLLECTION_MODULE = COLLECTION_MODULE_RECENT  # w100
COLLECTION_FILE = COLLECTION_FILE_RECENT      # w100
COLLECTION_TASK = COLLECTION_TASK_ALL         # all
```

To change which collections are used, modify these constants in `config.py`.

## Related Documentation

- `TASK_EMBEDDINGS_FIX.md` - How the task embeddings bug was fixed
- `SIMPLE_AGENT_COLLECTIONS.md` - Collections used by simple agent
- `QUICKSTART_DUAL_INDEXING.md` - How to create all collections

## Summary

Collection existence checks provide:

- **Fail Fast**: Errors at startup, not during execution
- **Clear Messages**: Friendly collection names, not SQL table names
- **Actionable Fixes**: Exact commands to resolve the issue
- **Better UX**: Users know immediately what's wrong and how to fix it

This makes the system more robust and user-friendly, especially when recovering from ETL script bugs or incomplete migrations.
