# Simple Agent (local_agent) - Collection Configuration

## Current Configuration

The simple agent (`local_agent.py` and `local_agent_web.py`) uses `mcp_server_postgres.py` which searches these collections from `config.py`:

```python
# From config.py lines 39-41:
COLLECTION_MODULE = COLLECTION_MODULE_RECENT  # rag_exp_desc_module_w100_modn_bge-small
COLLECTION_FILE = COLLECTION_FILE_RECENT      # rag_exp_desc_file_w100_modn_bge-small
COLLECTION_TASK = COLLECTION_TASK_ALL         # task_embeddings_all_bge-small
```

## Collections Used

| Tool | Collection Name | Description |
|------|----------------|-------------|
| `search_modules` | `rag_exp_desc_module_w100_modn_bge-small` | RECENT modules (last 100 tasks) |
| `search_files` | `rag_exp_desc_file_w100_modn_bge-small` | RECENT files (last 100 tasks) |
| `search_similar_tasks` | `task_embeddings_all_bge-small` | ALL tasks (complete history) |

## Verify Collections Exist

### Quick Check (Windows):
```bash
cd ragmcp
check_collections.bat
```

### Detailed Check:
```bash
psql -h localhost -U postgres -d semantic_vectors -f ragmcp/check_collections.sql
```

### Expected Collections

After running `run_etl_dual_postgres.bat/sh`, you should have 6 collections:

**RECENT (w100 - last 100 tasks):**
- `rag_exp_desc_module_w100_modn_bge-small`
- `rag_exp_desc_file_w100_modn_bge-small`
- `task_embeddings_w100_bge-small`

**ALL (complete history):**
- `rag_exp_desc_module_all_modn_bge-small`
- `rag_exp_desc_file_all_modn_bge-small`
- `task_embeddings_all_bge-small`

## If Collections Don't Exist

If you haven't run the dual indexing ETL yet:

```bash
cd exp3
run_etl_dual_postgres.bat  # Windows
# or
./run_etl_dual_postgres.sh  # Linux/Mac
```

This takes 35-45 minutes on CPU, 12-18 minutes on GPU.

## Changing Collections

If you want the simple agent to use different collections, edit `config.py` lines 39-41:

```python
# To use ALL collections instead of RECENT:
COLLECTION_MODULE = COLLECTION_MODULE_ALL  # Use complete history
COLLECTION_FILE = COLLECTION_FILE_ALL      # Use complete history
COLLECTION_TASK = COLLECTION_TASK_ALL      # Already uses ALL

# Or use specific collection names directly:
COLLECTION_MODULE = 'your_module_collection_name'
COLLECTION_FILE = 'your_file_collection_name'
COLLECTION_TASK = 'your_task_collection_name'
```

## Comparison: Simple Agent vs Two-Phase Agent

| Feature | Simple Agent | Two-Phase Agent |
|---------|-------------|-----------------|
| Collections | Single (RECENT or ALL) | Dual (RECENT + ALL) |
| Search Strategy | One collection per search | Parallel search in both |
| Speed | Faster (single search) | Slower (dual search) |
| Coverage | Good for recent work | Comprehensive |
| Analysis | Single-pass RAG | 3-phase with reflection |
| UI Port | 7861 | 7860 |

## Troubleshooting

### Error: "No results found in collection"

1. **Check if collections exist:**
   ```bash
   cd ragmcp
   check_collections.bat
   ```

2. **Verify collection has vectors:**
   ```sql
   SELECT COUNT(*) FROM vectors."rag_exp_desc_module_w100_modn_bge-small";
   ```

3. **If empty, run ETL:**
   ```bash
   cd exp3
   run_etl_dual_postgres.bat
   ```

### Error: "Table does not exist"

The collection name in `config.py` doesn't match what's in PostgreSQL.

**Fix:**
1. Check what collections you have (use `check_collections.bat`)
2. Update `config.py` to match the actual collection names

### Using Old Collections

If you have older collections with different naming:

```python
# Example: If you have these older collections:
# - rag_module_embeddings
# - rag_file_embeddings
# - task_embeddings

COLLECTION_MODULE = 'rag_module_embeddings'
COLLECTION_FILE = 'rag_file_embeddings'
COLLECTION_TASK = 'task_embeddings'
```

## Performance Notes

- **RECENT collections** (w100): ~100 tasks worth of vectors, faster search
- **ALL collections**: Complete history, more comprehensive but slower
- Simple agent uses RECENT by default for balance of speed and coverage
- Two-phase agent searches both for best results
