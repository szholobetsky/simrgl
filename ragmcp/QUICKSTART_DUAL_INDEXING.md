# Quick Start: Dual Indexing System

## TL;DR

Run these commands to set up the dual indexing system:

```batch
REM 1. Clear existing data (if needed)
cd exp3
clear_postgres_vectors.bat

REM 2. Create dual collections (RECENT + ALL)
run_etl_dual_postgres.bat

REM 3. Migrate RAWDATA to PostgreSQL
cd ..\ragmcp
migrate_rawdata_to_postgres.bat

REM 4. Run the agent - WEB INTERFACE (Recommended)
launch_two_phase_web.bat
```

**Or use CLI:**
```batch
two_phase_agent.bat
```

**Done!** The agent now searches both RECENT and ALL collections.

## What You Get

### RECENT Collections (Last 100 Tasks)
- **High precision** for current development work
- **Fast queries** with focused context
- Perfect for recent bugs and new features

### ALL Collections (Complete History)
- **Comprehensive coverage** of entire codebase
- Finds rare or old functionality
- Ensures nothing is missed

## Example Usage

### Web Interface

1. **Launch:** `launch_two_phase_web.bat`
2. **Open browser:** http://127.0.0.1:7860
3. **Enter query:** "Fix memory leak in network buffer pool"
4. **Click:** 🚀 Analyze with Two-Phase Agent
5. **View results in tabs:**
   - 📊 Summary
   - 🔍 Phase 1: File Selection
   - 🔬 Phase 2: Deep Analysis
   - 💡 Phase 3: Reflection

### Command Line

```
> Fix memory leak in network buffer pool

[1.1] Searching for similar tasks (DUAL: recent + all)...
[1.2] Searching for relevant modules (DUAL: recent + all)...
[1.3] Searching for relevant files (DUAL: recent + all)...

DUAL SEARCH RESULTS:

🎯 RECENT Collection (Last 100 Tasks)
1. NetworkBufferPool.java (similarity: 0.89)
2. BufferPool.java (similarity: 0.85)
...

📚 ALL Collection (Complete History)
1. legacy/OldBufferPool.java (similarity: 0.92) ← Found old fix!
2. NetworkBufferPool.java (similarity: 0.89)
...

💡 Recommendation:
Check ALL collection - shows old fix with higher similarity!
```

## When to Use Which?

| Scenario | Best Collection |
|----------|----------------|
| Recent bug fix | RECENT (high precision) |
| New feature development | RECENT (current context) |
| Finding old implementation | ALL (comprehensive) |
| Rare functionality | ALL (historical) |
| **Don't know?** | **DUAL (both!)** |

## Timing

### ETL (One-time setup)
- **CPU:** 35-45 minutes
- **GPU:** 12-18 minutes

### Queries (Per search)
- **RECENT only:** ~100ms
- **ALL only:** ~400ms
- **DUAL (both):** ~500ms

**Trade-off:** Slightly slower, but much better coverage!

## How It Works

```
User Query
    ↓
    ├─→ Search RECENT (last 100 tasks) → High precision results
    └─→ Search ALL (complete history)  → Comprehensive results
    ↓
Merge & Present Both
    ↓
LLM analyzes both sets
    ↓
Recommendations
```

## Configuration

Already configured! Default settings in `ragmcp/config.py`:

```python
# RECENT collections (last 100 tasks)
COLLECTION_MODULE_RECENT = 'rag_exp_desc_module_w100_modn_bge-small'
COLLECTION_FILE_RECENT = 'rag_exp_desc_file_w100_modn_bge-small'

# ALL collections (complete history)
COLLECTION_MODULE_ALL = 'rag_exp_desc_module_all_modn_bge-small'
COLLECTION_FILE_ALL = 'rag_exp_desc_file_all_modn_bge-small'
```

## Verify Setup

Check that collections exist:

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres -d semantic_vectors

# List all collections
\dt vectors.*

# Should see 6 tables:
# - rag_exp_desc_module_w100_modn_bge-small (RECENT modules)
# - rag_exp_desc_file_w100_modn_bge-small (RECENT files)
# - task_embeddings_w100_bge-small (RECENT tasks)
# - rag_exp_desc_module_all_modn_bge-small (ALL modules)
# - rag_exp_desc_file_all_modn_bge-small (ALL files)
# - task_embeddings_all_bge-small (ALL tasks)
# - rawdata (file content & diffs)
```

## Troubleshooting

### ETL fails at Phase 2
**Issue:** Ran out of memory

**Fix:**
```bash
# Create RECENT only (lighter)
python etl_pipeline.py --backend postgres --split_strategy modn \
  --sources desc --targets module file --windows w100 --model bge-small
```

### No RECENT results
**Normal!** Query might refer to older code

**Action:** Check ALL results - should show matches

### Slow queries
**Expected:** Dual search queries both collections

**Speed up:**
- Use `search_modules` (RECENT only) instead of `search_modules_dual`
- Or accept slight delay for better coverage

## Advanced: Adjust Window Size

Want different window? Edit `exp3/run_etl_dual_postgres.sh`:

```bash
# Change from w100 to w1000 (last 1000 tasks)
python3 etl_pipeline.py \
  --backend postgres \
  --windows w1000 \  # ← Change this
  ...
```

Then update `ragmcp/config.py` collection names to match.

## Comparison

| Setup | RECENT Only | ALL Only | DUAL (Both) |
|-------|-------------|----------|-------------|
| **ETL Time** | ~10 min | ~30 min | ~40 min |
| **Storage** | ~50 MB | ~100 MB | ~150 MB |
| **Query Speed** | Fast | Medium | Medium |
| **Precision (recent)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Coverage (old)** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Overall** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation:** Use DUAL for best results!

## Interface Comparison

| Feature | Web UI | CLI |
|---------|--------|-----|
| **Launch** | `launch_two_phase_web.bat` | `two_phase_agent.bat` |
| **Interface** | Browser (http://127.0.0.1:7860) | Terminal |
| **Output** | Formatted tabs with colors | Plain text |
| **History** | Automatic tracking | Manual |
| **Best For** | Interactive use, demos | Automation, scripts |

## What's Next?

1. ✅ Run setup commands above
2. ✅ Launch web interface (recommended for first use)
3. ✅ Test with example queries
4. ✅ Review all 3 phases in separate tabs
5. ✅ Compare RECENT vs ALL results
6. 📊 Observe which collection works better for different query types
7. 🔄 Re-index RECENT weekly to keep it fresh

## Help

- **Web interface guide:** See `WEB_INTERFACE_GUIDE.md`
- **Full documentation:** See `DUAL_INDEXING_GUIDE.md`
- **Agent documentation:** See `TWO_PHASE_AGENT_README.md`
- **ETL details:** See `../exp3/README_EXPERIMENTS.md`

## Quick Commands Reference

```batch
# Setup (one-time)
cd exp3
clear_postgres_vectors.bat           # Clear old data
run_etl_dual_postgres.bat            # Create collections (40 min)
cd ..\ragmcp
migrate_rawdata_to_postgres.bat      # Migrate file data (5 min)

# Run Agent
launch_two_phase_web.bat             # Web UI (recommended)
two_phase_agent.bat                  # CLI

# Old simple agent (for comparison)
local_agent.bat                      # Simple single-phase agent
```

## Summary

✅ **DUAL = Best of both worlds**
- High precision (RECENT)
- Comprehensive coverage (ALL)
- Automatic merging & ranking
- Clear source labels
- Intelligent recommendations

Just run the setup and enjoy better search results!
