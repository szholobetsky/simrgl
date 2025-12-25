# 🚀 Quick Test Guide - RAG System in 10 Minutes

## Overview

This guide shows how to **test the complete RAG system** in just **5-10 minutes** using the last 1000 tasks instead of all tasks.

## Why Test Mode?

| Mode | Tasks | Time | Use Case |
|------|-------|------|----------|
| **Test (w1000)** | Last 1000 | 5-8 min | Quick testing, development |
| **Production (all)** | All ~9,799 | 60-70 min | Full deployment |

**Recommendation**: Start with test mode to verify everything works, then run production mode overnight.

## Quick Test (10 Minutes Total)

### Step 1: Start Vector Database (1 minute)

**Choose one:**

**For Qdrant:**
```bash
cd exp3
podman-compose -f qdrant-compose.yml up -d
```

**For PostgreSQL:**
```bash
cd exp3
podman-compose -f postgres-compose.yml up -d
```

### Step 2: Run Test ETL (5-8 minutes)

**For Qdrant:**
```bash
cd exp3
run_etl_test_qdrant.bat
```

**For PostgreSQL:**
```bash
cd exp3
run_etl_test_postgres.bat
```

**What happens:**
- Processes last 1000 tasks only
- Creates ~20-30 modules
- Creates ~3,000-5,000 files
- Takes 5-8 minutes on CPU, 2-3 minutes on GPU

### Step 3: Switch to Test Collections (30 seconds)

```bash
cd ragmcp
use_test_mode.bat
```

**What it does:**
- Updates config.py automatically
- Points to w1000 collections
- No manual editing needed

### Step 4: Test RAG System (1 minute)

**Without LLM (Vector Search Only):**
```bash
cd ragmcp
python gradio_ui.py
```

1. Open http://localhost:7860
2. Go to "Search" tab
3. Try: "Fix memory leak in buffer pool"
4. See results instantly!

**With LLM (Full RAG):**

1. Install Ollama (if not already):
   ```bash
   # Download from https://ollama.ai
   ollama pull qwen2.5-coder
   ollama serve
   ```

2. In Gradio UI:
   - Go to "RAG + LLM" tab
   - Enter task description
   - Click "Run RAG + LLM"
   - Get AI recommendations!

### Step 5: Verify It Works

**Check Search Tab:**
- Should find relevant modules
- Should show similarity scores
- Should be fast (<1 second)

**Check RAG + LLM Tab:**
- Should find modules, files, tasks
- Should retrieve code context
- Should generate LLM recommendations (3-5 seconds)

## Example Test Queries

### 1. Memory Leak
```
Fix memory leak in network connection pool
```

**Expected Results:**
- Relevant network/connection modules
- ConnectionPool.java or similar files
- Similar historical task about resource leaks

### 2. Feature Addition
```
Add authentication to API endpoints
```

**Expected Results:**
- Authentication/security modules
- API handler files
- Historical authentication tasks

### 3. Performance
```
Optimize query execution performance
```

**Expected Results:**
- Query processing modules
- Execution engine files
- Past performance optimization tasks

## Switching Between Test and Production

### Use Test Mode (Fast)

```bash
cd ragmcp
use_test_mode.bat
```

**When to use:**
- Development
- Testing new features
- Quick iterations
- Learning the system

### Use Production Mode (Complete)

```bash
cd ragmcp
use_production_mode.bat
```

**When to use:**
- Final deployment
- Production use
- Maximum accuracy
- Complete history

## Manual Verification

### Check Vector Database

**For Qdrant:**
```bash
# Open Qdrant dashboard
http://localhost:6333/dashboard

# Check collections exist:
# - rag_exp_desc_module_w1000_modn_bge-small
# - rag_exp_desc_file_w1000_modn_bge-small
```

**For PostgreSQL:**
```bash
podman exec -it semantic_vectors_db psql -U postgres -d semantic_vectors

# List tables:
\dt vectors.*

# Check counts:
SELECT COUNT(*) FROM vectors.rag_exp_desc_module_w1000_modn_bge_small;
SELECT COUNT(*) FROM vectors.rag_exp_desc_file_w1000_modn_bge_small;
```

### Check Config

```bash
cd ragmcp
cat config.py | grep COLLECTION
```

**Should show (test mode):**
```python
COLLECTION_MODULE = 'rag_exp_desc_module_w1000_modn_bge-small'
COLLECTION_FILE = 'rag_exp_desc_file_w1000_modn_bge-small'
```

## Troubleshooting Test Mode

### "Collection not found"

**Problem**: Test collections don't exist

**Solution**:
```bash
cd exp3
run_etl_test_qdrant.bat  # or run_etl_test_postgres.bat
```

### "Using wrong collections"

**Problem**: Config still points to production

**Solution**:
```bash
cd ragmcp
use_test_mode.bat
```

### "No results found"

**Problem**: Collections empty or corrupted

**Solution**:
```bash
# Recreate test collections
cd exp3
run_etl_test_qdrant.bat
```

### "LLM not working"

**Problem**: Ollama not running

**Solution**:
```bash
ollama serve
ollama pull qwen2.5-coder
```

## Complete Test Workflow

### Full System Test (10 Minutes)

```bash
# 1. Start database (choose one)
cd exp3
podman-compose -f qdrant-compose.yml up -d
# OR
podman-compose -f postgres-compose.yml up -d

# 2. Run test ETL (5-8 min)
run_etl_test_qdrant.bat
# OR
run_etl_test_postgres.bat

# 3. Switch to test mode
cd ../ragmcp
use_test_mode.bat

# 4. Start Ollama (optional, for LLM)
ollama serve
# (in new terminal)
ollama pull qwen2.5-coder

# 5. Launch UI
python gradio_ui.py

# 6. Test in browser
# http://localhost:7860
# Try "Search" tab first
# Then try "RAG + LLM" tab

# 7. Verify results
# - Search finds relevant items
# - LLM generates recommendations
# - Everything works fast
```

### Expected Times

| Step | Time |
|------|------|
| Start database | 30 seconds |
| Run test ETL | 5-8 minutes |
| Switch config | 10 seconds |
| Start Ollama | 1 minute |
| Pull model (first time) | 2-3 minutes |
| Launch UI | 10 seconds |
| Test query (no LLM) | < 1 second |
| Test query (with LLM) | 3-5 seconds |
| **Total** | **~10-15 minutes** |

## After Testing Successfully

### Option 1: Keep Using Test Mode

Good if:
- Fast iterations needed
- Limited compute resources
- Recent tasks most relevant
- Development/learning phase

### Option 2: Switch to Production

When ready:
```bash
# 1. Run full ETL (60-70 min)
cd exp3
run_etl_practical.bat  # or run_etl_postgres.bat

# 2. Switch to production
cd ../ragmcp
use_production_mode.bat

# 3. Restart UI
python gradio_ui.py
```

## Test Checklist

- [ ] Vector database running
- [ ] Test ETL completed (5-8 min)
- [ ] Config switched to test mode
- [ ] Gradio UI launches
- [ ] Search tab works
- [ ] Results are relevant
- [ ] (Optional) Ollama running
- [ ] (Optional) RAG + LLM tab works
- [ ] (Optional) LLM recommendations make sense

## Next Steps

After successful test:

1. **Explore Features**:
   - Try different queries
   - Adjust LLM temperature
   - Compare different models
   - Check augmented context

2. **Create Task Embeddings** (optional):
   ```bash
   cd exp3
   python create_task_collection.py
   ```

3. **Switch to Production** (when ready):
   ```bash
   cd exp3
   run_etl_practical.bat
   cd ../ragmcp
   use_production_mode.bat
   ```

4. **Customize**:
   - Add custom LLM models
   - Tune search parameters
   - Set code_root for file retrieval

## Comparison: Test vs Production

### Test Mode (w1000)

**Pros:**
- ✅ Fast setup (5-8 min)
- ✅ Quick iterations
- ✅ Good for testing
- ✅ Recent tasks most relevant

**Cons:**
- ⚠️ Fewer modules (~20-30 vs 64)
- ⚠️ Fewer files (~3k vs 63k)
- ⚠️ Limited history (1000 tasks)

### Production Mode (all)

**Pros:**
- ✅ Complete coverage (all modules)
- ✅ Full file search (63k files)
- ✅ Complete history (9,799 tasks)
- ✅ Maximum accuracy

**Cons:**
- ⚠️ Slow setup (60-70 min)
- ⚠️ More disk space
- ⚠️ More RAM usage

## Scripts Created

### ETL Scripts

**Test Mode (w1000):**
- `exp3/run_etl_test_qdrant.bat` - Qdrant test ETL
- `exp3/run_etl_test_qdrant.sh` - Linux version
- `exp3/run_etl_test_postgres.bat` - PostgreSQL test ETL
- `exp3/run_etl_test_postgres.sh` - Linux version

**Production Mode (all):**
- `exp3/run_etl_practical.bat` - Qdrant full ETL
- `exp3/run_etl_postgres.bat` - PostgreSQL full ETL

### Config Switchers

- `ragmcp/switch_to_test_collections.py` - Python script
- `ragmcp/use_test_mode.bat` - Switch to test (Windows)
- `ragmcp/use_production_mode.bat` - Switch to production (Windows)

## Tips

1. **Always test first**: Run test mode before full ETL
2. **Monitor progress**: Watch ETL output for errors
3. **Check results**: Verify search quality
4. **Test LLM**: Try with and without LLM
5. **Compare modes**: See if test mode is sufficient

---

**Happy Testing! 🚀**

**Estimated total time from zero to working RAG system: 10-15 minutes**
