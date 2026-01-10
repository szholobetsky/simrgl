# Quick Start Guide: Two-Phase RAG Agent

## Prerequisites

✅ PostgreSQL with pgvector extension running
✅ Ollama running with qwen2.5-coder model
✅ Python dependencies installed

## Step-by-Step Setup

### Step 1: Migrate Data (One-Time Setup)

```bash
cd ragmcp
python migrate_rawdata_to_postgres.py
```

**Expected Output:**
```
Starting RAWDATA migration: SQLite → PostgreSQL
✓ Connected to PostgreSQL
Creating RAWDATA table in PostgreSQL...
Migrating records: 100%|████████| 469836/469836
✓ Migration completed: 469,836 records migrated
```

### Step 2: Verify Services

```bash
# Check PostgreSQL
psql -h localhost -U postgres -d semantic_vectors -c "SELECT COUNT(*) FROM vectors.rawdata;"

# Check Ollama
curl http://localhost:11434/api/tags

# Check vector collections exist
psql -h localhost -U postgres -d semantic_vectors -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'vectors';"
```

### Step 3: Run the Two-Phase Agent

```bash
cd ragmcp
python two_phase_agent.py
```

### Step 4: Try Example Queries

```
> Fix memory leak in network buffer pool
> Add support for custom SQL aggregations
> Improve error handling in authentication module
> Optimize checkpoint performance
```

## What Happens During Execution

### Phase 1: Reasoning & File Selection (5-10 seconds)
- Searches for similar tasks
- Finds relevant modules and files
- LLM selects top files to examine

### Phase 2: Deep Analysis (20-40 seconds)
- Fetches actual file content
- LLM analyzes the code
- Determines if more files are needed

### Phase 3: Final Reflection (10-15 seconds)
- Self-critique of analysis
- Confidence scoring
- Alternative approaches

**Total Time:** ~35-65 seconds per query

## Key Differences from Simple Agent

| Simple Agent | Two-Phase Agent |
|--------------|-----------------|
| `simple_mcp_client.py` | `two_phase_agent.py` |
| `mcp_server_postgres.py` | `mcp_server_two_phase.py` |
| Search results only | Full file content |
| No reflection | Self-reflection with confidence |
| Single pass | Three-phase analysis |

## Configuration

Edit `config.py` if needed:

```python
# Database
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = 5432

# Code repository location
CODE_ROOT = r'C:\Project\codeXplorer\capestone\repository\SONAR\sonarqube'

# LLM
OLLAMA_URL = 'http://localhost:11434'
MODEL = 'qwen2.5-coder:latest'
```

## Troubleshooting

### No files selected in Phase 1
**Cause:** LLM didn't format JSON correctly
**Fix:** Agent automatically falls back to top search results

### File content not found
**Cause:** `CODE_ROOT` path is incorrect
**Fix:** Update `CODE_ROOT` in `config.py`

### Slow performance
**Cause:** Large files or slow model
**Fix:**
- Use faster model (e.g., `llama3:8b`)
- Files are auto-truncated to 5000 chars

## Next Steps

1. ✅ Run migration script
2. ✅ Test with example queries
3. ✅ Review output and confidence scores
4. 📝 Read full documentation in `TWO_PHASE_AGENT_README.md`
5. 🔬 Experiment with different queries

## Keep the Simple Agent

The original simple agent is still available:

```bash
# Simple agent (unchanged)
python local_agent.py

# Simple MCP client (unchanged)
python simple_mcp_client.py --interactive
```

Both agents can coexist and serve different purposes!
