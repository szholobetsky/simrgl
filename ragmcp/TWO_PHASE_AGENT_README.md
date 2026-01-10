# Two-Phase Reflective RAG Agent

## Overview

The Two-Phase Reflective RAG Agent is an advanced AI coding assistant that uses a structured three-phase approach to analyze codebases and provide recommendations:

1. **Phase 1: Reasoning & File Selection** - Semantic search and intelligent file selection
2. **Phase 2: Deep Analysis** - Detailed code examination and solution assessment
3. **Phase 3: Final Reflection** - Self-critique and confidence scoring

## Architecture

```
User Query
    ↓
┌─────────────────────────────────────────────────┐
│ PHASE 1: REASONING & FILE SELECTION             │
│ - Search similar tasks                          │
│ - Search relevant modules                       │
│ - Search relevant files                         │
│ - LLM selects top 5-10 files to examine         │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ PHASE 2: DEEP ANALYSIS                          │
│ - Fetch content of selected files               │
│ - LLM analyzes actual code                      │
│ - Assess if more files are needed               │
│ - Provide solution recommendations              │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│ PHASE 3: FINAL REFLECTION                       │
│ - Self-critique of analysis                     │
│ - Confidence scoring (0-100%)                   │
│ - Identify strengths and weaknesses             │
│ - Suggest alternative approaches                │
└─────────────────────────────────────────────────┘
    ↓
Final Recommendations
```

## Setup

### 1. Migrate Data from SQLite to PostgreSQL

First, run the migration script to copy RAWDATA from SQLite to PostgreSQL:

```bash
cd ragmcp
python migrate_rawdata_to_postgres.py
```

This will:
- Create the `rawdata` table in PostgreSQL
- Copy TASK_NAME, PATH, MESSAGE, and DIFF columns
- Create indexes for fast lookups
- Verify data integrity

Expected output:
```
Starting RAWDATA migration: SQLite → PostgreSQL
✓ Connected to PostgreSQL
Creating RAWDATA table in PostgreSQL...
Table created successfully
Migrating records: 100%|████████████| 469836/469836
✓ Verification successful: counts match
Migration completed: 469,836 records migrated
```

### 2. Start Required Services

Ensure the following services are running:

```bash
# 1. PostgreSQL (with pgvector extension)
docker run -d \
  --name postgres-vector \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  ankane/pgvector

# 2. Ollama with Qwen2.5-Coder model
ollama serve
ollama pull qwen2.5-coder:latest
```

### 3. Verify Configuration

Check `config.py` settings:

```python
# PostgreSQL Configuration
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = 5432
POSTGRES_DB = 'semantic_vectors'
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'postgres'
POSTGRES_SCHEMA = 'vectors'

# Code Repository Path
CODE_ROOT = r'C:\Project\codeXplorer\capestone\repository\SONAR\sonarqube'

# LLM Configuration
OLLAMA_URL = 'http://localhost:11434'
MODEL = 'qwen2.5-coder:latest'
```

## Usage

### Interactive Mode

```bash
cd ragmcp
python two_phase_agent.py
```

### Example Session

```
================================================================================
TWO-PHASE REFLECTIVE RAG AGENT
================================================================================
[INIT] MCP Server: mcp_server_two_phase.py
[INIT] LLM Model: qwen2.5-coder:latest
[OK] Ollama is running
[OK] MCP Server connected (8 tools available)

================================================================================
INTERACTIVE MODE
================================================================================

Enter your task description to begin analysis.
================================================================================

> Fix memory leak in network buffer pool

================================================================================
QUERY: Fix memory leak in network buffer pool
================================================================================

================================================================================
PHASE 1: REASONING & FILE SELECTION
================================================================================

[1.1] Searching for similar tasks...
[1.2] Searching for relevant modules...
[1.3] Searching for relevant files...
[1.4] LLM reasoning and file selection...
[OK] Selected 7 files for detailed analysis

================================================================================
PHASE 2: DEEP ANALYSIS
================================================================================

[2.1] Fetching content for 7 files...
  [1/7] flink-runtime/src/main/java/org/apache/flink/runtime/io/network/buffer/NetworkBufferPool.java
  [2/7] flink-runtime/src/main/java/org/apache/flink/runtime/io/network/buffer/BufferPool.java
  ...
[OK] Files fetched

[2.2] LLM analyzing file contents...
[OK] Analysis complete

================================================================================
PHASE 3: FINAL REFLECTION
================================================================================

[3.1] LLM self-reflection...
[OK] Reflection complete (Confidence: 85%)

================================================================================
FINAL RESULTS
================================================================================

📋 QUERY: Fix memory leak in network buffer pool
⏱️  PROCESSING TIME: 45.23s

================================================================================
PHASE 1: FILE SELECTION
================================================================================
Selected 7 files:
  1. flink-runtime/src/main/java/.../NetworkBufferPool.java
  2. flink-runtime/src/main/java/.../BufferPool.java
  ...

================================================================================
PHASE 2: ANALYSIS
================================================================================
After analyzing the code, I found that the memory leak is likely caused by...

================================================================================
PHASE 3: REFLECTION & RECOMMENDATION
================================================================================
**Final Recommendation:**
1. Add proper cleanup in NetworkBufferPool.destroy()
2. Ensure all buffers are returned before pool destruction
3. Add monitoring for unreturned buffers

**Strengths:**
- Identified the root cause in buffer lifecycle
- Found similar historical fixes

**Weaknesses:**
- Could have examined more test files
- Need to verify thread safety

**Alternative Approaches:**
- Use weak references for buffer tracking
- Implement automatic buffer reclamation

📊 CONFIDENCE: 85%
⏱️  TOTAL TIME: 45.23s
================================================================================
```

## Enhanced MCP Server Tools

The `mcp_server_two_phase.py` provides these additional tools:

### Original Tools (from simple MCP)
- `search_modules` - Semantic search for modules
- `search_files` - Semantic search for files
- `search_similar_tasks` - Find similar historical tasks

### New Tools (for Two-Phase Agent)

#### 1. `list_tasks`
Get a list of all available tasks with file counts.

```python
# Example usage
result = await session.call_tool("list_tasks", {"limit": 100})
```

Returns:
```json
{
  "total_tasks": 1234,
  "tasks": [
    {
      "task_name": "SONAR-12345",
      "total_changes": 45,
      "unique_files": 38,
      "message_preview": "Fix memory leak in buffer pool..."
    }
  ]
}
```

#### 2. `get_task_files`
Get all files changed in a specific task.

```python
result = await session.call_tool("get_task_files", {"task_name": "SONAR-12345"})
```

Returns:
```json
{
  "task_name": "SONAR-12345",
  "commit_message": "Fix memory leak...",
  "total_files": 38,
  "files": [
    {"path": "src/main/NetworkBufferPool.java", "diff_size": 1234}
  ]
}
```

#### 3. `get_file_diff`
Get the diff for a specific file in a task.

```python
result = await session.call_tool("get_file_diff", {
    "task_name": "SONAR-12345",
    "file_path": "src/main/NetworkBufferPool.java"
})
```

#### 4. `get_file_content`
Get the actual current content of a file from the repository.

```python
result = await session.call_tool("get_file_content", {
    "file_path": "src/main/NetworkBufferPool.java"
})
```

#### 5. `get_task_summary`
Get a comprehensive summary of a task.

```python
result = await session.call_tool("get_task_summary", {"task_name": "SONAR-12345"})
```

## Key Features

### 1. **Intelligent File Selection**
- Uses semantic search to find relevant files
- LLM selects the most important files to examine
- Focuses analysis on high-value code

### 2. **Iterative Refinement**
- Phase 2 can request additional files if needed
- Agent adapts based on what it learns

### 3. **Self-Reflection**
- Phase 3 provides honest self-assessment
- Confidence scoring helps gauge reliability
- Identifies weaknesses in the analysis

### 4. **Complete Transparency**
- All phases are visible to the user
- Clear reasoning for file selection
- Explicit confidence levels

### 5. **Historical Learning**
- Searches for similar past tasks
- Learns from previous solutions
- Provides context-aware recommendations

## Comparison: Simple Agent vs Two-Phase Agent

| Feature | Simple Agent | Two-Phase Agent |
|---------|-------------|-----------------|
| Search Strategy | Single-pass | Multi-phase |
| File Selection | Top-K from search | LLM-guided selection |
| Code Access | Search results only | Full file content |
| Analysis Depth | Surface-level | Deep code analysis |
| Self-Reflection | None | Comprehensive |
| Confidence Scoring | No | Yes (0-100%) |
| Iterative Refinement | No | Yes |
| Transparency | Limited | Complete |

## Files Overview

### Core Implementation
- **`two_phase_agent.py`** - Main agent implementation
- **`mcp_server_two_phase.py`** - Enhanced MCP server with new tools
- **`migrate_rawdata_to_postgres.py`** - Data migration script

### Supporting Files (unchanged)
- **`simple_mcp_client.py`** - Original simple client
- **`local_agent.py`** - Original simple agent
- **`config.py`** - Configuration
- **`vector_backends.py`** - PostgreSQL backend

## Troubleshooting

### Issue: Migration fails
```
Error: relation "rawdata" does not exist
```
**Solution:** Ensure PostgreSQL schema exists:
```sql
CREATE SCHEMA IF NOT EXISTS vectors;
```

### Issue: MCP server connection fails
```
Error: Failed to initialize MCP server
```
**Solution:** Check that:
1. PostgreSQL is running
2. Vector collections are created (run ETL pipeline)
3. Python dependencies are installed

### Issue: File content not found
```
File not found: src/main/File.java
```
**Solution:** Verify `CODE_ROOT` in `config.py` points to your repository

### Issue: Ollama timeout
```
Error: Ollama error: timeout
```
**Solution:**
- Increase timeout in `two_phase_agent.py`
- Use a smaller/faster model
- Check CPU/GPU resources

## Performance Considerations

### Typical Processing Times
- **Phase 1 (Search & Selection):** 5-10 seconds
- **Phase 2 (File Analysis):** 20-40 seconds (depends on file count)
- **Phase 3 (Reflection):** 10-15 seconds
- **Total:** 35-65 seconds per query

### Optimization Tips
1. Use faster models for initial testing (e.g., `llama3:8b`)
2. Limit file selection to 5-7 files in Phase 1
3. Truncate large files to first 5000 characters
4. Use GPU-accelerated Ollama if available

## Advanced Usage

### Programmatic API

```python
from two_phase_agent import TwoPhaseRAGAgent

async def analyze_task(query: str):
    agent = TwoPhaseRAGAgent()
    await agent.initialize()

    try:
        result = await agent.process_query(query)
        print(f"Confidence: {result['phase3']['confidence_score']}%")
        print(f"Recommendation: {result['phase3']['final_recommendation']}")
    finally:
        await agent.cleanup()
```

### Custom LLM Configuration

```python
agent = TwoPhaseRAGAgent(
    model="llama3:70b",  # Use larger model
    ollama_url="http://gpu-server:11434"  # Remote Ollama
)
```

## Future Enhancements

Potential improvements:
- [ ] Support for multiple concurrent file analysis
- [ ] Integration with code execution/testing
- [ ] Automatic code fix generation
- [ ] Learning from user feedback
- [ ] Multi-agent collaboration
- [ ] Integration with IDE plugins

## License

Same as parent project.

## Support

For issues or questions:
1. Check this README
2. Review `TWO_PHASE_REFLECTIVE_AGENT.md` for architectural details
3. Examine the code comments in `two_phase_agent.py`
