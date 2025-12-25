# 🔍 Semantic Search MCP Server & Local Offline AI Agent

**Complete offline AI coding assistant** using MCP semantic search + PostgreSQL vector database + Ollama local LLM.

## 🎯 What is This?

A comprehensive system that provides:

✅ **MCP Server** - Model Context Protocol server for semantic code search
✅ **Local AI Agent** - 100% offline coding assistant (CLI + Web interface)
✅ **PostgreSQL Backend** - pgvector-based vector storage (27 modules, 12,532 files, 9,799 tasks)
✅ **Backup/Restore** - Full backup and restore capabilities
✅ **Multi-Interface** - Works with Claude Desktop, VS Code, CLI, and Web browser
✅ **Privacy-First** - All data stays on your machine, no cloud services needed

---

## 🚀 Quick Start

### Prerequisites

1. **PostgreSQL with pgvector** running:
   ```bash
   podman start semantic_vectors_db
   ```

2. **Ollama** running with qwen2.5-coder model:
   ```bash
   ollama serve
   ```

3. **Python dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```

### Option 1: Local Offline Agent (Recommended)

**CLI Mode:**
```bash
start_local_agent.bat          # Windows
./start_local_agent.sh         # Linux/Mac
```

**Web Interface:**
```bash
start_local_agent_web.bat      # Windows
./start_local_agent_web.sh     # Linux/Mac
# Then open: http://127.0.0.1:7861
```

### Option 2: MCP Server Integration

**For Claude Desktop** - See `MCP_SETUP_GUIDE.md`
**For VS Code (Cline/Continue)** - See `MCP_SETUP_GUIDE.md`

---

## 📊 Available Collections

| Collection | Vectors | Dimension | Description |
|------------|---------|-----------|-------------|
| **Modules** | 27 | 384 | Folder-level semantic search |
| **Files** | 12,532 | 384 | File-level semantic search |
| **Tasks** | 9,799 | 384 | Historical task similarity |

**Embedding Model:** BAAI/bge-small-en-v1.5
**Total Storage:** 112 MB (backed up)
**Backend:** PostgreSQL + pgvector

---

## 🏗️ Architecture

### Local Offline Agent

```
You ask a question
       ↓
Local Agent (Python)
       ↓
   ┌───┴───┐
   │       │
   ↓       ↓
MCP Server  Ollama
   ↓       ↓
PostgreSQL  qwen2.5-coder
 (search)   (reasoning)
   ↓       ↓
   └───┬───┘
       ↓
 AI Recommendations
```

**Everything runs on your machine:**
- PostgreSQL: localhost:5432
- Ollama: localhost:11434
- MCP Server: stdio connection (auto-started)
- No network calls to external services

### MCP Server

```
┌─────────────────────────────────────────────┐
│  AI Application (Claude / VS Code / Local)  │
│  ┌──────────────────────────────────────┐  │
│  │ MCP Client                           │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
              ↓ MCP Protocol (JSON-RPC over stdio)
┌─────────────────────────────────────────────┐
│  MCP Server (mcp_server_postgres.py)       │
│  ┌──────────────────────────────────────┐  │
│  │ Tools:                               │  │
│  │ • search_modules()                   │  │
│  │ • search_files()                     │  │
│  │ • search_similar_tasks()             │  │
│  │ • get_collections_info()             │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │ PostgreSQL + pgvector                │  │
│  │ • Module Embeddings (384-dim)        │  │
│  │ • File Embeddings (384-dim)          │  │
│  │ • Task Embeddings (384-dim)          │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## 🛠️ Available Tools (MCP Server)

### 1. `search_modules`
Search for relevant code modules based on task description.

**Parameters:**
- `task_description` (str): Task description to search for
- `top_k` (int): Number of modules to return (default: 5, max: 20)

**Returns:** Top-K most relevant modules with similarity scores

**Example:**
```json
{
  "task_description": "Fix memory leak in network buffer pool",
  "top_k": 5
}
```

**Response:**
```
1. **server**
   - Similarity: 0.6611
   - Relevance: Medium

2. **sonar-server**
   - Similarity: 0.6351
   - Relevance: Medium
```

### 2. `search_files`
Search for relevant code files based on task description.

**Parameters:**
- `task_description` (str): Task description
- `top_k` (int): Number of files (default: 10, max: 50)

**Returns:** Top-K most relevant files with paths and similarity scores

### 3. `search_similar_tasks`
Find historical tasks similar to the given description.

**Parameters:**
- `task_description` (str): Task description
- `top_k` (int): Number of tasks (default: 5, max: 20)

**Returns:** Historical similar tasks with task IDs (e.g., SONAR-12345) and titles

**Example Response:**
```
1. **SONAR-18729**
   - Title: Slow Compute Engine queue processing during re-indexation
   - Similarity: 0.7455

2. **SONAR-18168**
   - Title: Improve Python and Java DBD analysis
   - Similarity: 0.7447
```

### 4. `get_collections_info`
Get information about available collections and statistics.

**Parameters:** None

**Returns:** Collection metadata (count, dimension, backend info)

---

## 🤖 Local Offline Agent Features

### CLI Mode
- Interactive command-line interface
- Real-time semantic search
- LLM-powered recommendations
- Commands: `help`, `tools`, `exit`

### Web Interface
- Browser-based UI (http://127.0.0.1:7861)
- Toggle modules/files/tasks search
- View results in separate tabs
- Example queries provided
- 100% offline, no public sharing

### VS Code Integration
- Run in terminal (`` Ctrl+` ``)
- VS Code tasks configuration
- Keyboard shortcuts
- Future: Custom VS Code extension

See `LOCAL_AGENT_GUIDE.md` for complete documentation.

---

## 📁 Project Structure

```
ragmcp/
├── README.md                           # This file
├── README_UA.md                        # Ukrainian version
├── config.py                           # Configuration settings
│
├── MCP Server
│   ├── mcp_server_postgres.py          # PostgreSQL MCP server
│   ├── test_mcp_server.py              # MCP server tests
│   ├── start_mcp_server.bat/sh         # MCP server launchers
│   └── mcp_config_postgres.json        # Config example
│
├── Local Offline Agent
│   ├── local_agent.py                  # CLI agent
│   ├── local_agent_web.py              # Web interface (Gradio)
│   ├── start_local_agent.bat/sh        # CLI launchers
│   ├── start_local_agent_web.bat/sh    # Web launchers
│   └── test_local_agent.py             # Agent tests
│
├── Backup/Restore
│   ├── backup_data_from_postgree.bat/sh    # Backup scripts
│   └── restore_data_to_postgree.bat/sh     # Restore scripts
│
├── Documentation
│   ├── LOCAL_AGENT_GUIDE.md            # Complete local agent guide
│   ├── MCP_SETUP_GUIDE.md              # MCP setup for Claude & VS Code
│   ├── MCP_QUICKSTART.md               # Quick 5-minute setup
│   ├── STARTUP_CHECKLIST.md            # What to start manually
│   └── READY_TO_USE.md                 # Summary of what's ready
│
└── Gradio UI (Original)
    ├── gradio_ui.py                    # Original Gradio interface
    ├── llm_integration.py              # LLM integration module
    ├── utils.py                        # Utility functions
    └── start_ui.bat/sh                 # UI launchers
```

---

## 💾 Backup and Restore

### Backup PostgreSQL Collections

**Windows:**
```bash
backup_data_from_postgree.bat
```

**Linux/Mac:**
```bash
./backup_data_from_postgree.sh
```

Creates timestamped backup: `vectors_backup_YYYYMMDD_HHMMSS.sql` (112 MB)

### Restore from Backup

**Windows:**
```bash
restore_data_to_postgree.bat
```

**Linux/Mac:**
```bash
./restore_data_to_postgree.sh
```

Restores from latest backup in `backups/` folder.

### Deploy to Another Machine

1. Copy these files:
   - Backup SQL file (`vectors_backup_*.sql`)
   - All Python scripts
   - `requirements.txt`

2. Start PostgreSQL on new machine

3. Restore data:
   ```bash
   ./restore_data_to_postgree.sh
   ```

4. Run the agent:
   ```bash
   ./start_local_agent.bat
   ```

---

## 📖 Usage Examples

### Example 1: Bug Investigation

**Query:**
```
Fix memory leak in network buffer pool
```

**Agent provides:**
- Relevant modules: `server`, `sonar-core`
- Relevant files: `BufferPool.java`, `NetworkManager.java`
- Similar historical tasks: SONAR-8493, SONAR-11066
- AI debugging suggestions

### Example 2: New Feature Planning

**Query:**
```
Add OAuth 2.0 authentication support
```

**Agent provides:**
- Authentication modules
- Related files (auth, security)
- Similar implementations from history
- AI recommendations on implementation

### Example 3: Performance Optimization

**Query:**
```
Improve database query performance
```

**Agent provides:**
- Database-related modules
- Query optimization files
- Similar performance tasks: SONAR-18729, SONAR-18168
- AI optimization strategies

---

## 🔧 Configuration

### Change LLM Model

Edit `local_agent.py`:
```python
agent = LocalCodingAgent(
    model="qwen2.5-coder:1.5b"  # Faster, less memory
    # model="qwen2.5-coder:latest"  # Better quality
)
```

Available models (must be installed in Ollama):
- `qwen2.5-coder:1.5b` - Fast, 4GB RAM
- `qwen2.5-coder:latest` - Better, 8GB RAM
- `codellama:latest` - Alternative
- `deepseek-coder:latest` - Another option

### Adjust Search Results

In `local_agent.py`, modify:
```python
modules = await self.search_modules(user_query, top_k=10)  # More modules
files = await self.search_files(user_query, top_k=20)      # More files
tasks = await self.search_similar_tasks(user_query, top_k=5)  # More tasks
```

### Change LLM Temperature

In `local_agent.py`, find `call_ollama` method:
```python
"options": {
    "temperature": 0.3,  # More focused (0.0-1.0)
    # "temperature": 0.7,  # Balanced
    # "temperature": 1.0,  # More creative
    "num_predict": 2000
}
```

---

## 🆚 Comparison: Local Agent vs Cloud Solutions

| Feature | Local Agent | Claude Desktop | GitHub Copilot |
|---------|------------|----------------|----------------|
| **Offline** | ✅ Yes | ❌ No | ❌ No |
| **Cost** | ✅ Free | ❌ $20/month | ❌ $10/month |
| **Privacy** | ✅ 100% local | ❌ Cloud | ❌ Cloud |
| **Speed** | ⚡ Fast | 🌐 Network | 🌐 Network |
| **Codebase Search** | ✅ Yes (MCP) | ✅ Yes (MCP) | ❌ Limited |
| **Historical Tasks** | ✅ Yes (9,799) | ❌ No | ❌ No |
| **Customizable** | ✅ Fully | ❌ Limited | ❌ No |
| **Data Stays Local** | ✅ Yes | ❌ No | ❌ No |

---

## 🐛 Troubleshooting

### "Cannot connect to Ollama"

```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve

# Verify it's accessible
curl http://localhost:11434/api/tags
```

### "MCP server failed to initialize"

```bash
# Check PostgreSQL is running
podman ps | grep semantic_vectors_db

# Start if needed
podman start semantic_vectors_db

# Verify collections exist
python test_mcp_server.py
```

### "Agent is slow"

**Solutions:**
1. Use smaller model: `qwen2.5-coder:1.5b`
2. Reduce search results: `top_k=3` instead of `top_k=10`
3. First query is slow (model loading) - subsequent queries are faster

### "No results found"

Make sure ETL pipeline ran and collections exist:
```bash
cd ../exp3
python create_task_collection.py --backend postgres
```

---

## ⚡ Performance Tips

1. **First query is slow** - Model loading takes time (~30 seconds). This is normal.
2. **Subsequent queries are fast** - Model stays in memory (~5-10 seconds)
3. **Use smaller models** - For faster responses
4. **Reduce context** - Fewer search results = faster LLM processing
5. **Keep services running** - Don't restart Ollama/PostgreSQL unnecessarily

---

## 🔒 Security & Privacy

✅ **Completely Private:**
- All code analyzed stays on your machine
- No data sent to cloud services
- No telemetry, no tracking
- No API keys needed or stored

✅ **Safe for Sensitive Code:**
- Use on proprietary codebases without concerns
- No license restrictions
- No data leaks possible
- Full control over your data

---

## 📚 Documentation

- **Local Agent Guide**: `LOCAL_AGENT_GUIDE.md` - Complete guide with CLI, Web, and VS Code integration
- **MCP Setup Guide**: `MCP_SETUP_GUIDE.md` - Detailed setup for Claude Desktop and VS Code
- **MCP Quickstart**: `MCP_QUICKSTART.md` - Quick 5-minute setup
- **Startup Checklist**: `STARTUP_CHECKLIST.md` - What to start manually vs automatic
- **Ready to Use**: `READY_TO_USE.md` - Summary of all features and status

---

## 🎓 Research Background

This project is based on semantic fingerprinting research for task-to-code retrieval:

- **Embedding Model**: BAAI/bge-small-en-v1.5 (384 dimensions)
- **Aggregation**: Centroid-based (average of all task embeddings per module/file)
- **Vector Search**: pgvector with cosine similarity
- **Dataset**: 9,799 historical tasks from real projects

See `../exp3/README.md` for complete research methodology and results.

---

## 🔗 Related Projects

### exp3 - Embedding-Based RAG
**Location**: `../exp3/`

The research experiments that created the embeddings:
- ETL pipeline for embedding generation
- Systematic evaluation of different approaches
- Streamlit UI for interactive exploration
- Support for multiple backends (PostgreSQL, Qdrant)

### Data Gathering Tool
**Location**: `../../data_gathering/refactor/`

Creates the database used for generating embeddings:
- Extracts Git commits
- Fetches Jira task details
- Links commits to tasks

---

## 🤝 Contributing

Improvements welcome:

1. **New Features**: Additional MCP tools, better LLM integration
2. **Documentation**: Clarifications and examples
3. **Performance**: Optimization suggestions
4. **Integrations**: New IDE plugins, CI/CD integration

---

## 📄 License

This is academic research code. Use for educational and research purposes.

---

## 🎉 What's New

### v2.0 (December 2024)
- ✅ PostgreSQL backend support (replacing Qdrant)
- ✅ MCP server for PostgreSQL
- ✅ Local offline AI agent (CLI + Web)
- ✅ Backup/Restore scripts
- ✅ Fixed task name display (SONAR-12345 instead of internal IDs)
- ✅ LLM timeout fixes (600s for CPU processing)
- ✅ Async cleanup error fixes
- ✅ Complete documentation (5 guides)

### v1.0 (Initial)
- ✅ Gradio UI for semantic search
- ✅ RAG+LLM integration with Ollama
- ✅ Multiple embedding model support

---

## 📧 Contact & Support

For questions:
- **Local Agent**: See `LOCAL_AGENT_GUIDE.md`
- **MCP Server**: See `MCP_SETUP_GUIDE.md`
- **Research**: See `../exp3/README.md`

---

**Built with**: Python • PostgreSQL • pgvector • MCP Protocol • Ollama • Gradio • Sentence Transformers

**Research conducted**: 2024-2025
**Authors**: Stanislav Zholobetskyi, Oleg Andriichuk
