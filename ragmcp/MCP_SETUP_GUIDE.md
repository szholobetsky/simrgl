# MCP Server Setup Guide

Complete guide for integrating the Semantic Module Search MCP Server with Claude Desktop and VS Code.

## Table of Contents
- [Quick Start](#quick-start)
- [Claude Desktop Integration](#claude-desktop-integration)
- [VS Code Integration](#vs-code-integration)
- [Testing the MCP Server](#testing-the-mcp-server)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

1. **PostgreSQL with pgvector** must be running:
   ```bash
   # Windows
   podman start semantic_vectors_db

   # Linux/Mac
   podman start semantic_vectors_db
   ```

2. **Vector collections** must be created (from exp3 ETL pipeline):
   - Module embeddings: `rag_exp_desc_module_w1000_modn_bge-small`
   - File embeddings: `rag_exp_desc_file_w1000_modn_bge-small`
   - Task embeddings: `task_embeddings_all_bge-small`

3. **Python dependencies** installed:
   ```bash
   pip install mcp sentence-transformers psycopg2-binary
   ```

### Test the MCP Server

```bash
# From the ragmcp directory
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp

# Test the server (should start without errors)
python mcp_server_postgres.py
```

If successful, you'll see:
```
Starting Semantic Module Search MCP Server (PostgreSQL Edition)
Backend: PostgreSQL
Host: localhost:5432
Database: semantic_vectors
Model: BAAI/bge-small-en-v1.5
```

Press `Ctrl+C` to stop.

---

## Claude Desktop Integration

### Step 1: Install Claude Desktop

Download and install Claude Desktop from: https://claude.ai/download

### Step 2: Locate Configuration File

Claude Desktop configuration file location:

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```
Full path: `C:\Users\<YourUsername>\AppData\Roaming\Claude\claude_desktop_config.json`

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### Step 3: Configure MCP Server

Edit `claude_desktop_config.json` and add the MCP server configuration:

**Windows Configuration:**
```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "python",
      "args": [
        "C:\\Project\\codeXplorer\\capestone\\simrgl\\ragmcp\\mcp_server_postgres.py"
      ],
      "env": {
        "PYTHONPATH": "C:\\Project\\codeXplorer\\capestone\\simrgl\\ragmcp"
      }
    }
  }
}
```

**macOS/Linux Configuration:**
```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "python3",
      "args": [
        "/path/to/simrgl/ragmcp/mcp_server_postgres.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/simrgl/ragmcp"
      }
    }
  }
}
```

**Important Notes:**
- Use **absolute paths** (not relative)
- On Windows, use **double backslashes** (`\\`) or forward slashes (`/`)
- Ensure Python is in your PATH
- Update paths to match your installation directory

### Step 4: Restart Claude Desktop

1. Close Claude Desktop completely
2. Restart Claude Desktop
3. The MCP server should now be available

### Step 5: Verify Integration

In Claude Desktop, you should see a tools icon (🔧) indicating MCP tools are available.

Try these commands:

**Search for modules:**
```
Find modules related to: "Fix memory leak in buffer pool"
```

**Search for files:**
```
Search files for: "Add support for SQL window functions"
```

**Find similar historical tasks:**
```
Show me similar tasks to: "Improve query performance"
```

**Check collections:**
```
What collections are available in the semantic search system?
```

Claude will automatically use the MCP tools to answer these queries!

---

## VS Code Integration

### Option 1: Using Cline Extension (Recommended)

Cline is a VS Code extension that supports MCP servers.

#### Step 1: Install Cline

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Cline"
4. Click Install

#### Step 2: Configure Cline

1. Open Cline settings (click Cline icon in sidebar)
2. Go to Settings → MCP Servers
3. Add the MCP server configuration:

**Windows:**
```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "python",
      "args": [
        "C:\\Project\\codeXplorer\\capestone\\simrgl\\ragmcp\\mcp_server_postgres.py"
      ],
      "env": {
        "PYTHONPATH": "C:\\Project\\codeXplorer\\capestone\\simrgl\\ragmcp"
      }
    }
  }
}
```

**macOS/Linux:**
```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "python3",
      "args": [
        "/path/to/simrgl/ragmcp/mcp_server_postgres.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/simrgl/ragmcp"
      }
    }
  }
}
```

#### Step 3: Use in VS Code

Open Cline chat and ask:
```
Search for modules related to authentication
```

Cline will use the MCP server to search your semantic collections!

### Option 2: Using Continue Extension

Continue is another AI coding assistant for VS Code with MCP support.

#### Step 1: Install Continue

1. Open VS Code Extensions
2. Search for "Continue"
3. Install the extension

#### Step 2: Configure Continue

1. Open Continue settings
2. Add MCP server to configuration
3. Follow similar steps as Cline

---

## Testing the MCP Server

### Manual Test Script

Create a test script to verify the MCP server works:

**test_mcp.py:**
```python
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_search():
    """Test the MCP server"""

    # Connect to the MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_postgres.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Test search_modules
            print("\nTesting search_modules:")
            result = await session.call_tool(
                "search_modules",
                arguments={
                    "task_description": "Fix memory leak in network buffer pool",
                    "top_k": 5
                }
            )
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(test_search())
```

Run:
```bash
python test_mcp.py
```

### Test with Simple MCP Client

Use the included simple client:

```bash
python simple_mcp_client.py
```

This will demonstrate all available tools.

---

## Available Tools

### 1. search_modules

**Description:** Search for relevant code modules based on task description.

**Input:**
- `task_description` (string, required): Task description
- `top_k` (integer, optional): Number of results (default: 10, max: 50)

**Example:**
```json
{
  "task_description": "Fix authentication bug",
  "top_k": 5
}
```

**Output:** List of modules with similarity scores.

### 2. search_files

**Description:** Search for relevant files (more granular than modules).

**Input:**
- `task_description` (string, required): Task description
- `top_k` (integer, optional): Number of results (default: 10)

**Example:**
```json
{
  "task_description": "Add OAuth support",
  "top_k": 10
}
```

**Output:** List of files with similarity scores.

### 3. search_similar_tasks

**Description:** Find historical tasks similar to the description.

**Input:**
- `task_description` (string, required): Task description
- `top_k` (integer, optional): Number of results (default: 5, max: 20)

**Example:**
```json
{
  "task_description": "Improve database query performance",
  "top_k": 5
}
```

**Output:** List of similar historical tasks with titles and descriptions.

### 4. get_collections_info

**Description:** Get information about available collections.

**Input:** None

**Output:** Information about module, file, and task collections including counts and dimensions.

---

## Troubleshooting

### Issue: "Connection refused" or "Server not starting"

**Solution:**
1. Check PostgreSQL is running:
   ```bash
   podman ps | grep semantic_vectors_db
   ```
2. Verify collections exist:
   ```bash
   podman exec semantic_vectors_db psql -U postgres -d semantic_vectors -c "\dt vectors.*"
   ```

### Issue: "No module named 'mcp'"

**Solution:**
```bash
pip install mcp
```

### Issue: "No module named 'sentence_transformers'"

**Solution:**
```bash
pip install sentence-transformers
```

### Issue: Claude Desktop doesn't show tools

**Solution:**
1. Check `claude_desktop_config.json` syntax is valid JSON
2. Verify absolute paths are correct
3. Check Python is in PATH: `python --version`
4. Restart Claude Desktop completely
5. Check Claude Desktop logs:
   - Windows: `%APPDATA%\Claude\logs\`
   - macOS: `~/Library/Logs/Claude/`
   - Linux: `~/.local/share/Claude/logs/`

### Issue: "Collection not found" errors

**Solution:**
1. Run ETL pipeline to create collections:
   ```bash
   cd ../exp3
   run_etl_postgres.bat  # Windows
   ./run_etl_postgres.sh # Linux/Mac
   ```
2. Create task embeddings:
   ```bash
   python create_task_collection.py --backend postgres
   ```

### Issue: Slow responses

**Solution:**
1. First query is slow due to model loading (normal)
2. Reduce `top_k` parameter
3. Check CPU/RAM usage
4. Consider using smaller model in config.py

---

## Configuration Options

Edit `config.py` to customize:

```python
# Vector Backend
VECTOR_BACKEND = 'postgres'

# PostgreSQL Configuration
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = 5432
POSTGRES_DB = 'semantic_vectors'

# Embedding Model
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'  # Fast, good quality
# EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'  # Slower, better quality

# Search Configuration
DEFAULT_TOP_K = 10
MAX_TOP_K = 50
```

---

## Example Use Cases

### Use Case 1: Developer Starting New Task

**Scenario:** Developer receives task "Add rate limiting to API endpoints"

**In Claude Desktop:**
```
I need to add rate limiting to API endpoints. Which modules should I look at?
```

**Claude will:**
1. Use `search_modules` tool
2. Return relevant modules with similarity scores
3. Use `search_similar_tasks` to show how it was done before
4. Provide recommendations

### Use Case 2: Code Review

**Scenario:** Reviewing PR about authentication changes

**In VS Code with Cline:**
```
Search for authentication-related modules and show similar historical tasks
```

**Cline will:**
1. Search modules
2. Find similar tasks
3. Help understand context and potential issues

### Use Case 3: Bug Investigation

**Scenario:** Memory leak reported in network layer

**Query:**
```
Find files related to: "memory management in network buffer pool"
Show similar historical bugs
```

**Result:**
- List of relevant files
- Historical similar bugs
- How they were fixed

---

## Advanced: Creating Custom MCP Clients

You can create custom clients that use the MCP server:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def search_for_task(task_description: str):
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_postgres.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Search modules
            result = await session.call_tool(
                "search_modules",
                {"task_description": task_description, "top_k": 5}
            )

            return result.content[0].text
```

---

## Performance Tips

1. **First startup is slow** - Model loading takes time
2. **Subsequent queries are fast** - Model stays in memory
3. **Use appropriate top_k** - Smaller values are faster
4. **Database indexes** - HNSW indexes are already created
5. **RAM requirements** - Model needs ~1GB RAM when loaded

---

## Security Notes

1. **Local only** - MCP server connects to localhost PostgreSQL
2. **No external connections** - All data stays on your machine
3. **No API keys needed** - Uses local models and database
4. **Safe for sensitive code** - Everything runs locally

---

## Next Steps

1. **Try it in Claude Desktop** - Ask questions about your codebase
2. **Install VS Code extension** - Use during development
3. **Create backups** - Use backup scripts regularly
4. **Explore similar tasks** - Learn from historical data
5. **Customize** - Adjust config.py for your needs

---

## Support

**Issues:**
- Check logs in Claude Desktop logs directory
- Run test scripts to verify setup
- Ensure PostgreSQL is running
- Verify collections exist

**Documentation:**
- MCP Server: https://modelcontextprotocol.io/
- Sentence Transformers: https://www.sbert.net/
- PostgreSQL+pgvector: https://github.com/pgvector/pgvector

---

**Happy Searching! 🚀**
