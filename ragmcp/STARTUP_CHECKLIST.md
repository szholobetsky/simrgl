# Local Agent Startup Checklist

## Prerequisites (One-Time Setup)
- [x] PostgreSQL container created
- [x] Ollama installed with qwen2.5-coder model
- [x] Python packages installed
- [x] Vector embeddings created

---

## Every Time You Use the Agent

### Step 1: Start PostgreSQL (Manual)
```bash
podman start semantic_vectors_db
```

**Check if running:**
```bash
podman ps | findstr semantic_vectors_db
```

**Expected output:**
```
semantic_vectors_db   Up X minutes   0.0.0.0:5432->5432/tcp
```

---

### Step 2: Start Ollama (Manual)
```bash
ollama serve
```

**Check if running:**
```bash
curl http://localhost:11434/api/tags
```

**Expected output:**
```json
{"models":[{"name":"qwen2.5-coder:latest",...}]}
```

---

### Step 3: Start Local Agent (Manual)

**The agent will AUTOMATICALLY start the MCP server for you!**

#### Option A: CLI Mode
```bash
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp
start_local_agent.bat
```

#### Option B: Web Interface
```bash
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp
start_local_agent_web.bat
```

---

## What Happens Automatically

When you run `start_local_agent.bat`, the agent:

1. ✅ Checks if PostgreSQL is running
2. ✅ Checks if Ollama is running
3. ✅ **Automatically launches MCP server** (`mcp_server_postgres.py`)
4. ✅ Connects to MCP server via stdio
5. ✅ Loads embedding model (BAAI/bge-small-en-v1.5)
6. ✅ Ready to answer your questions!

**You see:**
```
[INIT] Starting Local Coding Agent...
[INIT] MCP Server: mcp_server_postgres.py
[INIT] LLM Model: qwen2.5-coder:latest
[OK] Ollama is running
[OK] MCP Server connected (4 tools available)
```

---

## MCP Server Lifecycle

### When Agent Starts:
```
local_agent.py starts
       ↓
Creates subprocess: python mcp_server_postgres.py
       ↓
Connects via stdio (stdin/stdout)
       ↓
MCP server initializes:
  - Connects to PostgreSQL
  - Loads embedding model
  - Registers 4 tools
       ↓
Agent: "[OK] MCP Server connected"
```

### When Agent Stops:
```
You type: exit
       ↓
Agent cleanup() called
       ↓
MCP server subprocess terminated
       ↓
Agent exits cleanly
```

**You DO NOT need to manually start/stop the MCP server!**

---

## When DO You Need to Manually Start MCP Server?

### For Claude Desktop Integration
If you want to use the MCP server with Claude Desktop (not the local agent):

1. Claude Desktop config points to MCP server
2. Claude Desktop launches MCP server automatically
3. Same as local agent - automatic!

### For VS Code Cline/Continue Extension
If you want to use the MCP server with VS Code extensions:

1. Extension config points to MCP server
2. Extension launches MCP server automatically
3. Same as local agent - automatic!

### For Testing MCP Server Directly
Only if you want to test the MCP server independently:

```bash
python test_mcp_server.py
```

This test script also automatically launches the MCP server!

---

## Summary: What to Start Manually

| Component | Manual Start? | Why |
|-----------|--------------|-----|
| PostgreSQL | ✅ YES | External database service |
| Ollama | ✅ YES | External LLM service |
| MCP Server | ❌ NO | Auto-started by agent |
| Local Agent | ✅ YES | This is what you run |

---

## Quick Start (Copy-Paste)

```bash
# 1. Start PostgreSQL (if not running)
podman start semantic_vectors_db

# 2. Start Ollama (if not running)
ollama serve

# 3. Start Local Agent (MCP server starts automatically!)
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp
start_local_agent.bat

# That's it! Agent is ready.
```

---

## Troubleshooting

### "MCP server failed to initialize"

**Check PostgreSQL:**
```bash
podman ps | findstr semantic_vectors_db
```

If not running:
```bash
podman start semantic_vectors_db
```

**Check Ollama:**
```bash
curl http://localhost:11434/api/tags
```

If not running:
```bash
ollama serve
```

### "Cannot find mcp_server_postgres.py"

The agent looks for `mcp_server_postgres.py` in the same directory. Make sure you're in:
```bash
C:\Project\codeXplorer\capestone\simrgl\ragmcp
```

---

## Advanced: How the Agent Launches MCP Server

The agent uses **stdio connection** (standard input/output):

```python
# From local_agent.py
from mcp.client.stdio import stdio_client

# Launch MCP server as subprocess
server_params = StdioServerParameters(
    command="python",                    # Python interpreter
    args=[self.mcp_server_path],        # mcp_server_postgres.py
    env=None                             # Use current environment
)

# Connect via stdio (automatic subprocess management)
self._client_context = stdio_client(server_params)
read, write = await self._client_context.__aenter__()
```

**Key Points:**
- MCP server runs as a subprocess of the agent
- Communication via stdin/stdout (not HTTP)
- Subprocess lifecycle managed automatically
- When agent exits, MCP server subprocess is terminated

---

## Comparison: Different MCP Usage Scenarios

### 1. Local Agent (Your Current Setup)
```
You → start_local_agent.bat
    → Agent launches MCP server subprocess
    → Agent uses MCP for search + Ollama for LLM
    → Results displayed in CLI/Web
```
**Manual steps:** Start PostgreSQL + Ollama + Agent

### 2. Claude Desktop
```
Claude Desktop config → Points to MCP server
    → Claude Desktop launches MCP server subprocess
    → Claude uses MCP for search + Claude API for LLM
    → Results in Claude Desktop UI
```
**Manual steps:** Start PostgreSQL + Configure Claude Desktop

### 3. VS Code Extension (Cline/Continue)
```
Extension config → Points to MCP server
    → Extension launches MCP server subprocess
    → Extension uses MCP for search + Extension's LLM
    → Results in VS Code sidebar
```
**Manual steps:** Start PostgreSQL + Configure extension

### 4. Direct Testing
```
You → python test_mcp_server.py
    → Test script launches MCP server subprocess
    → Runs tests, shows results
    → Terminates MCP server
```
**Manual steps:** Start PostgreSQL + Run test script

---

## Bottom Line

**You NEVER need to manually start the MCP server!**

Just make sure PostgreSQL and Ollama are running, then run the agent.

The agent handles everything else automatically.

🚀 **Ready to use!**
