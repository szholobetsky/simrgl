
# Local Offline Coding Agent - Complete Guide

Your **100% offline AI coding assistant** using MCP semantic search + Ollama local LLM.

## What is This?

A completely offline coding agent that:
- ✅ Works WITHOUT cloud services (no Claude, no OpenAI, no API keys)
- ✅ Uses MCP server for semantic search (your embeddings)
- ✅ Uses Ollama for local LLM reasoning (qwen2.5-coder)
- ✅ All data stays on your machine
- ✅ Free to use, no costs, no limits

## Quick Start

### Prerequisites

1. **PostgreSQL running:**
   ```bash
   podman start semantic_vectors_db
   ```

2. **Ollama running:**
   ```bash
   # Start Ollama (if not already running)
   ollama serve

   # Verify qwen2.5-coder is installed
   ollama list | grep qwen2.5-coder
   ```

3. **Python dependencies installed:**
   ```bash
   pip install mcp httpx-sse pydantic-settings sse-starlette requests gradio
   ```

### 3 Ways to Use the Local Agent

---

## Option 1: CLI Mode (Terminal/Command Line)

**Best for:** Quick queries, scripting, automation

### Start the Agent

```bash
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp

# Windows
start_local_agent.bat

# Or directly
python local_agent.py
```

### Usage

```
> Fix authentication bug in login module
[QUERY] Fix authentication bug in login module
============================================================

[1/3] Searching semantic database...
[OK] Retrieved semantic context

[2/3] Building context for LLM...

[3/3] Generating LLM recommendations...
[OK] LLM response generated

============================================================
RECOMMENDATIONS
============================================================

Based on the semantic search results, I recommend...
[AI recommendations here]
============================================================

> exit
```

### Commands

- `help` - Show help
- `tools` - List available MCP tools
- `exit` or `quit` - Exit the agent

---

## Option 2: Web Interface (Browser-based)

**Best for:** Visual interface, easier to use, better for reviewing results

### Start the Web UI

```bash
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp

# Windows
start_local_agent_web.bat

# Or directly
python local_agent_web.py
```

### Access the UI

1. Open browser: http://127.0.0.1:7861
2. Enter your coding task
3. Click "[OFFLINE] Analyze with Local AI"
4. View results in tabs:
   - AI Recommendations
   - Module Search
   - File Search
   - Similar Tasks

### Features

- ✅ Toggle modules/files/tasks search
- ✅ View results in separate tabs
- ✅ Example queries provided
- ✅ 100% offline
- ✅ No public sharing

---

## Option 3: VS Code Integration

**Best for:** Development workflow, integrated with code editor

### Method A: Terminal in VS Code (Simplest)

1. Open VS Code
2. Open Terminal (`` Ctrl+` ``)
3. Run the agent:
   ```bash
   python local_agent.py
   ```
4. Use the CLI within VS Code terminal

### Method B: VS Code Task (Recommended)

Create a VS Code task to launch the agent quickly.

**1. Create `.vscode/tasks.json` in your project:**

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start Local AI Agent",
      "type": "shell",
      "command": "python",
      "args": ["C:/Project/codeXplorer/capestone/simrgl/ragmcp/local_agent.py"],
      "presentation": {
        "reveal": "always",
        "panel": "dedicated"
      },
      "problemMatcher": []
    },
    {
      "label": "Start Local AI Agent (Web)",
      "type": "shell",
      "command": "python",
      "args": ["C:/Project/codeXplorer/capestone/simrgl/ragmcp/local_agent_web.py"],
      "presentation": {
        "reveal": "always",
        "panel": "dedicated"
      },
      "problemMatcher": []
    }
  ]
}
```

**2. Run the task:**
- Press `Ctrl+Shift+P`
- Type "Tasks: Run Task"
- Select "Start Local AI Agent"

### Method C: Keyboard Shortcut (Advanced)

**1. Add to `.vscode/keybindings.json`:**

```json
[
  {
    "key": "ctrl+shift+a",
    "command": "workbench.action.tasks.runTask",
    "args": "Start Local AI Agent"
  }
]
```

**2. Press `Ctrl+Shift+A` to launch instantly!**

### Method D: Custom VS Code Extension (Future)

We could create a dedicated VS Code extension that:
- Shows agent results in sidebar
- Provides code actions
- Integrates with editor
- Offers inline suggestions

(This would require extension development - let me know if you want this!)

---

## How It Works

### Architecture

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

### Process Flow

1. **You ask:** "Fix authentication bug"
2. **MCP Search:** Finds relevant modules, files, similar tasks
3. **Context Building:** Combines search results
4. **LLM Reasoning:** Ollama analyzes and generates recommendations
5. **You get:** AI-powered suggestions based on your codebase

### What Makes It Offline?

- **MCP Server:** Runs locally, searches local PostgreSQL
- **PostgreSQL:** Local database with your embeddings
- **Ollama:** Local LLM running on your machine
- **No network:** All processing happens on your computer
- **No API calls:** No cloud services contacted

---

## Example Use Cases

### Use Case 1: New Feature Planning

**Query:**
```
Add OAuth 2.0 authentication support
```

**Agent provides:**
- Relevant authentication modules
- Related files (auth, security)
- Similar historical tasks
- AI recommendations on implementation

### Use Case 2: Bug Investigation

**Query:**
```
Fix memory leak in network buffer pool
```

**Agent provides:**
- Network-related modules
- Buffer management files
- Historical memory leak fixes
- AI debugging suggestions

### Use Case 3: Performance Optimization

**Query:**
```
Improve database query performance
```

**Agent provides:**
- Database-related modules
- Query optimization files
- Similar performance tasks
- AI optimization strategies

### Use Case 4: Code Review Preparation

**Query:**
```
Understand changes needed for rate limiting
```

**Agent provides:**
- Rate limiting modules
- Relevant middleware files
- Past rate limiting implementations
- AI implementation guide

---

## Configuration

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

### Install New Model

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull codellama
```

### Adjust Search Results

In CLI mode, you can modify the code:

```python
# In local_agent.py, modify these lines:
modules = await self.search_modules(user_query, top_k=10)  # More modules
files = await self.search_files(user_query, top_k=20)      # More files
tasks = await self.search_similar_tasks(user_query, top_k=5)  # More tasks
```

### Change LLM Temperature

In `local_agent.py`, find the `call_ollama` method:

```python
"options": {
    "temperature": 0.3,  # More focused (0.0-1.0)
    # "temperature": 0.7,  # Balanced
    # "temperature": 1.0,  # More creative
    "num_predict": 2000
}
```

---

## Comparison: Local Agent vs Cloud Solutions

| Feature | Local Agent | Claude Desktop | GitHub Copilot |
|---------|------------|----------------|----------------|
| **Offline** | ✅ Yes | ❌ No | ❌ No |
| **Cost** | ✅ Free | ❌ $20/month | ❌ $10/month |
| **Privacy** | ✅ 100% local | ❌ Cloud | ❌ Cloud |
| **Speed** | ⚡ Fast | 🌐 Network | 🌐 Network |
| **Codebase Search** | ✅ Yes (MCP) | ✅ Yes (MCP) | ❌ Limited |
| **Historical Tasks** | ✅ Yes | ❌ No | ❌ No |
| **Customizable** | ✅ Fully | ❌ Limited | ❌ No |
| **Data Stays Local** | ✅ Yes | ❌ No | ❌ No |

---

## Troubleshooting

### Issue: "Cannot connect to Ollama"

**Solution:**
```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve

# Verify it's accessible
curl http://localhost:11434/api/tags
```

### Issue: "MCP server failed to initialize"

**Solution:**
```bash
# Check PostgreSQL is running
podman ps | grep semantic_vectors_db

# Start if needed
podman start semantic_vectors_db

# Verify collections exist
python test_mcp_server.py
```

### Issue: "Agent is slow"

**Solutions:**
1. Use smaller model: `qwen2.5-coder:1.5b`
2. Reduce search results: `top_k=3` instead of `top_k=10`
3. Ensure Ollama uses CPU efficiently (it's normal for first query to be slow)

### Issue: "No results found"

**Solution:**
Make sure ETL pipeline ran and collections exist:
```bash
cd ../exp3
python create_task_collection.py --backend postgres
```

### Issue: "Python module not found"

**Solution:**
```bash
pip install mcp httpx-sse pydantic-settings sse-starlette requests gradio
```

---

## Performance Tips

1. **First query is slow** - Model loading takes time (normal)
2. **Subsequent queries are fast** - Model stays in memory
3. **Use smaller models** - For faster responses
4. **Reduce context** - Fewer search results = faster LLM
5. **Keep services running** - Don't restart Ollama/PostgreSQL unnecessarily

---

## Security & Privacy

✅ **Completely Private:**
- All code stays on your machine
- No data sent to cloud
- No telemetry, no tracking
- No API keys needed

✅ **Safe for Sensitive Code:**
- Use on proprietary codebases
- No license concerns
- No data leaks
- Full control

---

## Advanced Usage

### Scripting

Use the agent in scripts:

```python
from local_agent import LocalCodingAgent
import asyncio

async def automated_analysis():
    agent = LocalCodingAgent()
    await agent.initialize()

    queries = [
        "Fix authentication bugs",
        "Improve database performance",
        "Add rate limiting"
    ]

    for query in queries:
        result = await agent.process_query(query)
        print(f"Query: {query}")
        print(f"Response: {result['llm_response']}\n")

    await agent.cleanup()

asyncio.run(automated_analysis())
```

### Integration with CI/CD

Add to your build pipeline:

```yaml
# .github/workflows/ai-review.yml
- name: AI Code Analysis
  run: |
    python local_agent.py <<EOF
    Analyze security concerns in this PR
    EOF
```

### Custom Prompts

Modify system prompt in `local_agent.py`:

```python
system_prompt = """You are a security expert.
Focus on finding vulnerabilities and security issues.
Always mention OWASP Top 10 concerns."""
```

---

## Files Created

- ✅ `local_agent.py` - CLI agent
- ✅ `local_agent_web.py` - Web interface
- ✅ `start_local_agent.bat` - CLI launcher
- ✅ `start_local_agent_web.bat` - Web launcher
- ✅ `LOCAL_AGENT_GUIDE.md` - This guide

---

## Next Steps

1. ✅ Start the agent (CLI or Web)
2. ✅ Try example queries
3. ✅ Set up VS Code task (optional)
4. ✅ Customize for your workflow
5. ✅ Use during development!

---

## FAQ

**Q: Do I need internet?**
A: No! Works 100% offline after initial setup.

**Q: Can I use other LLMs?**
A: Yes! Any Ollama model works. Try `codellama`, `deepseek-coder`, etc.

**Q: Is this better than Claude/Copilot?**
A: Different trade-offs:
- Offline & free vs online & paid
- Your codebase context vs general knowledge
- Privacy vs convenience

**Q: Can I share this with my team?**
A: Yes! It's your code, your data, your agent.

**Q: Does this work on Mac/Linux?**
A: Yes! Just use the .sh scripts instead of .bat

**Q: Can I add more features?**
A: Absolutely! The code is open and customizable.

---

**Happy Coding! 🚀**

Your AI assistant, running locally, respecting your privacy, working offline.
