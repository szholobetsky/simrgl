# RAG MCP Server - Setup Guide

## Prerequisites

✅ **Completed:**
1. Qdrant running (`podman-compose up -d` in exp3 folder)
2. ETL pipeline completed (creates collections in Qdrant)

⏳ **Wait for:**
- ETL pipeline to finish (~60-70 min on CPU, ~7-12 min on GPU)
- Collections created: `rag_exp_desc_module_all_modn` and `rag_exp_desc_file_all_modn`

## Quick Start

### Option 1: Gradio UI (Recommended) 🎨

**Windows:**
```bash
cd ragmcp
launch_ui.bat
```

**Linux:**
```bash
cd ragmcp
./launch_ui.sh
```

Then open: **http://localhost:7860**

### Option 2: MCP Server (for Claude Desktop integration)

**1. Test the server:**
```bash
cd ragmcp
python mcp_server_simple.py
```

**2. Configure Claude Desktop:**

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "semantic-search": {
      "command": "python",
      "args": [
        "C:\\full\\path\\to\\ragmcp\\mcp_server_simple.py"
      ],
      "env": {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

**3. Restart Claude Desktop**

## How to Use

### Gradio UI

1. **Search Tab:**
   - Enter task description
   - Choose number of results (1-20)
   - Select granularity (Module or File)
   - Click "Search"

2. **Collections Tab:**
   - View available Qdrant collections
   - Check number of points (modules/files)

3. **About Tab:**
   - Learn how the system works
   - View configuration details

### MCP Server in Claude

Once configured, you can ask Claude:

```
"Find modules related to memory management in SonarQube"
"Which files should I look at for fixing buffer pool leaks?"
"Search for code related to SQL query optimization"
```

Claude will use the MCP server to search your codebase!

## Troubleshooting

### Error: Collection not found

**Problem:** ETL pipeline hasn't completed yet

**Solution:**
```bash
cd ../exp3
# Check if ETL is still running
# Wait for it to complete, then come back
```

### Error: Cannot connect to Qdrant

**Problem:** Qdrant not running

**Solution:**
```bash
cd ../exp3
podman-compose up -d
# Wait 5 seconds, then try again
```

### Error: No results found

**Problem:** Collections are empty

**Solution:**
```bash
cd ../exp3
# Re-run ETL pipeline
run_etl_practical.bat  # Windows
./run_etl_practical.sh # Linux
```

## Architecture

```
User Query
    ↓
Gradio UI / MCP Server
    ↓
Embedding Model (BGE-small)
    ↓
Qdrant Vector Search
    ↓
Ranked Results
```

## Configuration

Edit `config.py` to customize:
- Qdrant connection
- Collection names
- Embedding model
- Search parameters

## Files

- `config.py` - Configuration
- `gradio_ui.py` - Gradio web interface
- `mcp_server_simple.py` - MCP server for Claude
- `launch_ui.bat/sh` - UI launchers
- `SETUP_GUIDE.md` - This file

## Next Steps

Once the ETL completes:
1. ✅ Launch Gradio UI and test search
2. ✅ Try different task descriptions
3. ✅ Compare Module vs File granularity
4. ✅ (Optional) Configure Claude Desktop MCP integration

---

**Questions?** Check the main README or exp3 documentation.
