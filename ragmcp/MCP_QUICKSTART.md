# MCP Server Quick Start

Get your semantic search working in Claude Desktop or VS Code in 5 minutes!

## What is This?

An MCP (Model Context Protocol) server that lets AI assistants like Claude search your codebase semantically using the embeddings you created in exp3.

**Available Tools:**
- 🔍 **search_modules** - Find relevant code modules
- 📄 **search_files** - Find specific files
- 📋 **search_similar_tasks** - Find historical similar tasks
- ℹ️ **get_collections_info** - Check database status

## Quick Setup (3 Steps)

### 1. Make Sure PostgreSQL is Running

```bash
# Check if running
podman ps | grep semantic_vectors_db

# If not running, start it
podman start semantic_vectors_db
```

### 2. Test the MCP Server

```bash
# From ragmcp directory
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp

# Run test
python test_mcp_server.py
```

If successful, you'll see test results for all tools!

### 3. Add to Claude Desktop

**Location of config file:**
- Windows: `C:\Users\<YourUsername>\AppData\Roaming\Claude\claude_desktop_config.json`
- Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Add this configuration:**

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

**Important:** Update the path to match your installation!

Restart Claude Desktop, and you're done! 🎉

## Try It Out

In Claude Desktop, try asking:

```
Find modules related to: "Fix authentication bug"
```

```
Show me files for: "Add SQL support"
```

```
Find similar historical tasks to: "Improve query performance"
```

Claude will automatically use your semantic search! 🚀

## Troubleshooting

**"Server not starting"**
- Check PostgreSQL is running
- Run test: `python test_mcp_server.py`
- Check logs in Claude Desktop logs folder

**"No results found"**
- Make sure you ran the ETL pipeline (exp3)
- Check collections exist: `python test_mcp_server.py`

**"Module not found"**
- Install dependencies: `pip install mcp sentence-transformers psycopg2-binary`

## Full Documentation

See `MCP_SETUP_GUIDE.md` for:
- Complete setup instructions
- VS Code integration (Cline, Continue)
- Advanced configuration
- Custom client development
- Troubleshooting guide

## Files Created

- ✅ `mcp_server_postgres.py` - PostgreSQL-compatible MCP server
- ✅ `test_mcp_server.py` - Test script
- ✅ `mcp_config_postgres.json` - Example configuration
- ✅ `start_mcp_server.bat/sh` - Manual start scripts
- ✅ `MCP_SETUP_GUIDE.md` - Complete documentation
- ✅ `MCP_QUICKSTART.md` - This file

## What's Next?

1. ✅ Test with `test_mcp_server.py`
2. ✅ Add to Claude Desktop
3. ✅ Try searches in Claude
4. 📖 Read `MCP_SETUP_GUIDE.md` for VS Code integration
5. 🚀 Use semantic search during development!

---

**Questions?** Check `MCP_SETUP_GUIDE.md` for detailed help.
