# Two-Phase RAG Agent - Web Interface Guide

## Overview

The web interface provides a user-friendly Gradio UI for the dual indexing two-phase reflective RAG agent.

## Quick Start

```bash
# Windows
cd ragmcp
launch_two_phase_web.bat

# Linux/Mac
cd ragmcp
python two_phase_agent_web.py
```

The web UI will open at: **http://127.0.0.1:7860**

## Features

### 📊 **Summary Tab**
- Quick overview of the analysis
- Confidence score
- Files selected and analyzed
- Quick recommendation

### 🔍 **Phase 1: File Selection**
- Shows selected files for analysis
- Displays file relevance scores from dual search
- Lists results from both RECENT and ALL collections

### 🔬 **Phase 2: Deep Analysis**
- LLM analysis of actual file content
- Shows which files were examined
- Indicates if additional files are needed

### 💡 **Phase 3: Reflection**
- Confidence score (0-100%)
- Final recommendations
- Strengths and weaknesses of analysis
- Alternative approaches
- Lessons learned

### 📜 **Query History**
- Recent queries with confidence scores
- Timestamps
- Click "Refresh History" to update

## Interface Layout

```
┌─────────────────────────────────────────────────────┐
│  Two-Phase Reflective RAG Agent                    │
│  With Dual Collection Search                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Task Description:                                  │
│  ┌───────────────────────────────────────────┐    │
│  │ Fix memory leak in network buffer pool    │    │
│  └───────────────────────────────────────────┘    │
│                                                     │
│  [🚀 Analyze with Two-Phase Agent]                 │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Tabs:                                              │
│  ┌─────┬──────────┬──────────┬──────────┐         │
│  │ 📊  │   🔍     │   🔬     │   💡     │         │
│  │Sum- │ Phase 1  │ Phase 2  │ Phase 3  │         │
│  │mary │ Select   │ Analyze  │ Reflect  │         │
│  └─────┴──────────┴──────────┴──────────┘         │
│                                                     │
│  [Results displayed here with formatting]          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Example Usage

### 1. Enter Query

In the "Task Description" box:
```
Fix memory leak in network buffer pool
```

### 2. Click "Analyze"

The agent will:
1. Search both RECENT and ALL collections
2. Select top files to examine
3. Analyze actual file content
4. Provide reflection with confidence

### 3. View Results

**Summary Tab** shows:
```
Query: Fix memory leak in network buffer pool
Confidence: 85%
Files Selected: 7
Files Analyzed: 7

Quick Recommendation:
The memory leak is likely caused by buffers not being
properly released in NetworkBufferPool.destroy()...
```

**Phase 1 Tab** shows:
```
Selected Files for Analysis:
1. NetworkBufferPool.java
2. BufferPool.java
3. NetworkBufferAllocator.java
...

File Relevance Scores:
| Rank | File | Similarity | Relevance |
|------|------|------------|-----------|
| 1 | NetworkBufferPool.java | 0.8923 | High |
| 2 | BufferPool.java | 0.8512 | High |
...
```

**Phase 2 Tab** shows:
```
Files analyzed: 7

LLM Analysis:
After analyzing the code, I found that the memory leak
occurs in the NetworkBufferPool cleanup logic...
```

**Phase 3 Tab** shows:
```
Confidence Score: 🟢 85%

Final Recommendation:
1. Add proper cleanup in NetworkBufferPool.destroy()
2. Ensure all buffers are returned before destruction
3. Add monitoring for unreturned buffers

✅ Strengths:
- Found similar historical fixes
- Identified root cause in buffer lifecycle

⚠️ Weaknesses:
- Could examine more test files
- Need to verify thread safety

🔄 Alternative Approaches:
- Use weak references for buffer tracking
- Implement automatic buffer reclamation
```

## Customization

### Change Port

Edit `two_phase_agent_web.py`:

```python
demo.launch(
    server_port=8080,  # Change from 7860
    ...
)
```

### Change Model

Edit the TwoPhaseRAGAgent initialization:

```python
self.agent = TwoPhaseRAGAgent(
    model="llama3:70b",  # Use different model
    use_dual_search=True
)
```

### Disable Dual Search

To use only RECENT collections:

```python
self.agent = TwoPhaseRAGAgent(
    use_dual_search=False,
    mcp_server_path="mcp_server_two_phase.py"
)
```

## Troubleshooting

### Issue: Web UI won't start

**Error:**
```
ModuleNotFoundError: No module named 'gradio'
```

**Solution:**
```bash
pip install gradio
```

### Issue: "Failed to initialize agent"

**Causes:**
- PostgreSQL not running
- Ollama not running
- Collections not created
- RAWDATA not migrated

**Solution:**
```bash
# 1. Check PostgreSQL
psql -h localhost -U postgres -d semantic_vectors -c "\dt vectors.*"

# 2. Check Ollama
curl http://localhost:11434/api/tags

# 3. Verify collections exist
# Should see: rag_exp_desc_module_w100_modn_bge-small, etc.

# 4. Run migrations if needed
cd ragmcp
python migrate_rawdata_to_postgres.py
```

### Issue: Slow responses

**Expected:** First query is slow (model loading)
- First query: 45-60 seconds
- Subsequent queries: 35-45 seconds

**To speed up:**
- Use GPU-accelerated Ollama
- Use smaller/faster model (e.g., `llama3:8b`)
- Reduce file selection in Phase 1

### Issue: Low confidence scores

**Normal:** Agent is being honest about uncertainty

**To improve:**
- Check if RECENT collection has relevant data
- Verify ALL collection is comprehensive
- Try different query phrasing
- Review Phase 2 analysis for gaps

## Advanced Features

### Access from Network

**Default:** localhost only (127.0.0.1)

**To allow network access:**

Edit `two_phase_agent_web.py`:

```python
demo.launch(
    server_name="0.0.0.0",  # Listen on all interfaces
    server_port=7860,
    share=False  # Don't create public Gradio link
)
```

Then access from other devices: `http://YOUR_IP:7860`

⚠️ **Security Warning:** Only do this on trusted networks!

### Enable Authentication

Add to `two_phase_agent_web.py`:

```python
demo.launch(
    auth=("admin", "your_password"),  # Add authentication
    server_port=7860
)
```

### Custom Themes

Change the theme:

```python
with gr.Blocks(
    theme=gr.themes.Glass(),  # Try: Glass, Monochrome, Soft
    ...
) as demo:
```

## Keyboard Shortcuts

- **Enter** in text box: Submit query
- **Ctrl+C** in terminal: Stop server
- **Tab**: Navigate between fields
- **Click on tab headers**: Switch between phases

## Performance Tips

1. **First query is slow** - Model loading
2. **Use examples** - Pre-filled queries
3. **Watch terminal** - See progress logs
4. **Check history** - Review past queries
5. **Refresh history** - Update after queries

## Comparison: CLI vs Web

| Feature | CLI (two_phase_agent.bat) | Web Interface |
|---------|---------------------------|---------------|
| **Interface** | Terminal | Browser |
| **Ease of Use** | Moderate | Easy |
| **Output Format** | Plain text | Formatted tabs |
| **History** | Manual | Automatic |
| **Multi-user** | No | Yes (with auth) |
| **Performance** | Same | Same |
| **Best For** | Scripts, automation | Interactive use |

## Files Reference

- **`two_phase_agent_web.py`** - Main web interface
- **`launch_two_phase_web.bat`** - Windows launcher
- **`two_phase_agent.py`** - Core agent (shared with CLI)
- **`mcp_server_dual.py`** - Dual search MCP server
- **`config.py`** - Configuration

## Next Steps

1. ✅ Launch web interface
2. ✅ Try example queries
3. ✅ Review all three phases
4. ✅ Check confidence scores
5. ✅ Compare RECENT vs ALL results in Phase 1
6. 📊 Use for actual development tasks!

## Support

For issues:
- Check terminal for error messages
- Review `DUAL_INDEXING_GUIDE.md`
- See `TWO_PHASE_AGENT_README.md`
- Verify prerequisites (PostgreSQL, Ollama, collections)

## Summary

The web interface provides:
- ✅ **Easy-to-use** Gradio UI
- ✅ **Clear visualization** of all three phases
- ✅ **Formatted output** with confidence scores
- ✅ **Query history** tracking
- ✅ **Example queries** for quick testing
- ✅ **Dual search** results clearly labeled
- ✅ **Completely offline** (no cloud services)

Perfect for interactive development work and demonstrations!
