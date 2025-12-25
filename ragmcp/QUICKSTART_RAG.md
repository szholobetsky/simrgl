# 🚀 RAG System - 5 Minute Quick Start

## What You Get

A complete AI-powered code navigation system that:
- **Searches** your codebase semantically
- **Retrieves** actual code context
- **Generates** AI recommendations using LLM

## Prerequisites

✅ Already done:
- Python 3.8+
- Vector database running (Qdrant or PostgreSQL)
- Embeddings created from exp3

⏳ Need to do:
- Install Ollama (5 minutes)

## Step-by-Step Setup

### 1. Install Ollama (Easiest LLM option)

**Windows/Mac/Linux:**
```bash
# Download and install from https://ollama.ai
# Or use installer for your OS
```

**Verify installation:**
```bash
ollama --version
```

### 2. Download a Code Model

```bash
# Pull Qwen 2.5 Coder (recommended - 1.9GB)
ollama pull qwen2.5-coder

# This will take a few minutes depending on your internet speed
```

### 3. Start Ollama Server

```bash
ollama serve
```

Keep this terminal open.

### 4. Launch Gradio UI

**New terminal:**
```bash
cd C:\Project\codeXplorer\capestone\simrgl\ragmcp
python gradio_ui.py
```

### 5. Open in Browser

Navigate to: http://localhost:7860

### 6. Try RAG + LLM

1. Click **"RAG + LLM"** tab
2. Enter a task, e.g.:
   ```
   Fix memory leak in network connection pool
   ```
3. Click **"Run RAG + LLM"**
4. Wait 3-5 seconds
5. See AI recommendations!

## What Happens Behind the Scenes

```
Your Query
    ↓
1. Vector Search → Finds top 5 modules, 10 files, 5 tasks
    ↓
2. Code Retrieval → Gets actual code from top 3 files
    ↓
3. Context Building → Creates rich prompt with:
   - Relevant modules
   - Similar historical tasks
   - Actual code snippets
    ↓
4. LLM Analysis → Qwen generates recommendations
    ↓
Your Answer (with code context!)
```

## Example Output

**Input:**
> Fix memory leak in connection pool

**RAG Retrieves:**
- 5 relevant modules (e.g., `server/network`)
- 10 relevant files (e.g., `ConnectionPool.java`)
- 5 similar tasks (e.g., "Database pool leak fix")
- Actual code from top 3 files

**LLM Recommends:**
```
1. Most Relevant Locations:
   - server/network/ConnectionPool.java (similarity: 0.85)
   - Core connection lifecycle management

2. Recommended Approach:
   - Review release() method for unclosed connections
   - Check error paths in acquire()
   - Add connection leak detection

3. Similar Patterns (from Task #5234):
   - Used try-finally blocks
   - Added WeakReference tracking
   - Implemented timeout-based cleanup

4. Potential Concerns:
   - Thread safety in connection release
   - Race conditions during shutdown
   - Need leak detection tests

5. Code Insights:
   - Missing cleanup in exception paths (line 145)
   - No connection age tracking
   - Consider using Apache Commons Pool pattern
```

## Troubleshooting

### "Ollama is not available"

**Fix:**
```bash
# Make sure Ollama is running
ollama serve
```

### "Model not found: qwen2.5-coder"

**Fix:**
```bash
ollama pull qwen2.5-coder
```

### "No code snippets retrieved"

**Cause**: Code files not accessible from database paths

**Fix**: RAG will still work with metadata, just without actual code

### LLM Takes Too Long

**Current**: 3-5 seconds with Ollama
**If slower**:
- Close other applications
- Use smaller model
- Check CPU/RAM usage

## Next Steps

### 1. Try Different Models

```bash
# Faster, smaller (900MB)
ollama pull codellama:7b

# Better quality, larger (4GB)
ollama pull qwen2.5-coder:14b
```

Update UI dropdown to use new model.

### 2. Adjust Settings

In UI:
- **Temperature**: Lower (0.3) for focused, Higher (1.0) for creative
- **Max Tokens**: Increase for longer responses
- **Top Files**: Increase to see more code context

### 3. Create Task Embeddings (Optional)

For better historical task search:
```bash
cd ../exp3
python create_task_collection.py
```

### 4. Try Other Tabs

- **Search**: Quick module/file lookup
- **Task Search**: Find similar historical tasks
- **Collections**: View database status

## Alternative LLM Options

### LM Studio (GUI)

1. Download from https://lmstudio.ai
2. Load Qwen 2.5 Coder or any model
3. Start server
4. In UI, select "lmstudio" from dropdown

### Local Transformers (Advanced)

```bash
pip install torch transformers
```

In UI, select "qwen-2.5-coder-1.5b"

**Warning**: Slower, needs 8GB+ RAM

### OpenAI API (Best Quality)

1. Get API key from https://platform.openai.com
2. Add to `llm_integration.py`:
   ```python
   PREDEFINED_LLMS["openai-gpt4"].api_key = "your-key-here"
   ```
3. In UI, select "openai-gpt4"

**Warning**: Costs money (~$0.01 per query)

## Performance Tips

### Faster Searches
- Use **Module** search (faster than File)
- Reduce top_k values
- Disable LLM for quick lookups

### Better Recommendations
- Increase top_k for more context
- Use larger LLM model
- Enable historical tasks search

### Offline Usage
- Use Ollama (fully local)
- Or local transformers
- No internet needed after model download

## Common Use Cases

### 1. Bug Fixing
```
Query: "Fix null pointer exception in user authentication"
→ Finds auth modules, similar bugs, suggests fixes
```

### 2. Feature Addition
```
Query: "Add rate limiting to API endpoints"
→ Shows where to add, how others did it, potential issues
```

### 3. Code Understanding
```
Query: "How does the query optimizer work?"
→ Explains with actual code, shows key files
```

### 4. Refactoring
```
Query: "Refactor database connection management"
→ Suggests patterns, shows current code, best practices
```

## What Makes This Special?

Unlike ChatGPT or Copilot:
- ✅ Uses YOUR actual codebase
- ✅ Learns from YOUR historical tasks
- ✅ Retrieves REAL code, not hallucinations
- ✅ Works OFFLINE with local LLM
- ✅ Fully CUSTOMIZABLE
- ✅ 100% FREE with Ollama

## Success Checklist

- [ ] Ollama installed and running
- [ ] Model downloaded (qwen2.5-coder)
- [ ] Gradio UI launched
- [ ] RAG + LLM tab accessible
- [ ] First query successful
- [ ] LLM recommendations received

🎉 If all checked, you're ready to go!

## Need Help?

1. Check `RAG_SYSTEM.md` for detailed docs
2. Review error messages in UI
3. Check console logs
4. Verify Ollama with: `ollama list`

---

**Total Setup Time**: ~5 minutes
**First Query Time**: ~10 seconds
**Subsequent Queries**: ~3-5 seconds

**Enjoy your AI-powered code navigation! 🚀**
