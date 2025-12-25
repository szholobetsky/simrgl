# 🤖 Complete RAG System with LLM Integration

## Overview

This is a **production-ready RAG (Retrieval-Augmented Generation) system** for intelligent code navigation. It combines:

1. **Vector Search**: Semantic search across modules, files, and historical tasks
2. **Code Retrieval**: Actual code content from relevant files
3. **Context Augmentation**: Rich context with code snippets and historical data
4. **LLM Integration**: AI-powered recommendations from local or cloud LLMs

## Architecture

```
User Query
    ↓
Vector Search (Modules/Files/Tasks)
    ↓
Code Retrieval (Actual file content)
    ↓
Context Augmentation (Create rich prompt)
    ↓
LLM Generation (Qwen/Ollama/OpenAI)
    ↓
Recommendations
```

## Features

### ✅ What This RAG System Does

1. **Multi-Level Search**:
   - Module-level similarity (64 modules)
   - File-level similarity (63,069 files)
   - Task-level similarity (9,799 historical tasks)

2. **Code Context Retrieval**:
   - Retrieves actual code from top matching files
   - Up to 50 lines per file (configurable)
   - Fallback to database metadata if files not available

3. **Intelligent Context Building**:
   - Combines search results from all levels
   - Adds code snippets for concrete context
   - Includes similar historical tasks for learning
   - Truncates to fit LLM context window

4. **LLM Integration**:
   - **Local Models**: Qwen 2.5 Coder (1.5B, 7B)
   - **Ollama**: Any Ollama model (qwen, codellama, etc.)
   - **LM Studio**: Local server with any model
   - **OpenAI API**: GPT-4, GPT-3.5, etc.

5. **Smart Recommendations**:
   - Most relevant locations to start
   - Suggested implementation approach
   - Lessons from similar historical tasks
   - Potential concerns and edge cases
   - Specific code insights

## Quick Start

### 1. Install Dependencies

```bash
cd ragmcp

# Basic dependencies (already installed)
pip install gradio sentence-transformers

# For Ollama (Recommended - easiest)
# Download from https://ollama.ai and install

# For local models (Optional)
pip install torch transformers

# For OpenAI API (Optional)
pip install openai
```

### 2. Setup LLM (Choose One)

#### Option A: Ollama (Recommended - Easiest)

```bash
# Install Ollama from https://ollama.ai

# Start Ollama
ollama serve

# Pull a code model
ollama pull qwen2.5-coder

# Test it
ollama run qwen2.5-coder
```

#### Option B: LM Studio (Good for GUI users)

1. Download LM Studio from https://lmstudio.ai
2. Download a model (e.g., Qwen 2.5 Coder)
3. Load the model
4. Start the server (default: http://localhost:1234)

#### Option C: Local Transformers (For advanced users)

```python
# Model will be downloaded automatically on first use
# Requires: 8GB+ RAM for 1.5B model, 16GB+ for 7B model
```

### 3. Launch Gradio UI

```bash
cd ragmcp
python gradio_ui.py
```

### 4. Use RAG + LLM Tab

1. Go to **"RAG + LLM"** tab
2. Enter your task description
3. Adjust search and LLM settings
4. Click **"Run RAG + LLM"**
5. View results in 3 tabs:
   - **Search Results**: What was found
   - **Augmented Context**: What was sent to LLM
   - **LLM Recommendations**: AI analysis

## Usage Examples

### Example 1: Fix a Memory Leak

**Input:**
```
Fix memory leak in network connection pool
```

**RAG Process:**
1. Searches vector DB → Finds relevant modules/files
2. Retrieves code from top files
3. Finds similar historical tasks
4. Creates augmented prompt with all context
5. LLM analyzes and provides recommendations

**LLM Output:**
```
1. Most Relevant Locations:
   - server/network/ConnectionPool.java (0.8532 similarity)
   - Contains connection lifecycle management
   - Historical leak fixes in similar code

2. Recommended Approach:
   - Start with ConnectionPool.release() method
   - Check for unclosed connections in error paths
   - Review timeout handling in acquire()

3. Similar Patterns:
   - Task #5234 had similar issue with database pool
   - Fixed by adding try-finally blocks
   - Also added connection leak detection

4. Potential Concerns:
   - Thread safety in connection release
   - Race conditions during pool shutdown
   - Need comprehensive connection lifecycle tests

5. Code Insights:
   - Current code lacks proper cleanup in exception paths
   - Missing WeakReference tracking for leak detection
   - Consider implementing connection age-based eviction
```

### Example 2: Add New Feature

**Input:**
```
Add support for custom SQL functions in query execution engine
```

**RAG Process:**
1. Finds relevant query execution modules
2. Retrieves code from UDF registration and execution
3. Finds historical feature additions
4. LLM provides implementation guidance

### Example 3: Performance Optimization

**Input:**
```
Improve performance of large file processing in import module
```

**RAG Process:**
1. Finds import/processing modules
2. Retrieves relevant code sections
3. Finds past performance fixes
4. LLM suggests optimization strategies

## Configuration

### RAG Pipeline Settings

In Gradio UI, adjust:

**Search Settings:**
- **Top Modules**: 1-10 (default: 5)
  - More modules = broader context
- **Top Files**: 1-20 (default: 10)
  - More files = more code examples
- **Historical Tasks**: 1-10 (default: 5)
  - Learn from past solutions

**LLM Settings:**
- **Enable LLM**: Toggle AI recommendations
- **Model**: Select from predefined or custom
- **Temperature**: 0.0-2.0 (default: 0.7)
  - 0.0 = Deterministic, focused
  - 0.7 = Balanced
  - 1.5+ = Creative, diverse
- **Max Tokens**: 500-4000 (default: 2000)
  - Response length limit

### Code Retrieval Settings

In `rag_pipeline.py`:

```python
# Maximum lines per file
max_lines_per_file = 30  # Default: 30

# Maximum files to retrieve code from
max_code_files = 3  # Default: 3

# Maximum context length (chars)
max_context_length = 4000  # Default: 4000
```

### Custom LLM Configuration

In `llm_integration.py`, add your own:

```python
PREDEFINED_LLMS["my-custom-llm"] = LLMConfig(
    provider="ollama",  # or "local", "openai", "lmstudio"
    model_name="my-model:latest",
    temperature=0.7,
    max_tokens=2000,
    api_base="http://localhost:11434"  # For Ollama/LMStudio
)
```

## LLM Provider Comparison

| Provider | Pros | Cons | Best For |
|----------|------|------|----------|
| **Ollama** | Easy setup, Fast, Free | Needs local install | Most users |
| **LM Studio** | GUI, Model browser | Windows/Mac only | GUI preference |
| **Local (transformers)** | Full control, Offline | Complex setup, Slow | Advanced users |
| **OpenAI API** | Best quality, Fast | Costs money, Needs API key | Production/quality |

## Performance

### Vector Search
- **Modules**: < 50ms
- **Files**: < 100ms
- **Tasks**: < 50ms
- **Total search**: ~200ms

### Code Retrieval
- **From disk**: ~50ms per file
- **From database**: ~100ms per file

### LLM Generation
- **Ollama (Qwen 2.5 Coder)**: 2-5 seconds
- **Local (1.5B)**: 5-10 seconds (CPU)
- **Local (7B)**: 10-30 seconds (CPU)
- **OpenAI GPT-4**: 3-8 seconds

### Total RAG Pipeline
- **Without LLM**: ~500ms
- **With Ollama**: 3-6 seconds
- **With Local**: 5-30 seconds
- **With OpenAI**: 4-10 seconds

## Troubleshooting

### LLM Errors

#### "Ollama is not available"

**Problem**: Ollama not running

**Solution**:
```bash
ollama serve
```

#### "Model not found"

**Problem**: Model not downloaded

**Solution**:
```bash
ollama pull qwen2.5-coder
```

#### "Out of memory"

**Problem**: Model too large for RAM

**Solutions**:
1. Use smaller model (1.5B instead of 7B)
2. Use Ollama (more efficient)
3. Close other applications

#### "Connection refused"

**Problem**: LM Studio/Ollama not running or wrong port

**Solution**:
- Check LM Studio is running and server started
- Verify port in settings (default: 1234 for LMStudio, 11434 for Ollama)

### Code Retrieval Issues

#### "No code snippets retrieved"

**Problem**: Code files not accessible

**Solutions**:
1. Set `code_root` parameter in RAG pipeline
2. Ensure file paths in database are correct
3. Check file permissions

#### "Permission denied"

**Problem**: Cannot read code files

**Solution**:
```bash
# Check permissions
ls -la /path/to/code

# Fix if needed
chmod -R +r /path/to/code
```

### Search Quality Issues

#### "Irrelevant results"

**Solutions**:
1. Increase top_k to see more options
2. Check if embeddings are up-to-date
3. Try different query phrasing
4. Verify vector collections exist

#### "No historical tasks found"

**Problem**: Task collection not created

**Solution**:
```bash
cd ../exp3
python create_task_collection.py
```

## Advanced Usage

### Custom System Prompt

Modify `llm_integration.py`:

```python
def create_system_prompt(self) -> str:
    return """You are a specialized assistant for [YOUR DOMAIN].

    Focus on:
    - [Your specific requirements]
    - [Your coding standards]
    - [Your architecture patterns]
    """
```

### Custom Code Retrieval

Create custom retriever in `rag_pipeline.py`:

```python
class CustomCodeRetriever(CodeRetriever):
    def get_file_content(self, file_path: str, max_lines: int = 50):
        # Your custom logic
        # E.g., fetch from Git, S3, API, etc.
        pass
```

### Batch Processing

Process multiple queries:

```python
from rag_pipeline import RAGPipeline
from llm_integration import LLMFactory, PREDEFINED_LLMS, RAGWithLLM

pipeline = RAGPipeline()
llm = LLMFactory.create(PREDEFINED_LLMS["ollama-qwen"])
rag_with_llm = RAGWithLLM(llm)

queries = [
    "Fix memory leak in connection pool",
    "Add authentication to API endpoints",
    "Optimize database query performance"
]

for query in queries:
    result = pipeline.run(query)
    recommendations = rag_with_llm.generate_recommendations(result.augmented_prompt)
    print(f"\nQuery: {query}")
    print(f"Recommendations: {recommendations[:200]}...")
```

### Export Results

Save RAG results:

```python
import json

result = pipeline.run("Your query")

# Save to JSON
output = {
    'query': result.query,
    'modules': result.modules,
    'files': result.files,
    'tasks': result.tasks,
    'code_snippets': result.code_snippets,
    'augmented_prompt': result.augmented_prompt
}

with open('rag_result.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
```

## API Usage (Programmatic)

```python
from rag_pipeline import RAGPipeline
from llm_integration import LLMFactory, PREDEFINED_LLMS, RAGWithLLM

# Initialize
pipeline = RAGPipeline()
llm = LLMFactory.create(PREDEFINED_LLMS["ollama-qwen"])
rag_with_llm = RAGWithLLM(llm)

# Run RAG
result = pipeline.run(
    query="Fix memory leak in connection pool",
    top_k_modules=5,
    top_k_files=10,
    top_k_tasks=5,
    retrieve_code=True,
    max_code_files=3
)

# Get LLM recommendations
recommendations = rag_with_llm.generate_recommendations(result.augmented_prompt)

print("Recommendations:", recommendations)
```

## Integration with IDEs

### VS Code Extension (Future)

The RAG API can be wrapped in a VS Code extension to provide:
- Code navigation assistance
- Context-aware suggestions
- Historical task lookup
- AI-powered code review

### IntelliJ Plugin (Future)

Similar integration for JetBrains IDEs.

## Comparison with Other Tools

| Feature | This RAG System | GitHub Copilot | ChatGPT Code Interpreter |
|---------|----------------|----------------|--------------------------|
| **Semantic Search** | ✅ Yes | ❌ No | ❌ No |
| **Historical Tasks** | ✅ Yes | ❌ No | ❌ No |
| **Actual Code Context** | ✅ Yes | ⚠️ Limited | ❌ No |
| **Local/Privacy** | ✅ Yes | ❌ Cloud only | ❌ Cloud only |
| **Customizable** | ✅ Fully | ❌ No | ❌ No |
| **Cost** | ✅ Free (local) | 💰 $10/month | 💰 $20/month |
| **Offline** | ✅ Yes | ❌ No | ❌ No |

## Best Practices

1. **Query Formulation**:
   - Be specific about the problem
   - Include technical terms
   - Mention the domain/module if known

2. **LLM Selection**:
   - Ollama: Best for quick iterations
   - Local 1.5B: Good for basic recommendations
   - Local 7B: Better quality, slower
   - GPT-4: Best quality, costs money

3. **Context Tuning**:
   - Start with default settings
   - Increase top_k if results seem limited
   - Decrease if too much irrelevant context

4. **Code Root Configuration**:
   - Set code_root for best code retrieval
   - Keep code repository up-to-date
   - Ensure proper file permissions

## Future Enhancements

- [ ] Multi-turn conversation support
- [ ] Code diff analysis from historical tasks
- [ ] Integration with Git for blame/history
- [ ] Fine-tuned models for specific codebases
- [ ] Real-time code indexing
- [ ] API endpoints for IDE integration
- [ ] Caching for faster repeated queries

## Contributing

To extend the RAG system:

1. **Add new LLM provider** → Edit `llm_integration.py`
2. **Custom code retrieval** → Edit `rag_pipeline.py`
3. **UI enhancements** → Edit `gradio_ui.py`
4. **New search strategies** → Edit `vector_backends.py`

## Support

For issues:
1. Check this documentation
2. Review error messages in Gradio UI
3. Check logs in console
4. Verify all dependencies installed
5. Test individual components

---

**Last Updated**: 2025-12-22
**Version**: 1.0
**Author**: RAG System Team
