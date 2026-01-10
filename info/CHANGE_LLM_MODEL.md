# How to Change the LLM Model

## Current Configuration

Both agents currently use **Ollama** with the model **`qwen2.5-coder:latest`** (7B parameters).

## Quick Answer

To change the model, you need to:
1. Pull the new model in Ollama
2. Update the default model in the agent files

## Step-by-Step Guide

### Step 1: Check Available Models

See what models you already have:

```bash
ollama list
```

Expected output:
```
NAME                      ID              SIZE      MODIFIED
qwen2.5-coder:latest      abc123def456    4.7 GB    2 days ago
llama3.1:8b              def456ghi789    4.9 GB    1 week ago
```

### Step 2: Browse Available Models

Visit Ollama's model library: https://ollama.com/library

**Popular coding models:**
- `qwen2.5-coder:latest` (7B) - Current default, excellent for code
- `qwen2.5-coder:1.5b` - Smaller, faster, good for simple tasks
- `qwen2.5-coder:14b` - Larger, smarter, slower
- `qwen2.5-coder:32b` - Very large, best quality, needs GPU
- `codellama:7b` - Meta's code-focused model
- `codellama:13b` - Larger CodeLlama
- `deepseek-coder:6.7b` - DeepSeek's coding model
- `starcoder2:7b` - Hugging Face's StarCoder

**General purpose models:**
- `llama3.1:8b` - Good general reasoning
- `llama3.2:3b` - Fast and efficient
- `mistral:7b` - Good balance of speed/quality
- `phi3:3.8b` - Microsoft's small but capable model

### Step 3: Pull the New Model

```bash
# Example: Pull smaller Qwen model
ollama pull qwen2.5-coder:1.5b

# Example: Pull CodeLlama
ollama pull codellama:7b

# Example: Pull larger Qwen model
ollama pull qwen2.5-coder:14b
```

This will download the model (3-20 GB depending on size).

### Step 4: Update Agent Configuration

You have two options:

#### Option A: Change Default in Agent Files (Recommended)

This changes the model for all uses of the agent.

**For Simple Agent (`local_agent.py` line 25):**
```python
def __init__(
    self,
    mcp_server_path: str = "mcp_server_postgres.py",
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5-coder:1.5b"  # ← Change this
):
```

**For Two-Phase Agent (`two_phase_agent.py` line 71):**
```python
def __init__(
    self,
    mcp_server_path: str = "mcp_server_dual.py",
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5-coder:1.5b",  # ← Change this
    use_dual_search: bool = True,
    # ...
):
```

#### Option B: Pass Model When Creating Agent (For Advanced Use)

**In `local_agent_web.py` line 22:**
```python
self.agent = LocalCodingAgent(model="qwen2.5-coder:1.5b")
```

**In `two_phase_agent_web.py` initialize_agent():**
```python
self.agent = TwoPhaseRAGAgent(
    use_dual_search=True,
    model="qwen2.5-coder:1.5b",  # ← Add this parameter
    top_k_tasks_recent=self.config['top_k_tasks_recent'],
    # ... rest of config
)
```

### Step 5: Restart the Agent

```bash
# CLI mode
cd ragmcp
python local_agent.py

# Web mode
cd ragmcp
launch_local_agent_web.bat  # or launch_two_phase_web.bat
```

## Model Comparison

### Performance vs Quality Trade-offs

| Model | Size | Speed | Quality | Best For | RAM Needed |
|-------|------|-------|---------|----------|------------|
| qwen2.5-coder:1.5b | 1.5B | ⚡⚡⚡⚡ | ⭐⭐⭐ | Quick queries, Phase 1 | 4 GB |
| qwen2.5-coder:latest (7B) | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Default, balanced | 8 GB |
| qwen2.5-coder:14b | 14B | ⚡⚡ | ⭐⭐⭐⭐⭐ | Complex reasoning | 16 GB |
| qwen2.5-coder:32b | 32B | ⚡ | ⭐⭐⭐⭐⭐⭐ | Best quality (needs GPU) | 32 GB |
| codellama:7b | 7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Code completion | 8 GB |
| deepseek-coder:6.7b | 6.7B | ⚡⚡⚡ | ⭐⭐⭐⭐ | Code understanding | 8 GB |

### Speed Reference (Phase 1 LLM Call)

On CPU (Intel i7):
- 1.5B model: ~2-3 seconds
- 7B model: ~5-7 seconds
- 14B model: ~12-18 seconds
- 32B model: ~45-90 seconds (not recommended on CPU!)

On GPU (RTX 3080):
- 1.5B model: ~0.5 seconds
- 7B model: ~1-2 seconds
- 14B model: ~3-5 seconds
- 32B model: ~8-12 seconds

## Recommended Configurations

### Fast Configuration (For Speed)
```python
model: str = "qwen2.5-coder:1.5b"
```
- Phase 1: ~8 seconds total (2s LLM)
- Best for: Quick lookups, exploration, prototyping

### Balanced Configuration (Default)
```python
model: str = "qwen2.5-coder:latest"  # 7B
```
- Phase 1: ~12 seconds total (6s LLM)
- Best for: General use, good quality/speed balance

### Quality Configuration (For Accuracy)
```python
model: str = "qwen2.5-coder:14b"
```
- Phase 1: ~20 seconds total (14s LLM)
- Best for: Complex reasoning, architectural decisions

### GPU Configuration (If You Have GPU)
```python
model: str = "qwen2.5-coder:32b"
```
- Phase 1: ~15 seconds total (8s LLM on GPU)
- Best for: Maximum quality with GPU acceleration

## Using Different Models for Different Phases

You can use a **fast model for Phase 1** (file selection) and a **smart model for Phase 2** (analysis):

**In `two_phase_agent.py`:**

```python
def __init__(
    self,
    # ... other params ...
    model: str = "qwen2.5-coder:1.5b",  # Fast model for Phase 1
    model_phase2: str = "qwen2.5-coder:latest",  # Smart model for Phase 2
):
    self.model = model
    self.model_phase2 = model_phase2 if model_phase2 else model
```

Then in `call_ollama()`, add a parameter:
```python
def call_ollama(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
    """Call local Ollama LLM"""
    model_to_use = model if model else self.model

    payload = {
        "model": model_to_use,  # Use specified model
        # ... rest of payload
    }
```

And in phase methods:
```python
# Phase 1: Use fast model
response = self.call_ollama(prompt, system_prompt)  # Uses self.model (1.5b)

# Phase 2: Use smart model
response = self.call_ollama(prompt, system_prompt, model=self.model_phase2)  # Uses 7b
```

## Testing the New Model

After changing the model, test it:

```bash
cd ragmcp
python local_agent.py
```

Try a query:
```
> Fix authentication bug in login module
```

You should see:
```
[INIT] LLM Model: qwen2.5-coder:1.5b  ← Your new model
```

## Troubleshooting

### Model Not Found Error

```
Error: model 'qwen2.5-coder:1.5b' not found
```

**Solution:** Pull the model first:
```bash
ollama pull qwen2.5-coder:1.5b
```

### Out of Memory Error

```
Error: failed to allocate memory for model
```

**Solutions:**
1. Use a smaller model (try 1.5b instead of 7b)
2. Close other applications
3. Restart Ollama: `ollama serve` (restart terminal)

### Model Too Slow

If the model is taking too long:
1. Use a smaller model (1.5b or 3b)
2. Check if GPU is being used: `ollama ps`
3. Enable GPU if available (CUDA for NVIDIA, ROCm for AMD)

### Model Giving Poor Results

If the model's answers are not good:
1. Try a larger model (14b or 32b)
2. Adjust temperature in `call_ollama()`:
   ```python
   "temperature": 0.3,  # Lower = more focused (0.1-0.5)
   "temperature": 0.7,  # Higher = more creative (0.6-1.0)
   ```

## Model Size Guide

### Disk Space Required

- 1.5B models: ~1-2 GB
- 3B models: ~2-3 GB
- 7B models: ~4-5 GB
- 14B models: ~8-10 GB
- 32B models: ~18-20 GB

### RAM Requirements

**Minimum RAM = Model Size × 1.5**

- 1.5B: 4 GB RAM minimum
- 7B: 8 GB RAM minimum
- 14B: 16 GB RAM minimum
- 32B: 32 GB RAM minimum (GPU highly recommended)

## Advanced: Using OpenAI-Compatible APIs

If you want to use a cloud API instead of Ollama:

**Update `call_ollama()` to support both:**
```python
def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
    """Call LLM (Ollama or OpenAI-compatible API)"""

    if self.ollama_url.startswith("http://localhost"):
        # Use Ollama (local)
        return self.call_ollama(prompt, system_prompt)
    else:
        # Use OpenAI-compatible API
        return self.call_openai_api(prompt, system_prompt)
```

Then you can use:
- OpenAI API
- Anthropic Claude API
- Groq API
- Together.ai
- Any OpenAI-compatible endpoint

## Summary

**To change the model:**

1. **Pull new model:**
   ```bash
   ollama pull qwen2.5-coder:1.5b
   ```

2. **Update agent file:**
   ```python
   model: str = "qwen2.5-coder:1.5b"
   ```

3. **Restart agent:**
   ```bash
   launch_local_agent_web.bat
   ```

**Recommendations:**
- **CPU users:** Use 1.5b or 7b models
- **GPU users:** Use 7b, 14b, or 32b models
- **Speed priority:** 1.5b
- **Quality priority:** 14b or 32b
- **Balanced:** 7b (current default)

The model choice significantly affects:
- Response quality
- Processing time
- RAM usage
- Disk space

Choose based on your hardware and requirements!
