# Centralized LLM Configuration

## Overview

The LLM model configuration has been moved to `ragmcp/config.py` for centralized management. Now you can change the model in one place instead of editing multiple agent files.

## How It Works

### Configuration File (`ragmcp/config.py`)

```python
# LLM Configuration (Ollama)
OLLAMA_URL = 'http://localhost:11434'
OLLAMA_MODEL = 'gemma3:1b'  # Default model for all agents

# Alternative models (change OLLAMA_MODEL to any of these):
# - 'gemma3:1b' (815 MB) - Fastest, best for testing
# - 'qwen2.5-coder:1.5b' (986 MB) - Fast, better for code
# - 'qwen2.5-coder:latest' (4.7 GB) - Best quality, slower
# - 'qwen2.5-coder:14b' - Excellent quality, needs 16GB RAM
# - 'codellama:7b' - Alternative coding model
```

### Agent Implementation

Both `local_agent.py` and `two_phase_agent.py` now use these config values as defaults:

```python
def __init__(
    self,
    mcp_server_path: str = "mcp_server_postgres.py",
    ollama_url: str = None,  # Uses config.OLLAMA_URL if None
    model: str = None        # Uses config.OLLAMA_MODEL if None
):
    self.ollama_url = ollama_url if ollama_url else config.OLLAMA_URL
    self.model = model if model else config.OLLAMA_MODEL
```

## How to Change the Model

### Method 1: Edit config.py (Recommended)

This changes the model for **all agents** system-wide:

**File:** `ragmcp/config.py` (line 49)

```python
# Change this line:
OLLAMA_MODEL = 'gemma3:1b'

# To any available model:
OLLAMA_MODEL = 'qwen2.5-coder:1.5b'
# or
OLLAMA_MODEL = 'qwen2.5-coder:latest'
```

**Effect:** All agents (simple and two-phase) will use the new model.

**No code changes needed in agent files!**

### Method 2: Override in Web Interface (Advanced)

You can still override the model when creating an agent instance:

**File:** `local_agent_web.py`

```python
# Override config default
self.agent = LocalCodingAgent(model="qwen2.5-coder:latest")
```

**File:** `two_phase_agent_web.py`

```python
# Override config default
self.agent = TwoPhaseRAGAgent(
    model="qwen2.5-coder:latest",
    use_dual_search=True,
    # ... other params
)
```

### Method 3: Override in CLI

When using the agent directly:

```python
from local_agent import LocalCodingAgent

# Use config default
agent = LocalCodingAgent()

# Or override
agent = LocalCodingAgent(model="codellama:7b")
```

## Available Models (from your ollama list)

| Model | Size | Status | Speed |
|-------|------|--------|-------|
| `gemma3:1b` | 815 MB | ✅ Installed | ⚡⚡⚡⚡ Fastest |
| `qwen2.5-coder:1.5b` | 986 MB | ✅ Installed | ⚡⚡⚡⚡ Very Fast |
| `qwen2.5-coder:latest` | 4.7 GB | ✅ Installed | ⚡⚡⚡ Medium |

## Example: Switching to Faster Model

**Step 1:** Edit `ragmcp/config.py`:

```python
# Line 49: Change from
OLLAMA_MODEL = 'qwen2.5-coder:latest'

# To
OLLAMA_MODEL = 'gemma3:1b'  # Fastest testing model
```

**Step 2:** Restart agents:

```bash
cd ragmcp
launch_local_agent_web.bat
# or
launch_two_phase_web.bat
```

**Step 3:** Verify in output:

```
[INIT] LLM Model: gemma3:1b  ← Should show new model
```

Done! No code changes needed.

## Example: Switching to Better Code Model

**Step 1:** Edit `ragmcp/config.py`:

```python
# Line 49: Change from
OLLAMA_MODEL = 'gemma3:1b'

# To
OLLAMA_MODEL = 'qwen2.5-coder:1.5b'  # Better for code, still fast
```

**Step 2:** Restart agents (same as above)

**Expected change:**
- Speed: ~1s → ~2s per LLM call
- Quality: Better code understanding
- Still very fast for testing!

## Comparison: Before vs After

### Before (Hardcoded in Agents)

**To change model:**
1. Edit `local_agent.py` line 25
2. Edit `two_phase_agent.py` line 71
3. Edit `local_agent_web.py` (if overridden)
4. Edit `two_phase_agent_web.py` (if overridden)
5. Restart all agents

**Problems:**
- Must edit multiple files
- Easy to have inconsistent models
- Hard to maintain

### After (Centralized in config.py)

**To change model:**
1. Edit `config.py` line 49
2. Restart agents

**Benefits:**
- ✅ Single source of truth
- ✅ Consistent across all agents
- ✅ Easy to maintain
- ✅ Can still override per-agent if needed

## Configuration Hierarchy

1. **Highest Priority:** Explicit parameter in agent constructor
   ```python
   agent = LocalCodingAgent(model="codellama:7b")  # Uses codellama:7b
   ```

2. **Default:** Value from `config.py`
   ```python
   agent = LocalCodingAgent()  # Uses config.OLLAMA_MODEL
   ```

## Testing Different Models

You can quickly test different models by editing config once:

```python
# config.py
OLLAMA_MODEL = 'gemma3:1b'  # Test 1: Fastest
# Restart agents, test queries, note performance

OLLAMA_MODEL = 'qwen2.5-coder:1.5b'  # Test 2: Fast + code focus
# Restart agents, test queries, compare results

OLLAMA_MODEL = 'qwen2.5-coder:latest'  # Test 3: Best quality
# Restart agents, test queries, final comparison
```

## Troubleshooting

### Model Not Found After Changing Config

**Error:**
```
Error: model 'qwen2.5-coder:14b' not found
```

**Solution:** Pull the model first:
```bash
ollama pull qwen2.5-coder:14b
```

### Config Changes Not Applied

**Problem:** Changed `config.py` but agent still uses old model

**Solutions:**
1. Restart the agent (web interface or CLI)
2. Check you edited the right config.py (in `ragmcp/` directory)
3. Check for overrides in `*_web.py` files

### Different Models for Different Agents

If you want simple agent to use fast model but two-phase to use smart model:

**Option 1: Use overrides in web files**

`local_agent_web.py`:
```python
self.agent = LocalCodingAgent(model="gemma3:1b")  # Fast for simple
```

`two_phase_agent_web.py`:
```python
self.agent = TwoPhaseRAGAgent(model="qwen2.5-coder:latest")  # Smart for two-phase
```

**Option 2: Add separate config values**

`config.py`:
```python
OLLAMA_MODEL_SIMPLE = 'gemma3:1b'
OLLAMA_MODEL_TWOPHASE = 'qwen2.5-coder:latest'
```

Then in agents:
```python
# local_agent.py
self.model = model if model else config.OLLAMA_MODEL_SIMPLE

# two_phase_agent.py
self.model = model if model else config.OLLAMA_MODEL_TWOPHASE
```

## Related Files

- `ragmcp/config.py` - Main configuration file (line 47-56)
- `ragmcp/local_agent.py` - Simple agent (line 21-29)
- `ragmcp/two_phase_agent.py` - Two-phase agent (line 67-85)
- `concepts/CHANGE_LLM_MODEL.md` - Detailed model comparison guide

## Summary

**Key Changes:**
- ✅ LLM model now configured in `config.py`
- ✅ Both agents use config by default
- ✅ Can still override per-agent if needed
- ✅ Single place to change model for entire system

**To change model:**
1. Edit `ragmcp/config.py` line 49
2. Change `OLLAMA_MODEL = 'your-model-name'`
3. Restart agents

**Current default:** `gemma3:1b` (fastest for testing)

**Recommended for production:** `qwen2.5-coder:latest` (best balance)
