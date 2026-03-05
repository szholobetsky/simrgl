# 1bcoder

AI-assisted code editor designed for small (1B parameter) language models running locally via [Ollama](https://ollama.com).

**Core idea:** 1B models hallucinate badly when asked to rewrite large blocks of code. 1bcoder works around this by keeping changes small and structured — the model outputs a single-line fix (`LINE N: content`) or a minimal SEARCH/REPLACE block, which the tool then applies with a diff preview before writing to disk.

---

## Features

- Terminal UI built with [Textual](https://textual.textualize.io/)
- Inject any file (or a line range) into the AI context with `/read`
- AI proposes a **one-line fix** (`/fix`) or a **SEARCH/REPLACE patch** (`/patch`) — always shows a diff before applying
- Manual line editing with `/edit`
- Run shell commands and inject their output with `/run`
- Save AI replies to files with `/save` (with optional code-fence stripping)
- **Plans** — reusable sequences of commands, stored as `.txt` files and run step-by-step or fully automated
- **MCP support** — connect external tool servers (filesystem, web, git, database, browser…) via the Model Context Protocol
- **Parallel queries** — send the same prompt to multiple models simultaneously with `/parallel`
- Switch model or Ollama host at runtime without restarting
- Interrupt any streaming response with `ESC`

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| [Ollama](https://ollama.com) | any recent version |
| requests | ≥ 2.28 |
| textual | ≥ 0.80 |

Optional (for MCP servers):
- Node.js + npx (for `@modelcontextprotocol/*` servers)
- uv / uvx (for Python-based MCP servers)

---

## Installation

### 1. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2:1b       # fast, minimal RAM
ollama pull qwen2.5-coder:1b  # good for code
```

### 2. Clone and install 1bcoder

```bash
git clone <repo-url>
cd 1bcoder

# Install with pip (creates the `1bcoder` command)
pip install .

# Or install in editable/dev mode
pip install -e .

# Or just install dependencies and run directly
pip install -r requirements.txt
```

---

## Running

```bash
# Using the installed command (after pip install)
1bcoder

# Or run the script directly
python chat.py

# Windows quick launch
run.bat
```

### CLI options

```
1bcoder [--host URL] [--model NAME] [--init] [--planapply PLAN]

--host URL        Ollama host (default: http://localhost:11434)
--model NAME      Skip model selection, use this model directly
--init            Create .1bcoder/plans/ in the current directory
--planapply PLAN  Run a plan file headlessly (no UI), then exit
```

Examples:

```bash
# Use a remote Ollama instance
1bcoder --host http://192.168.1.50:11434

# Start with a specific model
1bcoder --model qwen2.5-coder:1b

# Auto-run a plan without UI
1bcoder --model llama3.2:1b --planapply my-fixes.txt
```

On startup (without `--model`) a numbered list of available Ollama models is shown — type the number to select one.

---

## UI Overview

```
┌─────────────────────────────────────────────────────┐
│  log — AI responses and command output              │
│                                                     │
│                                                     │
├───────────────────────────┬─────────────────────────┤
│  > input                  │  command reference bar  │
└───────────────────────────┴─────────────────────────┘
```

- Type a message and press **Enter** to chat with the AI.
- Press **ESC** to interrupt a streaming response.
- On Windows, hold **Shift** and drag with the left mouse button to select and copy text from the log.

---

## Command Reference

### File operations

| Command | Description |
|---|---|
| `/read <file> [start-end]` | Inject file content into AI context |
| `/edit <file> <line>` | Manually replace a single line |
| `/save <file> [mode]` | Save last AI reply to a file |

`/read` examples:
```
/read main.py
/read main.py 10-30
```

`/save` modes: `overwrite` (default), `append_above`, `append_below`, `add_suffix`, `code`
The `code` mode strips ` ```language ... ``` ` fences and saves only the code inside.

```
/save out.txt
/save out.txt add_suffix     # → out_1.txt, out_2.txt, …
/save main.py code           # strips ```python…``` wrapper
```

---

### AI edits

| Command | Description |
|---|---|
| `/fix <file> [start-end] [hint]` | AI proposes one-line fix, shows diff, asks to apply |
| `/patch <file> [start-end] [hint]` | AI proposes SEARCH/REPLACE block, shows unified diff |

`/fix` is designed for 1B models — the output format is strictly constrained to `LINE N: content`.
`/patch` works better with 7B+ models and can replace multiple consecutive lines.

```
/fix main.py
/fix main.py 2-2 wrong operator
/patch main.py 10-40 fix the loop logic
```

---

### Shell

```
/run <command>
```
Runs any shell command and injects stdout into the AI context.

```
/run python main.py
/run pytest tests/ -x
```

---

### Plans

Plans are `.txt` files that contain one command per line, stored in `.1bcoder/plans/`.
Lines starting with `[v]` are treated as already done and skipped.

| Command | Description |
|---|---|
| `/plan list` | List all plan files (`*` marks the current one) |
| `/plan open` | Select and load a plan (type number) |
| `/plan create [path]` | Create a new empty plan |
| `/plan show` | Display steps of the current plan |
| `/plan add <command>` | Append a step to the current plan |
| `/plan clear` | Wipe current plan completely |
| `/plan reset` | Unmark all done steps |
| `/plan refresh` | Reload plan from disk and show contents |
| `/plan apply [file]` | Run steps one by one (Y/n/q per step) |
| `/plan apply -y [file]` | Run all pending steps automatically |

```
/plan create personal/test/my-fixes.txt
/plan add /read main.py
/plan add /fix main.py 5-5 off-by-one error
/plan apply -y
```

You can also run a plan headlessly (no UI) from the command line:

```bash
1bcoder --model llama3.2:1b --planapply my-fixes.txt
```

---

### MCP (Model Context Protocol)

Connect external tool servers to give the AI access to filesystems, databases, web pages, and more.

```
/mcp connect <name> <command>
/mcp tools [name]
/mcp call <server/tool> [json_args]
/mcp disconnect <name>
```

Examples:

```
/mcp connect fs npx -y @modelcontextprotocol/server-filesystem .
/mcp connect web uvx mcp-server-fetch
/mcp call web/fetch {"url": "https://docs.python.org/3/"}
/mcp tools
/mcp disconnect fs
```

See [MCP.md](MCP.md) for a full list of ready-to-use servers (filesystem, web, git, SQLite, Postgres, GitHub, browser automation, Slack, and more).

---

### Parallel queries

Send the same prompt to multiple models at the same time. Each answer is saved to its own file.

```
/parallel <prompt> host:port|model|file [host:port|model|file ...]
```

```
/parallel review this function for bugs \
    localhost:11434|llama3.2:1b|answers/llm1.txt \
    localhost:11435|qwen2.5:1b|answers/llm2.txt
```

If no prompt words are given before the first spec, the current context is used as the prompt.

---

### Session controls

| Command | Description |
|---|---|
| `/model` | Switch AI model (shows numbered list) |
| `/host <url>` | Switch Ollama host without restarting |
| `/ctx <n>` | Set context window size in tokens (default 8192) |
| `/ctx` | Show current context window size |
| `/clear` | Clear conversation context and screen |
| `/init` | Create `.1bcoder/plans/` in current directory |
| `/help` | Show full command reference |
| `/exit` | Quit |

---

## Project layout

```
1bcoder/
├── chat.py          # main application (UI + all commands)
├── headless.py      # headless plan runner (no UI, used by --planapply)
├── requirements.txt # pip dependencies
├── pyproject.toml   # build metadata
├── run.bat          # Windows quick-launch script
├── MCP.md           # MCP server quick-reference
└── .1bcoder/        # created per project by /init or --init
    └── plans/       # plan .txt files live here
```

---

## Tips

- **Start small.** Load only the lines you want changed with `/read file.py 10-25` rather than the whole file. This keeps the context short and the model focused.
- **Use `/fix` for 1B models.** The strictly constrained output format (`LINE N: content`) is much more reliable than free-form code generation at this scale.
- **Use `/patch` for 7B+ models.** SEARCH/REPLACE gives more flexibility for multi-line edits.
- **Plans make repetitive work reproducible.** Build a plan once, then replay it with `/plan apply -y` or `--planapply`.
- **Press ESC** if the model starts going off-track — you can interrupt mid-stream.
