# 1bcoder

AI-assisted code editor designed for small (1B parameter) language models running locally via [Ollama](https://ollama.com), [LMStudio](https://lmstudio.ai), or [LiteLLM](https://litellm.ai).

**Core idea:** 1B models hallucinate badly when asked to rewrite large blocks of code. 1bcoder works around this by keeping changes small and structured — the model outputs a single-line fix (`LINE N: content`) or a minimal SEARCH/REPLACE block, which the tool then applies with a diff preview before writing to disk.

Planning and navigation are externalized: plans live in `.txt` files, project structure is indexed into a searchable map — so the model never has to hold the whole codebase in its head.

**Target:** programmers running `qwen2.5-coder:0.6b` or `llama3.2:1b` on a 4 GB machine — offline, no cloud, no subscription. The tool does the heavy lifting so the model doesn't have to.

---

## Features

- Plain terminal REPL — works in any shell, IDE terminal, or SSH session
- Inject one or more files into the AI context with `/read` — multi-file in one command, with optional line range for single-file reads
- AI proposes a **one-line fix** (`/fix`) or a **SEARCH/REPLACE patch** (`/patch`) — always shows a diff before applying
- **Apply AI code blocks directly** with `/edit <file> code` (new/full file) or `/patch <file> code` (SEARCH/REPLACE from reply, no line numbers needed) — preferred for agent mode
- **`<think>` tag support** — reasoning blocks shown in terminal, stripped from context by default; `/think include` passes model reasoning to the next turn
- Run shell commands and inject their output with `/run`
- Save AI replies to files with `/save` (code-fence stripping, multiple files, append modes)
- **Session persistence** — `/ctx save` / `/ctx load` dump and restore full conversations; `/ctx compact` summarizes and compresses the context via AI
- **Plans** — reusable sequences of commands stored as `.txt` files, run step-by-step or fully automated
- **Plan from history** — `/plan create ctx` captures this session's commands into a reusable plan automatically
- **Project map** — scan any codebase into a searchable index (`/map index`), query it (`/map find`), trace call chains (`/map trace`), and diff changes (`/map idiff`)
- **Agent mode** — `/agent <task>` runs an autonomous loop: the model picks tools, can execute multiple actions per turn, checks `/help` before using a command, and stops when done. Configurable via `.1bcoder/agent.txt`
- **Backup/restore** — `/bkup save` / `/bkup restore` for quick file snapshots before risky edits
- **MCP support** — connect external tool servers (filesystem, web, git, database, browser…) via the Model Context Protocol
- **Parallel queries** — send prompts to multiple models simultaneously with `/parallel`, with saved profiles
- Switch model or host at runtime without restarting (`/model gemma3:1b`, `/host openai://localhost:1234`)
- **Model parameters** — `/param temperature 0.2`, `/param enable_thinking false` — sent with every request, auto-cast to correct type
- **Multi-provider** — connect to Ollama, LMStudio, or LiteLLM using `ollama://` / `openai://` URL scheme; plain host defaults to Ollama

---

## Quick install

### Option 1 — Clone and install locally

```bash
git clone https://github.com/your-username/1bcoder.git
cd 1bcoder
pip install -e .
```

The `1bcoder` command will be available in your PATH immediately.

### Option 2 — Install directly from GitHub (no clone needed)

```bash
pip install git+https://github.com/your-username/1bcoder.git
```

Then run anywhere:

```bash
1bcoder
```

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| [Ollama](https://ollama.com) | any recent version |
| requests | ≥ 2.28 |

Instead of Ollama, any OpenAI-compatible backend works: [LMStudio](https://lmstudio.ai), [LiteLLM](https://litellm.ai), or any `/v1/chat/completions` proxy.

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
pip install -e .
```

---

## Running

```bash
# Using the installed command
1bcoder

# Or run directly
python chat.py
```

On startup a numbered list of available Ollama models is shown — type the number to select one. Use `--model` to skip the prompt.

### CLI options

```
1bcoder [--host URL] [--model NAME] [--init] [--planapply PLAN] [--param KEY=VALUE]

--host URL              Host URL — supports ollama:// and openai:// schemes (default: http://localhost:11434)
--model NAME            Skip model selection, use this model directly
--init                  Create .1bcoder/ scaffold in the current directory
--planapply PLAN        Run a plan file non-interactively, then exit
--param KEY=VALUE       Plan parameter substitution (repeatable)
```

Examples:

```bash
1bcoder --host http://192.168.1.50:11434
1bcoder --model qwen2.5-coder:1b
1bcoder --planapply my-fixes.txt --param file=calc.py --param range=1-4
```

---

## Quick start

```
 ██╗██████╗        ██████╗ ██████╗ ██████╗ ███████╗██████╗
███║██╔══██╗      ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
╚██║██████╔╝█████╗██║     ██║   ██║██║  ██║█████╗  ██████╔╝
 ██║██╔══██╗╚════╝██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
 ██║██████╔╝      ╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
 ╚═╝╚═════╝        ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝

  model    : gemma3:1b
  host     : http://localhost:11434
  provider : ollama
  dir      : /home/user/myproject

  /help for all commands   /init to create .1bcoder/ folder
  Ctrl+C interrupts stream   /exit to quit

> /init
> /map index .
> /read main.py 1-20
> what does the divide() function do?
> /fix main.py 5-5 wrong operator
```

---

## Command Reference

### File operations

| Command | Description |
|---|---|
| `/read <file> [file2 ...] [start-end]` | Inject one or more files into AI context; range only applies to single-file reads |
| `/edit <file> <line>` | Manually replace a single line |
| `/edit <file> code` | Apply last AI reply (code block) to whole file — creates file if missing, diff before applying |
| `/edit <file> <line> code` | Apply code block starting at `<line>` — creates file if missing |
| `/edit <file> <start>-<end> code` | Apply code block replacing exactly lines `start`–`end` |
| `/save <file> [mode]` | Save last AI reply to a file |
| `/bkup save <file>` | Save a backup as `<file>.bkup` |
| `/bkup restore <file>` | Replace `<file>` with its `.bkup` copy |
| `/diff <file_a> <file_b> [-y]` | Show colored unified diff between two files; `-y` auto-injects into context |

`/save` modes: `overwrite` (default), `append-above` / `-aa`, `append-below` / `-ab`, `add-suffix`, `code`

```
/diff main.py main.py.bkup          # colored diff, asks to inject into context
/diff v1/calc.py v2/calc.py -y      # auto-inject without confirmation
```

```
/save out.txt
/save out.txt add-suffix        # → out_1.txt, out_2.txt, …
/save main.py code              # strips ```python…``` wrapper
/save index.html style.css code # block 1 → index.html, block 2 → style.css
```

---

### AI edits

| Command | Description |
|---|---|
| `/fix <file> [start-end] [hint]` | AI proposes one-line fix, shows diff, asks to apply |
| `/patch <file> [start-end] [hint]` | AI proposes SEARCH/REPLACE block, shows unified diff |
| `/patch <file> code` | Apply SEARCH/REPLACE block from last AI reply (no new LLM call) |

`/fix` is designed for 1B models — output is strictly constrained to `LINE N: content`.
`/patch` works better with 7B+ models and can replace multiple consecutive lines.
`/patch <file> code` is the preferred agent mode edit — the agent writes the SEARCH/REPLACE block in its reply, then calls `/patch <file> code` to apply it without needing line numbers.

```
/fix main.py
/fix main.py 2-2 wrong operator
/patch main.py 10-40 fix the loop logic
/patch main.py code
```

---

### Shell

```
/run <command>
```
Runs any shell command and injects stdout + stderr into the AI context.

```
/run python main.py
/run pytest tests/ -x
```

---

### Project map

The map command scans your project with language-agnostic regex, extracts definitions (classes, functions, endpoints, tables…) and cross-references between files. No external dependencies — pure regex, works for Python, Java, SQL, HTML, Terraform, YAML, and anything else.

```
/map index [path] [depth]        — scan project, save to .1bcoder/map.txt
/map find [query] [-d N] [-y]    — search the map and inject results into context
/map trace <id> [-y]             — follow call chain backwards from an identifier (BFS)
/map idiff [path] [depth]        — re-index then show diff vs previous snapshot
/map diff                        — show diff without re-indexing (safe to repeat)
```

**`/map find` search syntax:**

| Token | Where | Effect |
|---|---|---|
| `term` | filename | include if filename contains term |
| `!term` | filename | exclude if filename contains term |
| `\term` | child lines | include only if a child line contains term |
| `\!term` | child lines | include block but hide child lines containing term |
| `\\!term` | child lines | exclude entire block if any child contains term |
| `-d 1` | — | filenames only |
| `-d 2` | — | filenames + defines/vars (no links) |
| `-d 3` | — | full blocks (default) |
| `-y` | — | skip "add to context?" confirmation |

```
/map find register                       — files named *register*
/map find \register                      — files that define/link "register"
/map find register \register             — both: in name AND in children
/map find \register !mock                — has "register" in children, skip mock files
/map find auth \UserService \\!deprecated -y
/map find password -d 1                  — just filenames, no details
/map find models -d 2                    — filenames + defines/vars only
```

**`/map trace`** follows the call graph backwards from any defined identifier:

```
/map trace register

auth/routes.py  [defines register(ln:45)]
  ← import:register  app/__init__.py
      ← import:init  main.py
  ← call:register  tests/test_auth.py
```

**`/map idiff`** re-indexes and shows what changed — use after every edit:

```
/map idiff

[map diff]  map.prev.txt → map.txt

  calc.py
  - defines: subtract(ln:12)
  ! WARNING: 1 identifier(s) removed
```

Standalone tools (usable without 1bcoder):
- `map_index.py` — scanner only: `python map_index.py [path] [depth]`
- `map_query.py` — query only: `python map_query.py find \register` / `python map_query.py trace register`

---

### Agent mode

`/agent` runs an autonomous loop: the model reads the task, decides which tools to use, and stops when it outputs plain text with no ACTION. The agent can emit **multiple ACTION lines per turn** — all are executed in order before the next turn. Before using any command, the agent calls `/help <cmd>` first to confirm the correct syntax.

```
/agent [-t N] [-y] <task description>
```

- **`-t N`** — override `max_turns` for this run only (e.g. `-t 1` for a quick read+explain)
- **`-y`** — skip per-action confirmation (execute all actions automatically)
- Without `-y`: each proposed action pauses and asks `[Y/n/q]` — `n` skips it, `q` stops the agent

```
/agent find and fix the divide by zero bug in calc.py
/agent -t 1 read models.py and explain the User class
/agent -y -t 5 refactor utils.py
```

The agent loop (default, with confirmation):
```
[agent] turn 1/10
AI:
ACTION: /help read
ACTION: /help edit

[agent] action: /help read
  execute? [Y/n/q]: Y
[tool result] /read <file> [file2 ...] [start-end] ...

[agent] action: /help edit
  execute? [Y/n/q]: Y
[tool result] /edit <file> code ...

[agent] turn 2/10
ACTION: /read calc.py
ACTION: /read README.md

[agent] action: /read calc.py
  execute? [Y/n/q]: Y
[agent] action: /read README.md
  execute? [Y/n/q]: Y

[agent] turn 3/10
ACTION: /fix calc.py 2

[agent] action: /fix calc.py 2
  execute? [Y/n/q]: Y

...

[agent] turn 5/10
AI: The bug is fixed. divide() now returns None when b is 0.
[agent] task complete
```

Configure in `.1bcoder/agent.txt`:

```ini
max_turns = 10
auto_apply = true

tools =
    read
    run
    edit
    save
    bkup
    diff
    patch
    map index
    map find
    map idiff
    map diff
    map trace
    help
```

Remove tools from the list to restrict a weaker model to a smaller instruction set.

---

### Plans

Plans are `.txt` files containing one command per line, stored in `.1bcoder/plans/`.
Lines starting with `[v]` are already done and skipped.

| Command | Description |
|---|---|
| `/plan list` | List all plan files (`*` marks the current one) |
| `/plan open` | Select and load a plan (type number) |
| `/plan create [path]` | Create a new empty plan |
| `/plan create ctx [path]` | **Create plan from this session's command history** |
| `/plan show` | Display steps of the current plan |
| `/plan add <command>` | Append a step to the current plan |
| `/plan clear` | Wipe current plan completely |
| `/plan reset` | Unmark all done steps |
| `/plan refresh` | Reload plan from disk and show contents |
| `/plan apply [file] [key=value ...]` | Run steps one by one (Y/n/q per step) |
| `/plan apply -y [file] [key=value ...]` | Run all pending steps automatically |

**`/plan create ctx`** captures all work commands typed this session (`/read`, `/edit`, `/fix`, `/patch`, `/run`, `/save`, `/bkup`, `/map`, `/model`, `/host`) into a ready-to-run plan:

```
> /host http://192.168.1.50:11434
> /model gemma3:1b
> /read calc.py
> /fix calc.py 11-11 divide by zero
> /run python calc.py

> /plan create ctx fix-calc.txt
[plan] created 'fix-calc.txt' from session history (5 step(s))
```

Plans support `{{key}}` placeholders:

```
/read {{file}} {{range}}
what is wrong in lines {{range}}?
/fix {{file}} {{range}} {{hint}}
```

```
/plan apply fix-fn.txt file=calc.py range=1-4 hint="wrong operator"
```

Run a plan non-interactively from the command line:

```bash
1bcoder --model llama3.2:1b --planapply my-fixes.txt --param file=calc.py
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

```
/mcp connect fs npx -y @modelcontextprotocol/server-filesystem .
/mcp connect web uvx mcp-server-fetch
/mcp call web/fetch {"url": "https://docs.python.org/3/"}
/mcp tools
/mcp disconnect fs
```

See [MCP.md](MCP.md) for a full list of ready-to-use servers.

---

### Parallel queries

Send prompts to multiple models at the same time. Each answer is saved to its own file.

```
/parallel ["prompt"] [profile <name>] [host:port|model|file ...]
```

```
/parallel "review this for bugs" \
    localhost:11434|llama3.2:1b|answers/llm1.txt \
    localhost:11435|qwen2.5:1b|answers/llm2.txt
```

**Profiles** — save a set of workers for reuse:

```
/parallel profile create          # interactive wizard
/parallel profile list            # show all profiles
/parallel "explain this" profile review
```

Profiles stored in `.1bcoder/profiles.txt`:
```
review: localhost:11434|ministral3:3b|ans/review.txt localhost:11435|cogito:3b|ans/tests.txt  # code review + unit tests
fast:   localhost:11434|qwen2.5-coder:0.6b|ans/q.txt                                          # quick sanity check
```

---

### Session controls

| Command | Description |
|---|---|
| `/model [-sc]` | Switch AI model interactively |
| `/model <name> [-sc]` | Switch directly by name (e.g. `/model gemma3:1b`) |
| `/host <url> [-sc]` | Switch host and provider (see below); `-sc` keeps context |
| `/ctx <n>` | Set context window size in tokens (default 8192) |
| `/ctx` | Show current usage vs limit |
| `/ctx cut` | Remove oldest messages until context fits |
| `/ctx compact` | Ask AI to summarize the conversation, replace context with summary |
| `/ctx save <file>` | Save full conversation to file |
| `/ctx load <file>` | Restore a saved conversation |
| `/think exclude` | Strip `<think>` blocks from context — shown in terminal only (default) |
| `/think include` | Keep `<think>` blocks in context (pass model reasoning to next turn) |
| `/param <key> <value>` | Set a model parameter for every request (e.g. `temperature`, `enable_thinking`) |
| `/param` | Show currently set params |
| `/param clear` | Remove all params |
| `/clear` | Clear conversation context |
| `/help` | Show full command reference |
| `/help <command>` | Show help for one command (e.g. `/help map`) |
| `/help <command> ctx` | Same, and inject into AI context |
| `/init` | Create `.1bcoder/` scaffold in current directory |
| `/exit` | Quit |

### Providers

1bcoder supports **Ollama** (default) and any **OpenAI-compatible** endpoint (LMStudio, LiteLLM, etc.).
The provider is encoded in the URL scheme — no separate flag needed.

| URL | Provider |
|---|---|
| `localhost:11434` | Ollama (default, no scheme needed) |
| `ollama://localhost:11434` | Ollama (explicit) |
| `openai://localhost:1234` | LMStudio |
| `openai://localhost:4000` | LiteLLM |
| `openai://api.example.com` | Any OpenAI-compatible proxy |

```
/host openai://localhost:1234          # switch to LMStudio, clear context
/host openai://localhost:4000 -sc      # switch to LiteLLM, keep context
/host localhost:11434                  # back to Ollama
```

**`/parallel` with mixed providers** — each worker carries its own scheme:

```
/parallel "review this" \
    ollama://localhost:11434|llama3.2:1b|ans/ollama.txt \
    openai://localhost:1234|qwen2.5:7b|ans/lmstudio.txt \
    openai://localhost:4000|gpt-4o-mini|ans/litellm.txt
```

On startup the active provider is shown:
```
  model    : llama3.2:1b
  host     : http://localhost:11434
  provider : ollama
```

---

## Project layout

```
1bcoder/
├── chat.py           # entire application — REPL, all commands
├── map_index.py      # standalone project scanner (usable without 1bcoder)
├── map_query.py      # standalone map query tool (find + trace)
├── map_query_help.txt # full map_query usage reference
├── requirements.txt  # pip dependencies (requests only)
├── pyproject.toml    # build metadata
├── run.bat           # Windows quick-launch
├── MCP.md            # MCP server quick-reference
└── .1bcoder/         # created per project by /init
    ├── plans/        # plan .txt files
    ├── agent.txt     # agent mode config (max_turns, auto_apply, tools)
    ├── profiles.txt  # /parallel worker profiles
    ├── map.txt       # generated by /map index
    └── map.prev.txt  # previous snapshot (for /map diff)
```

---

## Tips for 1B models

- **Start small.** Use `/read file.py 10-25` instead of loading the whole file. Short context = better focus.
- **Use `/fix` not `/patch`.** The `LINE N: content` format is much more reliable at 1B scale than free-form generation.
- **Build a map first.** Run `/map index .` at the start of a session, then use `/map find` to load only the relevant parts into context.
- **Use plans.** Plans make multi-step work reproducible — the model only needs to handle one step at a time.
- **Capture workflows.** After solving a task manually, run `/plan create ctx` to save the exact steps as a reusable plan.
- **Agent mode needs a bigger model.** `/agent` works best with 7B+ models. For 1B, use plans instead.
- **Ctrl+C** interrupts streaming if the model starts going off-track.

## Tips for reasoning models (Qwen3, DeepSeek-R1, etc.)

- **Disable thinking for simple tasks.** `/param enable_thinking false` speeds up responses when reasoning isn't needed.
- **Use `/think include` to chain reasoning.** Pass one model's `<think>` output as context to another model or the next turn.
- **`/patch <file> code` over `/edit`.** Reasoning models write precise SEARCH/REPLACE blocks — no line numbers needed, no full-file rewrites.
- **`/ctx compact` after long sessions.** Reasoning models produce verbose output; compact regularly to stay within context limits.
- **Connect via LMStudio.** `/host openai://localhost:1234` — full parameter control including `enable_thinking`, `temperature`, `seed`.
