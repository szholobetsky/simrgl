# 1bcoder

AI-assisted code editor designed for small (1B parameter) language models running locally via [Ollama](https://ollama.com).

**Core idea:** 1B models hallucinate badly when asked to rewrite large blocks of code. 1bcoder works around this by keeping changes small and structured — the model outputs a single-line fix (`LINE N: content`) or a minimal SEARCH/REPLACE block, which the tool then applies with a diff preview before writing to disk.

Planning and navigation are also externalized: plans live in `.txt` files, project structure is indexed into a searchable map — so the model never has to hold the whole codebase in its head.

---

## Features

- Plain terminal REPL — works in any shell, IDE terminal, or SSH session
- Inject any file (or a line range) into the AI context with `/read`
- AI proposes a **one-line fix** (`/fix`) or a **SEARCH/REPLACE patch** (`/patch`) — always shows a diff before applying
- **Apply AI code blocks directly** with `/edit <file> code` — diff preview before writing
- Run shell commands and inject their output with `/run`
- Save AI replies to files with `/save` (code-fence stripping, multiple files, append modes)
- **Session persistence** — `/ctx save` / `/ctx load` dump and restore full conversations
- **Plans** — reusable sequences of commands stored as `.txt` files, run step-by-step or fully automated
- **Plan from history** — `/plan create ctx` captures this session's commands into a reusable plan automatically
- **Project map** — scan any codebase into a searchable index (`/map index`), query it (`/map find`), trace call chains (`/map trace`), and diff changes (`/map idiff`)
- **Agent mode** — `/agent <task>` runs an autonomous loop: the model picks tools, executes them one at a time, and stops when done. Configurable via `.1bcoder/agent.txt`
- **Backup/restore** — `/bkup save` / `/bkup restore` for quick file snapshots before risky edits
- **MCP support** — connect external tool servers (filesystem, web, git, database, browser…) via the Model Context Protocol
- **Parallel queries** — send prompts to multiple models simultaneously with `/parallel`, with saved profiles
- Switch model or Ollama host at runtime without restarting (`/model gemma3:1b`)

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| [Ollama](https://ollama.com) | any recent version |
| requests | ≥ 2.28 |

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

--host URL              Ollama host (default: http://localhost:11434)
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

  model : gemma3:1b
  host  : http://localhost:11434

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
| `/read <file> [start-end]` | Inject file content into AI context |
| `/edit <file> <line>` | Manually replace a single line |
| `/edit <file> code` | Apply last AI reply (code block) to whole file — diff before applying |
| `/edit <file> <line> code` | Apply code block starting at `<line>` |
| `/edit <file> <start>-<end> code` | Apply code block replacing exactly lines `start`–`end` |
| `/save <file> [mode]` | Save last AI reply to a file |
| `/bkup save <file>` | Save a backup as `<file>.bkup` |
| `/bkup restore <file>` | Replace `<file>` with its `.bkup` copy |

`/save` modes: `overwrite` (default), `append-above` / `-aa`, `append-below` / `-ab`, `add-suffix`, `code`

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

`/fix` is designed for 1B models — output is strictly constrained to `LINE N: content`.
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
Runs any shell command and injects stdout + stderr into the AI context.

```
/run python main.py
/run pytest tests/ -x
```

---

### Project map

The map command scans your project with language-agnostic regex, extracts definitions (classes, functions, endpoints, tables…) and cross-references between files. No external dependencies — pure regex, works for Python, Java, SQL, HTML, Terraform, YAML, and anything else.

```
/map index [path] [depth]   — scan project, save to .1bcoder/map.txt
/map find [query] [-y]      — search the map and inject results into context
/map trace <id> [-y]        — follow call chain backwards from an identifier (BFS)
/map idiff [path] [depth]   — re-index then show diff vs previous snapshot
/map diff                   — show diff without re-indexing (safe to repeat)
```

**`/map find` search syntax:**

| Token | Where | Effect |
|---|---|---|
| `term` | filename | include if filename contains term |
| `!term` | filename | exclude if filename contains term |
| `\term` | child lines | include only if a child line contains term |
| `\!term` | child lines | include block but hide child lines containing term |
| `\\!term` | child lines | exclude entire block if any child contains term |
| `-y` | — | skip "add to context?" confirmation |

```
/map find register                       — files named *register*
/map find \register                      — files that define/link "register"
/map find register \register             — both: in name AND in children
/map find \register !mock                — has "register" in children, skip mock files
/map find auth \UserService \\!deprecated -y
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

`/agent` runs an autonomous loop: the model reads the task, decides which tools to use, executes them one at a time, and stops when it outputs plain text with no ACTION.

```
/agent <task description>
```

```
/agent find and fix the divide by zero bug in calc.py
/agent add input validation to the register endpoint
```

The agent loop:
```
[agent] turn 1/10
AI: I need to read the file first.
ACTION: /read calc.py

[tool result]  1: def divide(a, b):
               2:     return a / b

[agent] turn 2/10
AI: Found it. Line 2 has no zero check.
ACTION: /bkup save calc.py

[agent] turn 3/10
ACTION: /edit calc.py 2 code

[agent] turn 4/10
ACTION: /map idiff

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
| `/host <url> [-sc]` | Switch Ollama host; `-sc` keeps context |
| `/ctx <n>` | Set context window size in tokens (default 8192) |
| `/ctx` | Show current usage vs limit |
| `/ctx cut` | Remove oldest messages until context fits |
| `/ctx save <file>` | Save full conversation to file |
| `/ctx load <file>` | Restore a saved conversation |
| `/clear` | Clear conversation context |
| `/help` | Show full command reference |
| `/help <command>` | Show help for one command (e.g. `/help map`) |
| `/help <command> ctx` | Same, and inject into AI context |
| `/init` | Create `.1bcoder/` scaffold in current directory |
| `/exit` | Quit |

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
