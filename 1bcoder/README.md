# 1bcoder

AI-assisted code editor designed for small (1B parameter) language models running locally via [Ollama](https://ollama.com), [LMStudio](https://lmstudio.ai), or [LiteLLM](https://litellm.ai).

---

**(c) 2026 Stanislav Zholobetskyi**
Institute for Information Recording, National Academy of Sciences of Ukraine, Kyiv

*Створено в рамках аспірантського дослідження на тему:
«Інтелектуальна технологія підтримки розробки та супроводу програмних продуктів»*

---

**Core idea:** 1B models hallucinate badly when asked to rewrite large blocks of code. 1bcoder works around this by keeping changes small and structured — the model outputs a single-line fix (`LINE N: content`) or a minimal SEARCH/REPLACE block, which the tool then applies with a diff preview before writing to disk.

Planning and navigation are externalized: plans live in `.txt` files, project structure is indexed into a searchable map — so the model never has to hold the whole codebase in its head.

**Target:** programmers running `qwen2.5-coder:0.6b` or `llama3.2:1b` on a 4 GB machine — offline, no cloud, no subscription. The tool does the heavy lifting so the model doesn't have to.

---

## What 1B models are actually good at

1B models fail at open-ended tasks. They excel at **bounded tasks** — where the answer is almost fully determined by the input you give them. The key is to use 1bcoder's navigation tools to narrow context to exactly what the model needs, then ask a specific question.

### Generate

| Task | How |
|---|---|
| Dockerfile / docker-compose | Describe the stack, ask the model to write it |
| SQL script | Load schema with `/plan apply SQLiteSchema.txt`, ask for INSERT / SELECT / migration |
| Simple algorithm | Bubble sort, binary search, Newton gradient — model knows these cold |
| Function stub → implementation | Write the signature + docstring, ask the model to fill the body |
| Unit test for one function | `/read` the function, ask for a test |
| Type annotations | `/read` a Python or TypeScript file, ask to add types |
| Regex pattern | Describe the pattern precisely, ask for the regex |
| Config file | nginx server block, systemd service, GitHub Actions single-job workflow |
| Shell script / Makefile | Repetitive structure the model handles reliably |
| argparse / click parser | Very formulaic — model knows these patterns exactly |
| Port a function to another language | Paste 10–20 lines of Python/Go/Java, ask for the equivalent |

### Explain

| Task | How |
|---|---|
| What does this function do? | `/read file.py 10-30` → ask; works even for unfamiliar languages |
| What does this error mean? | `/run python main.py` → output injected automatically → ask |
| What is this project? | `/tree ctx` → injects directory tree → ask for overview |
| What does this module do? | `/find ClassName -c ctx` or `/map find \ClassName -y` → inject → ask |
| What changed after my edit? | `/map idiff` → inject diff → ask what the structural change means |
| Explain this SQL / regex / config | Paste the fragment, ask; no file context needed |

### The rule behind the list

These tasks share one property: the model never needs to **search** for information. You deliver exactly the right context using `/read`, `/run`, `/find`, `/tree`, or `/map find` — then the model executes a single well-defined subtask. Context under 2K tokens, one clear question, deterministic output format: this is where 1B models are reliable.

Tasks that require the model to decide *what to look at* — refactoring across files, debugging a multi-file interaction, writing a new feature from scratch — need 32B+ models with `/agent advance`.

---

## Features

- Plain terminal REPL — works in any shell, IDE terminal, or SSH session; status line before each prompt shows active model, disk size, quantization, native context limit, and context fill %
- **`/read`** injects files without line numbers (clean text, ideal for `notes.txt` and structured data); **`/readln`** injects with line numbers (use before `/fix` or `/patch` when line references matter)
- **Command autocorrection** — typos in command names, file paths, and keywords are detected and fixed automatically before execution, for both human input and agent actions
- **`/tree [path]`** — display directory tree of the whole project or any subtree; ask to inject into context (or pass `ctx` to skip the prompt)
- **`/find <pattern>`** — search filenames and file content with regex; supports `-f`/`-c`/`-i`/`--ext` flags; highlights matches, asks to inject results into context
- AI proposes a **one-line fix** (`/fix`) or a **SEARCH/REPLACE patch** (`/patch`) — always shows a diff before applying
- **Apply AI code blocks directly** with `/edit <file> code` (new/full file) or `/patch <file> code` (SEARCH/REPLACE from reply, no line numbers needed) — preferred for agent mode
- **`<think>` tag support** — reasoning blocks shown in terminal, stripped from context by default; `/think include` passes model reasoning to the next turn
- Run shell commands and inject their output with `/run`
- Save AI replies to files with `/save` (code-fence stripping, multiple files, append modes)
- **Session persistence** — `/ctx save` / `/ctx load` dump and restore full conversations; `/ctx compact` summarizes and compresses the context via AI
- **Plans** — reusable sequences of commands stored as `.txt` files, run step-by-step or fully automated
- **Plan from history** — `/plan create ctx` captures this session's commands into a reusable plan automatically
- **Project map** — scan any codebase into a searchable index (`/map index`), query it (`/map find`), trace call chains (`/map trace`), and diff changes (`/map idiff`) — now includes `ORPHAN_DRIFT` alert (dead code delta) and `GHOST ALERT` (deleted file that other files depended on)
- **Agent mode** — `/agent <task>` runs an autonomous loop: the agent prompt instructs the model to emit one ACTION per turn (multiple ACTIONs are executed in order if the model emits them), checks `/help` before using a command, and stops when done. Configurable via `.1bcoder/agent.txt`
- **Backup/restore** — `/bkup save` rotates existing backups (`file.bkup` → `file.bkup(1)`, `file.bkup(2)`…) so no snapshot is ever overwritten; `/bkup restore` always restores the latest
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

  model    : gemma3:1b [815M Q4_K 32K]
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
| `/read <file> [file2 ...] [start-end]` | Inject file(s) into AI context **without line numbers** (clean text) |
| `/readln <file> [file2 ...] [start-end]` | Same as `/read` but **with line numbers** — use before `/fix` or `/patch` |
| `/insert <file> <line>` | Insert last AI reply before line N (full text) |
| `/insert <file> <line> code` | Insert extracted code block from last AI reply before line N |
| `/insert <file> <line> <text>` | Insert literal text directly, preserving indentation (e.g. `/insert main.py 14    x = 1`) |
| `/edit <file> <line>` | Manually replace a single line |
| `/edit <file> code` | Apply last AI reply (code block) to whole file — creates file if missing, diff before applying |
| `/edit <file> <line> code` | Apply code block starting at `<line>` — creates file if missing |
| `/edit <file> <start>-<end> code` | Apply code block replacing exactly lines `start`–`end` |
| `/save <file> [mode]` | Save last AI reply to a file |
| `/bkup save <file>` | Save a backup as `<file>.bkup`; rotates existing backup to `<file>.bkup(N)` |
| `/bkup restore <file>` | Replace `<file>` with its `.bkup` copy (always the latest) |
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
`/patch` works better with larger models (7B+) and can replace multiple consecutive lines.
`/patch <file> code` is the preferred agent mode edit — the agent writes the SEARCH/REPLACE block in its reply, then calls `/patch <file> code` to apply it without needing line numbers.

When `/patch` fails to find the SEARCH text it shows a diagnostic diff — the SEARCH lines vs the nearest matching lines in the file — so you can see immediately what doesn't match (e.g. trailing commas, wrong indentation). It also detects no-op patches where SEARCH and REPLACE are identical.

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

### Codebase navigation

#### `/tree` — directory structure

```
/tree                     whole project from current directory (depth 4)
/tree src                 subtree rooted at src/
/tree src/java/com -d 6   deep subtree, 6 levels
/tree static ctx          show and auto-inject into context (no prompt)
```

Displays a Unicode box-drawing tree. Skips `.git`, `node_modules`, `__pycache__`, `.venv`, and other noise automatically. When depth is cut, shows `… (N entries)` so you know something's there. After displaying results asks `Add tree to context? [Y/n]` — pass `ctx` to skip.

#### `/find` — search filenames and content

```
/find <pattern> [-f] [-c] [-i] [--ext <ext>] [ctx]
```

| Flag | Effect |
|---|---|
| *(none)* | search filenames **and** file content |
| `-f` | filenames only |
| `-c` | content only |
| `-i` | case-insensitive |
| `--ext py` | restrict to `.py` files |
| `ctx` | auto-inject results into context (no Y/n prompt) |

Pattern is a full **regex**. Matches highlighted in the terminal output.

```
/find MyClass                    filenames + content
/find user_id -c -i              content only, case-insensitive
/find config --ext py ctx        .py files, inject automatically
/find \.connect\(                regex: literal .connect(
```

After showing results asks `Add results to context? [Y/n]` (suppressed when there are no matches).

---

### Project map

The map command scans your project with language-agnostic regex, extracts definitions (classes, functions, endpoints, tables…) and cross-references between files. No external dependencies — pure regex, works for Python, Java, SQL, HTML, Terraform, YAML, and anything else.

```
/map index [path] [depth]                      — scan project, save to .1bcoder/map.txt
/map find [query] [-d N] [-y]                  — search the map and inject results into context
/map trace <id> [-d N] [-y]                    — BFS backwards: who depends on this identifier?
/map trace deps <id> [-d N] [-leaf] [-y]       — forward: what does this identifier depend on?
/map trace <start> <end> [-y]                  — shortest dependency path between two points
/map idiff [path] [depth]                      — re-index then show diff vs previous snapshot
/map diff                                      — show diff without re-indexing (safe to repeat)
/map keyword index                             — build keyword vocabulary from map.txt
/map keyword extract <text> [-f] [-a] [-n] [-c] — extract real identifiers from keyword.txt matching text/file
```

**Partial / incremental indexing** — for large codebases where a full re-scan is slow:

```
/map index sonar_core/src/main/java/org/sonar/core/util
```

When `path` is a subfolder of the working directory, 1bcoder:
1. Scans only that subtree (fast)
2. Adjusts all paths to be relative to the project root
3. Saves a named segment file: `.1bcoder/map_sonar_core_src_main_java_org_sonar_core_util.txt`
4. Patches `map.txt` in-place — removes stale blocks for that subtree, appends the fresh ones
5. Backs up the previous `map.txt` to `map.prev.txt` before patching

This lets you re-index a changed module in seconds instead of hours.

**`/map find` search syntax:**

| Token | Where | Effect |
|---|---|---|
| `term` | filename | include if filename contains term |
| `!term` | filename | exclude if filename contains term |
| `\term` | child lines | include block if any child line contains term |
| `\!term` | child lines | exclude entire block if any child contains term |
| `-term` | child lines | show ONLY child lines containing term |
| `-!term` | child lines | hide child lines containing term |
| `-d 1` | — | filenames only |
| `-d 2` | — | filenames + defines/vars (no links) |
| `-d 3` | — | full blocks (default) |
| `-y` | — | skip "add to context?" confirmation |

```
/map find register                       — files named *register*
/map find \register                      — files that define/link "register"
/map find register \register             — both: in name AND in children
/map find \register !mock                — has "register" in children, skip mock files
/map find auth \UserService -!deprecated -y
/map find \py -!import                   — show only non-import lines in .py files
/map find password -d 1                  — just filenames, no details
/map find models -d 2                    — filenames + defines/vars only
```

**`/map trace`** — three modes:

**1. Backwards BFS** (who depends on this?):
```
/map trace register -d 2

auth/routes.py  [defines register(ln:45)]
  ← import:register  app/__init__.py
  ← call:register    tests/test_auth.py
```

**2. Forward deps** (what does this depend on?):
```
/map trace deps UserService -leaf

deps (leaves): UserService(ln:5)  [UserService.java]
  UserEntity.java   [firstName, lastName, accountNumber]
  UserRepo.java     [findById, findByEmail, save]
```

**3. Path between two points** (shortest dependency chain):
```
/map trace AccountNumber UserController

path 1: AccountNumber(ln:15) → UserController

  AccountEntity.java
    ↓ import:AccountNumber
  UserDAO.java
    ↓ import:UserDAO
  UserController.java

  [Y]es add + next / [s]kip next / [n]o stop:
```

After each path:
- `Y` — add to context, find next alternative path
- `s` — skip (don't add), find next alternative path
- `l N` or `l` then enter N — auto-collect the next N paths without prompting (e.g. `l 10`)
- `n` — stop and inject everything collected so far

**`/map keyword`** — build and query a keyword vocabulary from the map:

```
/map keyword index
```

Scans `map.txt`, extracts all real code identifiers, saves to `.1bcoder/keyword.txt` as CSV (`word, count, line_numbers`). Run once after `/map index`.

```
/map keyword extract <text or file> [-f] [-a] [-n] [-c]
```

Finds identifiers from `keyword.txt` that match words in the given text or file. Output is always **real identifiers from keyword.txt** — never synthetic splits.

| Flag | Effect |
|---|---|
| *(none)* | Exact match: `rule` matches `rule` only, not `RuleIndex` |
| `-f` | Fuzzy subword match: splits both query and keyword into parts |
| `-a` | Alphabetical order (default: order of appearance in text) |
| `-n` | Show codebase count: `RuleIndex(25)` |
| `-c` | Comma-separated output |

**Fuzzy subword rules (`-f`):**
- `rule` → matches `RuleIndex`, `RuleName`, `RuleUpdater` (all contain subword `rule`)
- `RuleIndex` → matches `RuleIndex` only (requires both `rule` AND `index`)
- `RuleIndex` does **not** match `Rule` (missing `index`) or `Index` (missing `rule`)
- Combined identifiers are more specific — no false positives from their parts

Handles all naming conventions at match time: `camelCase`, `PascalCase`, `snake_case`, `UPPER_SNAKE_CASE`, `kebab-case`.

```
/map keyword extract "fix rule search performance" -f -c
→ RuleIndex, RuleSearch, RuleUpdater, SearchService

/map keyword extract notes.txt -f -n
→ RuleIndex(47)
  RuleName(23)
  SearchService(18)
```

**`/map idiff`** re-indexes and shows what changed — use after every edit:

```
/map idiff

[map diff]  map.prev.txt → map.txt

  calc.py
  - defines: subtract(ln:12)
  ! WARNING: 1 identifier(s) removed

ORPHAN_DRIFT = +1  [DEGRADATION]
  before: 3 orphans
  after:  4 orphans
  + subtract    ← calc.py

! GHOST ALERT — 1 deleted file(s) had active callers:
  ! services/UserService.java
    called: findUser, createUser
```

`ORPHAN_DRIFT` catches dead code accumulation and deleted caller files. `GHOST ALERT` catches deleted callee files that other files still depend on — the blind spot that structural diff alone misses.

Standalone tools (usable without 1bcoder):
- `map_index.py` — scanner only: `python map_index.py [path] [depth]`
- `map_query.py` — query only: `python map_query.py find \register` / `python map_query.py trace register`

---

### Agent mode

`/agent` runs an autonomous loop: the model reads the task and decides which tool to use. The agent prompt instructs the model to emit one ACTION per turn; if it emits multiple, all are executed in order. Stops when the model outputs plain text with no ACTION.

The agent system prompt is kept minimal for small model compatibility. Two file operations are taught explicitly:
- **Modify existing file**: write SEARCH/REPLACE block → `ACTION: /patch <file> code`
- **Create or replace file**: write full code block → `ACTION: /edit <file> code`

If the model writes a code block without an ACTION, the agent detects this and offers to save the code to a file.

```
/agent [-t N] [-y] <task> [plan step1, step2, ...]
```

- **`-t N`** — override `max_turns` for this run only (e.g. `-t 1` for a quick read+explain)
- **`-y`** — skip per-action confirmation (execute all actions automatically)
- Without `-y`: each proposed action pauses and asks `[Y/n/e/f/q]`:

| Key | Action |
|---|---|
| `Y` / Enter | Execute the action |
| `n` | Skip this action |
| `e` | Edit the command before executing (copies to clipboard on Windows) |
| `f` | Send feedback to the AI and skip the action (redirect the model mid-loop) |
| `q` | Stop the agent |
- **`plan step1, step2, ...`** — optional comma-separated list of items injected as hints one per turn; lets you drive the agent through multiple files or sub-tasks in sequence. If a turn produces an empty reply or no ACTION, the agent continues to the next plan step automatically

```
/agent find and fix the divide by zero bug in calc.py
/agent -t 1 read models.py and explain the User class
/agent -y -t 5 refactor utils.py
/agent read file plan models.py, views.py, urls.py
```

**`/agent advance`** uses the full advanced toolset and system prompt — designed for larger models (32B+), includes `run`, `diff`, `map`, `bkup`, and all edit tools:

```
/agent advance refactor the auth module
/agent advance read and summarise plan models.py, views.py
```

**`/agent continue`** resumes the last session (saved automatically on stop, Ctrl+C, or max_turns). Optionally pass a follow-up instruction:

```
/agent continue
/agent continue now also add error handling
/agent continue -t 5 -y
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

# tools: used by /agent — minimal set for small models
tools =
    read
    insert
    save
    patch

# advanced_tools: used by /agent advance — full set for larger models
advanced_tools =
    read
    run
    insert
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

Remove tools from either list to restrict what the model can use.

---

### Plans

Plans are `.txt` files containing one command per line, stored in `.1bcoder/plans/`.
Lines starting with `[v]` are already done and skipped. Lines starting with `#` are comments and skipped.

| Command | Description |
|---|---|
| `/plan list` | List all plan files (`*` = current). Shows global plans `(g:)` and project plans |
| `/plan open` | Select and load a plan (type number). Includes global and project plans |
| `/plan create [path]` | Create a new empty plan |
| `/plan create ctx [path]` | **Create plan from this session's command history** |
| `/plan show` | Display steps of the current plan |
| `/plan add <command>` | Append a step to the current plan |
| `/plan clear` | Wipe current plan completely |
| `/plan reset` | Unmark all done steps |
| `/plan reapply [key=value ...]` | Reset all done steps then apply automatically |
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

### Prompt templates

Save any useful message as a reusable template and load it later with `{{param}}` substitution.

```
/prompt save ConvertJavaToPy     # saves last user message as ConvertJavaToPy.txt
/prompt load                     # numbered list, select by number, fill {{params}} interactively
```

Templates stored in `<install>/.1bcoder/prompts/`. Use `{{keyword}}` placeholders — values are prompted on load.

---

### Post-processors (`/proc`)

Run a Python script against the last LLM reply. Useful for extracting filenames, validating identifiers against `map.txt`, collecting data across turns, and more.

```
/proc list              # list available processors
/proc run extract-files # one-shot: run against last reply, confirm ACTION if present
/proc on grounding-check  # persistent: run after every reply automatically
/proc off               # stop persistent processor
/proc new my-proc       # create a new processor from template
```

**Processor protocol:** `stdin` = last LLM reply · `stdout` = result · `key=value` lines = extracted params · `ACTION: /command` = confirmed and executed (run mode only) · exit 1 = failure.

Built-in processors in `<install>/.1bcoder/proc/`:

| Processor | Purpose | Best mode |
|---|---|---|
| `extract-files` | Extract filenames, `ACTION: /read` if one found | one-shot |
| `extract-code` | Extract code blocks; `ACTION: /save <file>` if one block + filename detected | one-shot |
| `extract-list` | Convert first bullet/numbered list in reply to comma-separated line | one-shot |
| `grounding-check` | Score identifiers against `map.txt`, warn if <50% | persistent |
| `collect-files` | Accumulate filenames to `.1bcoder/collected-files.txt` | persistent |

---

### Team runs (`/team`)

Spawn multiple 1bcoder workers in parallel, each running a different plan against the same project. Each worker gets its own context (e.g. `/tree`, `/find`, `/map`) and saves results to `.1bcoder/results/`.

```
/team list                                           # list team definitions
/team show code-analysis                             # show workers in a team
/team new my-team                                    # create team yaml from template
/team run code-analysis --param keyword=auth --param task="404 on login"
```

Team definition (`.1bcoder/teams/<name>.yaml`):
```yaml
workers:
  - host: localhost:11434
    model: qwen2.5-coder:1.5b
    plan: team-tree-worker.txt
  - host: openai://localhost:1234
    model: qwen2.5-coder:1.5b
    plan: team-search-worker.txt
  - host: 192.168.0.10:11434
    model: gemma3:4b
    plan: team-map-worker.txt
```

`--param` values are forwarded to every worker plan as `{{placeholders}}`. Each worker log goes to `.1bcoder/team-logs/`. After all workers finish, aggregate with a summary plan:

```
/plan apply team-summarize.txt --param keyword=auth --param task="404 on login"
```

Built-in team plans in `<install>/.1bcoder/plans/`:

| Plan | Worker role |
|---|---|
| `team-tree-worker.txt` | `/tree` → where does the keyword live structurally? |
| `team-search-worker.txt` | `/find` → which functions implement it? |
| `team-map-worker.txt` | `/map find` → what depends on it? |
| `team-summarize.txt` | reads all results, produces unified answer |

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
| `/param timeout <seconds>` | Set HTTP read timeout — increase when models are slow on large contexts (default: 120s) |
| `/param` | Show currently set params including timeout |
| `/param clear` | Remove all params and reset timeout to 120s |
| `/format <description>` | Inject a strict output format constraint and call the AI (e.g. `JSON array`, `one word`) |
| `/format clear` | Remove the active format constraint from context |
| `/clear` | Clear conversation context, reset params, and reload model metadata |
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
  model    : llama3.2:1b [1.3G Q4_K 131K]
  host     : http://localhost:11434
  provider : ollama
```

The status line before each `>` prompt shows the same info in compact form:
```
 llama3.2:1b [1.3G Q4_K 131K]  │  ctx 245 / 8192 (3%)
 gpt-4o-mini [128K]  │  ctx 1536 / 128000 (1%)     ← OpenAI (no disk size)
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
└── .1bcoder/              # global install assets
    ├── plans/             # built-in plan .txt files (team workers, examples)
    ├── teams/             # /team yaml definitions
    ├── prompts/           # /prompt saved templates
    └── proc/              # /proc post-processor scripts

Project folder (created by /init or /team run):
.1bcoder/
    ├── plans/             # project-specific plan .txt files
    ├── teams/             # project-specific team yamls (override global)
    ├── agent.txt          # agent mode config (max_turns, auto_apply, tools)
    ├── profiles.txt       # /parallel worker profiles
    ├── map.txt            # generated by /map index
    ├── map.prev.txt       # previous snapshot (for /map diff)
    ├── results/           # worker output files (tree-analysis.txt, etc.)
    └── team-logs/         # per-worker logs from /team run
```

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Submit message |
| `Shift+Enter` | Insert newline (requires Kitty keyboard protocol support) |
| `Ctrl+N` | Insert newline (reliable fallback for all terminals) |
| `ESC` | Interrupt AI response mid-stream |
| `Ctrl+C` | Interrupt streaming or exit prompt |

On Windows: hold `Shift` and drag with the left mouse button to select and copy text from the terminal.

---

## Tips for 1B models

- **Start small.** Use `/read file.py 10-25` instead of loading the whole file. Short context = better focus.
- **Use `/fix` not `/patch`.** The `LINE N: content` format is much more reliable at 1B scale than free-form generation.
- **Build a map first.** Run `/map index .` at the start of a session, then use `/map find` to load only the relevant parts into context.
- **Use plans.** Plans make multi-step work reproducible — the model only needs to handle one step at a time.
- **Capture workflows.** After solving a task manually, run `/plan create ctx` to save the exact steps as a reusable plan.
- **Agent mode needs a bigger model.** `/agent advance` works best with 32B+ models. For 1B–7B, use plans instead.
- **Ctrl+C** interrupts streaming if the model starts going off-track.
- **Use `/readln` before `/patch`.** Reading with line numbers lets you reference exact locations; `/patch` then sends the file without line numbers so the model's SEARCH block matches the real content.
- **Timeout errors?** If you see `Read timed out` with a large context, run `/param timeout 300` to extend the limit to 5 minutes.
- **Model behaving oddly after a long session?** Run `/clear` — it clears context, resets all params, and reloads model metadata, equivalent to a restart.

## Command autocorrection

1bcoder automatically detects and fixes common typos in commands before executing them — for both human input and agent-generated `ACTION:` lines.

| Error type | Example | Fixed to |
|---|---|---|
| Command name typo | `/insrt models.py 14` | `/insert models.py 14` |
| File path typo | `/read models.p` | `/read models.py` |
| Keyword prefix | `/insert main.py 14 co` | `/insert main.py 14 code` |
| Subcommand prefix | `/map fnd auth` | `/map find auth` |
| Subcommand typo | `/bkup resore calc.py` | `/bkup restore calc.py` |

For human input, the corrected command is shown with `[fix?]` and you are asked to confirm. For agent actions (`auto` mode), the fix is applied silently with a `[fix]` warning. Prefix matching is used for keywords to avoid false positives on hint words.

---

## Tips for reasoning models (Qwen3, DeepSeek-R1, etc.)

- **Disable thinking for simple tasks.** `/param enable_thinking false` speeds up responses when reasoning isn't needed.
- **Use `/think include` to chain reasoning.** Pass one model's `<think>` output as context to another model or the next turn.
- **`/patch <file> code` over `/edit`.** Reasoning models write precise SEARCH/REPLACE blocks — no line numbers needed, no full-file rewrites.
- **`/ctx compact` after long sessions.** Reasoning models produce verbose output; compact regularly to stay within context limits.
- **Connect via LMStudio.** `/host openai://localhost:1234` — full parameter control including `enable_thinking`, `temperature`, `seed`.
