# 1bcoder — The Human-in-the-Loop Anthill Prototype

## Abstract

**1bcoder** is a terminal-based AI coding tool built on Ollama, designed specifically for 1B-parameter language models. It is the first working implementation within the Anthill research lineage, functioning as a **human-in-the-loop prototype** of the Anthill Distributed Cognitive OS. Rather than automating the full agent pipeline, 1bcoder externalizes the cognitive roles of Router, Architect, Notary, and Auditor onto the human operator, while providing a structured set of tool primitives that directly instantiate the Anthill's core architectural concepts: constrained generation, curated context injection, plan-based task decomposition, adversarial verification via diff preview, a flat-file project map as a primitive OKG seed, a single-model agentic loop as a stub Router, and MCP-mediated external tool access. This document describes the project in full and maps each of its components to their theoretical counterparts in the Anthill architecture.

---

## 1. Project Description

### 1.1 Problem Statement

Small language models (1B parameters) fail not because they cannot reason, but because their context windows are small and their generation is unconstrained. Given a 600-line file and the instruction "fix the bug," a 1B model will hallucinate. Given the 15 relevant lines and the instruction "fix the bug; output only `LINE N: corrected content`," the same model frequently succeeds.

This is the empirical core of the Anthill hypothesis as stated in [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §1.1: *"the quality of the answer is determined primarily by the quality of the context, not the size of the model."*

1bcoder is a practical tool built around this single insight.

### 1.2 Core Design Principles

| Principle | Expression in 1bcoder |
|---|---|
| **Minimal context** | `/read file.py 10-30` injects only the relevant line range, not the full file |
| **Constrained generation** | `FIX_SYSTEM` prompt forces output to `LINE N: content` — one line, no freeform |
| **Externalized verification** | Every AI-proposed change shows a diff and requests confirmation before writing |
| **Structured decomposition** | Plans decompose tasks into discrete, independently executable steps |
| **Externalized project knowledge** | `/map` indexes the codebase into a searchable flat-file structure — the OKG seed |
| **MCP as nervous system** | External tool servers provide structured context without raw file access |

### 1.3 Technical Stack

- **Language**: Python 3.10+
- **UI framework**: Plain terminal REPL — no TUI framework; works in any shell, IDE terminal, or SSH session
- **AI backend**: Ollama HTTP API (`POST /api/chat`, NDJSON streaming) **and** OpenAI-compatible API (`POST /v1/chat/completions`, SSE streaming) — selected via URL scheme prefix
- **Provider abstraction**: `parse_host(s)` returns `(http_url, provider)`. Scheme `ollama://` (default) or `openai://` selects protocol branch. Plain `http://` URLs default to Ollama. This enables transparent use of LMStudio, LiteLLM, vLLM, or any OpenAI-compatible server without code changes.
- **Terminal output**: ANSI color helpers (`_ok`, `_err`, `_info`, `_warn`, `_cdiff`) applied only in `print()` calls — never injected into AI context. Colors are cosmetic; context remains plain text.
- **Dependencies**: `requests>=2.28` only (standard library for everything else)
- **Entry point**: `1bcoder` command (via `pyproject.toml` + `py-modules` build)

The deliberate absence of a TUI framework is a design choice: 1bcoder is optimized for portability, low overhead, and compatibility with CI/headless use. It runs identically whether launched interactively, piped from a script, or driven by `--planapply`.

### 1.4 Project Files

| File | Role |
|---|---|
| `chat.py` | Entire application — REPL, all command handlers, MCP client, streaming logic (~4500+ lines) |
| `map_index.py` | Standalone project scanner: AST-free regex-based identifier extractor → `map.txt` |
| `map_query.py` | Standalone map query tool: `find` (filter blocks) and `trace` (BFS call chain) modes |
| `map_query_help.txt` | Full CLI reference for `map_query.py` |
| `pyproject.toml` | Build metadata; py-modules = [chat, map_index, map_query]; entry point `1bcoder = "chat:main"` |
| `requirements.txt` | Single line: `requests>=2.28` |
| `run.bat` | Windows quick-launch |
| `MCP.md` | Catalog of ready-to-use MCP servers (filesystem, git, web, database, browser…) |
| `<install_dir>/.1bcoder/plans/` | Global plan library — shipped with the tool; available in every project |
| `.1bcoder/plans/` | Per-project plan `.txt` files; local file overrides global plan of same name |
| `.1bcoder/map.txt` | Generated project map (definitions, cross-references) |
| `.1bcoder/map.prev.txt` | Previous map snapshot — enables structural diff without re-indexing |
| `.1bcoder/agent.txt` | Agent configuration: max_turns, auto_apply, tool whitelist |
| `.1bcoder/profiles.txt` | Saved `/parallel` worker profiles |

---

## 2. Feature Architecture

### 2.1 Context Injection — `/read` and `/readln`

```
/read <file> [start-end]
/read <file1> <file2> <file3> ...
/readln <file> [start-end]
```

The user selects which lines to inject into the AI's context window. Multiple files may be listed in a single command — each is read sequentially and appended to the context in order. `/readln` injects with line numbers — use before `/fix` or `/patch` when the model needs to reference exact line positions. `/read` (without numbers) produces cleaner output for prose files, notes, and structured data. This is the manual implementation of what the Anthill's **Librarian agent** (GPU 1) would do automatically by querying Object Passports from the OKG. The human operator plays the Librarian role: knowing which files and lines are relevant and loading only those.

This command directly validates the Anthill's context efficiency claim. A 1B model given `/read file.py 10-25` performs measurably better than the same model given the full file, because it cannot be distracted by irrelevant surrounding code. Multi-file reading enables loading a call chain (interface + implementation + test) in a single step — a direct approximation of the Librarian's `get_neighbors()` traversal.

### 2.2 Constrained AI Edits — `/fix` and `/patch`

Two edit modes, calibrated for different model scales:

**`/fix` — designed for 1B models**

Uses `FIX_SYSTEM` prompt:
```
"Respond with ONLY the single most important fix in this exact format:
LINE <number>: <corrected line content>
One fix only. No explanation. No other text. Preserve indentation."
```

The model's output format is constrained to a single line. This is the Church-Turing "tape management" principle from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §1.2: the model's capacity is not increased — its required output is compressed to the point where 1B parameters are sufficient.

The extracted fix is parsed with `re.search(r'LINE\s+(\d+)\s*: ?(.*)')`, applied to the file, and shown as a unified diff before the user confirms.

**`/patch` — designed for 7B+ models**

Uses `PATCH_SYSTEM` prompt enforcing a strict SEARCH/REPLACE block format:
```
<<<<<<< SEARCH
exact lines to replace
=======
replacement lines
>>>>>>> REPLACE
```

A fuzzy matching algorithm (`_find_in_lines`) handles three cases: exact match, indentation-tolerant match, and line-number-prefixed match (when a model echoes `/read` output). This makes the patch system robust to the common hallucination pattern of models copying visible context artifacts.

**`/edit <file> <line> code`** — INSERT mode: a single-line target inserts new lines *before* that line rather than overwriting. A range `start-end` replaces those lines. This distinction is critical for the agent loop: the Coder can add new functions at a given position without accidentally destroying existing code.

Both `/fix` and `/patch` display a **unified diff before writing**, implementing the Anthill's Crooked Wall Principle: the human is the spirit level.

### 2.3 Shell Integration — `/run`

```
/run <command>
```

Runs any shell command and injects stdout+stderr into the AI context. This is the manual implementation of the **Kantian Observer Loop** from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §6.2: the user runs the code, observes its actual behavior (noumenal reality), and injects the output as grounding evidence for the next AI interaction. Tracebacks, test failures, and print output all become part of the phenomenal representation the model reasons from.

### 2.4 Plan System — `/plan`

Plans are `.txt` files in `.1bcoder/plans/`, one command per line. Lines prefixed `[v]` are marked done and skipped. Plans can be executed interactively (Y/n/q per step), automatically (`/plan apply -y`), or headlessly from the CLI (`--planapply`).

This is the manual implementation of the **Anthill orchestration pipeline**. The user writes what the Architect + Router would generate:

```
# example plan: fix the authentication timeout
/read src/auth/session.py 45-90
/fix src/auth/session.py 45-90 timeout not reset on activity
/run pytest tests/auth/ -x
/save ans/fix_result.txt
```

Plans support `{{key}}` parameter placeholders, substituted at apply time (immediate substitution — not deferred) — enabling reusable templates analogous to the Anthill's parameterized task schemas.

**`/plan create ctx`** captures all work commands typed in the current session into a ready-to-run plan automatically. This closes the loop between ad-hoc exploration and reproducible procedure — the same transition the Anthill makes from interactive to pipeline mode.

**Global plan library**: Plans shipped with 1bcoder (next to `chat.py`) are available in every project without copying. Current global plans include DevOps utilities (PipFreeze, CheckRequirements, GitIgnorePython, EnvTemplate, MySQLDump, SQLiteSchema, DockerMySQL, DockerNginx, DockerPython, DockerStack), research tools (WikiSearch, WikiPage, DuckDuckGoInstant, PyPI), and code workflow templates (AddFunction, RunAndFix, NewScript, Explain, Refactor). Local project plans override global plans of the same name. This implements a two-tier SKILL.md analog: global behavioral templates + per-project specializations.

### 2.5 Project Map — `/map`

The `/map` command suite provides a flat-file, regex-based implementation of the OKG's core function: making the codebase's structure legible to agents without requiring them to read raw source files.

```
/map index [path] [depth]    — scan project, extract definitions → .1bcoder/map.txt
/map find [query] [-y]       — filter map blocks by filename or child-line content
/map trace <identifier> [-y] — BFS backwards through call graph from a defined identifier
/map diff                    — diff map.txt vs map.prev.txt (no re-index)
/map idiff [path] [depth]    — re-index then diff vs previous snapshot
```

**`map_index.py`** scans the project tree with language-agnostic regex patterns, extracting:
- Defined identifiers (functions, classes, endpoints, SQL tables, Terraform resources, HTML ids…)
- Cross-file references with relationship type classification: `import`, `call`, `ref`, `expr`
- Optionally: module-level variables and function parameters (depth 3)

The output (`map.txt`) is a structured text file — a flat, human-readable shadow of the OKG. Each entry looks like:

```
auth/routes.py
  defines : register(ln:45), login(ln:62), logout(ln:78)
  links  → auth/models.py (import:User, call:validate_token)
  links  → db/session.py (call:get_session, import:Session)
```

**`map_query.py`** provides two query modes:
- **`find`**: filter file blocks by filename terms (`term`, `!term`) and child-line content (`\term`, `\!term`, `\\!term`) — equivalent to semantic coordinate lookup in the Anthill's OKG
- **`trace`**: BFS backwards through the reference graph from any defined identifier — equivalent to a Librarian agent traversing the OKG's dependency edges

**`/map diff` / `/map idiff`**: After any code change, re-indexing and diffing the map reveals which identifiers were added, removed, or moved. Two additional structural integrity signals are now computed and shown after the diff:

- **`ORPHAN_DRIFT`**: delta in orphan count (defined but never called) between snapshots. `+N [DEGRADATION]` → new dead code or deleted caller file. `−N [HEALING]` → dead code removed.
- **`GHOST ALERT`**: cross-snapshot detection of files that were link targets in the previous map but no longer exist in the current map. Catches the case where an agent deletes a callee file whose names vanish from `global_index` — making the deletion invisible to structural diff alone.

These are the manual equivalent of the Notary agent's post-commit OKG consistency check (M2: OCS). `detect_ghosts()` in `map_query.py` implements the cross-snapshot comparison logic.

The `/map` system is not an OKG — it has no graph database, no RDF triplets, no runtime persistence of semantic relationships. But it is the **OKG seed**: it externalizes exactly the knowledge the OKG would contain, in a queryable form, without requiring any graph infrastructure.

This is the most significant architectural advance in 1bcoder since the initial design.

### 2.6 Autonomous Agent Loop — `/agent`

```
/agent <task description>
```

Runs an autonomous agentic loop configured by `.1bcoder/agent.txt`:
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

The model receives a system prompt listing available tools (2-line summaries extracted from `HELP_TEXT` by `get_help_list()`), followed by the task. On each turn, the model either emits `ACTION: /command` (tool call) or plain text (completion). The tool call is executed via `_agent_exec()`, which captures output using a `_Tee` class (stdout is both shown to the user and captured as a string for context injection). The result is injected as `[tool result]` and the loop continues.

`auto_apply` bypasses confirmation prompts by replacing `self._confirm` with a lambda returning `True`. The agent loop runs in its own message thread (copies current context), so the main conversation is not polluted unless the user confirms at the end.

**Multi-ACTION per turn**: the agent loop uses `ACTION_RE.findall(reply)` (not `.search()`) to extract and execute *all* ACTION lines from a single model reply. A model may emit several tool calls in one response; each is executed in sequence, results injected, then the loop continues. This materially increases throughput for models capable of emitting compound plans.

**Code preview before execution**: before each ACTION is executed, the agent shows a preview of the command (and the last AI reply for `/edit code`, `/patch code` operations), followed by `execute? [Y/n/q]` — skip or abort without executing. The `-y` flag suppresses this confirmation. This is the Crooked Wall Principle applied to the agent execution layer.

**`/agent continue`**: the agent saves its state (`self._agent_state`) at the end of each loop — including the thread's message history, remaining turns, and last task description. `/agent continue` restores this state and resumes from the stopping point. This implements primitive **session persistence** for the agent role specifically, analogous to checkpointing an Anthill pipeline mid-execution.

**`/agent advance`**: uses the full `advanced_tools` set and a more capable system prompt — designed for 7B+ models. Includes `run`, `diff`, `map`, `bkup`, and all edit tools.

**`/agent -y`**: skips per-action confirmation — all ACTIONs execute automatically. Works at any position in the command (`/agent -y task`, `/agent task -y`). Equivalent to `auto_apply = true` in `agent.txt` but scoped to a single run.

**`/agent -t N`**: overrides `max_turns` from `agent.txt` for a single run.

The agent system prompt (`AGENT_SYSTEM`) is intentionally minimal — two explicit file operations with inline SEARCH/REPLACE format example, no instruction noise. This was redesigned after observing that small models reliably fail when given too many command variants to choose from. Orphan code detection: if the model writes a code block without an ACTION line, the agent detects this, warns the user, and offers to save the code to a named file.

`/edit` command parsing now tolerates `code:`, `code,`, `code.` suffixes — small models frequently append punctuation to the keyword when writing ACTION lines.

This is a stub implementation of the **Router + Architect + Coder pipeline** from Anthill Phase 1. It is single-model (no role separation), linear (no conditional branching), and relies on the same model to plan and execute. Its value is proving that the primitives are sufficient: the model can reason about which tool to use, call it correctly, and chain multiple steps to completion — given enough context and the right task scope.

The `agent.txt` tool whitelist implements a primitive **SKILL.md**: it constrains which verbs the agent can invoke, preventing it from attempting operations outside its tested capability. Removing tools from the list restricts a weaker model to a smaller instruction set.

### 2.7 Backup/Restore — `/bkup`

```
/bkup save <file>      — copy file to <file>.bkup; rotates existing backup to <file>.bkup(N)
/bkup restore <file>   — replace file with <file>.bkup (always the latest)
```

Lightweight file snapshot before any risky edit. If `file.bkup` already exists, it is renamed to `file.bkup(1)` (incrementing N if needed) before writing the new backup — no snapshot is ever silently overwritten. The agent is instructed to call `/bkup save` before modifying important files. This is the Anthill's pre-change state preservation, implemented without a version control abstraction layer.

### 2.8 Parallel Queries — `/parallel`

```
/parallel ["prompt1"] ["prompt2"] [profile <name>] [host|model|file ...]
```

Sends prompts to multiple models simultaneously using `concurrent.futures`. Each response is saved to its designated file. Current context (`/read` injections) is automatically included. Profiles are saved in `.1bcoder/profiles.txt` for reuse.

This directly implements **Multi-Instance Debate** from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §4.3. The user reviews multiple outputs and plays the Critic role — identifying where all models agree (structural signal) versus where they diverge (implementation choice). The profile system mirrors the Anthill's GPU farm worker configuration.

### 2.9 Post-processors — `/proc`

```
/proc list
/proc run <name> [args...]
/proc on <name> [args...]     — persistent: runs after every LLM reply
/proc off [name]              — stop one or all persistent processors
/proc new <name>              — create processor from template
```

Post-processors are Python scripts that receive the last LLM reply on `stdin` and write results to `stdout`. The protocol:
- `stdout` lines are displayed and optionally injected into context
- `key=value` lines are extracted as named parameters
- `ACTION: /command` lines are confirmed with the user then executed (in `run` mode)
- Exit code non-zero = failure; `stderr` shown as warning

Built-in processors:

| Processor | Purpose | Mode |
|---|---|---|
| `extract-files` | Extract filenames; `ACTION: /read` if exactly one found | one-shot |
| `extract-code` | Extract code blocks; `ACTION: /save` if one block + filename detected | one-shot |
| `extract-list` | Convert first bullet/numbered list to comma-separated line | one-shot |
| `grounding-check` | Score identifiers against `map.txt`, warn if <50% real | persistent |
| `collect-files` | Accumulate filenames to `.1bcoder/collected-files.txt` | persistent |
| `regexp-extract` | Extract all regex matches: `regexp-extract <pattern> [-i] [-u] [-g N]` | one-shot |
| `add-save.py` | Accumulate code blocks across turns into a target file | persistent |

**`/proc on` supports multiple simultaneous processors**: each is run in order after every reply. `/proc off <name>` stops one; `/proc off` stops all.

**`regexp-extract`** is the map-reduce primitive for free-form model output. Because 1B models "think while they write" (generation IS reasoning), constraining output length constrains thinking. The correct pattern: let the model generate freely, then extract structure programmatically:

```
/proc run regexp-extract \b[0-9]{3}\b          # find 3-digit numbers
/proc run regexp-extract "def (\w+)\(" -g 1 -u  # extract function names
/proc run regexp-extract [\w./\\-]+\.py -u       # collect .py paths
```

In Anthill terms, `/proc` is the **discriminator layer**: a lightweight post-generation filter that validates or extracts structure from free-form output without requiring the model to constrain its own generation. This is architecturally equivalent to a Process Reward Model operating on finished output rather than token-by-token.

### 2.10 Team Runs — `/team`

```
/team list
/team show <name>
/team run <name> [--param k=v ...]
/team new <name>
```

Spawns multiple 1bcoder workers in parallel, each running a different plan against the same project. Each worker gets its own model, host, and plan. Results are saved to `.1bcoder/results/`; logs to `.1bcoder/team-logs/`.

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

`--param` values are forwarded to every worker plan as `{{placeholders}}`. After all workers finish, results are aggregated with a summary plan.

Built-in team plans implement the **map-reduce pattern** for codebase analysis: `team-tree-worker.txt` (structural location), `team-search-worker.txt` (function-level search), `team-map-worker.txt` (dependency graph), `team-summarize.txt` (aggregation).

In Anthill terms, `/team` is the **parallel agent dispatch** mechanism — the same function as the LangGraph orchestrator spawning role-specialized agents on separate GPUs, implemented as parallel subprocesses with plan-driven task assignment.

### 2.11 Prompt Templates — `/prompt` and `/format`

```
/prompt save <name>    — save last user message as reusable template
/prompt load           — numbered list, select, fill {{params}} interactively
/format <description>  — inject strict output format constraint into context
/format clear          — remove active format constraint
```

**`/prompt`** stores reusable message templates with `{{key}}` placeholders in `<install>/.1bcoder/prompts/`. Values are prompted interactively on load. Equivalent to parameterized task schemas in the Anthill's plan system — but at the message level rather than the command sequence level.

**`/format`** injects a strict output format constraint as a system-level instruction before the next generation. Examples: `/format JSON array`, `/format one word`, `/format LINE N: content`. This complements `FIX_SYSTEM`/`PATCH_SYSTEM` by allowing ad-hoc format enforcement without modifying the system prompt. Particularly useful for forcing structured output from models that drift toward prose.

```
/diff <file_a> <file_b> [-y]
```

Computes and displays a unified diff between any two files — not just before/after versions of the same file. Typical use: `file.py` vs `file.py.bkup`, two branches of the same module, or pre/post refactor snapshots. The diff output is injected into context, allowing the model to reason about what changed between two states.

This is the Anthill's **Notary consistency check** applied manually and on-demand, extended to arbitrary file pairs rather than only OKG snapshots.

### 2.13 Structural Diff — `/diff`

```
/diff <file_a> <file_b> [-y]
```

Computes and displays a unified diff between any two files. Typical use: `file.py` vs `file.py.bkup`, two branches of the same module, or pre/post refactor snapshots. The diff output is injected into context, allowing the model to reason about what changed between two states.

This is the Anthill's **Notary consistency check** applied manually and on-demand, extended to arbitrary file pairs rather than only OKG snapshots.

### 2.14 Model Parameters — `/param`

```
/param <key> <value>
```

Injects arbitrary model parameters into every subsequent generation request — `temperature`, `top_p`, `top_k`, `repeat_penalty`, `seed`, or any provider-specific key. Values are auto-cast (bool → bool, integer → int, float → float, else str). LMStudio and vLLM pass these through their OpenAI-compatible endpoint.

This implements the Anthill's **per-agent behavioral calibration**: a Coder agent running in `/fix` mode may be configured with `temperature 0.1` for determinism; a brainstorming run uses `temperature 0.9`. The human manually sets what the Router would select automatically per role.

### 2.15 Think Block Management — `/think`

```
/think include
/think exclude
```

Some reasoning models (QwQ, DeepSeek-R1, certain Qwen variants) emit `<think>...</think>` blocks containing chain-of-thought before the answer. By default (`exclude`), these blocks are shown during streaming (visible to the user) but stripped before storing in `self.messages` and `self.last_reply`. This prevents think-block content from consuming context tokens across turns or contaminating `/patch code` extractions.

`/think include` retains think blocks in context — useful when the reasoning chain itself is the target of analysis or debugging.

This is the Anthill's **context hygiene layer**: intermediate cognitive artifacts are shown but not persisted unless explicitly requested.

### 2.16 MCP Integration

A full `MCPClient` class implements JSON-RPC over stdio, compatible with any standard MCP server. The `/mcp` command suite provides:

```
/mcp connect <name> <command>   — start and connect a server
/mcp tools [name]               — list available tools
/mcp call <server/tool> [args]  — call a tool, inject result into context
/mcp disconnect <name>          — shut down server
```

This is the **nervous system** described in [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §8.1. The filesystem, git, web, database, and memory servers available in `MCP.md` are exactly the external knowledge sources the Anthill agents use to access structured context without reading raw files. When Phase 1's `get_passport()` MCP server is built, the existing `/mcp connect` command will connect to it without any code changes.

### 2.17 Output Management — `/save`

```
/save <file> [mode]
```

Modes: `overwrite`, `append-above` / `-aa`, `append-below` / `-ab`, `add-suffix`, `code` (strips ` ``` ` fences). Supports multiple files in one command — mapping code blocks to files by position.

The `code` mode implements the Anthill's principle of treating model outputs as structured artifacts rather than raw text: the markdown wrapper is discarded and only the code payload is persisted.

### 2.18 Context and Session Controls

| Command | Role |
|---|---|
| `/ctx <n>` | Set context window size in tokens (default 8192) |
| `/ctx cut` | Remove oldest messages until context fits within limit |
| `/ctx compact` | AI-assisted context compression: model summarizes the full conversation into a compact digest, which replaces the message history. Preserves semantic continuity while freeing tokens for next task. |
| `/ctx save <file>` | Dump full conversation to a file — session persistence |
| `/ctx load <file>` | Restore a saved conversation — cross-session memory |
| `/model <name> [-sc]` | Switch model at runtime; `-sc` keeps context (switch Coder agent) |
| `/host <url> [-sc]` | Switch provider/host at runtime; `-sc` keeps context; accepts `ollama://`, `openai://`, or plain URLs |
| `/clear` | Reset conversation context (new task isolation) |
| `Ctrl+C` | Interrupt streaming response (agent interrupt in Anthill terms) |
| `/help <cmd> [ctx]` | Show per-command help; `ctx` injects the help text into AI context |

`/ctx save` / `/ctx load` implement primitive **session memory** — the ability to resume a working context across launches. In Anthill terms, this is a manual L2 memory tier (conversation history) without the L3 (OKG) or L4 (vector RAG) tiers.

### 2.19 Named Agents and Aliases — `/agent <name>`, `/alias`

```
/agent <name> [-t N] [-y] <task>
/<name> <task>                    — direct dispatch shorthand
/alias /name = expansion          — define a command alias
/alias save /name                 — persist to aliases.txt
```

The named agent system is the most architecturally significant development since the original `/agent` loop. It moves agent definition **out of code and into files**, enabling specialization without modifying `chat.py`.

**Agent definition files** (`.1bcoder/agents/<name>.txt`):

```ini
description = What this agent does
max_turns = 8
auto_exec = true
auto_apply = true

system =
    You are a ... Complete the task using the available tools.
    ...
    Available tools:
    {tool_list}

tools =
    run
    find

aliases =
    /schema = /run sqlite3 {{args}} ".schema"
    /query  = /run sqlite3 {{args}}
```

Four fields define an agent's complete behavioral envelope:

| Field | Role |
|---|---|
| `system =` | Inline multiline system prompt; `{tool_list}` substituted at runtime |
| `tools =` | Whitelist of allowed tools — controls both what the agent knows and what it can do |
| `aliases =` | Agent-scoped command shortcuts; active only during this agent's run |
| `max_turns`, `auto_exec`, `auto_apply` | Execution parameters |

**Agent-scoped aliases** are the critical primitive: they let the agent file define domain-specific shorthand commands (e.g. `/schema db` → `sqlite3 db ".schema"`) that are merged into `self._aliases` before the loop and fully restored after. The agent operates with an augmented vocabulary; the main session is unaffected.

**Shared `_run_agent_loop`**: `/ask`, `/agent <name>`, and `/agent` all execute through a single shared loop. The `_in_agent` flag (True for the entire duration, regardless of `-y`) enables the code keyword warning system on `/patch`, `/save`, `/insert`, `/edit` — catching agent-generated commands that would fail silently.

**Global alias table** (`aliases.txt`): loaded at startup, survives `/clear`. Global file ships `/ask = /agent ask` and `/advance = /agent advance` — turning what were hardcoded commands into first-class named agents. New agents are added by creating a file, not modifying code.

**Context integration after completion**: when any agent loop finishes, the user is prompted `[s]ummary / [a]ll / [n]one` — pulling the agent's last reply, full conversation, or nothing into main context. This enables seamless chaining: `/ask` discovers the relevant function, then the main session modifies it.

**The Thin Agent Principle** (empirical finding, 2026-03-23): the SQLite agent (2 tools: `run`, `find`) with `qwen3:1.7b` completed DB research tasks reliably. The same model running `aider`'s general agent (12+ tools, large system prompt) with `nemotron-3-nano:4b` failed to start due to context overflow; when it did start, it looped the system prompt. The finding: **for models ≤4B parameters, a narrow tool surface and a domain-specific system prompt are more predictive of task success than model capability alone**. See [1BCODER_SPECIAL_AGENTS.md](1BCODER_SPECIAL_AGENTS.md) for the full experimental design.

In Anthill terms, each named agent is a **minimal SKILL.md instantiation**: system prompt = behavioral specification, tool list = allowed verb space, aliases = domain vocabulary. The agent file is what the Dynamic SKILL.md (Gap 5.1) would load and update automatically. The named agent system is the missing link between the static `agent.txt` whitelist and a fully dynamic per-role behavioral configuration.

---

## 3. Relation to the Anthill Distributed Cognitive OS

### 3.1 Mapping: 1bcoder Commands → Anthill Components

| 1bcoder Feature | Anthill Component | Implementation Status |
|---|---|---|
| `/read file.py 10-30` | Librarian agent + Object Passport retrieval | **Manual** — human selects context |
| `FIX_SYSTEM` + `/fix` | Coder agent (constrained output format) | **Implemented** — format enforced by prompt |
| `PATCH_SYSTEM` + `/patch` | Coder agent (SEARCH/REPLACE mode) | **Implemented** — fuzzy match applied |
| Diff preview before apply | Crooked Wall Principle / Auditor pre-check | **Implemented** — human is the spirit level |
| `/run` + output injection | Kantian Observer Loop | **Manual** — human triggers and injects |
| `/plan` system | Router + Architect task decomposition | **Manual** — human authors the plan |
| `/plan create ctx` | Anthill session → plan crystallization | **Implemented** — auto-captures session commands |
| `/plan apply -y` / `--planapply` | LangGraph pipeline batch execution | **Stub** — linear, no conditional routing |
| `/map index` / `map_index.py` | Notary agent (passive) — structure extraction | **Primitive** — regex flat-file, no runtime update |
| `/map find` / `map_query.py find` | Librarian agent — semantic coordinate lookup | **Primitive** — text filter, no embedding |
| `/map trace` / `map_query.py trace` | Librarian agent — OKG graph traversal | **Primitive** — BFS on cross-ref index |
| `/map diff` / `/map idiff` | Observer Loop + Notary consistency check | **Primitive+** — structural diff + ORPHAN_DRIFT + GHOST ALERT |
| `.1bcoder/map.txt` | OKG (flat-file seed) | **Primitive** — text, no RDF, no graph DB |
| `/agent` | Router + Architect + Coder loop (single model) | **Stub** — no role separation, linear only |
| `agent.txt` tool whitelist | SKILL.md (behavioral constraint) | **Primitive** — static config, no Researcher update |
| `FIX_SYSTEM` / `PATCH_SYSTEM` | SKILL.md (output format constraint) | **Primitive** — hardcoded, not dynamic |
| `/bkup save/restore` | Pre-change state preservation | **Implemented** — rotating snapshots, no silent overwrite |
| `/readln <file>` | Librarian retrieval with line-number anchoring (for `/fix`/`/patch` targeting) | **Implemented** — line numbers in context |
| `/parallel` | Multi-Instance Debate (Section 4.3) | **Implemented** — concurrent model queries |
| `/parallel profile` | GPU farm worker configuration | **Implemented** — saved profiles in profiles.txt |
| `/proc run <name>` | Post-generation discriminator / structured extraction layer | **Implemented** — stdin/stdout pipeline, ACTION protocol |
| `/proc on <name>` (multiple) | Persistent discriminator chain after every reply | **Implemented** — list of active procs, ordered execution |
| `regexp-extract` proc | Map-reduce extraction: free generation → programmatic structure | **Implemented** — regex with -i/-u/-g flags |
| `grounding-check` proc | Identifier grounding validation against map.txt | **Implemented** — warns if <50% real identifiers |
| `/team run` | Parallel agent dispatch — role-specialized workers on separate plans | **Implemented** — concurrent subprocesses, per-worker logs |
| `/prompt save/load` | Parameterized task schema templates | **Implemented** — {{key}} substitution, interactive fill |
| `/format <description>` | Ad-hoc output format constraint injection | **Implemented** — appended to system context before generation |
| `/agent advance` | Full toolset agent loop for 7B+ models | **Implemented** — advanced_tools whitelist, richer system prompt |
| `/agent -y` | Autonomous execution mode (no per-action confirmation) | **Implemented** — any position in command |
| MCP client + servers | MCP nervous system (Section 8.1) | **Implemented** — full JSON-RPC client |
| `/ctx save` / `/ctx load` | L2 memory tier (conversation history) | **Implemented** — file-serialized context |
| `/ctx compact` | Context compression / session summarization | **Implemented** — AI-distills history into compact digest |
| `parse_host` + `openai://` scheme | Multi-provider API abstraction | **Implemented** — LMStudio, LiteLLM, vLLM, OpenAI API |
| `/diff <file_a> <file_b>` | Notary consistency check (on-demand, arbitrary files) | **Implemented** — unified diff injected into context |
| `/param <key> <value>` | Per-agent behavioral calibration (temperature, top_p…) | **Implemented** — auto-cast, passed to provider |
| `/think exclude/include` | Context hygiene — chain-of-thought artifact filtering | **Implemented** — shown but not persisted by default |
| Agent multi-ACTION + code preview | Compound tool call per turn + Crooked Wall execution gate | **Implemented** — findall + Y/n/q per action |
| Named agent files (`agents/<name>.txt`) | Dynamic SKILL.md per role — system prompt + tool whitelist + aliases | **Implemented** — file-driven, no code change needed |
| Agent-scoped aliases (`aliases =` in agent file) | Per-role domain vocabulary — active only during agent run | **Implemented** — saved/restored around `_run_agent_loop` |
| Global `aliases.txt` | Session-persistent command shortcuts; survives `/clear` | **Implemented** — loaded at startup, global then local |
| `/ask = /agent ask` alias | Router-level dispatch to named agent | **Implemented** — `ask.txt` defines system + tools + aliases |
| `/sqlite` agent (2 tools) | Minimal SKILL.md — domain specialist with aliased primitives | **Implemented** — first empirical thin agent, validated 2026-03-23 |
| `[s]/[a]/[n]` context prompt | Agent result integration gate — human decides what enters main context | **Implemented** — after every agent loop |
| `_in_agent` flag | Agent execution context signal — enables code keyword warnings | **Implemented** — True for full loop duration, regardless of -y |
| Shared `_run_agent_loop` | Single orchestration substrate for all agent types | **Implemented** — /ask, /agent <name>, /agent all share the loop |
| Global plan library | Two-tier SKILL.md (global templates + local overrides) | **Implemented** — 20 built-in plans shipped with tool |
| `.1bcoder/` directory | BCODER_DIR / project-scoped workspace | **Stub** — flat files, no graph |
| Status line (model + ctx%) | Operator situational awareness | **Implemented** — printed before each prompt |
| Human operator | Router + Architect + Notary + Auditor | **Human-in-the-loop** |

### 3.2 Position in the Anthill Roadmap

1bcoder sits at **Phase 0 (complete) → Phase 1 entry** of the Anthill implementation roadmap from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §11.

**Phase 0 (complete):**
- Constrained generation format (`LINE N: content`, SEARCH/REPLACE)
- Context injection via `/read` (selective line range; multiple files in one command)
- `/edit <line>` INSERT mode (single line = insert before, range = replace)
- Post-change verification via diff preview (Crooked Wall)
- `/diff <file_a> <file_b>` — arbitrary file comparison injected into context
- MCP substrate installed and operational
- Plan execution engine (linear, manual-authored, `--planapply`)
- Global plan library (20 built-in DevOps + workflow templates; local override)
- Multi-instance parallel queries with profiles
- Session persistence (`/ctx save` / `/ctx load`; `/ctx compact` AI summarization)
- Flat-file project map with identifier extraction and BFS trace (OKG seed)
- Single-model agentic loop with tool whitelist (Router+Architect stub)
  - Multi-ACTION per turn, code preview gate, `/agent continue` checkpoint
- Pre-change backup/restore with rotation (no silent overwrite)
- Status line: active model (truncated) + context fill % before each prompt
- Structural integrity signals in `/map idiff`: ORPHAN_DRIFT + GHOST ALERT
- Agent system prompt simplified for small model compatibility
- `/edit` keyword parsing tolerates trailing punctuation (`code:`, `code,`)
- Multi-provider API abstraction (`ollama://`, `openai://` schemes)
- Model parameter injection (`/param`) — temperature, top_p, seed, etc.
- Think block hygiene (`/think`) — shown, not persisted by default
- `/readln` — context injection with line numbers for precise `/fix`/`/patch` targeting
- `/proc` post-processor pipeline: stdin/stdout, ACTION protocol, persistent mode
- Multiple simultaneous `/proc on` processors — ordered chain after every reply
- Built-in processors: `extract-files`, `extract-code`, `extract-list`, `grounding-check`, `collect-files`, `regexp-extract`, `add-save`
- `/team` parallel worker dispatch — yaml-defined, per-worker plan + log, `--param` forwarding
- `/prompt` reusable message templates with `{{key}}` substitution
- `/format` ad-hoc output format constraint injection
- `/agent advance` — full toolset for 7B+ models (`advanced_tools` whitelist)
- `/agent -y` flag — autonomous execution, any position in command
- Two-tier `agent.txt`: `tools` (small models) + `advanced_tools` (7B+ models)
- Named agent system: `.1bcoder/agents/<name>.txt` with inline system prompt, tool list, agent-scoped aliases
- Shared `_run_agent_loop`: single orchestration substrate for `/ask`, `/agent <name>`, `/agent`
- Global `aliases.txt`: session-persistent shortcuts, survive `/clear`; `/ask`, `/advance`, `/sqlite` defined as named agents
- `[s]/[a]/[n]` context integration gate after every agent loop
- `_in_agent` flag: code keyword warning system for `/patch`, `/save`, `/insert`, `/edit` in agent context
- **Thin Agent Principle** (2026-03-23): empirical finding that ≤5 domain-specific tools outperform large general toolsets for models ≤4B; `sqlite` agent (2 tools) validated with `qwen3:1.7b`
- `/help <alias>`: resolves alias and shows target agent file metadata (description, tools, aliases)

**Phase 1 (missing — next step):**
- `ontology_extractor.py` — AST parser generating Object Passports from source files (map_index.py is the regex precursor)
- Object Passport YAML schema
- Notary agent — file watcher that triggers incremental OKG updates on save (map idiff is the manual version)
- Minimal MCP server exposing `get_passport()` and `get_neighbors()` (1bcoder's MCP client is ready to connect)
- Graph database (Neo4j/FalkorDB) replacing the flat `map.txt`

The critical observation: 1bcoder already has the **verbs** (edit, run, read, fix, patch, plan, map, agent, parallel) and a **proto-memory** (map.txt, ctx save/load). Phase 1 upgrades the memory from flat text to a live graph and makes the Notary autonomous.

### 3.3 The Human as Agent Network

In the current state of 1bcoder, the human operator is running the following Anthill cognitive roles simultaneously:

| Anthill Role | Human Activity |
|---|---|
| **Router** | Decides which file to work on next; chooses between `/fix`, `/patch`, `/agent` |
| **Architect** | Writes the plan; decides which lines are relevant; frames task scope |
| **Librarian** | Chooses what to `/read` and at what range; uses `/map find` to locate relevant files |
| **Notary** | Runs `/map idiff` after edits to verify what changed structurally |
| **Auditor** | Reviews diffs before approving each change; reads `/map diff` warnings |
| **Tester** | Decides when to `/run` the code and what to check |
| **Observer Loop** | Injects `/run` output as ground truth; interprets failures and reformulates |

This is not a weakness — it is the **design philosophy** of the prototype phase. The human runs the roles manually, gains understanding of what each role does and where it fails, and the automation of each role becomes a well-specified engineering task rather than a speculative design.

### 3.4 The SKILL.md Analog

1bcoder has two SKILL.md analogs operating at different levels:

**Static system prompts** (`FIX_SYSTEM`, `PATCH_SYSTEM`): encode behavioral constraints ("One fix only. No explanation.") and output format specifications. These are hardcoded strings — the primitive form.

**`agent.txt` tool whitelist**: encodes the agent's allowed action space. Removing a tool from the list restricts the agent's behavioral envelope — this is the SKILL.md's tool manifest in its simplest form.

The evolution path is clear: replace the hardcoded `FIX_SYSTEM` string with a `SKILL.md` file loaded at startup, and add a mechanism to update it when the model repeatedly fails a particular output format. The `agent.txt` whitelist becomes a section of the same file.

---

## 4. Relation to the SIMARGL Research Program

### 4.1 1bcoder as Empirical Validation Platform

The SIMARGL research asks: *can context quality substitute for model scale?* 1bcoder is a working instrument for testing this empirically. Every `/fix` command is an experiment: same model, different context quality (full file vs. selected range), measurable outcome (fix accepted or rejected). Aggregate data from 1bcoder sessions would provide direct empirical evidence for or against the Anthill's core hypothesis.

The `/map` system extends this: does injecting a `map find` result (structured identifier index) as context improve fix quality compared to raw line injection? This is a direct test of the Object Passport hypothesis (Exp A1 from [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md)) using only existing 1bcoder infrastructure.

### 4.2 Concept Correspondences

| SIMARGL Concept | 1bcoder Expression |
|---|---|
| **Minimum Description Length (Object Passport)** | `/read file.py 10-30` — human-selected MDL representation of the relevant code |
| **Semantic Coordinates** (positive space) | Lines selected by `/read`; blocks matched by `/map find` — the positive semantic space of the task |
| **Negative Space filtering** | Lines *not* read; `/map find` `!term` tokens — deliberately excluded to prevent context saturation |
| **Reduction Ladder** | `/read` selects Level 6 (code) → `/map find` selects Level 5 (passport-like structure) → model reasons at Level 3–4 |
| **Symbol Grounding** | `/run` output grounds abstract model reasoning in concrete execution results |
| **Hermeneutic Circle** | Edit → diff → confirm → `/run` → re-read → `/map idiff` — the iterative code↔context loop |
| **Novelty/Structurality** | A human reviewer of diffs performs the same classification informally; `/map diff` warnings flag structural changes |
| **SIMARGL Evolution Zone** | Plans that `/fix` within a single module are EVOLUTION; cross-module patches flagged by `/map diff` are DISRUPTION |
| **Entity Map** | `map.txt` — the Entity Map made queryable; direct precursor to the OKG |
| **Cross-reference graph** | `/map trace` BFS — the call graph traversal that the OKG's Librarian performs over graph edges |
| **Observer Loop** | `/map idiff` is the manual observer loop: re-index after change, diff vs. previous state, validate consistency |

### 4.3 Contribution to the Anthill Experimental Program

[ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) defines twelve experiments across four families. 1bcoder provides a practical testbed for **Experiment Family A** (OKG Effectiveness) without requiring the full OKG infrastructure:

- **Exp A1 (Passport vs Raw — Context Compression Battle)**: Compare `/fix` success rate with (a) full-file `/read` vs. (b) selected-range `/read` vs. (c) a `/map find` result (structured identifier block) injected as context. The three conditions map directly to A1's Condition 1 (raw), Condition 2 (OKG/passport), and Condition 3 (Oracle) without requiring a graph database.

- **Exp A2 (OKG Density and Memoization Maturity)**: Run the same task after progressively enriching `map.txt` — cold (just indexed), warm (after 25 tasks), mature (after 100 tasks). Use MHR-equivalent: what fraction of `/map find` queries return useful results vs. empty.

- **Exp B2 (Specialization vs Generality)**: Use `/parallel` to run the same task across multiple model sizes simultaneously; measure fix acceptance rate across sizes at controlled context quality. 1bcoder already implements the parallelism; the measurement is a logging addition.

These experiments require only 1bcoder, an Ollama instance, and a set of test files — no OKG, no graph database, no agent infrastructure.

---

## 5. Gap Analysis: 1bcoder → Full Anthill

| Gap | Current State | Required Component | Complexity |
|---|---|---|---|
| **Object Passports** | `map.txt` blocks are a regex primitive (defines + links, no types, no signatures) | `ontology_extractor.py`: AST parser → full YAML passport with method signatures, types, dependencies | Medium |
| **OKG** | `.1bcoder/map.txt` — flat text, rebuilt from scratch on each `/map index` | Neo4j/FalkorDB graph with RDF triplets; incremental updates without full re-scan | High |
| **Notary agent** | Human runs `/map idiff` after edits | File watcher → incremental OKG update on save; triggered automatically, not manually | Medium |
| **Observer Loop** | Human runs `/run` + `/map idiff` manually | Post-edit trigger: re-scan changed file → diff vs. OKG → flag inconsistencies → block if OCS < threshold | Medium |
| **Automated plan generation** | Human writes plans manually; `/plan create ctx` captures session commands | Architect agent: NL intent → plan steps with file/range selection | High |
| **Conditional routing** | Linear plan execution, no branching | LangGraph: conditional edges, retry on failure, escalation to Oracle | High |
| **Role separation in agent loop** | `/agent` uses one model for all roles | Separate Coder, Critic, Auditor models; adversarial integrity (B1 experiment setup) | High |
| **Qualia detection** | None | Router classifies tasks by Qualia Boundary markers; routes to Oracle gateway | High |
| **Oracle gateway** | None (user manually switches `/host` to a cloud endpoint) | Privacy-preserving passport-only Oracle interface; never sends source code | Medium |
| **Dynamic SKILL.md** | Hardcoded `FIX_SYSTEM`/`PATCH_SYSTEM` + static `agent.txt` | Dynamic per-agent SKILL.md loaded at startup; Researcher agent updates on failure patterns | Medium |
| **SIMARGL metrics as runtime monitor** | Human reads `/map diff` output and classifies changes manually | Automated Novelty/Structurality computation on each OKG delta; Auditor escalation on DISRUPTION | Medium |

The minimum viable step from 1bcoder to Phase 1 Anthill is the **Notary + Passport + minimal MCP server** trio:
1. Replace `map_index.py` regex extractor with an AST-based `ontology_extractor.py` producing YAML Object Passports
2. Deploy a lightweight file watcher (watchdog) that calls the extractor on file save → updates `map.txt` / graph
3. Wrap the map file in a minimal MCP server exposing `get_passport(file)` and `get_neighbors(identifier)`
4. The existing 1bcoder `/mcp connect` command connects to it immediately — no other code changes required

The plan execution engine, context injection, diff verification, parallel queries, and MCP client are all already production-ready for Phase 1.

---

## 6. Summary

1bcoder is the Anthill OS running on one substrate: human intelligence. It provides a complete set of edit primitives, a context injection mechanism, a flat-file project map (OKG seed), a single-model agentic loop, a plan execution engine, a multi-instance query system, and an MCP client — all the verbs of the Anthill system, with the human playing every cognitive role that has not yet been automated.

Its relationship to the Anthill architecture is not merely analogical. Each 1bcoder command is a direct, working implementation of a specific Anthill component, held in manual or primitive operation until the corresponding agent infrastructure is built to replace the human. The tool is designed to be progressively automated: each role the human currently performs is a well-defined, independently implementable component in the Anthill roadmap.

The `/map` system is the most significant structural advance: it externalizes the codebase's identifier topology into a queryable artifact — the same function the OKG serves, without the graph database. This makes the Notary's job (Phase 1) a concrete engineering task: replace `map.txt` with a live graph and add a file watcher.

The `/agent` loop proves the orchestration substrate works: the model can select tools, chain them, and converge on a result. The missing piece is not orchestration capability but role separation and conditional routing.

The core thesis — that context quality substitutes for model scale — is validated every time a 1B model successfully applies a `/fix` that it would fail without `/read`'s selective context injection, or every time `/map find` helps the agent locate the relevant file without reading the entire codebase.

---

**Document Version**: 1.5
**Created**: 2026-03-05
**Updated**: 2026-03-23
**Project**: SIMARGL / Anthill Research Program
**Relation to**:
- [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) — parent architecture document
- [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) — experimental validation framework
- [FINAL_PRODUCT.md](FINAL_PRODUCT.md) — SIMARGL product architecture (codeXpert = 1bcoder's planned MCP evolution)
- [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md) — agent architecture precursor
- [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) — philosophical grounding (Observer Loop, symbol grounding)

**Changelog v1.5** (2026-03-23):
- §2.19 (new): Named agents and aliases — agent files, shared loop, thin agent principle, Anthill mapping
- §3.1: Added 8 new mapping table rows (named agents, aliases, sqlite agent, _in_agent, shared loop, context gate)
- §3.2: Updated Phase 0 completion list with named agent system and thin agent principle
- See [1BCODER_SPECIAL_AGENTS.md](1BCODER_SPECIAL_AGENTS.md) for experimental design

**Changelog v1.4** (2026-03-18):
- §1.4: Updated `chat.py` line count (~4500+)
- §2.1: Added `/readln` — line-number injection variant
- §2.6: Updated `agent.txt` to two-tier `tools`/`advanced_tools`; added `/agent advance`, `/agent -y`
- §2.9 (new): `/proc` post-processor pipeline — full protocol description, all built-in processors, map-reduce framing, Anthill discriminator mapping
- §2.10 (new): `/team` parallel worker dispatch — yaml format, built-in plans, Anthill mapping
- §2.11 (new): `/prompt` templates and `/format` constraint injection
- §2.13 (new): `/diff` moved and renumbered from old §2.9
- §3.1: Added `/readln`, `/proc`, `/team`, `/prompt`, `/format`, `/agent advance`, `/agent -y` to mapping table
- §3.2: Updated Phase 0 completion list with all new features

**Changelog v1.3** (2026-03-09):
- §2.5: `/map idiff` now computes ORPHAN_DRIFT + GHOST ALERT; `detect_ghosts()` in `map_query.py`
- §2.6: Agent system prompt simplified; orphan code detection (code block without ACTION); `/edit` tolerates `code:` suffix
- §2.7: `/bkup save` now rotates existing backups — no silent overwrite
- §3.1: Updated mapping table entries for `/map idiff`, `/bkup`, added status line row
- §3.2: Phase 0 list updated with all new features

**Changelog v1.2** (2026-03-08):
- §1.3: Added multi-provider abstraction (`ollama://` / `openai://` schemes); ANSI color note
- §1.4: Added global plan library directory to file table
- §2.1: `/read` now accepts multiple files in one command
- §2.2: Added note on `/edit <line>` INSERT mode vs range REPLACE
- §2.4: Added global plan library (20 built-in plans, two-tier SKILL.md analog)
- §2.6: Added multi-ACTION per turn, code preview gate, `/agent continue`, `/agent -t N`
- §2.9 (new): `/diff <file_a> <file_b>` — structural diff as Notary on-demand check
- §2.10 (new): `/param <key> <value>` — per-agent model parameter calibration
- §2.11 (new): `/think include/exclude` — context hygiene for reasoning models
- §2.14: Added `/ctx compact`; updated `/host` to note openai:// support
- §3.1: Extended mapping table with all new Phase 0 components
- §3.2: Updated Phase 0 completion list with all new features

**Changelog v1.1** (2026-03-07):
- Corrected technical stack: plain terminal REPL, no Textual TUI, `requests` only
- Removed `headless.py` (does not exist); added `map_index.py`, `map_query.py`
- Added Section 2.5 `/map` system — OKG seed, flat-file project knowledge
- Added Section 2.6 `/agent` — Router+Architect stub, SKILL.md analog
- Added Section 2.7 `/bkup`; expanded §2.11 with `/ctx save/load`, `/help ctx`, `/model -sc`
- Updated mapping table (Section 3.1) with all new commands
- Updated Phase 0 completion list and Phase 1 gap definition (Section 3.2)
- Updated concept correspondences (Section 4.2) with map/Entity Map/Observer Loop entries
- Added `/map`-based Exp A1/A2 testbed to Section 4.3
- Updated Gap Analysis (Section 5) — Object Passports now "Primitive" not "None"; added role separation row
