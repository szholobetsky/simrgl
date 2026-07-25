---
name: 1bcoder recent changes
description: Commands and fixes added to 1bcoder in recent sessions
type: project
originSessionId: 457af3c4-1663-4f2f-a188-730d645cdea6
---
## Changes made in session 2026-04-23/24

### /flow system (new)
Deterministic Python pipelines in `_bcoder_data/flows/`. Run with `/flow <name> [args]`, list with `/flow list`.
- `_cmd_flow` dispatcher: loads `<name>.py` from BCODER_DIR/flows, HOME_BCODER_DIR/flows, INSTALL_BCODER_DIR/flows
- Pattern: `run(chat, args)` → collect data (narrow first) → temp context → inject only summary into `chat.messages`
- `chat._agent_exec(cmd, auto_apply=True)` — runs any /command, returns string
- `/flow` added to `_KNOWN_CMDS`, `_CMD_SPEC`, dispatch
- `_bcoder_data/doc/FLOWS.md` — full developer guide (template, chat API table, arg parsing, prompt tips)

### Built-in flows
- `webask.py` — DDG search → fetch top URLs → temp LLM
- `grounding.py` — keyword extract (exact first, fuzzy fallback) → progressive 4-level search → temp LLM asks "which files?"
- `simargl_files.py` — `/run simargl search --mode task --sort rank "{task}"` → loop /read → temp LLM
- `py_error_trace.py` — `-f file` / inline args / fallback to `chat._last_output` (last /run) → extract File:line → read ±10 lines → temp LLM
- `commit_message.py` — staged diff → unstaged diff → minimal one-line prompt

### /web command (new)
- `/web search <term>` — DuckDuckGo HTML POST, returns titles+URLs+snippets; routes to agent context or main context
- `/web fetch <url>` — fetch page, strip HTML, truncate to 8000 chars; same context routing
- Context routing: `self._agent_msgs if (self._in_agent and self._agent_msgs) else self.messages`

### websearch agent
- `_bcoder_data/agents/websearch.txt` — uses `ACTION: /web search` and `ACTION: /web fetch`
- Alias `/websearch = /agent websearch {{args}}` in `_bcoder_data/aliases.txt`
- Alias `/webask = /flow webask {{args}}` (replaces old built-in /webask command)

### /translate fixes
- Removed old positional `mode` / `lang` subcommands — key:value syntax only
- Fixed `/translate last mode:mini lang:uk` split bug: `tokens = " ".join(parts[2:]).split()`
- Code block preservation in mini/offline modes: stash ```` ``` ``` ` blocks as `[CODEBLK_N]` before translation, restore after

### /run fix
- `_cmd_run` now sets `self._last_output = output` after capturing output — enables `/flow py_error_trace` with no args

### Agent done prompt fix
- `[s]ummary / [a]ll / [n]one` prompt was missing from normal done branch (only existed in proc-only branch). Fixed.

### README.md
- Added `/flow` section, `/web` command, `/websearch` alias

## Changes made in session 2026-04-21/22

### /translate command (new)
Four-mode translation pipeline — user input translated en before LLM call, reply translated back after:
- `online` — Google Translate via deep-translator
- `mini` — argostranslate (~100MB per language pair); was called `offline` before this session
- `offline` — NLLB-200-distilled-600M via ctranslate2 (~2.4GB download → ~600MB int8); sentence-level chunking (line-by-line) to prevent repetition loops; flores200 language code map; source files deleted after conversion
- `lm` — Ollama / LM Studio (translategemma:4b default); supports `openai://` prefix for LM Studio
- Config: `~/.1bcoder/translate.json`; key:value syntax for `/translate setup`
- `/translate last` retranslates last reply; `translated` qualifier for `/proc run mdx translated`
- `[translating en→uk...]` spinner + colored `─── UK ───` separator (`_LBLUE`)
- `/translate` added to `_KNOWN_CMDS` and `_CMD_SPEC`
- `lmtrans` profile added to `_bcoder_data/profiles.txt`
- `_bcoder_data/scripts/Translate.txt` script added
- `_bcoder_data/doc/TRANSLATE.md` created (language code reference, four modes)

### /script run N fix
- Numeric index now resolves global script paths correctly
- Root cause: `shlex.split` strips backslashes on Windows; fixed with forward slash conversion
- `run` subcommand now resets `[v]` markers before applying (always runs all steps fresh)

### md proc fix
- `C:/Project/1bcoder/_bcoder_data/proc/md.py` — force UTF-8 via `io.TextIOWrapper` to avoid cp1252 UnicodeEncodeError on Windows with Cyrillic content

### pyproject.toml
- Was emptied (cause unknown); restored from git history; current version: `0.1.7`
- Next publish to PyPI should bump to `0.1.8`

## Changes made in session 2026-04-06/07/08

### New commands
- `/script run <file> [key=value ...]` — shorthand for `apply -y`, added to `_CMD_SPEC` keywords
- `/script show N` — open script N from list then show its steps
- `/prompt load N` — select prompt by number inline (still shows list first)
- `/mcp connect <name> <cmd> --cwd <dir>` — sets subprocess working directory

### Fixes
- `MCPClient.__init__` now accepts `cwd=` parameter, passed to `subprocess.Popen`
- `_CMD_SPEC["/prompt"]` keywords updated: added `list`, `delete`
- `_CMD_SPEC["/script"]` keywords updated: added `run`
- All usage: strings updated for /script, /prompt, /mcp

### Changes made in session 2026-04-09

- `/proj` — project management built into 1bcoder (see project_1bcoder_proj_compose.md)
- `/ctx compose` — context composer with dedup (see project_1bcoder_proj_compose.md)
- `/ctx compact N` — compact last N messages in place
- `DEFAULT_AGENT_TOOLS` now includes `tree` and `find`
- `AGENT_SYSTEM_BASIC` hints to start with `/tree`
- Startup tip line added: `/agent ask`, `/read`, `/tree`
- `/doc <path>` — accepts direct file path, not only doc/ folder names
- `/proc run md <file>` and `/proc run mdx <file>` — accept filename argument directly
- `info/lectures/index.md` updated: /plan→/script, added /proj+/ctx compose lecture, agent ask, thinking models
- `_write_config_yml` bug fixed: was silently dropping `active_project` key

### simargl MCP fixes (C:\Project\codeXplorer\capestone\simrgl\simargl\)
- Scores >1 bug fixed: was dividing by 127 instead of 127² in numpy_backend.py
- Hanging bug fixed: print() in embedder.py was writing to stdout (MCP channel) → moved to stderr
- show_progress_bar=False when not tty (prevents hang under MCP stdio)
- Pre-warm model at server startup in main() — reads meta.json, loads embedder
- Fast-fail before model load: check index exists before calling get_embedder()
- `--store-dir` and `--project-id` startup args added to simargl-mcp
- `_resolve()` now called in all MCP tools
