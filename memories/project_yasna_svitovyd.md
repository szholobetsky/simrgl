---
name: yasna-and-svitovyd-plans
description: Status and design decisions for yasna (memory system) and svitovyd (code map service)
metadata: 
  node_type: memory
  type: project
  originSessionId: 59f4e9c6-779f-4f0a-9009-ff73ef6ea5a3
---

## yasna — Session Search Tool (pivoted from original plan)

Original plan: `C:\Project\codeXplorer\capestone\simrgl\info\yasna_plan.md` (2026-04-08) — a curated,
1bcoder-only project wiki (SQLite, manual keyword/note/file curation, `/project` alias in 1bcoder).

**As of 2026-07-07 the implemented tool at `C:\Project\yasna\` is a different, broader design** —
the plan was superseded, not followed. Confirmed by reading current source (v0.1.8, on PyPI as `yasna`):

- **Scope**: fully automatic session indexer/search across 9 AI coding agents via a pluggable
  adapter registry (`yasna/adapters/__init__.py`): claude, opencode, continue.dev, aider, nanocoder,
  1bcoder, gemini, codex, copilot. Not 1bcoder-only, not manually curated.
- **Storage**: flat text files under `~/.yasna/index/<agent>/<id>.txt` (`:::key: value` metadata
  header + full conversation body) — no SQLite at all (`core.py` has no DB, just `Session` dataclass
  + `write_session`/`read_meta`).
- **Commands**: only `index`, `find`, `list`, `about` (`yasna/cli.py`). None of the plan's manual-curation
  commands exist — no `init`, `switch`, `status`, `save`, `note`, `file`, `keyword add/list`, `show`.
- **No manual keyword/file tagging** — search is full-text substring match (`re.escape(query)`,
  case-insensitive) over the whole indexed conversation, not curated keywords/files/notes.
- **New concept not in the plan**: CWD-aware auto-scoping — `find`/`list`/`index` default to filtering
  by the session's `project_path` matching the current working directory (`scope="auto"`), with `-g`/
  `--global` to search everything. Replaces the plan's manual "project key" concept entirely.
- **Not implemented from plan**: regex file-extraction from 1bcoder command args, `/project` 1bcoder
  alias, v0.2 items (MCP server, wake-up, FTS5, ontology).
- Zero deps (stdlib only, `tqdm` optional for progress bar).

**Status:** Implemented and in active use — this is how the user finds/resumes past Claude Code
sessions (`yasna find <keyword>` → `claude --resume <session-id>`).

---

## svitovyd — Code Structure Map Service

**Why:** Світовид — чотириликий бог, бачить увесь світ одночасно = scans whole codebase.
`pip install svitovyd` (likely free on PyPI)

**Core concept:** Extract `map_index.py` and `map_query.py` from 1bcoder into standalone CLI + MCP server.
Same dual-mode pattern as simargl: `svitovyd` CLI + `svitovyd-mcp` MCP server.

**Status:** Discussed, NOT planned in detail, NOT implemented.
Decision: `/map` stays in 1bcoder for now — extract svitovyd later after yasna is done.

---

## Three-tool ecosystem

| Tool | God | Role |
|---|---|---|
| simargl | Симаргл — охоронець | git/task history → which files |
| yasna | Ясна — богиня долі | memory between sessions |
| svitovyd | Світовид — чотириликий | code structure, dependency graph |

simargl acronym: **S**emantic **I**ndex: **M**ap **A**rtifacts, **R**etrieve from **G**it **L**og
(added to simargl README.md)
