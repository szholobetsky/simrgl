# AGENTS.md — Project Context for AI Assistants

> This file is the single source of truth for any AI coding assistant working in this repo.
> Read it fully before making any suggestions or edits.

---

## What This Is

**SIMARGL** is a doctoral dissertation research platform + emerging product ecosystem.

**Core research question**: Given a natural language task description (from Jira / GitHub Issues), can we automatically predict which code modules/files need to be modified?

This is **task-to-code retrieval** — a semantic search problem at the intersection of NLP and software engineering. The dissertation is being written at the Institute for Information Recording, National Academy of Sciences of Ukraine.

**Author**: Stanislav Zholobetskyi  
**Copyright**: (c) 2026 Stanislav Zholobetskyi, Institute for Information Recording, NAS Ukraine  
**Dissertation title**: «Інтелектуальна технологія підтримки розробки та супроводу програмних продуктів»

---

## Repository Map

```
C:\Projects\simrgl\   ← THIS REPO (working dir)
├── exp0/        TF-IDF baseline (failed, historical reference only)
├── exp1/        Statistical analysis (Bradford, HHI distributions)
├── exp2/        Embedding models comparison (Word2Vec → BGE-large), module-level
├── exp3/        Full RAG pipeline — sentence-transformers + pgvector/Qdrant
├── exp4/        PLANNED — architecture families (ModernBERT, ColBERT, Qwen3)
├── exp5/        PLANNED — cross-vocabulary symbol grounding (TF-IDF on identifiers)
├── exp3.1 … exp14.1/  later experiments (ontology, agents, triangulation, health, glossary) — see the table below
├── datasets/    dataset lists: projects.csv (infrastructure / dev tools), bizprojects.csv (domain-rich: fineract, ofbiz, …)
├── ragmcp/      Production MCP server + local Ollama agent
├── concepts/    Research concept documents (see list below)
├── info/        Results files, leaderboard, planning docs
├── 1bcoder/     git submodule — PINNED SNAPSHOT, do not edit here (see below)
├── vyrii/       git submodule — PINNED SNAPSHOT, do not edit here (see below)
├── simargl/     git submodule — PINNED SNAPSHOT, do not edit here (see below)
├── radogast/    git submodule — PINNED SNAPSHOT, do not edit here (see below)
├── yasna/       git submodule — PINNED SNAPSHOT, do not edit here (see below)
├── alkonost/    git submodule — PINNED SNAPSHOT, do not edit here (see below)
├── svitovyd/    git submodule — PINNED SNAPSHOT, do not edit here (see below)
└── (syryn/)     moved out — independent clone at C:\Projects\syryn\

Not under this repo at all (independent folders only, never were submodules):
  C:\Projects\gamayun\   statistical ontology, no LLM
  C:\Projects\mokosh\    dataset gathering (git + tracker -> SQLite)
```

### Policy: independent clone always wins

Every submodule tool is migrating, one at a time, to an **independent clone** at `C:\Projects\<tool>\` (own git repo, own `origin` remote on GitHub). Once that independent clone exists for a tool, **it is the only place to edit that tool — never edit the submodule copy under `C:\Projects\simrgl\<tool>\` again.** The submodule copy becomes a pinned, read-only snapshot at whatever commit it was last synced to; it will silently drift from the independent clone until someone manually re-syncs the submodule pointer, and edits made there are easy to lose or fork by accident (this has already happened once with vyrii — see below).

**Before editing any tool, always check first whether `C:\Projects\<tool>\` exists as its own independent git clone.** If it does, edit there, full stop — even if the submodule copy looks equally "live" (same files, recent-looking history). If it does not exist yet, the submodule copy under `C:\Projects\simrgl\<tool>\` is still the live one, until it migrates.

| Tool | Path | Notes |
|---|---|---|
| 1bcoder | `C:\Projects\1bcoder\` | independent clone — **edit here**. `C:\Projects\simrgl\1bcoder\` (submodule) is a pinned snapshot, never edit. |
| vyrii | `C:\Projects\vyrii\` | independent clone — **edit here**. `C:\Projects\simrgl\vyrii\` (submodule) is a pinned snapshot, never edit. |
| simargl | `C:\Projects\simargl\` | independent clone — **edit here**. `C:\Projects\simrgl\simargl\` (submodule) is a pinned snapshot, never edit. Note: this is the `simargl` *tool/submodule*, distinct from `C:\Projects\simrgl\` the top-level dissertation repo (confusingly similar names — one letter apart). |
| radogast | `C:\Projects\radogast\` | independent clone — **edit here**. `C:\Projects\simrgl\radogast\` (submodule) is a pinned snapshot, never edit. |
| yasna | `C:\Projects\yasna\` | independent clone — **edit here**. `C:\Projects\simrgl\yasna\` (submodule) is a pinned snapshot, never edit. |
| svitovyd | `C:\Projects\svitovyd\` | independent clone — **edit here**. `C:\Projects\simrgl\svitovyd\` (submodule) is a pinned snapshot, never edit. Installed editable (`pip install -e C:\Projects\svitovyd`). The exp8/exp9* scripts still import svitovyd from the submodule path on purpose (`sys.path`, `cwd`) — that keeps the experiments reproducible; leave them. |
| syryn | `C:\Projects\syryn\` | independent clone — **edit here**. No submodule copy exists anymore under `C:\Projects\simrgl\syryn\`. |
| alkonost | `C:\Projects\alkonost\` | independent clone — **edit here**. `C:\Projects\simrgl\alkonost\` (submodule) is a pinned snapshot, never edit. Installed `alkonost` command may run a frozen `pip`-installed copy separate from either git copy (same three-copies trap as 1bcoder) — after editing, `pip install -e "C:\Projects\alkonost"` to actually see changes. |
| gamayun | `C:\Projects\gamayun\` | independent repo — **edit here**; never a submodule. Git initialised (branch `main`), no commits and no remote yet. Installed editable (`pip install -e`). |
| mokosh | `C:\Projects\mokosh\` | independent repo — **edit here**; never a submodule. Git initialised (branch `main`), no commits and no remote yet. Installed editable (`pip install -e`). |
| simrgl (this repo) | `C:\Projects\simrgl\` | top-level dissertation repo itself, not a tool submodule |

Every tool has migrated to the independent-clone pattern (1bcoder first, then vyrii/simargl/radogast/yasna/alkonost/syryn/svitovyd; gamayun and mokosh were born independent). A new tool starts as its own folder `C:\Projects\<tool>\`, never as a submodule; update this table when a tool is added.

---

## Experiment History and Status

| Exp | Method | Status | Key result |
|-----|--------|--------|------------|
| exp0 | TF-IDF | Failed | MAP 0.5–1.5%; too slow (4–48h), no semantics |
| exp1 | Statistical analysis | Complete | Term distribution insights; not predictive |
| exp2 | Word2Vec/FastText/GloVe/BERT/CodeBERT/BGE | Complete | BGE-large best (MAP=0.37 Sonar, 0.36 Flask) |
| exp3 | Sentence-transformers + pgvector/Qdrant | Complete | Full RAG pipeline; modn split honest baseline |
| exp4 | ModernBERT + ColBERT + Qwen3/jina/e5 | Planned | Architecture families comparison |
| exp5 | Symbol grounding on code identifiers | Planned | Cross-vocabulary bridge |
| ragmcp | MCP server + Ollama agent | Production | Real-time tool, deployable |
| exp3.1 | Task unit: ticket vs commit message | Complete | → its README |
| exp3.2 | Cross-vocabulary: commit-built index, ticket-language queries | Complete | → its README |
| exp6 | RAG retrieval modes via the simargl CLI | Complete | → its README |
| exp7 | Ontology from the tracker + git history, no LLM (gamayun) | Complete | method: concepts/GAMAYUN.md |
| exp8, exp8.1 | simargl + gamayun + svitovyd synergy; code ontology beside the history ontology | Complete | → their READMEs |
| exp9 … exp9.8 | Agents (Haiku; small local models, 4k context, CPU) with the pantheon's search vs plain code search; compass, companions, "deep" tasks | Complete | → each README |
| exp10 | Triangulation: simargl + gamayun + svitovyd, equal-weight RRF, 200 modn sonar tasks | Complete | the fusion beat the best single tool on every measure (hit@10 0.62 → 0.755) |
| exp11 | Triangulation on the 20 latest tasks, today's code | Complete | → its README |
| exp12 | Embedding models for past-task search (CPU) | Interim | granite-embedding:30m best measured (MAP 0.200; bge-small 0.171); bge-large not scored |
| exp13 | Six indexes per project (svitovyd, simargl units/files, gamayun db/files), latest tasks held out | sqlfluff done | embeddings win (MAP ≈ 0.40); fusion not better than the best single; code indexes flattered (the change is already in today's code) |
| exp13.1 | Future tasks: open PRs that close an issue (the issue is the query) | sqlfluff done | the code index drops (hit@1 0.77 → 0.38); fusion beats the best single (MAP 0.41 vs 0.35) |
| exp13.2 | Where a task creates files: rules "anchor file → (folder, extension)" | sqlfluff, 10 tasks | the right folder first for 4–5 of 10 against 1 of 10 for the prior |
| exp13.3 | Identifiers a task touches, new links it makes | sqlfluff | identifiers of the most similar past tasks: hit@10 0.71; new links not predictable yet |
| exp14 | What the health square measures (concepts/HEALTH_EVALUATION.md) | sqlfluff, sonar, flink | a unit's zone follows its place, not its state; new files are 0.20–0.35 of the pairs; docs/tasks/code vocabularies |
| exp14.1 | Business terms from glossary: docs glossary with big vs small models; the docs vocabulary in the code and the tasks | Running | — |

### Locked experimental findings (do not contradict without new evidence)

1. **Aggregation strategy has zero effect** — avg/sum/median/weighted/cluster produce identical MAP. Always use `avg`.
2. **Split strategy matters critically** — `recent` split inflates module MAP (~0.80) due to temporal proximity; `modn` split is the honest baseline (~0.45).
3. **Source field**: `desc` (description) outperforms `title` and `comments`. Optimal: `title+description`.
4. **Window**: w1000 (1000 days) best for modn split; w100 best for recent split.
5. **CodeBERT worst** — domain mismatch: tasks are natural language, not code. General models beat code-specific models.
6. **bge-m3 OOM on 6GB VRAM** at batch_size=32; fix: batch_size=4.
7. **File-level retrieval is much harder** than module-level (MAP <0.03 vs ~0.45).

---

## Exp3 Codebase (main research code)

- Language: Python
- Key files: `config.py`, `etl_pipeline.py`, `run_experiments.py`, `vector_backends.py`, `utils.py`
- Backends: PostgreSQL + pgvector OR Qdrant (switchable)
- Results schema: `model, split_strategy, experiment_id, source, target, window, MAP, MRR, P@1, R@1, ...`

---

## Product Architecture (SIMARGL Platform)

The research is being refactored into four independent deployable components:

```
codeXtract  → git/Jira/GitHub → SQLite DB
codeXplorer → tokens + embeddings → pgvector
codeXpert   → RAG tools, Gradio UI, MCP server, structural analysis
codeXport   → AST graph + ontology: business terms ↔ code identifiers
```

**codeXtract** (current location: `codeXplorer/` folder) — data gathering pipeline:
- `git_connector.py` → `task_extractor.py` → `jira_api_connector` → `db_manager.py`

**codeXport** — phenomenological symbol grounding:
- Key idea: "Code is not a description of business — it is its digital emanation."
- Pipeline: camelCase tokenize → Word2Vec → keyword_index (pgvector) → entity_map (business↔code)
- Philosophical grounding: Husserl (intentionality), Heidegger (Zuhandenheit), Speech Act Theory

---

## SIMARGL Core Metrics

Two antagonistic axes for evaluating code change recommendations:

**Novelty@K** = fraction of top-K recommendations that are NEW (not in existing codebase relations)  
**Structurality@K** = fraction of top-K where source and target are in the SAME module

```
                   Novelty
             LOW          HIGH
         +-------------+-------------+
   HIGH  | MAINTENANCE | EVOLUTION   |  ← target zone
Struct.  | safe, no    | new + local |
         | innovation  | (ideal)     |
         +-------------+-------------+
   LOW   | STAGNATION  | DISRUPTION  |
         | no value    | new + cross |
         +-------------+-------------+
```

**SES** (Structural Evolution Score) = sqrt(Novelty × Structurality)  
**HES** (Harmonic Evolution Score) = harmonic mean of Novelty and Structurality

Goal: maximize EVOLUTION zone (high Novelty + high Structurality).

---

## Tool Ecosystem (mythological naming)

| Tool | Slavic god | Role | Status |
|---|---|---|---|
| simargl | Симаргл — guardian | Semantic Index: Map Artifacts, Retrieve from Git Log | Research complete |
| 1bcoder | — | CLI coding assistant, local LLM interface | Production (PyPI v0.1.x) |
| vyrii | Вирій | Local AI web UI (Flask default, port 5000) | Production |
| yasna | Ясна — goddess of fate | Session memory — indexes conversations from all AI agents | Production (PyPI v0.1.8) |
| svitovyd | Світовид — four-faced god | Project map — structural map of definitions/cross-file deps, MCP server, Flask web UI with a layered interactive map | Production |
| radogast | Радогост | Context drift monitor — measures how far an agent's conversation has drifted from the task | Active |
| syryn | Сирин — sorrow/night bird | Bluetooth identity beacon for headless devices — IP shown as the BT name `P:<ip>` (no app needed) | Production |
| alkonost | Алконост — joy/day bird, paired with Сирин | Task-board viewer/editor for `tasks.md`/`planMD`/`spec` (1bcoder's deepagent_* output) | Production |
| gamayun | Гамаюн — prophetic bird, messenger of the gods | Statistical ontology without an LLM — triples of terms, identifiers, files, modules from task history or a folder; search, compass, companions | Active |
| mokosh | Мокош — goddess of the harvest and of spinning | Dataset gathering — git commits + tracker tasks (Jira, GitHub, GitLab, YouTrack) into one SQLite db (`TASK`, `COMMIT`) that simargl and gamayun read | Active |

### 1bcoder (`C:\Projects\1bcoder\` — independent clone, NOT `C:\Projects\simrgl\1bcoder\`)
- PyPI: `pip install 1bcoder` — but the actually-installed package (`site-packages\_bcoder_data\`) is a separate, frozen snapshot from whatever version was last published; it does NOT auto-pick-up edits made in either git copy. To test local changes with the real `1bcoder` command, `pip install -e "C:\Projects\1bcoder"` (editable install) or manually copy changed files into `site-packages`.
- Key commands: `/agent`, `/flow`, `/proc`, `/parallel`, `/map`, `/translate`, `/role`, `/ctx compact`, `/script`, `/prompt`
- Packaging: `_bcoder_data/` = wheel defaults; `~/.1bcoder/` = user global; `.1bcoder/` = project-local
- Always edit `_bcoder_data/` for default changes, never `~/.1bcoder/`
- `/flow glossary` (2026-10-11): `--terms-only` (one LLM call per chunk, mentions in `SEEN:`), incremental `_index.json` (unchanged files skipped, a changed one forgotten and re-read, a run stopped halfway resumes without doubles), `--jev` / `--jev-model` / `--jev-host` (KIND: domain/technology/noise, definitions and facts checked against the chunk), a two-thread pipeline when the decision model is on another host. Exports for reading without alkonost or for git: `html` (glossary.html + a page per term, sources as links) and `_contents.md` (every term as a link; written by index/relink/html and `/flow glossary contents`; glossary.md stays the plain name list every reader parses). A folder source is walked without dot folders (`.1bcoder` — its own output —, `.git`, `.simargl`) and `node_modules`/`__pycache__`/`venv`. Design and competitor review: `concepts/GLOSSARY.md`. vyrii keeps its own older copy on purpose (see tool roles).
- `/note` (2026-10-16): titled notes of the whole project in `.1bcoder/note/<term>.md` (KEYWORDS, DEFINITION, FACTS; pure helpers in `note_store.py`) — `def`/`fact <term> <text|$>` (a bare `$` = the last reply, tagged `[llm: <model> · date]`, else `[user · date]`; `/note` is in `_NO_EXPAND_CMDS` so a reply's `~`/`@` stay text), `keyword`, `list`, `term [ctx]`, `search <words> [ctx]` (every definition, only the matching facts; no match → how many notes exist), `load N,M`. Deliberately not task-bound and not the glossary flow. Agents get `note list/term/search`; in an agent the text goes to the agent's own context, never `self.messages`.
- `/command` (2026-10-17): named scripts with help in `.1bcoder/command/` (project) over `~/.1bcoder/command/` (defaults in `_bcoder_data/command/`: summary, action-items, commit-msg) — `list`, `help <name|N>`, `<name> [args]` (`{{args}}`, `{{1}}`…); everything it switches on (/param, /proc on|before|gate, /role, /think, /model, /var, the open script) is restored when it ends, also on an error; no #POS written into it; a command inside a command is refused, scripts are allowed. Help convention for scripts and commands = the leading `#` lines (`# name — what it does`, When, Model, Usage), shown by `/script help N` and in `list`. Multi-line input everywhere: ` \` at a line end goes on (line break kept), ` \t` at the prompt moves the typed text into the `/text` editor (prompt_toolkit `default=`), `<<<` … `>>> [-> var]` is a literal block in scripts (only `{{name}}` expands; an unclosed block stops the script before it runs). `/text`'s handler is `_cmd_text` (it had been shadowed by `/note`'s `_cmd_note` for a day).
- Three distinct copies exist on this machine — know which one you're touching: (1) `C:\Projects\1bcoder\` — independent clone, live dev, edit here; (2) `C:\Projects\simrgl\1bcoder\` — git submodule inside this repo, pinned snapshot, do not edit; (3) `site-packages\1bcoder-*` — installed PyPI package, only updates via `pip install`, not by editing files in place.

### vyrii (`C:\Projects\vyrii\` — independent clone, NOT `C:\Projects\simrgl\vyrii\`)
- Default: Flask API (`flask_api.py`, port 5000) + static HTML/JS UI in `vyrii/ui/`
- Pure Python dependencies: requests, flask, flask-cors, apscheduler, waitress
- Optional extras: `vyrii[gradio]`, `vyrii[api]`, `vyrii[full]`
- OpenAI-compatible endpoints + `/vyrii/*` endpoints, basic auth
- Process tab (2026-10-25): Files → a folder → **Run** starts only the commands of `process_tools.json` (packaged: alkonost, simargl/gamayun/svitovyd/mokosh `ui`, syryn on Linux; `~/.config/vyrii/process_tools.json` replaces it whole for another stack — deliberately outside `~/.vyrii`, which Files → Upload writes) on a folder inside `~/.vyrii` only — no free command line, no folders outside. Registry `~/.vyrii/process/processes.json` (tool, folder, port, pid, log); same tool + folder runs once; a busy default port (a copy started elsewhere) → the person picks the port, a free one outside every listed default is suggested; Stop kills only what vyrii started. Tools bind like vyrii (`--bind`). radogast is left out on purpose (needs many parameters)
- Port and restart (2026-10-25): port = `-p` → `config.json "port"` (Settings → Port) → 5000. Restart (`/vyrii/system/restart-cmd`, and `/vyrii/system/restart`, which no longer runs `vyrii_auto_api.bat`) re-runs `sys.argv`, dropping `-p/--port` when a port is saved; it never takes a command from the browser (it used to — any program for anyone who reached vyrii). Files (Upload / New folder / Delete) never write vyrii's own state in the root of `~/.vyrii`: `config.json`, `scheduler.json`, `process_tools.json`, `history.db*`, `process/`, `scheduler_logs/`, nor the root itself. The Scheduler tab runs any shell command by design (`scheduler.py`, `shell=True`), so its API answers 403 `need_auth` unless vyrii asks for a password (jobs already scheduled keep running)
- Password and network warning (2026-10-25): password = `--auth` or Settings → Authentication → Require password (`config.json "auth"`, restart; a saved `false` drops `--auth` on restart); admin/admin stays the default login. `/vyrii/system/exposure` lists where vyrii is reachable (`netinfo.py`: Windows `Get-NetAdapter -Physical` via PowerShell, else psutil / `ip -o -4 addr`; kind ethernet / wifi / other), looked up once per start; without a password and with any non-Ethernet address the page shows a big modal "THIS COMPUTER IS VISIBLE ON THE NETWORK — Ethernet a:5000, Wi-Fi b:5000 — turn on the password in Settings" (never names the default login), once per vyrii start per browser. vyrii does not guess which network is safe: any address the person has not marked warns; the modal's "This is a private network" saves the ticked addresses in `config.json "trusted_ips"` (only this machine's current addresses, via `POST /vyrii/system/trust`; Settings lists them, each with its own Remove), and a new address — e.g. the mini-PC plugged into a router — warns again
- Files → Unzip / Git (2026-10-25, `vyrii/fileops.py`, `/vyrii/files/unzip`, `/vyrii/files/git`): upload and New folder go into the folder selected in the tree; a `.zip` unpacks next to itself (one top folder → that folder, else a folder named after the zip; nothing overwritten; no paths outside, size/count cap); a folder has Git clone (URL + optional name), and a repository Git pull (`--ff-only`) and Checkout (branch / tag / commit; one the clone lacks is fetched from origin → detached). Only values reach the server; git runs without a shell, hooks and fsmonitor off, only https/http/ssh/git protocols (no `ext::`, no local paths), no password prompt. `.git` is written by git alone: upload, mkdir and unzip refuse any `.git` part — whoever writes `.git/hooks` or `.git/config` decides what the next pull runs. The tree shows `.git/` without listing its objects
- Depends on `simargl` for RAG search (`simargl.searcher`) — make sure that's resolved from `C:\Projects\simargl\` (independent clone), not the pinned submodule, or newer searcher APIs (e.g. `get_chunk_text`) will be missing.

### yasna (`C:\Projects\yasna\` — independent clone, NOT `C:\Projects\simrgl\yasna\`)
- PyPI: `pip install yasna`
- Adapters in `yasna/adapters/` index sessions per AI coding agent: Claude Code, opencode, continue.dev, aider, nanocoder, 1bcoder, gemini, codex, copilot, kiro (Kiro CLI, `~/.kiro/sessions/cli/*.json`+`*.jsonl`), pi (Pi CLI, `~/.pi/agent/sessions/<sanitized-cwd>/*.jsonl` — project name is read from the session's own `cwd` field, not decoded from the sanitized directory name), antigravity (Antigravity CLI `agy`, `~/.gemini/antigravity-cli/brain/<id>/.system_generated/logs/transcript_full.jsonl` — the short `transcript.jsonl` truncates long fields; text only inside `<USER_REQUEST>`, model text from `PLANNER_RESPONSE.content`, `search_web` queries kept; workspace from `history.jsonl` `conversationId`, else the most frequent `run_command` `Cwd`; resume `agy --conversation <id>`)
- Adding an agent = one adapter file exposing `AGENT_NAME` + `sessions()`, registered in `yasna/adapters/__init__.py`'s `ALL` dict — CLI/indexer/searcher need no changes
- Kiro CLI itself stores sessions only globally under `~/.kiro/sessions/cli/`, never project-local — no `.kiro/` folder is left behind inside a project directory after running it
- `codex.py` deliberately ignores the `$CODEX_HOME` env var and always reads `~/.codex/sessions/**/rollout-*.jsonl` — on this machine a multi-agent orchestrator ("orca"/Claude Agent Teams) sets `CODEX_HOME` globally to its own sandboxed runtime dir with no real session history, which silently broke discovery when the adapter trusted it. Rollout format is JSONL `{"type":"response_item","payload":{"type":"message","role":...,"content":[...]}}` (current Codex CLI, v0.154+) — the older `{"type":"user_message"}`/top-level-`role` shape the adapter previously assumed is stale and no longer produced by the CLI.
- `codex.py` and `copilot.py` were originally written from documentation only, with no real local files to verify against on this machine and zero test coverage — both have since been rewritten and verified against real session data (`~/.codex/sessions/`, `~/.copilot/session-state/`). `copilot.py` real event format: JSONL with payload nested under `"data"` (`session.start`→`data.context.cwd`, `user.message`/`assistant.message`→`data.content` as a plain string, often `""` for tool-only assistant turns) — the old adapter checked top-level `type`/`role`/`content` fields that don't exist in this schema, so it silently found 0 sessions.

### alkonost (`C:\Projects\alkonost\` — independent clone, NOT `C:\Projects\simrgl\alkonost\`)
- PyPI: `pip install alkonost`; CLI entry `alkonost --port 8090 --root <dir>`
- Flask (`flask_api.py`, `waitress.serve`, default port 8090) + static HTML/JS UI in `alkonost/ui/`, no build step
- Reads/writes `.1bcoder/planMD/`, `.1bcoder/spec/`, `.1bcoder/tasks/` produced by 1bcoder's `deepagent_md`/`deepagent_spec`/`deepagent_task` — no import dependency on 1bcoder, independently re-implements the same file formats
- The alkonost process itself makes no LLM call, but it is the cockpit of 1bcoder flows: `runner.py` launches the real 1bcoder (`--scriptapply`) — deepagent_md/spec/task, flow help — and streams the output back. Glossary tab (2026-10-12): launches `/flow glossary index` (build / update / resume, `--terms-only`, `--jev*`), browses `.1bcoder/glossary/<name>/` articles (filters by name, KIND, SEEN, multi-word; article panel), builds a simargl index inside the glossary folder via `/flow rag index` and searches it with `simargl search --mode file --format json` (articles only) — `alkonost/glossary.py`, `/api/glossary/*`; every other `/flow glossary` command has a button (`/api/glossary/run`: relink [--synonyms], html, contents, find, extract, show, filter). The article panel renders its own links (LINK:/SYNONYM: open the article, `[file: path:line]` and SEEN open the source lines in the panel) — it never reuses the html/`_contents.md` exports, which are for reading without alkonost or for git
- Launches (2026-10-12): `runner.render_script` substitutes the values into a per-launch copy of the script (`~/.alkonost/logs/<id>.script.txt`), no `--param` (1bcoder's own `--param` quoting changed and doubled the quotes), and starts it with `/ctx` + `/param k v` lines from Settings (`settings.resolve_session`) — with `--host/--model` given, `--scriptapply` reads no config.yml, so ctx/timeout/`think=false` never reached a run before. Settings has an "Other params" row (`key=value, …`).
- Tool roles (2026-10-11): vyrii stays the simple out-of-the-box panel for a broad user; features that combine 3–4 tools go to simargl (triangulation) or alkonost; `~/.1bcoder` is the end user's single source of truth, updated by deploying `_bcoder_data` (`deploy_bcoder_data.bat`)
- Zero-config: recursively discovers every `.1bcoder/` project under wherever it's started, two-level project→plan picker; **＋** (or the banner shown when none is found) makes a folder under the root a project by creating an empty `.1bcoder/` in it (`/api/projects/folders`, `/api/projects/create`)
- 58 pytest + 48 `node --test` (client-side JS, no build step) — see `concepts/ALKONOST.md` for the full design rationale and `README.md` in the tool itself for user-facing docs

### gamayun (`C:\Projects\gamayun\`)
- `gamayun index --db <db> | --files <folder> --project P` — ontology (`triples.txt`) without an LLM; method in `concepts/GAMAYUN.md`, experiments exp7, exp8, exp8.1, exp9.x
- Projects live next to what they describe: `<folder>/.gamayun/<project>/`; `~/.gamayun/projects.json` only registers where. The first projects (`concepts`, `flink`, `sonar`, `sonar-code`) still live in `~/.gamayun/<project>/` ("home") and keep working. `$GAMAYUN_HOME` is for the tests only.
- Commands: `search`, `compass` (rare literal anchor via `git grep`, else the ranking; confidence high/medium/low), `companions` (co_changes_with of the files named in a text), `show`, `eval`, `ui` (Flask, port 8095; one project at a time)
- Experiments use the held-out ontology `exp7/out/sonar` through `exp9/tools.py`, never the `sonar` project (which contains every task — answers leak)

### simargl (`C:\Projects\simargl\`)
- Core install has no torch: `pip install simargl` = numpy, pyyaml, tqdm, mokosh, Flask, waitress. Extras: `[torch]` (sentence-transformers: bge-small/bge-large/jina-code, fast bulk indexing), `[mcp]` (simargl serve / simargl-mcp — moved out of the core so the phone install works), `[gradio]` (old UI, `simargl ui --gradio`), `[http]`, `[all]`.
- `DEFAULT_MODEL = ollama://granite-embedding:30m` for NEW indexes (exp12: best MAP of the models measured on CPU, 0.200; bge-large not scored yet); an index records `model_key` in meta.json and search always uses it; metas without `model_key` are read as `LEGACY_MODEL = bge-small` (never today's default). Ollama embeds `OLLAMA_BATCH` = 16 texts per /api/embed request (was 1 per request).
- Web UI: `simargl ui` (Flask, port 8099, 127.0.0.1 unless `--host 0.0.0.0`) = `simargl/ui/server.py` + `simargl/ui/webui/` (8 themes, 6 languages); long actions (index, vacuum, ollama pull) run the CLI as child processes with a streamed log. **simargl gathers nothing** (since 2026-10-11): no Setup page, no ingest, no `simargl init/ingest/export/plaintext`, no `simargl.ingest` / `simargl.exporter` shims — it only reads the database mokosh writes (`simargl.schema` stays: it is mokosh.schema). The older Gradio UI lost its Init tab and re-ingest too.
- **Triangulation** page (`/api/triangulate`): simargl + gamayun + svitovyd on one task, equal-weight RRF (k=60), agreement strip; moved-path resolution through the svitovyd map, `gone` = history path absent today, `share` = companion (file changed by ≥ 1 % of past tasks — marked, never dropped). simargl depends on gamayun and svitovyd for this. Evidence: exp10 (200 modn tasks), exp11 (20 latest tasks, today's code).
- Machine-wide registries: `~/.simargl/stores.json`, `~/.svitovyd/maps.json`, `~/.gamayun/projects.json` — the CLI index commands register, the UIs list them from any folder.

### svitovyd (`C:\Projects\svitovyd\`)
- `svitovyd index | find | trace | deps | sym | keywords | idiff | ui | serve` — map file `.svitovyd/map.txt` (files, `defines`, `links → target (kind:name)`, kinds import/call/ref/expr; expr is a bare word match and dominates large maps — the web UI leaves it out by default)
- Web UI: `svitovyd ui` = Flask + waitress (core dependencies, so `pip install svitovyd` is enough — phones too), port 8097, `127.0.0.1` unless `--host 0.0.0.0`; `svitovyd/graph.py` answers the UI with data (layers, edge detail, file card, trace/deps trees, find, keywords, sym, idiff), `svitovyd/server.py` the API, `svitovyd/webui/` the pages (8 themes, 6 languages, same look as gamayun). The server reads and sends only files that are svitovyd maps.
- The older Gradio UI (`ui.py`) stays: `svitovyd ui --gradio`, needs `pip install svitovyd[gradio]`.
- Task text → files (`graph.files_for_text`, used by simargl's Triangulation): one mode, keywords first. A word that is a defined identifier means exactly it; any other word expands to the defined identifiers containing its ≥ 5-letter subwords (as `keywords extract -f`). Weight `log(N/files)`; a built-in list of common English words is skipped. **svitovyd reads only its map — never a task database** (a version that weighed words by past tasks was tried and dropped). Evidence: simrgl exp11 (`single_mode.py`).

### mokosh (`C:\Projects\mokosh\`)
- `mokosh init | ingest | plaintext | export | status | fixdiff | ui` — the only writer of the task/commit SQLite db and the only place the dataset is set up (moved out of simargl entirely; 1bcoder `/rag init|ingest` and the `history` flows of 1bcoder/vyrii call mokosh)
- Web UI: `mokosh ui` (Flask + waitress, port 8093) = `mokosh/ui/server.py` + `mokosh/ui/webui/` (8 themes, 6 languages): Setup (the init form, commit masks tried on the repository), Ingest (streamed log), Status (db stats, project.yaml masked, ingest_state). Projects: folders with `.mokosh/project.yaml` (or `.simargl/project.yaml`), registry `~/.mokosh/projects.json` (`$MOKOSH_HOME` for tests)
- `COMMIT.CHANGE_TYPE` (A/M/D/R) and `DIFF` parent → commit (fixed 2026-10-10; older dbs are turned once by `ingest` / `fixdiff`, marked in `MOKOSH_META`); commit masks include `regex-key` (the project key(s) of `jira_project`, e.g. `ABC,ABD`)
- Settings in `.mokosh/project.yaml`; an existing `.simargl/project.yaml` is used as it is (no re-ingest of sonar/flink)
- `mokosh.schema` is the one table-name contract (TASK/TASKS, COMMIT/COMMITS resolved per file); simargl and gamayun both depend on mokosh. Publish mokosh to PyPI before the simargl / gamayun releases that require it.

### radogast (`C:\Projects\radogast\`)
- Context drift monitor: a static analyzer of AI agent sessions — how far a conversation has drifted from its task. `radogast init`, `radogast watch`; reads sessions of Claude Code, 1bcoder, aider, Gemini CLI, opencode, … (found through yasna when it is installed). Core deps `click`, `numpy`, `pyyaml`; `sentence-transformers` optional (without it: marker-word analysis only).

### syryn (`C:\Projects\syryn\`)
- Bluetooth identity beacon for headless Linux devices: while it runs the device's Bluetooth name is `P:<ip>`, so its IP is visible from any phone with no app. BlueZ through D-Bus (no PyBluez); `syryn install [--dry-run]` (system service under `/opt/syryn`), `syryn status`. Concept: `concepts/SYRYN.md`.

### jev (module of 1bcoder: `C:\Projects\1bcoder\jev.py`)
- Adapter for decision models (Ollama `/v1/systemone`, e.g. `tev1:0.8b`, `tev1:4b`): they never write text — they answer closed questions (`noul` yes/no, `choice`, `score`) with probabilities, and the thresholds live in Python ("LLM generates → jev perceives → Python decides"). Rubrics: `~/.1bcoder/rubrics/`; settings: `~/.1bcoder/jev.json` (model, host, threshold, timeout). Used by `/jev`, `/loop`, `/flow glossary --jev`. Concept: `concepts/JEV.md`.
- One request = ONE prompt holding the text and ALL its questions; tev1 reads at most 2 050 tokens and never truncates (HTTP 400 "… tokens …") — a choice question with three levels is ~110 tokens, so ask in groups (glossary: 8). On a CPU host a group takes ~10–60 s: the interactive 30 s timeout is too short for batch work.
- Quality (one test sentence, 2026-10-11): `tev1:0.8b` called every word *technology* and accepted an invented fact; `tev1:4b` told noise apart and rejected it — exp14.1 measures this properly on the sqlfluff docs.

---

## Datasets

All SQLite with a task table (ID, NAME, TITLE, DESCRIPTION, COMMENTS) and a commit table, one row per (commit, file): SHA, CMT_DATE, PATH, TASK_NAME, … Table names vary by age — `sonar.db` has `TASKS` + `COMMITS`; new databases from `mokosh ingest` have `TASK` + `COMMIT`; `mokosh.schema` resolves either. (Older notes call the commit table `RAWDATA`.)

| Dataset | Project | Tasks | Files | Modules |
|---|---|---|---|---|
| sonar.db | SonarQube | ~9,799 | 12,532 | 27 |
| kafka.db | Apache Kafka | available | — | — |
| spark.db | Apache Spark | available | — | — |

Plan: validate across 10 datasets (different languages, team sizes, project cultures) to establish "conditions of applicability", not just "it works".

---

## Hardware Constraints

- GPU: 6GB VRAM
- bge-m3 (568M, 2.2GB): use batch_size=4
- Qwen3-4B (7.7GB): may not fit; try batch_size=1 or skip
- Qwen3-8B via Ollama Q4 quantization (~4–5GB): should fit
- Local LLM stack: Ollama + qwen2.5-coder

---

## Concept Documents (`concepts/`)

Research ideas, not yet implemented unless stated:

| File | Topic |
|---|---|
| `FINAL_PRODUCT.md` | Full product vision and architecture |
| `PROJECT_OVERVIEW.md` | Executive summary |
| `AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md` | External logical supervision for small-model agents |
| `COMPOSITIONAL_CODE_EMBEDDINGS.md` | Vector arithmetic on function semantics |
| `CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md` | Data flow tracking across architectural layers |
| `MAGURA.md` | Magura (new goddess, concept only): field-level data lineage through every layer (ingress → DTO → mapper → entity → DB → query/logic → output) without a build — an entity as an "image" (glossary term + code identifiers + columns + JSON keys + report labels), mentions classified per line (define / copy / transform / consume / output / unknown), swim-lane diagram with evidence per arrow (svitovyd, gamayun code+history, simargl, mokosh, glossary, jev); finds write-only fields (KYC weight), ghost fields, renames, PII map. Structural half of CLTE: its lines are CLTE's missing paths and gold standard, CLTE gives bridges over renames and task → line retrieval |
| `KEYWORD_ENTITY_MAPPING.md` | Business terms ↔ code identifier mapping |
| `DUAL_MCP_SERVER_ARCHITECTURE.md` | Historical vs recent context servers |
| `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` | Husserl/Heidegger applied to code navigation |
| `DEEPAGENT_AND_DEEPAGENT_MD.md` | Recursive markdown generation with parallel workers |
| `DEEPAGENT_SPEC.md` | deepagent_md tree → atomic WHO/WHAT/spec units → flat tasks.md rollup |
| `DEEPAGENT_ARCHITECT.md` | Spec → interface/abstract/approach/class design, research-grounded |
| `DEEPAGENT_TREE.md` | deepagent_tree/deepagent_tree_md — structured decomposition vs. prose, RRTrack empirical case study |
| `ANTHILL_DISTRIBUTED_COGNITIVE_OS.md` | Distributed agent OS concept |
| `CTXTIMER.md` | Empirical context-limit measurement for local LLMs (`/flow ctxtimer` in 1bcoder) |
| `IGPU_OFFLOAD_THRESHOLD.md` | When Vulkan/iGPU offload helps vs hurts local LLM inference — empirical hardware threshold |
| `LIMITED_CONTEXT.md` | Sliding-window context (`/ctx window`) — RS/BM25/DP/TextRank algorithms for retaining gist beyond the ctxtimer ceiling |
| `HEALTH_CONCEPT.md` / `HEALTH_EVALUATION.md` | What the Novelty × Structurality square (gamayun health) really measures: character of change, not quality; unit verdicts are an artifact of a unit's place; growth (new files) apart; duplication as hidden coupling; architecture judged against the task flow and the task vocabulary (Lehman, Parnas, Evans…). Numbers: `exp14/health_eval.py` on sqlfluff, sonar, flink |
| `GAMAYUN.md` | gamayun: statistical ontology of task history and code without an LLM (G², lift, HHI, naive Bayes search) |
| `ONTOLOGY.md` | Triple knowledge base with contradiction checking (1bcoder `/flow ontology`) |
| `GLOSSARY.md` | `/flow glossary` — LLM-wiki adapted to 1B models; `--terms-only`, incremental `_index.json`, `--jev`, two-host pipeline; competitor review (llmwiki, GraphRAG, DeepWiki, MiniCheck) and what is its own |
| `NOTE.md` | `/note`: good LLM answers of lasting value, filed by the person under a concept (term = its name; the entry = a subject heading) with definitions, keywords and facts, each tagged with its origin (`[llm: model]`, `[user]`, `[output]`); a subject index of the project, not of a task, and not the glossary flow; word search, agents read it in their own context |
| `JEV.md` | jev: perception as the third role — decision models answer closed questions with probabilities, Python decides |
| `LOOP.md` | `/loop`: a stream from anything, an action on every item |
| `HISTORY_FLOW.md` | Conceptual blame for 1bcoder (`/flow history`) |
| `ENCYCLOPEDIA.md` | Unbounded article, as opposed to the compact glossary |
| `CONVERSATIONS.md` | Multi-agent deliberation and ensembling for small-model robustness |
| `INCIDENCE.md` | The common frame of the pantheon: one incidence table objects × attributes (cell provenance: syntax, history statistics, embedding, jev, LLM, human) and three operations — derivation = search, implication = symmetry (hand-written Noether table ≡ gamayun's learned co-change rules), closure = completeness; tools as ants, index maps of aggregation levels instead of passports; structuralism (anthill, OKG) → post-structuralism (trace, archive, rhizome); hypothesis of multiple readings H1–H5 against exp10/exp13/exp13.1 |
| `SWARM.md` | A swarm of small agents on different devices (phones, GPU, indexers): 12 dimensions; recipes (`code-trace`, `fact-check`, `review`, `bridge`) set once instead of per-task setup — fleet = where, recipe = how, task = what; A2A as a call with a contract, Python verifies |
| `ALKONOST_ADVANCE.md` | alkonost as a cockpit for the sequential tool chain, not a task board |
| `DEEPAGENT_SEARCH.md` / `DEEPAGENT_CODE.md` | Iterative search over an unbounded space; test-driven recursive code decomposition for small models |
| `KEYWORD_INDEXING.md` | Keyword indexing: semantic coordinates for code understanding |
| `SYRYN.md` | syryn: Bluetooth identity beacon |

The folder holds ~60 documents; this table lists the main ones. `INDEX.md` indexes the metrics and approaches.

---

## External Supervision Concept (partially implemented in 1bcoder)

**Thesis**: Small models (0.5–2B) cannot maintain multi-step task threads; thread management moves to a deterministic Python supervisor. "Model generates moves — automaton tracks the field."

**6 aspects**: (1) supervision via existing 1bcoder gates/before/tempctx; (2) milestone = structural condition (ladder: presence → witnessing → co-occurrence → node); (3) plan = belief field (TMS/ATMS); (4) descriptive/evaluative distinction (Pospelov); (5) text = full knowledge; (6) synonym island detector + HITL merge.

**MVP implemented** (2026-06-13):
- `chat.py`: dumps tempctx → `agent_ctx_{pid}.json`
- `_bcoder_data/proc/ladder.py`: 4-rung ladder gate, auto-extracts terms
- `_bcoder_data/agents/terminator.txt`: agent config with ladder gate

**Next step**: measure context recall against ground truth from sonar.db.

---

## Working Conventions

- **Do not suggest open-sourcing code** until publications are complete (Авторське Свідоцтво → conference → open source).
- **Before editing**: verify the path exists before editing. Every tool is edited in its own folder `C:\Projects\<tool>\` (independent clone, or gamayun/mokosh which never were submodules); the submodule copies inside `simrgl` (1bcoder, vyrii, simargl, radogast, yasna, alkonost, svitovyd) are pinned snapshots, never edit them. Check the tool table above.
- **Never commit**: AI agents do not run `git commit` (nor push, amend, or tag) in this repo or any tool repo — the author commits. Staging, diffs and commit message drafts are fine.
- **Before destructive git ops**: explicit confirmation required, not implied by prior discussion.
- **Show findings first**: report what existing code already handles before proposing changes.
- **No legacy code removal** without explicit permission — every line was added for a reason.
- **Hardware-aware suggestions**: keep 6GB VRAM limit in mind for any model recommendations.
- **Academic framing matters**: this is dissertation research; contributions need to be novel and citable.
