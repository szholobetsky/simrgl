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
- **AI backend**: Ollama HTTP API (`POST /api/chat`, NDJSON streaming)
- **Dependencies**: `requests>=2.28` only (standard library for everything else)
- **Entry point**: `1bcoder` command (via `pyproject.toml` + `py-modules` build)

The deliberate absence of a TUI framework is a design choice: 1bcoder is optimized for portability, low overhead, and compatibility with CI/headless use. It runs identically whether launched interactively, piped from a script, or driven by `--planapply`.

### 1.4 Project Files

| File | Role |
|---|---|
| `chat.py` | Entire application — REPL, all command handlers, MCP client, streaming logic (~2200 lines) |
| `map_index.py` | Standalone project scanner: AST-free regex-based identifier extractor → `map.txt` |
| `map_query.py` | Standalone map query tool: `find` (filter blocks) and `trace` (BFS call chain) modes |
| `map_query_help.txt` | Full CLI reference for `map_query.py` |
| `pyproject.toml` | Build metadata; py-modules = [chat, map_index, map_query]; entry point `1bcoder = "chat:main"` |
| `requirements.txt` | Single line: `requests>=2.28` |
| `run.bat` | Windows quick-launch |
| `MCP.md` | Catalog of ready-to-use MCP servers (filesystem, git, web, database, browser…) |
| `.1bcoder/plans/` | Per-project plan `.txt` files |
| `.1bcoder/map.txt` | Generated project map (definitions, cross-references) |
| `.1bcoder/map.prev.txt` | Previous map snapshot — enables structural diff without re-indexing |
| `.1bcoder/agent.txt` | Agent configuration: max_turns, auto_apply, tool whitelist |
| `.1bcoder/profiles.txt` | Saved `/parallel` worker profiles |

---

## 2. Feature Architecture

### 2.1 Context Injection — `/read`

```
/read <file> [start-end]
```

The user selects which lines to inject into the AI's context window. This is the manual implementation of what the Anthill's **Librarian agent** (GPU 1) would do automatically by querying Object Passports from the OKG. The human operator plays the Librarian role: knowing which lines are relevant and loading only those.

This command directly validates the Anthill's context efficiency claim. A 1B model given `/read file.py 10-25` performs measurably better than the same model given the full file, because it cannot be distracted by irrelevant surrounding code.

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

Both commands display a **unified diff before writing**, implementing the Anthill's Crooked Wall Principle: the human is the spirit level.

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

Plans support `{{key}}` parameter placeholders, substituted at apply time — enabling reusable templates analogous to the Anthill's parameterized task schemas.

**`/plan create ctx`** captures all work commands typed in the current session into a ready-to-run plan automatically. This closes the loop between ad-hoc exploration and reproducible procedure — the same transition the Anthill makes from interactive to pipeline mode.

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

**`/map diff` / `/map idiff`**: After any code change, re-indexing and diffing the map reveals which identifiers were added, removed, or moved — the manual equivalent of the Notary agent's post-commit OKG consistency check (M2: OCS). A warning is printed when identifiers disappear, mirroring the Auditor's structural drift alert.

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

The model receives a system prompt listing available tools (2-line summaries extracted from `HELP_TEXT` by `get_help_list()`), followed by the task. On each turn, the model either emits `ACTION: /command` (tool call) or plain text (completion). The tool call is executed via `_agent_exec()`, which captures output using a `_Tee` class (stdout is both shown to the user and captured as a string for context injection). The result is injected as `[tool result]` and the loop continues.

`auto_apply` bypasses confirmation prompts by replacing `self._confirm` with a lambda returning `True`. The agent loop runs in its own message thread (copies current context), so the main conversation is not polluted unless the user confirms at the end.

This is a stub implementation of the **Router + Architect + Coder pipeline** from Anthill Phase 1. It is single-model (no role separation), linear (no conditional branching), and relies on the same model to plan and execute. Its value is proving that the primitives are sufficient: the model can reason about which tool to use, call it correctly, and chain multiple steps to completion — given enough context and the right task scope.

The `agent.txt` tool whitelist implements a primitive **SKILL.md**: it constrains which verbs the agent can invoke, preventing it from attempting operations outside its tested capability. Removing tools from the list restricts a weaker model to a smaller instruction set.

### 2.7 Backup/Restore — `/bkup`

```
/bkup save <file>      — copy file to <file>.bkup (overwrites)
/bkup restore <file>   — replace file with <file>.bkup
```

Lightweight file snapshot before any risky edit. The agent is instructed to call `/bkup save` before modifying important files — making the human's recovery path explicit in the plan/agent workflow. This is the Anthill's pre-change state preservation, implemented without a version control abstraction layer.

### 2.8 Parallel Queries — `/parallel`

```
/parallel ["prompt1"] ["prompt2"] [profile <name>] [host|model|file ...]
```

Sends prompts to multiple models simultaneously using `concurrent.futures`. Each response is saved to its designated file. Current context (`/read` injections) is automatically included. Profiles are saved in `.1bcoder/profiles.txt` for reuse.

This directly implements **Multi-Instance Debate** from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §4.3. The user reviews multiple outputs and plays the Critic role — identifying where all models agree (structural signal) versus where they diverge (implementation choice). The profile system mirrors the Anthill's GPU farm worker configuration.

### 2.9 MCP Integration

A full `MCPClient` class implements JSON-RPC over stdio, compatible with any standard MCP server. The `/mcp` command suite provides:

```
/mcp connect <name> <command>   — start and connect a server
/mcp tools [name]               — list available tools
/mcp call <server/tool> [args]  — call a tool, inject result into context
/mcp disconnect <name>          — shut down server
```

This is the **nervous system** described in [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §8.1. The filesystem, git, web, database, and memory servers available in `MCP.md` are exactly the external knowledge sources the Anthill agents use to access structured context without reading raw files. When Phase 1's `get_passport()` MCP server is built, the existing `/mcp connect` command will connect to it without any code changes.

### 2.10 Output Management — `/save`

```
/save <file> [mode]
```

Modes: `overwrite`, `append-above` / `-aa`, `append-below` / `-ab`, `add-suffix`, `code` (strips ` ``` ` fences). Supports multiple files in one command — mapping code blocks to files by position.

The `code` mode implements the Anthill's principle of treating model outputs as structured artifacts rather than raw text: the markdown wrapper is discarded and only the code payload is persisted.

### 2.11 Context and Session Controls

| Command | Role |
|---|---|
| `/ctx <n>` | Set context window size in tokens (default 8192) |
| `/ctx cut` | Remove oldest messages until context fits within limit |
| `/ctx save <file>` | Dump full conversation to a file — session persistence |
| `/ctx load <file>` | Restore a saved conversation — cross-session memory |
| `/model <name> [-sc]` | Switch model at runtime; `-sc` keeps context (switch Coder agent) |
| `/host <url> [-sc]` | Switch Ollama host at runtime; `-sc` keeps context (route to different GPU) |
| `/clear` | Reset conversation context (new task isolation) |
| `Ctrl+C` | Interrupt streaming response (agent interrupt in Anthill terms) |
| `/help <cmd> [ctx]` | Show per-command help; `ctx` injects the help text into AI context |

`/ctx save` / `/ctx load` implement primitive **session memory** — the ability to resume a working context across launches. In Anthill terms, this is a manual L2 memory tier (conversation history) without the L3 (OKG) or L4 (vector RAG) tiers.

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
| `/map diff` / `/map idiff` | Observer Loop + Notary consistency check | **Primitive** — file diff, no OCS enforcement |
| `.1bcoder/map.txt` | OKG (flat-file seed) | **Primitive** — text, no RDF, no graph DB |
| `/agent` | Router + Architect + Coder loop (single model) | **Stub** — no role separation, linear only |
| `agent.txt` tool whitelist | SKILL.md (behavioral constraint) | **Primitive** — static config, no Researcher update |
| `FIX_SYSTEM` / `PATCH_SYSTEM` | SKILL.md (output format constraint) | **Primitive** — hardcoded, not dynamic |
| `/bkup save/restore` | Pre-change state preservation | **Implemented** — file-level snapshot |
| `/parallel` | Multi-Instance Debate (Section 4.3) | **Implemented** — concurrent model queries |
| `/parallel profile` | GPU farm worker configuration | **Implemented** — saved profiles in profiles.txt |
| MCP client + servers | MCP nervous system (Section 8.1) | **Implemented** — full JSON-RPC client |
| `/ctx save` / `/ctx load` | L2 memory tier (conversation history) | **Implemented** — file-serialized context |
| `.1bcoder/` directory | BCODER_DIR / project-scoped workspace | **Stub** — flat files, no graph |
| Human operator | Router + Architect + Notary + Auditor | **Human-in-the-loop** |

### 3.2 Position in the Anthill Roadmap

1bcoder sits at **Phase 0 (complete) → Phase 1 entry** of the Anthill implementation roadmap from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §11.

**Phase 0 (complete):**
- Constrained generation format (`LINE N: content`, SEARCH/REPLACE)
- Context injection via `/read` (selective line range)
- Post-change verification via diff preview (Crooked Wall)
- MCP substrate installed and operational
- Plan execution engine (linear, manual-authored, `--planapply`)
- Multi-instance parallel queries with profiles
- Session persistence (`/ctx save` / `/ctx load`)
- Flat-file project map with identifier extraction and BFS trace (OKG seed)
- Single-model agentic loop with tool whitelist (Router+Architect stub)
- Pre-change backup/restore

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

**Document Version**: 1.1
**Created**: 2026-03-05
**Updated**: 2026-03-07
**Project**: SIMARGL / Anthill Research Program
**Relation to**:
- [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) — parent architecture document
- [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) — experimental validation framework
- [FINAL_PRODUCT.md](FINAL_PRODUCT.md) — SIMARGL product architecture (codeXpert = 1bcoder's planned MCP evolution)
- [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md) — agent architecture precursor
- [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) — philosophical grounding (Observer Loop, symbol grounding)

**Changelog v1.1**:
- Corrected technical stack: plain terminal REPL, no Textual TUI, `requests` only
- Removed `headless.py` (does not exist); added `map_index.py`, `map_query.py`
- Added Section 2.5 `/map` system — OKG seed, flat-file project knowledge
- Added Section 2.6 `/agent` — Router+Architect stub, SKILL.md analog
- Added Section 2.7 `/bkup`; expanded Section 2.11 with `/ctx save/load`, `/help ctx`, `/model -sc`
- Updated mapping table (Section 3.1) with all new commands
- Updated Phase 0 completion list and Phase 1 gap definition (Section 3.2)
- Updated concept correspondences (Section 4.2) with map/Entity Map/Observer Loop entries
- Added `/map`-based Exp A1/A2 testbed to Section 4.3
- Updated Gap Analysis (Section 5) — Object Passports now "Primitive" not "None"; added role separation row
