# 1bcoder — The Human-in-the-Loop Anthill Prototype

## Abstract

**1bcoder** is a terminal-based AI coding tool built on Ollama, designed specifically for 1B-parameter language models. It is the first working implementation within the Anthill research lineage, functioning as a **human-in-the-loop prototype** of the Anthill Distributed Cognitive OS. Rather than automating the full agent pipeline, 1bcoder externalizes the cognitive roles of Router, Architect, Notary, and Auditor onto the human operator, while providing a structured set of tool primitives that directly instantiate the Anthill's core architectural concepts: constrained generation, curated context injection, plan-based task decomposition, adversarial verification via diff preview, and MCP-mediated external tool access. This document describes the project in full and maps each of its components to their theoretical counterparts in the Anthill architecture.

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
| **MCP as nervous system** | External tool servers provide structured context without raw file access |

### 1.3 Technical Stack

- **Language**: Python 3.10+
- **UI framework**: [Textual](https://textual.textualize.io/) — terminal TUI with RichLog, Input, and sidebar
- **AI backend**: Ollama HTTP API (`POST /api/chat`, NDJSON streaming)
- **Dependencies**: `requests>=2.28`, `textual>=0.80`
- **Entry point**: `1bcoder` command (via `pyproject.toml` + setuptools)

### 1.4 Project Files

| File | Role |
|---|---|
| `chat.py` | Main application: TUI, all command handlers, MCP client, streaming logic |
| `headless.py` | Headless plan runner — executes plans without UI, imports core logic from `chat.py` |
| `pyproject.toml` | Build metadata, entry point declaration |
| `requirements.txt` | Minimal dependency list |
| `run.bat` | Windows quick-launch |
| `MCP.md` | Catalog of ready-to-use MCP servers |
| `.1bcoder/plans/` | Per-project directory for plan `.txt` files |

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

The extracted fix is parsed with `re.search(r'LINE\s+(\d+)\s*: ?(.*)')`, applied to the file, and shown as a character-level diff with changed characters highlighted in red before the user confirms.

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

Runs any shell command and injects stdout into the AI context. This is the manual implementation of the **Kantian Observer Loop** from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §6.2: the user runs the code, observes its actual behavior (noumenal reality), and injects the output as grounding evidence for the next AI interaction. Tracebacks, test failures, and print output all become part of the phenomenal representation the model reasons from.

### 2.4 Plan System — `/plan`

Plans are `.txt` files in `.1bcoder/plans/`, one command per line. Lines prefixed `[v]` are marked done and skipped. Plans can be executed interactively (Y/n/q per step), automatically (`/plan apply -y`), or headlessly from the CLI (`--planapply`).

This is the manual implementation of the **Anthill orchestration pipeline** (Section 9). The user writes what the Architect + Router would generate:

```
# example plan: fix the authentication timeout
/read src/auth/session.py 45-90
/fix src/auth/session.py 45-90 timeout not reset on activity
/run pytest tests/auth/ -x
/save ans/fix_result.txt
```

The `headless.py` module runs plans without the TUI — this is the Anthill pipeline running in batch mode. The `HeadlessRunner` class executes the same step handlers as the interactive UI, sharing all core logic via imports from `chat.py`.

### 2.5 Parallel Queries — `/parallel`

```
/parallel <prompt> host:port|model|file [host:port|model|file ...]
```

Sends the same prompt to multiple models simultaneously using `concurrent.futures`. Each response is saved to its designated file. Current context (`/read` injections) is automatically included.

This directly implements **Multi-Instance Debate** from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §4.3. The user reviews the multiple outputs and plays the Critic role — identifying where all models agree (structural signal) versus where they diverge (implementation choice).

### 2.6 MCP Integration

A full `MCPClient` class implements JSON-RPC over stdio, compatible with any standard MCP server. The `/mcp` command suite provides:

```
/mcp connect <name> <command>   — start and connect a server
/mcp tools [name]               — list available tools
/mcp call <server/tool> [args]  — call a tool, inject result into context
/mcp disconnect <name>          — shut down server
```

This is the **nervous system** described in [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §8.1. The filesystem, git, web, database, and memory servers available in `MCP.md` are exactly the external knowledge sources the Anthill agents use to access structured context without reading raw files.

### 2.7 Output Management — `/save`

```
/save <file> [mode]
```

Modes: `overwrite`, `append_above`, `append_below`, `add_suffix`, `code` (strips ` ``` ` fences).

The `code` mode implements the Anthill's principle of treating model outputs as structured artifacts rather than raw text: the markdown wrapper is discarded and only the code payload is persisted.

### 2.8 Context and Session Controls

| Command | Role |
|---|---|
| `/ctx <n>` | Set context window size in tokens (default 8192) |
| `/model` | Switch model at runtime (corresponds to switching the Coder agent in Anthill) |
| `/host <url>` | Switch Ollama host at runtime (corresponds to routing to a different GPU node) |
| `/clear` | Reset conversation context (new task isolation) |
| `ESC` | Interrupt streaming response (agent interrupt in Anthill terms) |

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
| `--planapply` / `headless.py` | LangGraph pipeline batch execution | **Stub** — linear, no conditional routing |
| `/parallel` | Multi-Instance Debate (Section 4.3) | **Implemented** — concurrent model queries |
| MCP client + servers | MCP nervous system (Section 8.1) | **Implemented** — full JSON-RPC client |
| `.1bcoder/plans/` directory | `BCODER_DIR` / OKG project store seed | **Stub** — flat files, no graph |
| Human operator | Router + Architect + Notary + Auditor | **Human-in-the-loop** |

### 3.2 Position in the Anthill Roadmap

1bcoder sits between **Phase 0** (the core edit primitive exists) and **Phase 1** (the Notary Awakens) of the Anthill implementation roadmap from [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) §11.

**Phase 0 (complete):**
- Constrained generation format (`LINE N: content`, SEARCH/REPLACE)
- Context injection via `/read`
- Post-change verification via diff preview
- MCP substrate installed and operational
- Plan execution engine (linear, manual-authored)
- Multi-instance parallel queries

**Phase 1 (missing — next step):**
- `ontology_extractor.py` — AST parser generating Object Passports from source files
- Object Passport YAML schema
- Notary agent — file watcher that triggers incremental OKG updates on save
- Minimal MCP server exposing `get_passport()` and `get_neighbors()`

The critical observation: 1bcoder already has the **verbs** (edit, run, read, fix, patch, plan, parallel). Phase 1 adds the **memory** (the OKG tape). The plan system is ready to accept automated steps once the Notary and Passport infrastructure exists.

### 3.3 The Human as Agent Network

In the current state of 1bcoder, the human operator is running the following Anthill cognitive roles simultaneously:

| Anthill Role | Human Activity |
|---|---|
| **Router** | Decides which file to work on next |
| **Architect** | Writes the plan, decides which lines are relevant |
| **Librarian** | Chooses what to `/read` and at what range |
| **Notary** | Re-reads files after edits to verify correctness |
| **Auditor** | Reviews diffs before approving each change |
| **Tester** | Decides when to `/run` the code and what to check |
| **Observer Loop** | Manually injecting `/run` output as ground truth |

This is not a weakness — it is the **design philosophy** of the prototype phase. The human runs the roles manually, gains understanding of what each role does and where it fails, and the automation of each role becomes a well-specified engineering task rather than a speculative design.

### 3.4 The SKILL.md Analog

1bcoder's `FIX_SYSTEM` and `PATCH_SYSTEM` constants are primitive SKILL.md analogs. They encode behavioral constraints ("One fix only. No explanation.") and output format specifications into the system prompt. The difference from the full Anthill SKILL.md is that 1bcoder's prompts are static; the Anthill's SKILL.md files are dynamic documents maintained and updated by the Researcher agent when failure patterns emerge.

The evolution path from 1bcoder to Anthill is clear: replace the hardcoded `FIX_SYSTEM` string with a SKILL.md file loaded at startup, and add a mechanism to update it when the model repeatedly fails a particular output format.

---

## 4. Relation to the SIMARGL Research Program

### 4.1 1bcoder as Empirical Validation Platform

The SIMARGL research asks: *can context quality substitute for model scale?* 1bcoder is a working instrument for testing this empirically. Every `/fix` command is an experiment: same model, different context quality (full file vs. selected range), measurable outcome (fix accepted or rejected). Aggregate data from 1bcoder sessions would provide direct empirical evidence for or against the Anthill's core hypothesis.

### 4.2 Concept Correspondences

| SIMARGL Concept | 1bcoder Expression |
|---|---|
| **Minimum Description Length (Object Passport)** | `/read file.py 10-30` — human-selected MDL representation of the relevant code |
| **Semantic Coordinates** (positive space) | Lines selected by `/read` — the positive semantic space of the task |
| **Negative Space filtering** | Lines *not* read — deliberately excluded to prevent context saturation |
| **Reduction Ladder** | `/read` selects Level 6 (code) → model reasons at Level 3–4 (pattern recognition) |
| **Symbol Grounding** | `/run` output grounds abstract model reasoning in concrete execution results |
| **Hermeneutic Circle** | Edit → diff → confirm → `/run` → re-read — the iterative code↔context loop |
| **Novelty/Structurality** | A human reviewer of diffs is performing the same classification informally |
| **SIMARGL Evolution Zone** | Plans that `/fix` within a single module are EVOLUTION; cross-module patches are DISRUPTION — the same 2×2 matrix applies |

### 4.3 Contribution to the Anthill Experimental Program

[ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) defines twelve experiments across four families. 1bcoder provides a practical testbed for **Experiment Family A** (Context Quality Experiments) without requiring the full OKG infrastructure:

- **Exp A-1 (Context Window Saturation)**: Compare `/fix` success rate with full-file `/read` vs. selected-range `/read`
- **Exp A-2 (Passport Substitution)**: Compare `/fix` success rate with raw code vs. a manually-written Object Passport injected via `/read`
- **Exp A-3 (Model Sufficiency)**: Run the same task via `/parallel` across multiple model sizes; measure fix acceptance rate across sizes at controlled context quality levels

These experiments require only 1bcoder, an Ollama instance, and a set of test files — no OKG, no graph database, no agent infrastructure.

---

## 5. Gap Analysis: 1bcoder → Full Anthill

The following table identifies what must be built to advance from 1bcoder to a Phase 1 Anthill OS:

| Gap | Current State | Required Component | Complexity |
|---|---|---|---|
| **Object Passports** | None — context is raw code | `ontology_extractor.py`: AST parser → YAML passport | Medium |
| **OKG** | `.1bcoder/` flat files | Neo4j/FalkorDB graph with RDF triplets | High |
| **Notary agent** | Human re-reads after edits | File watcher → incremental OKG update on save | Medium |
| **Observer Loop** | Human runs `/run` manually | Post-edit trigger: re-scan file → diff vs. OKG | Medium |
| **Automated plan generation** | Human writes plans manually | Architect agent: NL intent → plan steps | High |
| **Conditional routing** | Linear plan execution | LangGraph: conditional edges, retry, escalation | High |
| **Qualia detection** | None | Router classifies tasks by Qualia Boundary markers | High |
| **SKILL.md system** | Hardcoded system prompts | Dynamic per-agent SKILL.md + Researcher update loop | Medium |

The minimum viable step from 1bcoder to Phase 1 Anthill is the **Notary + Passport + minimal MCP server** trio. This requires no multi-agent infrastructure — it is a single persistent background process that maintains Object Passports for all files in `.1bcoder/`, exposed via an MCP server that the existing 1bcoder MCP client can already connect to.

---

## 6. Summary

1bcoder is the Anthill OS running on one substrate: human intelligence. It provides a complete set of edit primitives, a context injection mechanism, a plan execution engine, a multi-instance query system, and an MCP client — all the verbs of the Anthill system, with the human playing every cognitive role that has not yet been automated.

Its relationship to the Anthill architecture is not merely analogical. Each 1bcoder command is a direct, working implementation of a specific Anthill component, held in manual operation until the corresponding agent infrastructure is built to replace the human. The tool is designed to be progressively automated: each role the human currently performs is a well-defined, independently implementable component in the Anthill roadmap.

The core thesis — that context quality substitutes for model scale — is validated every time a 1B model successfully applies a `/fix` that it would fail without `/read`'s selective context injection.

---

**Document Version**: 1.0
**Created**: 2026-03-05
**Project**: SIMARGL / Anthill Research Program
**Relation to**:
- [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) — parent architecture document
- [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) — experimental validation framework
- [FINAL_PRODUCT.md](FINAL_PRODUCT.md) — SIMARGL product architecture (codeXpert = 1bcoder's planned MCP evolution)
- [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md) — agent architecture precursor
- [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) — philosophical grounding (Observer Loop, symbol grounding)
