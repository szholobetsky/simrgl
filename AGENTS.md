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
C:\Project\codeXplorer\capestone\simrgl\   ← THIS REPO (working dir)
├── exp0/        TF-IDF baseline (failed, historical reference only)
├── exp1/        Statistical analysis (Bradford, HHI distributions)
├── exp2/        Embedding models comparison (Word2Vec → BGE-large), module-level
├── exp3/        Full RAG pipeline — sentence-transformers + pgvector/Qdrant
├── exp4/        PLANNED — architecture families (ModernBERT, ColBERT, Qwen3)
├── exp5/        PLANNED — cross-vocabulary symbol grounding (TF-IDF on identifiers)
├── ragmcp/      Production MCP server + local Ollama agent
├── concepts/    Research concept documents (see list below)
├── info/        Results files, leaderboard, planning docs
├── 1bcoder/     git submodule → szholobetsky/1bcoder (STALE SNAPSHOT)
├── vyrii/       git submodule (STALE SNAPSHOT)
└── ...other submodules (radogast, yasna, svitovyd) — all STALE SNAPSHOTS
```

**CRITICAL**: Submodules in this repo are pinned stale snapshots. All live code lives in `C:\Project\` (no "s"):

| Tool | Live path |
|---|---|
| 1bcoder | `C:\Project\1bcoder\` |
| vyrii | `C:\Project\vyrii\` |
| yasna | (not yet implemented) |
| svitovyd | (not yet implemented) |
| radogast | `C:\Project\radogast\` |
| simargl (this repo) | `C:\Project\codeXplorer\capestone\simrgl\` |

When editing any tool, always work in `C:\Project\<tool>\`, never in the submodule copy inside simrgl.

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
| vyrii | Вирій | Local LLM server (Flask default, port 5000) | Production |
| yasna | Ясна — goddess of fate | Knowledge memory system for ctx files | Planned only |
| svitovyd | Світовид — four-faced god | Code structure map service (extract from 1bcoder) | Discussed, not planned |
| radogast | Радогост | (supporting role) | Active |

### 1bcoder (`C:\Project\1bcoder\`)
- PyPI: `pip install 1bcoder`
- Key commands: `/agent`, `/flow`, `/proc`, `/parallel`, `/map`, `/translate`, `/role`, `/ctx compact`, `/script`, `/prompt`
- Packaging: `_bcoder_data/` = wheel defaults; `~/.1bcoder/` = user global; `.1bcoder/` = project-local
- Always edit `_bcoder_data/` for default changes, never `~/.1bcoder/`

### vyrii (`C:\Project\vyrii\`)
- Default: Flask API (`flask_api.py`, port 5000) + static HTML/JS UI in `vyrii/ui/`
- Pure Python dependencies: requests, flask, flask-cors, apscheduler, waitress
- Optional extras: `vyrii[gradio]`, `vyrii[api]`, `vyrii[full]`
- OpenAI-compatible endpoints + `/vyrii/*` endpoints, basic auth

---

## Datasets

All SQLite with `TASK` table (ID, NAME, TITLE, DESCRIPTION, COMMENTS) and `RAWDATA` table (ID, TASK_NAME, PATH):

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
| `KEYWORD_ENTITY_MAPPING.md` | Business terms ↔ code identifier mapping |
| `DUAL_MCP_SERVER_ARCHITECTURE.md` | Historical vs recent context servers |
| `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` | Husserl/Heidegger applied to code navigation |
| `DEEPAGENT_AND_DEEPAGENT_MD.md` | Recursive markdown generation with parallel workers |
| `ANTHILL_DISTRIBUTED_COGNITIVE_OS.md` | Distributed agent OS concept |

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
- **Before editing**: confirm which repo (live `C:\Project\<tool>\` vs stale submodule). On a new machine the live paths may differ — verify the path exists before editing.
- **Before destructive git ops**: explicit confirmation required, not implied by prior discussion.
- **Show findings first**: report what existing code already handles before proposing changes.
- **No legacy code removal** without explicit permission — every line was added for a reason.
- **Hardware-aware suggestions**: keep 6GB VRAM limit in mind for any model recommendations.
- **Academic framing matters**: this is dissertation research; contributions need to be novel and citable.
