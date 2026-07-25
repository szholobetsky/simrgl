# SIMARGL — Final Product Vision

> **SIMARGL**: Structural Integrity Metrics for Adaptive Relation Graph Learning
> A research platform + production tool for task-to-code retrieval and structural code analysis.

---

## 1. What We Have Built (Research History)

### The Core Research Question

> Given a natural language task description (from Jira, GitHub Issues, etc.), which code files/modules need to be modified?

This is **task-to-code retrieval**. It has applications in:
- Developer onboarding (where do I look?)
- Code review assistance (did the author touch the right files?)
- Automated impact analysis (what will this task affect?)
- Structural quality monitoring (is the codebase evolving or stagnating?)

### Experiment Timeline

| Exp | Method | Key Finding |
|-----|--------|-------------|
| **exp0** | TF-IDF + BM25 | Too noisy; failed on module retrieval |
| **exp1** | Statistical (Bradford, HHI) | Identified vocabulary concentration; useful for filtering |
| **exp2** | Word2Vec, FastText, GloVe, BERT, CodeBERT, BGE-large | **Aggregation strategy has zero effect** (MAP identical across avg/sum/median/weighted/cluster). BGE-large best (MAP 0.37 on Sonar). CodeBERT worst despite being code-specific — tasks are NL, not code. |
| **exp3** | Sentence-transformers + pgvector/Qdrant | Full RAG pipeline. bge-large > bge-small > bge-m3 (OOM). Best module MAP ~0.80 (recent split), ~0.45 (honest modn split). |
| **exp4** *(planned)* | ModernBERT, ColBERT, LLM-based (Qwen3, jina-v5) | Compare three architecture families across multiple datasets |
| **ragmcp** | Gradio UI + MCP server + two-phase agent | Production prototype for real-time code navigation |

### Key Experimental Findings (Locked)

1. **Aggregation strategy: irrelevant** — avg/sum/median/weighted/cluster produce identical MAP
2. **Split strategy matters** — `recent` inflates MAP (~0.80) due to temporal proximity; `modn` is honest (~0.45)
3. **Source variant** — `desc` (description) outperforms `title` and `comments`
4. **Window** — w1000 (1000 days) outperforms narrower windows for module retrieval
5. **CodeBERT worst** — domain mismatch: task descriptions are natural language, not code
6. **File-level retrieval** is much harder than module-level (MAP <0.03 vs ~0.45)

---

## 2. The Four Components (SIMARGL Product Architecture)

The research platform is being refactored into four distinct, independently deployable components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         SIMARGL Platform                         │
│                                                                   │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ codeXtract  │──▶│ codeXplorer  │──▶│      codeXpert       │  │
│  │             │   │              │   │                      │  │
│  │ git/Jira/   │   │ tokens,      │   │ RAG, Gradio UI,      │  │
│  │ GitHub →    │   │ embeddings,  │   │ MCP server,          │  │
│  │ SQLite DB   │   │ pgvector     │   │ structural analysis  │  │
│  └─────────────┘   └──────────────┘   └──────────────────────┘  │
│          │                │                      │               │
│          └────────────────┴──────────────────────┘               │
│                           │                                       │
│                  ┌────────▼────────┐                             │
│                  │   codeXport     │                             │
│                  │                 │                             │
│                  │ AST graph +     │                             │
│                  │ ontology +      │                             │
│                  │ business terms  │                             │
│                  │ ↔ identifiers   │                             │
│                  └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. codeXtract — Data Gathering Pipeline

**Purpose**: Extract git commits + task metadata into a structured SQLite database.

**Current location**: `codeXplorer/` folder (to be renamed/restructured)

### Architecture

```
git clone <repo>  →  git_connector.py  →  RAWDATA table (one row per file per commit)
                  →  task_extractor.py →  RAWDATA.TASK_NAME
                  →  jira/github_api_connector.py → TASK table (title, description, comments)
```

### SQLite Schema

```sql
RAWDATA: ID, SHA, AUTHOR_NAME, AUTHOR_EMAIL, CMT_DATE, MESSAGE, PATH, DIFF, TASK_NAME
TASK:    ID, NAME, TITLE, DESCRIPTION, COMMENTS
```

### Tracker Support

| Tracker | Status | Commit Pattern |
|---------|--------|----------------|
| Apache Jira | **Working** | `KAFKA-1234: description` → `re.match(r'^[A-Z]+-\d+')` |
| GitHub Issues | **To build** (~80 lines) | `Fixes #1234` → `re.search(r'(?:fix(?:e[sd])?|clos(?:e[sd]?)|resolv(?:e[sd]?))\s*#(\d+)')` |
| GitLab Issues | Planned | Same pattern as GitHub |
| Bugzilla | Low priority | `Bug NNNNN -` pattern |

**Critical bug** in `task_extractor.py`: `re.match()` → must be `re.search()` for GitHub (issue ref is not at position 0). Also use `group(1)` not `group(0)` when capture group present.

### GitHub API Connector (To Build)

```python
# connectors/github/github_api_connector.py — ~60 lines
class GitHubApiConnector:
    def fetch_task_details(self, issue_number: str) -> tuple:
        # GET /repos/{owner}/{repo}/issues/{number}
        # returns (title, description_markdown, comments_joined)
        # Must filter: if "pull_request" in data → skip (it's a PR not an issue)
```

Key differences vs Jira:
- Description: plain Markdown (better for embeddings than Jira's ADF JSON)
- Auth: token → 5000 req/hour; no token → 60/hour (unusable)
- Pandas commits have low link rate (~30%) → use GraphQL PR approach instead

### Dataset Diversity Strategy

**Current**: 3 Apache/Java/Jira projects (Sonar, Kafka, Spark) — all same culture

**Target**: 25+ projects across:
- Languages: Python, Ruby, Go, TypeScript, Rust, HCL, SQL, Scala
- Communities: Apache, GitHub-native, CNCF, Microsoft, Mozilla
- Trackers: Jira, GitHub Issues, GitLab

High-priority additions (zero code changes — same Jira tool):
- Apache Flink, Hive, Cassandra, Hadoop, Arrow, Beam

High-priority additions (need GitHub connector):
- Django, Pandas, Rails, Ansible, TypeScript, Prometheus, Terraform

Full project list: `datasets/projects.csv` (37 projects with metadata)

---

## 4. codeXplorer — Embedding & Vector Index

**Purpose**: Transform RAWDATA+TASK tables into vector embeddings stored in pgvector.

**Current location**: `exp3/` folder

### Architecture

```
SQLite (RAWDATA + TASK)
  ↓ ETL Pipeline
Token extraction → Embedding model → pgvector (PostgreSQL)
  ↓ Query
Task description → embed → cosine search → ranked files/modules
```

### Embedding Model Families (exp4 plan)

**Family 0 — Classical bi-encoders** (done):
- bge-small-en-v1.5 (127MB, 512 dims)
- bge-large-en-v1.5 (1.2GB, 1024 dims) ← best so far
- bge-m3 (2.2GB, 1024 dims) — needs batch_size=4 to avoid OOM on 6GB VRAM

**Family 1 — ModernBERT + ColBERT**:
- ModernBERT-base/large — new encoder (Dec 2024), 8192 token context, trained on code data
- ColBERT — late interaction (one vector per token, MaxSim at query time)
- Library: `pylate`

**Family 2 — LLM-based embedders**:
- Qwen3-Embedding-0.6B/4B/8B — decoder-only repurposed as embedder
- jina-embeddings-v5-text-small — 32768 token context
- Via Ollama for 8B+ models with 4-bit quantization

### Fixed Variables (exp2 proved these don't matter)

- Aggregation strategy: always use `avg` (others identical)
- Split: always `modn` (honest uniform sampling; `recent` inflates MAP)
- Source: `desc` (description-only)
- Window: `w1000` (1000-day)

### Multi-dataset Evaluation

Evaluate each model on: Sonar (Java, Jira), Kafka (Java, Jira), Spark (Scala+Java, Jira), + GitHub projects

Implementation notes: `exp4/README.md` (full skeleton code for all files)

---

## 5. codeXpert — RAG + Analysis Tools

**Purpose**: User-facing tools for querying the indexed codebase and analyzing structural quality.

**Current location**: `ragmcp/` folder

### Components

**Two-Phase Reflective Agent**:
- Phase 1: Reasoning → extract keywords → search modules/files/tasks → form initial hypothesis
- Phase 2: Reflection → validate hypothesis against code → refine → return ranked files

**Gradio UI**: Interactive web interface for task-to-code search

**MCP Server**: Claude integration for in-IDE code navigation

**Dual Collection System**:
- RECENT collection: last N tasks (temporal proximity)
- ALL collection: complete history (breadth of coverage)

**Structural Analysis** — SIMARGL metrics:

```
                      Novelty@K (new relationships)
                           HIGH
                             ▲
          DISRUPTION         │         EVOLUTION
        (new + cross-module) │       (new + within-module)
              ⚠️ RISKY       │            ✅ IDEAL
LOW ◄─────────────────────── ┼ ───────────────────────► HIGH
(Structurality@K)            │                  (Structurality@K)
         STAGNATION          │         MAINTENANCE
       (old + cross-module)  │       (old + within-module)
             ❌ USELESS      │            🔧 SAFE
                             ▼
                           LOW
```

Core metrics:
- **SES** = sqrt(Novelty@K × Structurality@K)
- **HES** = 2×N×S/(N+S) (harmonic, more sensitive to imbalance)

---

## 6. codeXport — Ontology & Symbol Grounding

**Purpose**: Extract a knowledge graph from code (AST + co-occurrence) and link business terms from task descriptions to code identifiers. Provides the "map" that makes codeXpert's LLM reason like a domain expert.

### The Core Problem (Philosophical Grounding)

Current RAG approach gives LLM a "list of random village names". What we need is a **map**.

Grounded in Husserlian phenomenology:
- **Noema** (object of intention): the business term as the user experiences it (`RULE`, `Act of disposal`)
- **Noesis** (act of perceiving): the agent's graph traversal finding the implementation
- **Lebenswelt** (life-world): the identifier graph + relations + commit history
- **Zuhandenheit** (ready-to-hand): code that works — not in search results
- **Vorhandenheit** (present-at-hand): code under examination — in search results
- **Symbol Grounding**: `business_term → identifier cluster → files`
- **Epoché**: ignore syntax, focus on co-occurrence patterns

**Key insight**: "Code is not a description of business — it is its digital emanation." We don't need to understand code syntax; we feel its patterns through co-occurrence statistics.

### The Approach: Language-Agnostic Identifier Analysis

Works with **any** programming language (Java, Python, SQL, PL/SQL, 1С, etc.) because it never parses grammar — only:
1. Extracts tokens (split camelCase, snake_case → individual words)
2. Counts co-occurrence (which tokens appear in the same files/classes)
3. Finds the semantic field of each token via Word2Vec trained on code
4. Links task description keywords to identifier clusters

### Pipeline

**Step 1: Train Word2Vec on the codebase**
```python
# Tokenize all code files: "RuleIndex" → ["Rule", "Index"]
# Train Word2Vec: model.most_similar("Rule") → [("Index", 0.87), ("Quality", 0.82)]
```

**Step 2: Build keyword index** (PostgreSQL `keyword_index` table)
```sql
keyword_index: keyword, vector(300), frequency, related_files[], related_keywords[]
```

**Step 3: Extract business terms from task history**
```python
# All tasks that modified RuleIndex.java → extract nouns → TF-IDF
# Result: RULE appears 100% → primary business term
```

**Step 4: Build entity map** (bidirectional mapping)
```json
{
  "RULE": {
    "business_terms": ["rule", "coding rule", "quality rule"],
    "files": ["RuleIndex.java", "RuleUpdater.java", ...],
    "related_entities": ["QUALITY_PROFILE", "INDEX", "ISSUE"]
  }
}
```

**Step 5: Positive + Negative Space filtering**
- Positive: task is about `RULE` → boost `RuleIndex.java`, `RuleUpdater.java`
- Negative: `RULE` is distant from `SERVER`, `PLUGIN` → penalize those files
- Result: Precision@5 goes from ~40% to ~80% in example scenarios

**Step 6: Handle obfuscated/poorly-named code**
```python
# If file is named identifier0001.java:
# Look at all tasks that modified it → extract recurring nouns
# Most common: "rule" (100%) → infer: identifier0001.java = RULE entity
```

### Storage Options

| Store | Use for |
|-------|---------|
| PostgreSQL + pgvector | keyword vectors (ivfflat index for cosine search) |
| NetworkX / graph DB | entity relationships, co-occurrence graph |
| JSON file | entity map (small enough for direct access) |
| SQLite | identifier ↔ file mappings |

### Integration with codeXpert

Enhanced search flow:
```
Task description
  ↓ Keyword extraction (LLM or KeyBERT or spaCy)
  ↓ Lookup keyword_index (pgvector cosine search)
  ↓ Get entity → files → related entities
  ↓ Boost matched files (+0.3–0.5 to similarity score)
  ↓ Penalize negative-space files (-0.3)
  ↓ LLM receives: entity context + ranked files + entity relationships
  ↓ LLM reasons with structural understanding, not disconnected file list
```

### Grounding Metrics

| Metric | Description |
|--------|-------------|
| **Grounding Accuracy** | % of business terms correctly linked to code identifiers |
| **Horizon Completeness** | % of actually changed files within the semantic horizon |
| **Noematic Precision** | % of grounded objects that match actual task intent |
| **Affordance Relevance** | % of suggested actions that were actually performed |
| **Negative Space Accuracy** | % of excluded files that were truly irrelevant |

---

## 7. Product UX: The Wizard Architecture

**Problem**: The data-gathering stage (codeXtract) takes 1–2 days. Users need feedback and can't just stare at a terminal.

**Solution**: Status-board wizard (not next/back linear wizard):

```
┌────────────────────────────────────────────────────────┐
│  SIMARGL Setup                              Step 2 of 4 │
├────────────────────────────────────────────────────────┤
│  ✅ 1. Connect Repository    (done)                     │
│  ⚙️  2. Extract Commits       ████░░░░ 43% (2h left)    │
│  ⏳ 3. Fetch Task Details    (waiting for step 2)       │
│  ⏳ 4. Build Vector Index    (waiting for step 3)       │
│                                                          │
│  [Pause]  [View Logs]  [Skip to Step 4 with saved data] │
└────────────────────────────────────────────────────────┘
```

Each stage runs as a background job with persistent progress. User can:
- Walk away and come back
- Skip already-done stages (idempotent)
- Monitor logs per stage
- Jump to search/analysis once indexing is done (even if still processing)

**Similar tools**: Sourcegraph (setup wizard), MLflow (experiment tracking), Elasticsearch + Kibana

---

## 8. Current Implementation State

### What Exists Today

| Component | Location | Status |
|-----------|----------|--------|
| codeXtract | `codeXplorer/` | Working for Jira; GitHub connector to build |
| exp2 results | `info/exp2_result/` | Aggregation strategy = irrelevant (proven) |
| exp3 pipeline | `exp3/` | Full pipeline: ETL + embedders + vector backends + evaluation |
| exp4 plan | `exp4/README.md` | Complete implementation guide (skeleton code included) |
| ragmcp | `ragmcp/` | Gradio UI + MCP server + two-phase agent |
| Dataset list | `datasets/projects.csv` | 37 projects with metadata |
| Dataset strategy | `datasets/README.md` | Tracker types, regex patterns, API endpoints |
| GitHub connector analysis | `datasets/GITHUB_CONNECTOR_ANALYSIS.md` | Full feasibility: ~80 lines needed |
| Concepts | `concepts/` | Phenomenological grounding, keyword indexing, SIMARGL metrics |

### What Needs to Be Built

**Short term (next session — different computer)**:
1. Fix `re.match` → `re.search` in `codeXplorer/task_extractor.py`
2. Create `connectors/github/github_api_connector.py` (~60 lines)
3. Implement exp4 code based on `exp4/README.md` (copy exp3, add model families)

**Medium term**:
4. Build codeXport pipeline (Word2Vec on code → keyword index → entity map)
5. Collect Apache Jira datasets (Flink, Hive, Cassandra, Hadoop, Arrow)
6. Collect GitHub Issues datasets (Django, Pandas, Rails, Prometheus, TypeScript)

**Long term**:
7. Refactor into SIMARGL monorepo (codeXtract, codeXplorer, codeXpert, codeXport)
8. Build wizard UI in Gradio
9. Publish datasets + results

### Proposed Monorepo Structure

```
simargl/
├── codeXtract/
│   ├── connectors/
│   │   ├── git/git_connector.py
│   │   ├── jira/jira_api_connector.py
│   │   └── github/github_api_connector.py   ← TO BUILD
│   ├── task_extractor.py                     ← BUG FIX NEEDED
│   ├── task_fetcher.py
│   └── db_manager.py
├── codeXplorer/
│   ├── config.py                             ← expand model registry
│   ├── embedders.py                          ← new file (family-aware)
│   ├── etl_pipeline.py
│   ├── vector_backends.py
│   ├── colbert_pipeline.py                   ← new file (pylate)
│   └── run_comprehensive_experiments.py
├── codeXpert/
│   ├── agents/two_phase_agent.py
│   ├── mcp_server.py
│   └── gradio_app.py                         ← wizard UX
├── codeXport/
│   ├── ast_parser.py                         ← tree-sitter multi-language
│   ├── word2vec_trainer.py                   ← train on codebase
│   ├── keyword_indexer.py                    ← build keyword_index table
│   ├── entity_mapper.py                      ← build entity_map
│   └── ontology_server.py                    ← query interface
└── shared/
    ├── db_schema.sql
    └── config_base.py
```

---

## 9. Academic Publication Status

**Presented at conference**: Embedding model comparison (exp2/exp3 findings)

**Key findings for papers**:
1. Aggregation strategy is irrelevant — simplifies architecture (one less hyperparameter)
2. `recent` split inflates MAP — warns community about evaluation bias
3. CodeBERT underperforms general sentence-transformers on task-to-code retrieval
4. BGE-large best practical choice at 6GB VRAM constraint
5. Module-level retrieval is much more tractable than file-level

**Planned papers**:
- Exp4: Architecture family comparison (classical → ModernBERT/ColBERT → LLM-based)
- Dataset diversity study: Does cross-language/community diversity change model ranking?
- codeXport: Phenomenological grounding for code navigation (symbol grounding via co-occurrence)
- SIMARGL metrics: Structural evolution measurement using Novelty × Structurality 2×2

---

## 10. Key References from concepts/ folder

| Concept | File | Key Idea |
|---------|------|----------|
| SIMARGL metrics (Novelty, Structurality, SES, HES) | `SIMARGL_concept.md` | 2×2 matrix: Evolution/Disruption/Maintenance/Stagnation |
| Keyword as semantic coordinates | `KEYWORD_INDEXING.md` | Word2Vec on code → positive/negative space → bounded search |
| Entity-to-file mapping | `KEYWORD_ENTITY_MAPPING.md` | Bidirectional: code identifiers ↔ business terms from task history |
| Phenomenological grounding | `PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md` | Language-agnostic; epoché = ignore syntax, feel co-occurrence |
| Philosophy of code understanding | `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` | Husserl (noema/noesis), Heidegger (zuhandenheit), Merleau-Ponty, symbol grounding |
| Compositional embeddings | `COMPOSITIONAL_CODE_EMBEDDINGS.md` | Additive/multiplicative composition for graph-structured code |
| Cross-layer transformation | `CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md` | UI→Service→Entity→DB path embeddings |
| Two-phase agent | `TWO_PHASE_REFLECTIVE_AGENT.md` | Reasoning + Reflection cycle (hermeneutic circle) |
| Dual MCP architecture | `DUAL_MCP_SERVER_ARCHITECTURE.md` | RECENT vs ALL collections; temporal horizon fusion |

---

## 11. Android Deployment — vyrii + Local LLM on Phones

**Goal**: run the full vyrii stack (UI + LLM server) on an Android phone — for field/shelter/offline use, and to turn a fleet of phones into parallel workers (Team profiles, nightly consolidation via apscheduler).

**Key enabler** (vyrii commit `5a5a96f` "create flask api to be able to run vyrii without rust"): the default vyrii mode is now **Flask + static HTML/JS UI** (port 5000) with pure-Python dependencies only (`flask`, `flask-cors`, `waitress`, `apscheduler`, `requests`). Gradio and FastAPI are optional extras. **Zero Rust, zero C extensions** — this single decision is what makes every Android path below cheap.

**Second enabler**: `llama-server` (llama.cpp) exposes an OpenAI-compatible `/v1/chat/completions`, and vyrii already supports `openai://` hosts. **Ollama is therefore not required on the phone** — llama-server replaces it entirely.

### Why Google Play blocks the Termux path

- Android 10+ W^X: apps with `targetSdkVersion ≥ 29` cannot `exec()` files from their writable home directory. Termux F-Droid stays on targetSdk 28; Play requires current targetSdk. The Play build of Termux survives on a `system_linker_exec` hack, is functionally frozen (~v0.108), and itself violates Play policy.
- Play "Device and Network Abuse" policy forbids downloading executable code — which is exactly what `pkg install` / `ollama pull` of binaries does.
- **Legal Play loophole**: native libraries shipped inside the APK as `jniLibs/*.so` are extracted to the read-only `nativeLibraryDir`, and `exec()` from there IS allowed at any targetSdk. Models downloaded at runtime are *data*, not code — also allowed.

### Variant A — Termux bootstrap script (1 day)

Not an app; a one-liner after installing Termux from F-Droid:

```bash
curl -fsSL https://.../vyrii-droid.sh | bash
# pkg install ollama python        (ollama is in official Termux repo since Apr 2025)
# pip install vyrii                (pure Python — installs cleanly)
# ollama pull qwen3:1.7b
# termux-wake-lock                 (prevents Android from killing the server)
# Termux:Boot autostart script
# → open http://localhost:5000
```

Natural continuation of the existing `vyrii_auto.sh`. Distribution: GitHub. Covers the fleet-of-phones scenario immediately.

### Variant B — Companion APK (~1 week)

A small sideloaded app that orchestrates Termux: checks Termux is installed, pushes the bootstrap via the `RUN_COMMAND` intent (`allow-external-apps`), then acts as a WebView on `localhost:5000`. Looks like a "vyrii app" with an icon; all machinery stays in Termux. Distribution: APK / F-Droid (not Play — it orchestrates downloading executables).

### Variant C — Termux fork with custom bootstrap (~weeks)

Self-contained APK "vyrii-droid": fork of termux-app (GPLv3 — fork must stay GPL; precedents: UserLAnd, Andronix) with python + ollama/llama-server + vyrii pre-installed in the bootstrap archive, launcher activity = WebView. One APK, zero commands for the user. Distribution: GitHub releases / F-Droid. Still not Play-eligible (same exec model).

### Variant D — Native Play-legal APK (feasibility 1–2 evenings; full build 1–2 weeks)

The only variant that can legitimately enter Google Play:

```
APK
├── jniLibs/arm64-v8a/
│   ├── libllamaserver.so          ← renamed llama-server (NDK build)
│   └── libllama.so, libggml*.so
├── Chaquopy: Python + pip install vyrii   ← pure Python, no wheel battles
└── Kotlin ForegroundService:
      exec(nativeLibraryDir + "/libllamaserver.so")   # port 8080, LD_LIBRARY_PATH set
      python thread: waitress → flask_api (port 5000), host = openai://localhost:8080
      WebView → http://localhost:5000                  # the same HTML UI as desktop
```

Packaging tool notes (the "PyToApk" question): there is **no true PyInstaller equivalent** for Android. Closest options: **Chaquopy** (Python SDK inside a normal Gradle project — best fit), **Briefcase/BeeWare** (one-command packaging, Chaquopy backend underneath), **Buildozer/python-for-android** (Kivy-centric, recipe pain — not recommended here).

Android-specific gotchas:
1. `os.environ["HOME"] = filesDir` before importing vyrii (redirects `~/.vyrii`);
2. `lxml` (web/html extras) is a C extension — Chaquopy has prebuilt wheels; optional anyway;
3. 16 KB page-size alignment required for native libs on Android 15+ — build llama.cpp with NDK r27+;
4. `extractNativeLibs=true` so the `.so` files land on the filesystem;
5. Foreground service + battery-optimization exemption to keep the server alive;
6. GGUF models downloaded at runtime into the app files dir (Play-legal: data, not code);
7. llama.cpp on Android: CPU by default; experimental Vulkan/OpenCL on Adreno is a potential GPU bonus over ollama-in-Termux (CPU-only).

**Phase-0 fallback** (if Python embedding stalls): llama-server ships its own built-in web chat UI — a Kotlin shell + jniLibs alone is already a working local-chat APK for Play. Not vyrii, but the skeleton vyrii drops onto later.

### Variant E — Play thin client (trivial)

A WebView/native client that only connects to a vyrii server running elsewhere (PC, home server). Executes nothing locally → passes Play review without any tricks. The one thing publishable to Play *today* with near-zero effort.

### Recommended sequence

| Step | Variant | Effort | Outcome |
|------|---------|--------|---------|
| 1 | A — Termux script | 1 day | Fleet of phone workers working immediately |
| 2 | D — feasibility spike | 1–2 evenings | Chaquopy hello-world + `pip install vyrii` + Flask in a thread + WebView |
| 3 | D — full APK | 1–2 weeks | Play-publishable native vyrii with local LLM |
| opt | C — Termux fork | weeks | Boxed APK for terminal-averse users (F-Droid) |
| opt | E — thin client | days | Play presence pointing at user's own server |

Ties to `AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md`: phones running vyrii become both parallel evaluation workers (pair-checking, dedupe) and hosts for nightly knowledge consolidation via the built-in apscheduler.

---

**Document Version**: 1.1
**Created**: 2026-02-28 · **Updated**: 2026-06-12 (added §11 Android Deployment)
**Purpose**: Final product vision synthesizing all conversations, experiments, and concepts
