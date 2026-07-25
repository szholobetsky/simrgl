# Project Memory: CodeXplorer / SimRGL

## What This Project Is
Research on **task-to-code retrieval**: given a Jira task description, predict which code modules/files need to be modified. Doctoral dissertation project (Ukraine).

- [Academic context + publication plan](project_academic_context.md)

## Repository Structure
**Live repos of ALL tools are in `C:\Project\` (no "s"):** 1bcoder, vyrii, yasna, radogast, svitovyd, simargl. Submodules inside simrgl are pinned stale snapshots — always check `C:\Project\<tool>` for current code.
```
C:\Project\codeXplorer\capestone\simrgl\
├── exp0/    TF-IDF (failed baseline)
├── exp1/    Statistical analysis
├── exp2/    Multiple embedding models (Word2Vec→BGE-large), module-level retrieval
├── exp3/    RAG with sentence-transformers + Qdrant/PostgreSQL — main research exp
├── exp4/    PLANNED — architecture families comparison (see README)
├── ragmcp/  Production MCP server + local Ollama agent
├── concepts/ Current working directory
└── info/    Results files, leaderboard data
```

## Key Confirmed Findings
- **BGE-large best so far**: MAP=0.371 (Sonar), MAP=0.361 (Flask) in exp2
- **Aggregation strategy has ZERO effect** — identical MAP across all strategies. Use `avg` always.
- **desc source beats title and comments** — title+description is optimal; comments add noise
- **w1000 window best for modn split**; w100 best for recent split
- **modn split is more honest** — recent split inflates module MAP to ~0.80
- **CodeBERT worse than general models** — domain pretraining on code doesn't help
- **bge-m3 OOM on 6GB VRAM** — batch_size=32 too large; fix: batch_size=4

## Exp3 Codebase Architecture
- Language: Python, sentence-transformers, PostgreSQL+pgvector (or Qdrant)
- Key files: `config.py`, `etl_pipeline.py`, `run_experiments.py`, `vector_backends.py`, `utils.py`
- Results schema: `model,split_strategy,experiment_id,source,target,window,MAP,MRR,P@1,R@1,...`

## Exp4 Plan (NOT YET IMPLEMENTED)
Location: `exp4/` (README exists, code not written yet). Full details: `memory/exp4_plan.md`

- Family 0: bge-large (done in exp3); Family 1: ModernBERT+ColBERT; Family 2: Qwen3-Embedding/jina/e5
- Fixed params: source=desc, window=w1000, split=modn, target=module+file
- Multi-dataset: Sonar + Kafka + Spark
- Key new file: `embedders.py` — unified BaseEmbedder; ColBERT via pylate library

## Hardware Constraints
- GPU: 6GB VRAM
- bge-m3 (568M, 2.2GB) needs batch_size=4
- Qwen3-4B (7.7GB) may not fit — try batch_size=1 or skip
- Qwen3-8B via Ollama Q4 quantization (~4-5GB) should fit

## Datasets
- sonar.db: SonarQube, ~9,799 tasks, 12,532 files, 27 modules
- kafka.db: Apache Kafka (available)
- spark.db: Apache Spark (available)
- All SQLite with TASK table (ID, NAME, TITLE, DESCRIPTION, COMMENTS) and RAWDATA table (ID, TASK_NAME, PATH)

## codeXplorer Data Gathering Tool
Architecture — 4 steps: `git_connector.py` → `task_extractor.py` → `jira_api_connector` → `db_manager.py`
- [GitHub connector analysis](datasets/GITHUB_CONNECTOR_ANALYSIS.md)

## SIMARGL Product Architecture
```
codeXtract  → git/Jira/GitHub → SQLite
codeXplorer → tokens + embeddings → pgvector
codeXpert   → RAG tools, Gradio UI, MCP server
codeXport   → AST graph + ontology: business terms ↔ code identifiers
```
- [Full product vision](concepts/FINAL_PRODUCT.md)

## SIMARGL Core Metrics (2×2 Matrix)
Axes: **Novelty@K** × **Structurality@K**
EVOLUTION (new+within) ✅ | DISRUPTION (new+cross) ⚠️ | MAINTENANCE (old+within) 🔧 | STAGNATION (old+cross) ❌
**SES** = sqrt(Novelty × Structurality), **HES** = harmonic mean

## exp5 Concept-Tag Bridge Refinement (2026-07-21)
- [Real-world precedent + PMI fix for cross-vocabulary co-occurrence](project_exp5_concept_tag_bridge.md) — past-job Oracle PL/SQL anecdote proves language/obfuscation-agnostic term↔identifier grounding via canonical concept-tag mediation, not lexical extraction; flags popularity-bias fix (PMI not raw counts) and software-traceability-recovery literature (Antoniol 2002, Marcus & Maletic 2003) missing from exp5 refs

## codeXport — Phenomenological Symbol Grounding
Key idea: "Code is not a description of business — it is its digital emanation."
Pipeline: camelCase tokenize → Word2Vec → keyword_index (pgvector) → entity_map (business↔code)
Philosophical grounding: Husserl, Heidegger, Symbol Grounding Problem, Speech Act Theory

## 1bcoder Tool
Full details: [memory_1bcoder.md](memory_1bcoder.md)

- **GitHub**: https://github.com/szholobetsky/1bcoder | **PyPI**: `pip install 1bcoder` (v0.1.0)
- Main dir: `C:\Project\1bcoder\` — separate repo from simrgl (submodule link in progress)
- Wheel defaults in `_bcoder_data/`; user global in `~/.1bcoder/` (bootstrapped on first run)
- New commands: `/role`, `/proc run -f <file>`, `/parallel profile show/add`, `md`/`mdx` procs, `/script run`, `/script show N`, `/prompt load N`, `/mcp connect --cwd`
- Publication plans: JOSS article, YouTube lectures, grant application
- [/parallel redesign spec](parallel_redesign_spec.md) — new syntax: no quotes, list:/file:/profile:/ctx:/collect:, /ctx compact profile:
- [1bcoder agent model capabilities](feedback_1bcoder_model_capabilities.md) — which local models work for /agent ACTION: format vs which fail
- [Recent changes](project_1bcoder_changes.md) — /translate (4 modes: online/mini/offline NLLB/lm), /script run N fix, md proc UTF-8 fix, pyproject.toml v0.1.7
- [/proj + /ctx compose session 2026-04-09](project_1bcoder_proj_compose.md) — project management, context composer, /ctx compact N

## deepagent_md Flow
- [deepagent_md spec](project_deepagent_md.md) — recursive md tree, BFS parallel workers, compose (flat/linked/html), --ctx/--ctx-worker/--profile/--web
- [deepagent_spec design + deepagent_md `continue`](project_deepagent_spec.md) — organic leaf-detection empirically falsified (3 real AnimalAlert runs, ~3-5x branching/level, never stops on its own); two-pass design (generate to external --maxdepth, then split leaves into specs); C4 decomposition ≠ Agile epic/story (orthogonal axes); `continue <plan_dir> plan: <label>` subcommand implemented, YAML meta persistence

## External Supervision Concept (2026-06-12)
- [Зовнішній логічний нагляд за агентами](project_external_supervision_concept.md) — concepts/AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md; 6 аспектів, MVP = terminator + ladder-gate, метрика context recall

## Tool Ecosystem (simargl + yasna + svitovyd)
- [vyrii: Flask default, live repo C:\Project\vyrii](project_vyrii_flask.md) — сабмодуль застарілий; чистий Python без Rust; Android APK план (Chaquopy + llama-server jniLibs)
- [yasna + svitovyd](project_yasna_svitovyd.md) — yasna pivoted from planned 1bcoder-only wiki to implemented multi-agent session search tool (v0.1.8, PyPI); svitovyd still not implemented
- simargl acronym: **S**emantic **I**ndex: **M**ap **A**rtifacts, **R**etrieve from **G**it **L**og (added to README)
- newer tool in ecosystem: **syryn** (сирин) — added alongside vyrii/simargl/svitovyd/yasna/radogast
- [Пантеон mini PC + Vulkan/iGPU finding](project_pantheon_hardware.md) — i5-6500T/32GB box; Vulkan/iGPU benefit is conditional on iGPU strength (HD 530 = slower than CPU-only; Iris Xe = faster) — NOT a universal win, verify per-machine with ctxtimer
- [ctxtimer flow](project_ctxtimer.md) — `/flow ctxtimer` in 1bcoder empirically measures max safe context per model/hardware; case study on chat.py's exception-swallowing convention in `_stream_chat`
- [LIMITED_CONTEXT / /ctx window concept](project_limited_context.md) — sliding-window context (RS/BM25/DP/TextRank mid-algorithms), reuses `_fts_rank` BM25 code, implement in 1bcoder first then centralize in vyrii engine.py — not yet implemented
- [Alkonost built](project_alkonost_build.md) — standalone Flask task-board viewer/editor for deepagent_task output, `C:\Project\alkonost\`, renamed from "alconost", 35 passing tests + real-data smoke test

## User Preferences
- Concise communication, no emojis
- Doctoral researcher — academic framing matters
- Ollama installed, uses qwen2.5-coder locally
- GPU: 6GB VRAM (important for model selection)
- [Full personal background](user_background.md) — origin story, career, PhD context, philosophy of code

## Collaboration Feedback
- [Explicit consent before destructive ops](feedback_explicit_consent_destructive.md) — пояснення/підтвердження фактів ≠ згода на дію; git rm/видалення — тільки після явного "так"
- [Show findings before editing](feedback_show_findings_before_edit.md) — report what existing code already handles before applying changes
- [1bcoder keybinding restrictions](feedback_1bcoder_keybindings.md) — Ctrl+S/Ctrl+Q intercepted by pyreadline3 on Windows; use text commands (/end, /save) instead
- [1bcoder default files location](feedback_1bcoder_defaults.md) — always edit `_bcoder_data/`, never `~/.1bcoder/` directly
- [1bcoder edit scope](feedback_1bcoder_edit_scope.md) — fix flows/procs only in `_bcoder_data\`, never in `~/.1bcoder/` even if that file is currently executing
- [Legacy code caution](feedback_legacy_code_caution.md) — every line was added for a reason; don't remove code to fix minor issues without explicit permission
- [Windows Terminal line dropping](feedback_windows_terminal_lines.md) — known quirk: loses banner/query/response lines at scroll boundary; not a 1bcoder bug, don't investigate unless asked
- [YAML over JSON](feedback_yaml_over_json.md) — standing preference for new config/metadata sidecar files; default to yaml.safe_dump/safe_load unless consumer requires JSON
- [Notepad indentation gotcha](feedback_notepad_indentation_gotcha.md) — verify raw bytes + depth math before assuming a deepagent_task/alkonost formatting bug; plain Windows Notepad can misrender correct files
- [Research vs hardware constraints](feedback_research_vs_hardware_constraints.md) — don't filter exp4/exp5 model choices by 6GB VRAM; that limit applies to production (ragmcp) only, not research design
