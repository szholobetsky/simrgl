# Plan: simargl Python Package — MCP Server + UI

## Context

`ragmcp/` contains working prototype code (MCP server, Gradio UI, vector backends) but it is not installable, has no pyproject.toml, mixes experiment code with production code, and duplicates `exp3/`. The goal is a clean, installable `simargl/` package that exposes:
- An MCP server callable from 1bcoder via `/mcp simargl find -t sort rank|freq <query>`
- A Gradio web UI
- CLI entry points
- `pip install simargl` — works out of the box, no PostgreSQL required

**Deployment decision**: Default backend is **numpy memmap + int8** — zero extra deps beyond numpy (already pulled by sentence-transformers). Vectors stored as int8 on disk, loaded on demand via memmap (OS manages what stays in RAM). Optional pgvector backend for research users who already have PostgreSQL (exp3 audience).

Key requirement: `-t sort rank` vs `-t sort freq` — two distinct ranking strategies for task-based search (rank = relevance-weighted, freq = frequency-weighted). This distinction matters because freq boosts universal files (pom.xml, build.gradle) while rank surfaces files specific to the most similar tasks.

## What to Reuse

| Source | What | Target |
|---|---|---|
| `exp3/vector_backends.py` | `PostgresBackend` — optional backend only | `simargl/backends/postgres_backend.py` |
| `exp3/config.py` | model registry dict (`MODELS`), `get_model_config()` | `simargl/config.py` |
| `exp3/utils.py` | `preprocess_text()`, `extract_file_path()`, `extract_module_path()` | `simargl/utils.py` |
| `ragmcp/mcp_server_dual.py` | MCP tool patterns, tool decorator style | reference only |
| `ragmcp/gradio_ui.py` | UI layout patterns | reference only |

Do NOT reuse `etl_pipeline.py` directly — tightly coupled to experiment runner. Write a thin `Indexer` class instead.
Default vector backend: **numpy memmap + int8**. PostgresBackend available as `simargl[postgres]` optional extra.

## Package Structure

```
simargl/                          ← repo root
├── pyproject.toml
├── simargl/                      ← package
│   ├── __init__.py
│   ├── config.py                 # model registry, defaults
│   ├── utils.py                  # copied from exp3/utils.py
│   ├── backends/
│   │   ├── __init__.py           # get_backend(type) factory
│   │   ├── numpy_backend.py      # DEFAULT — memmap + int8, zero extra deps
│   │   └── postgres_backend.py   # OPTIONAL — copied from exp3/vector_backends.py
│   ├── indexer.py                # index_files(), index_tasks()
│   ├── searcher.py               # search logic: files / tasks+rank / tasks+freq / aggregated
│   ├── mcp_server.py             # MCP tools: find, index_files, index_tasks, status
│   └── ui/
│       ├── __init__.py
│       ├── gradio_app.py         # Gradio web UI
│       └── cli.py                # CLI: simargl index / simargl serve / simargl ui
```

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "simargl"
version = "0.1.0"
description = "Task-to-code retrieval — MCP server and web UI"
requires-python = ">=3.10"
dependencies = [
    "sentence-transformers>=2.2",   # pulls numpy automatically
    "mcp>=1.0",
    "gradio>=4.0",
]

[project.optional-dependencies]
postgres = ["psycopg2-binary>=2.9", "pgvector>=0.2"]

[project.scripts]
simargl     = "simargl.ui.cli:main"
simargl-mcp = "simargl.mcp_server:main"
```

Data stored in `.simargl/` relative to working directory. No extra vector DB dependency.

## config.py

Model registry — only proven general-purpose models. Code-specific models (CodeBERT,
Code2Vec) excluded: exp2 proved domain mismatch. Qwen3-Embedding excluded until exp4 validates it.

Default model: **bge-small** — best RAM/quality tradeoff for daily-use tool.
bge-large available for research users who need max MAP.

```python
MODELS = {
    # DEFAULT — 384 dims, ~37MB vectors for 100k chunks (int8), MAP=0.34 on Sonar
    'bge-small':  {'name': 'BAAI/bge-small-en-v1.5',        'dim': 384},
    # Research — 1024 dims, ~100MB (int8), MAP=0.37 on Sonar
    'bge-large':  {'name': 'BAAI/bge-large-en-v1.5',        'dim': 1024},
}
DEFAULT_MODEL = 'bge-small'
```

# DB
PG_HOST = 'localhost'
PG_PORT = 5432
PG_DB   = 'semantic_vectors'
PG_USER = 'postgres'
PG_PASS = 'postgres'
PG_SCHEMA = 'simargl'             # separate schema from exp3's 'vectors'

DEFAULT_TOP_K = 10
```

## backends/numpy_backend.py

Zero extra deps. Vectors encoded as int8, stored via memmap. OS page cache manages RAM.

```
.simargl/
  {project_id}_files.int8        ← np.memmap, shape (N, 384), dtype int8
  {project_id}_files.meta.sqlite ← id, path, chunk_n, norm (float32 per vector)
  {project_id}_tasks.int8        ← np.memmap, shape (M, 384), dtype int8
  {project_id}_tasks.meta.sqlite ← id, task_name, norm
  {project_id}_task_files.sqlite ← task_name, file_path, module (no vectors)
```

**Write** (indexer calls this):
```python
# sentence-transformers encodes to float32, we quantize to int8
vectors_f32 = model.encode(texts, normalize_embeddings=True)  # already L2-normalized
norms = np.linalg.norm(vectors_f32, axis=1, keepdims=True)
vectors_int8 = (vectors_f32 * 127).clip(-127, 127).astype(np.int8)
# append to memmap file, store norms in sqlite
```

**Search** (cosine via int8 dot product):
```python
vectors = np.memmap(path, dtype='int8', mode='r', shape=(N, dim))
norms = load_norms_from_sqlite(...)                 # float32 array, tiny
q_int8 = (query_f32 * 127).clip(-127, 127).astype(np.int8)
scores = vectors.astype(np.float32) @ q_int8.astype(np.float32)
scores /= (norms * np.linalg.norm(q_int8))         # cosine normalize
top_idx = np.argpartition(scores, -top_k)[-top_k:]
```

RAM at search time: only accessed pages stay in RAM (OS page cache, ~4KB per page).
Disk size: 100k × 384 × 1B = **~37MB** for files index.

## Semantic Unit abstraction — tasks vs commits

Not all repositories have task tracking. The "semantic unit" concept generalises both:

| Repository type | Semantic unit | NL text to embed |
|---|---|---|
| Has Jira/GitHub tracker | Task | TITLE + DESCRIPTION |
| No tracker / NULL task_name | Commit | MESSAGE from RAWDATA |

Auto-detection in `index_units()`:
```python
coverage = SELECT COUNT(*) FROM RAWDATA WHERE TASK_NAME IS NOT NULL / total
if coverage > 0.5:
    mode = "tasks"   # embed TASK.TITLE + TASK.DESCRIPTION
else:
    mode = "commits" # embed RAWDATA.MESSAGE, group by SHA
```

Stored in `meta.json`: `{"unit_mode": "tasks"|"commits", ...}`

Both modes produce identical index structure — only the source query changes.
In WebUI/LLM output: label shown as "Similar task: SONAR-1234" or "Similar commit: abc123ef".

## indexer.py

```python
def index_files(path: str, glob: str = "**/*", model_key: str = DEFAULT_MODEL,
                project_id: str = "default", chunk_size: int = 400) -> dict:
    """Walk path, chunk text files, compute embeddings, store as files.int8.
    Skips binary files and hidden directories.
    Returns: {files_indexed, chunks_indexed}
    """

def index_units(db_path: str, model_key: str = DEFAULT_MODEL,
                project_id: str = "default", mode: str = "auto") -> dict:
    """Index semantic units from SQLite.
    mode="auto"    → detect: tasks if task coverage >50%, else commits
    mode="tasks"   → embed TASK.TITLE + TASK.DESCRIPTION
    mode="commits" → embed RAWDATA.MESSAGE grouped by SHA

    Writes:
      units.int8  — vectors (memmap)
      units.db    — (id, unit_id, unit_type, text_preview, norm)
      unit_files.db — (unit_id, file_path, module, sha, db_path)
        db_path: path to original SQLite — DIFF fetched live, never copied

    Returns: {units_indexed, mode_used}
    """
```

**Storage layout per project:**
```
.simargl/
  {project_id}/
    files.int8        ← np.memmap — file chunk vectors (N × dim, int8)
    files.db          ← SQLite: (id, path, chunk_n, norm)
    units.int8        ← np.memmap — task OR commit vectors (M × dim, int8)
    units.db          ← SQLite: (id, unit_id, unit_type, text_preview, norm)
    unit_files.db     ← SQLite: (unit_id, file_path, module, sha, db_path)
    meta.json         ← {model_key, unit_mode, db_path, indexed_at}
```

`unit_files.db` stores `db_path` — original SQLite path. DIFF fetched live at search time:
```sql
SELECT diff FROM RAWDATA WHERE task_name = ? AND path = ?   -- task mode
SELECT diff FROM RAWDATA WHERE sha = ? AND path = ?         -- commit mode
```

Vectors stay as flat files (memmap-friendly). All metadata and mappings in SQLite.

## searcher.py

```python
def search(query: str, mode: str, sort: str = "rank",
           top_n: int = 10, top_k: int = 20, top_m: int = 5,
           include_diff: bool = False,
           project_id: str = "default") -> dict:
    """
    mode="files":      embed query → cosine in files.int8 → top_n files
    mode="tasks":      embed query → cosine in units.int8 → top_k units
                       → JOIN unit_files.db
                       sort="rank": score(file) = max(unit_similarity for units containing file)
                       sort="freq": score(file) = count(units containing file)
                       → top_n files + top_m modules
    mode="aggregated": avg(top_k unit vectors) → cosine in files.int8
                       → top_n files + top_m modules

    include_diff=True: fetch DIFF from original SQLite for each result unit
                       (for WebUI display and LLM context)

    Returns:
    {
      "files": [{"path": ..., "score": ..., "module": ...}],
      "modules": [{"module": ..., "score": ...}],
      "units": [
        {
          "unit_id": "SONAR-12847",        # task name or commit SHA
          "unit_type": "task"|"commit",
          "text_preview": "Add buildString to...",
          "similarity": 0.94,
          "files": ["ProjectAnalysisService.java", ...],
          "diff": "+  buildString = ..."   # only if include_diff=True
        }
      ]
    }
    """
```

**rank vs freq:**
```python
# sort="rank": file inherits the highest similarity score among units that changed it
file_scores = {}
for unit_id, unit_score in top_k_units:
    for file_path in get_unit_files(unit_id):
        file_scores[file_path] = max(file_scores.get(file_path, 0), unit_score)

# sort="freq": file ranked by how many similar units changed it (pom.xml risk)
file_scores = Counter()
for unit_id, _ in top_k_units:
    for file_path in get_unit_files(unit_id):
        file_scores[file_path] += 1
```

## embedder.py — pluggable embedder factory

```python
def get_embedder(model_key: str = DEFAULT_MODEL) -> BaseEmbedder:
    """Factory — returns the right embedder for the model key.
    model_key examples:
      'bge-small'              → SentenceTransformerEmbedder('BAAI/bge-small-en-v1.5')
      'bge-large'              → SentenceTransformerEmbedder('BAAI/bge-large-en-v1.5')
      'ollama://nomic-embed'   → OllamaEmbedder('nomic-embed-text', host=...)
      'openai://text-emb-3-sm' → OpenAIEmbedder('text-embedding-3-small', api_key=...)
    """

class BaseEmbedder:
    dim: int
    def encode(self, texts: list[str]) -> np.ndarray: ...  # returns float32, normalized

class SentenceTransformerEmbedder(BaseEmbedder):
    # uses sentence-transformers, local GPU/CPU
    # encodes in batches, normalize_embeddings=True

class OllamaEmbedder(BaseEmbedder):
    # calls POST /api/embeddings on configured host
    # no local model download needed

class OpenAIEmbedder(BaseEmbedder):
    # calls POST /v1/embeddings
    # works with any OpenAI-compatible provider (LM Studio, etc.)
```

Model key stored in `{project_id}/meta.json` at index time — used at search time to re-embed the query with the same model.

## Combining modes — set operations

Any combination of `-t`, `-f`, `-a` can be joined with a set operator:

```
find "query" -t -f +        → union(task_results, file_results)
find "query" -t -f *        → intersection(task_results, file_results)
find "query" -t -a *        → intersection(task_results, aggregated_results)
find "query" -t -f -a *     → intersection of all three
find "query" -t -f -a +     → union of all three
```

`+` (union): broader recall — any file found by any mode  
`*` (intersection): higher precision — only files confirmed by all selected modes

Implementation: run each selected mode independently → apply set op on file lists → merge scores:
```python
results = [search(query, mode=m) for m in selected_modes]  # -t, -f, -a
file_sets = [set(r["files"]) for r in results]

if op == "+":
    combined = set.union(*file_sets)
    score = lambda f: max(r["scores"][f] for r in results if f in r["scores"])
if op == "*":
    combined = set.intersection(*file_sets)
    score = lambda f: prod(r["scores"][f] for r in results)  # multiply confirms each other
```

Intersection of all three (`-t -f -a *`) is the highest-confidence result:
files that are semantically similar to the query in code space (-f),
appeared in historically similar tasks (-t),
AND matched via aggregated task vectors (-a).

## mcp_server.py

Seven MCP tools. Flags `-f/-t/-a` map directly to `mode` parameter:

```python
@server.tool()
async def find(query: str,
               mode: str = "tasks",   # -t (tasks), -f (files), -a (aggregated)
               sort: str = "rank",    # rank | freq  — only for mode=tasks
               top_n: int = 10,       # top N files
               top_k: int = 20,       # top K tasks (intermediate, mode=tasks only)
               top_m: int = 5,        # top M modules
               project_id: str = "default") -> list[TextContent]:
    """
    -f (files):      embed query → cosine in files.int8 → top_n files
    -t (tasks):      embed query → cosine in tasks.int8 → top_k tasks
                     → JOIN task_files → rank|freq → top_n files + top_m modules
    -a (aggregated): avg(top_k task vectors) → cosine in files.int8
                     → top_n files + top_m modules
    """

@server.tool()
async def index_files(path: str, project_id: str = "default",
                      model_key: str = DEFAULT_MODEL) -> list[TextContent]:
    """Index code files. model_key stored in meta.json for consistent re-embedding."""

@server.tool()
async def index_tasks(db_path: str, project_id: str = "default",
                      model_key: str = DEFAULT_MODEL) -> list[TextContent]:
    """Index tasks from SQLite (TASK + RAWDATA tables)."""

@server.tool()
async def status(project_id: str = "default") -> list[TextContent]:
    """Show: file count, task count, model_key, index date."""

@server.tool()
async def embedding(text: str = "",
                    file: str = "",
                    project_id: str = "default") -> list[TextContent]:
    """Compute embedding for a text or file content.
    /mcp simargl embedding text -> vector1
    /mcp simargl embedding file: auth.py -> vector1
    Returns JSON array (float32). In 1bcoder: use -> varname to capture as {{vector1}}.
    Loads the model from project meta.json (same model used at index time).
    Source priority: text (inline) > file (read from disk).
    """

@server.tool()
async def distance(source1: str, source2: str,
                   project_id: str = "default") -> list[TextContent]:
    """Compute cosine similarity between two texts or pre-computed vectors.
    /mcp simargl distance file1.py file2.py
    /mcp simargl distance "add user auth" views.py
    /mcp simargl distance {{vector1}} {{vector2}}
    source can be:
      - a file path           → read file content → embed
      - inline text string    → embed directly
      - a JSON vector string  → parse as float array, skip embedding (from {{vector1}})
    Returns: {"similarity": 0.87, "source1_type": "text", "source2_type": "file"}
    """
```

## ui/cli.py

```
simargl index files <path> [--project sonar] [--model bge-large]
simargl index tasks <db.sqlite> [--project sonar]
simargl serve [--port 8765]          # start MCP server
simargl ui [--port 7860]             # start Gradio UI
simargl status [--project sonar]
```

## ui/gradio_app.py

```
Query: "add sonar.buildString to project_analyses search response"
Mode: [tasks ▼]  Sort: [rank ▼]  Project: [sonar ▼]  [Search]

── Files (top 10) ──────────────────────────────────────────
  0.94  ProjectAnalysisService.java
  0.91  AnalysisController.java
  0.87  BuildStringHandler.java

── Modules (top 5) ─────────────────────────────────────────
  0.94  server/analysis
  0.87  api/project

── Similar tasks ───────────────────────────────────────────
  [0.94] SONAR-12847 — "Add build string to analysis API"
         Files: ProjectAnalysisService.java (+12 -3), AnalysisController.java (+5 -1)
         [▶ Show diff for LLM]   ← copies diff to clipboard / injects into context

  [0.87] SONAR-11203 — "Extend project analysis response fields"
         Files: AnalysisController.java (+8 -2)
         [▶ Show diff for LLM]
```

"Show diff for LLM" button: displays full diff in a code box + copy button.
For commit mode: shows commit SHA, message, author, date instead of task ID.

Reuse layout patterns from `ragmcp/gradio_ui.py` but simplified.

## Files to Create

Progress legend: [ ] not started · [x] done · [~] stub/partial

### MVP (session 1)
- [x] `simargl/pyproject.toml`
- [x] `simargl/simargl/__init__.py`
- [x] `simargl/simargl/config.py`            — model registry, DEFAULT_MODEL='bge-small'
- [x] `simargl/simargl/embedder.py`          — SentenceTransformerEmbedder only; Ollama/OpenAI stubs
- [x] `simargl/simargl/utils.py`             — minimal (preprocess_text)
- [x] `simargl/simargl/backends/__init__.py` — get_backend() factory
- [x] `simargl/simargl/backends/numpy_backend.py`    — DEFAULT: memmap + int8
- [x] `simargl/simargl/indexer.py`           — index_files(), index_units() with auto-detect
- [x] `simargl/simargl/searcher.py`          — search() mode=tasks/files; aggregated + set ops deferred
- [x] `simargl/simargl/mcp_server.py`        — find, index_files, index_units, status, vacuum, embedding, distance
- [x] `simargl/simargl/mcp_server.py`        — HTTP/SSE transport (--http --port, starlette+uvicorn)
- [x] `simargl/simargl/indexer.py`           — incremental indexing (mtime, soft delete, --full flag)
- [x] `simargl/simargl/backends/numpy_backend.py` — soft delete, vacuum_files(), indexed_paths()
- [x] `simargl/simargl/ui/__init__.py`
- [~] `simargl/simargl/ui/cli.py`            — index + serve commands only

### Deferred (session 2+)
- [x] `simargl/simargl/embedder.py`          — OllamaEmbedder, OpenAICompatibleEmbedder
- [x] `simargl/simargl/searcher.py`          — mode=aggr (weighted centroid), mode aliases (file/task/aggr)
- [x] `simargl/simargl/backends/postgres_backend.py` — full simargl interface, HNSW index, soft delete via DELETE+VACUUM
- [x] `simargl/simargl/ui/gradio_app.py`     — Gradio web UI (query, mode/sort/project dropdowns, files+modules+units+diffs)

## Verification

```bash
cd simargl
pip install -e .                      # installs with chromadb, no PostgreSQL needed

# Index (creates .simargl/ in current directory)
simargl index tasks C:/path/to/sonar.db --project sonar
simargl index files C:/path/to/sonar-repo --project sonar

# Check
simargl status --project sonar

# Start MCP server
simargl-mcp --port 8765

# In 1bcoder (after adding simargl to MCP config):
/mcp simargl find -t sort rank add sonar.buildString to api/project_analyses/search
/mcp simargl find -t sort freq add sonar.buildString to api/project_analyses/search

# Research users with PostgreSQL (optional):
pip install -e ".[postgres]"
simargl index tasks sonar.db --project sonar --backend postgres
```
