# DeepAgent_Code: Test-Driven Recursive Code Decomposition for Small Local Models

**Component**: 1bcoder `/flow deepagent_code`
**Date**: 2026-06-24
**Status**: Concept

---

## 1. Problem

Small local models (1B–8B parameters) cannot write non-trivial programs. They fail at algorithm synthesis — producing code that looks plausible but doesn't work. They can reproduce LeetCode solutions from memory but cannot design a real algorithm for a real task.

At the same time, these models can reliably do two things:
- Write 5–15 lines of code for a well-specified function with a clear signature
- Write a unit test for a function when given its signature and description

DeepAgent_Code exploits this asymmetry: a human (or a stronger model) handles architecture, small models fill in the leaves, tests maintain integrity.

---

## 2. Core Idea

Recursive decomposition of a programming task into a tree of small functions, where:
- **Top levels (0–1)**: designed by the human — function names, signatures, contracts
- **Leaf level**: implemented by small models — one function per file, 5–15 lines
- **Verification**: unit tests written by a separate model call, executed by ladder gate
- **Integration**: bottom-up — leaf tests first, then integration tests climbing up the tree

Each function is a separate file. Each test is a separate file. The model never sees the full program — only the function it's implementing, its signature, and the signatures of its neighbors.

---

## 3. Why Not Full Automation

### 3.1 The Calendar Problem

A university assignment: four students each implement one function — `print_week()`, `print_month()`, `print_season()`, `print_year()`. Each function works in isolation. Integration fails because:
- `print_week(start, end, start_from)` expects a weekday number — but who computes it?
- `print_month` calls `print_week` in a loop — but how does it know each week's starting day?
- `print_year` needs leap year logic — tables or Zeller's formula
- Month boundaries: the last week of July bleeds into August

The problem is not in the leaves — it's in the **contracts between them**. A small model cannot hold two function signatures in its head simultaneously and reason about their compatibility.

### 3.2 The Naming Problem

A model asked to "decompose read_budgets into sub-functions" will produce `step_1()`, `do_stuff()`, `process_data()`. Meaningful names like `connect_to_db()`, `check_business_rules()`, `transform_to_report()` require domain understanding that small models lack.

### 3.3 The Shared Function Problem

Real decomposition produces a DAG, not a tree:

```
read_budgets ──→ connect_to_db ←── create_budget
             ──→ transform_to_report ←──
             ──→ log_result ←──
             ──→ prepare_query            (unique)
             ──→ fetch_data               (unique)

create_budget ─→ check_if_not_exists      (unique)
              ─→ check_business_rules     (unique)
              ─→ run_insert_query         (unique)
              ─→ check_data_in_db         (unique)
```

Recognizing that `connect_to_db` is the same function used by two callers — and that its signature must satisfy both — is an architectural decision beyond a small model's capability.

### 3.4 Honest Conclusion

Levels 0–1 are human work. Level N (leaves) is model work. This is less impressive than "AI writes your whole program" but it actually works.

---

## 4. Architecture

### 4.1 File Structure

```
project/
  deepagent_code.yaml       ← project state (the "micro-Jira")
  main.py                   ← level 0: orchestrator
  read_budgets.py           ← level 1: calls leaf functions
  create_budget.py          ← level 1: calls leaf functions
  connect_to_db.py          ← leaf (shared)
  connect_to_db_test.py     ← unit test
  prepare_query.py          ← leaf (unique to read_budgets)
  prepare_query_test.py     ← unit test
  check_business_rules.py   ← leaf (unique to create_budget)
  check_business_rules_test.py
  ...
```

Each file is 5–15 lines. Each test is 10–30 lines. The model never sees more than one file at a time.

### 4.2 Project State File

```yaml
task: "Budget management CRUD"
lang: py
depth: 2
status: in_progress

functions:
  connect_to_db:
    level: 2
    used_by: [read_budgets, create_budget]
    status: passed          # skeleton | implemented | test_written | passed | failed
    file: connect_to_db.py
    test: connect_to_db_test.py
    test_runs: 2
    bugs_fixed: 1

  prepare_query:
    level: 2
    used_by: [read_budgets]
    status: failed
    file: prepare_query.py
    test: prepare_query_test.py
    test_runs: 3
    last_error: "TypeError: query.format() missing argument"
    bugs_fixed: 2
    bugs_remaining: 1

  read_budgets:
    level: 1
    used_by: [main]
    status: skeleton
    children: [connect_to_db, prepare_query, fetch_data, transform_to_report, log_result]
    integration_test: read_budgets_test.py
```

Supervisor reads this file, finds the deepest node with status `implemented` or `failed`, and acts on it.

### 4.3 Execution Order

1. **Shared leaves first** — `connect_to_db` blocks two branches, implement it first
2. **Unique leaves** — `prepare_query`, `check_business_rules`, etc.
3. **Integration tests bottom-up** — when all children of `read_budgets` are `passed`, write and run `read_budgets_test.py`
4. **Top-level integration** — when all level-1 functions pass, test `main.py`

### 4.4 Pipeline Per Leaf Function

```
Step 1: IMPLEMENT
  system_prompt: {lang_implement_prompt}
  user: "Write the body of connect_to_db(host: str, port: int) -> Connection.
         It should establish a PostgreSQL connection and return it.
         Use psycopg2. No error handling — caller handles exceptions."
  → connect_to_db.py

Step 2: TEST (separate model call)
  system_prompt: {lang_test_prompt}
  user: "Write a test for connect_to_db(host: str, port: int) -> Connection.
         Signature and description above. Test happy path and invalid host."
  → connect_to_db_test.py

Step 3: LADDER (deterministic — no model involved)
  run: pytest connect_to_db_test.py
  if pass  → status: passed, move to next function
  if fail  → status: failed, save traceback
  
Step 4: RETRY (if failed, max N times)
  system_prompt: {lang_implement_prompt}
  user: "Fix connect_to_db(). Error: {traceback}. Previous code: {code}"
  → connect_to_db.py (overwrite)
  → re-run Step 3
```

Key: implement and test are **separate calls**. A model writing code should not simultaneously think about tests. A different model can write the test — independent verification.

### 4.5 Output Cleanup

Model output often contains explanation text around code fences. Pipeline:

```
model output
    ↓
extract code fences → function.py
    ↓
everything outside code fences → function.md (if non-empty)
    ↓
function.md is context for retry: corner cases, notes, rationale
```

---

## 5. Language Support

DeepAgent_Code is language-parametric. The language determines:
- File extension and import syntax
- System prompt for implement and test steps
- Test runner command

```yaml
languages:
  py:
    ext: .py
    test_ext: _test.py
    test_cmd: "pytest {test_file} -v"
    implement_prompt: _bcoder_data/deepagent_code/py_implement.txt
    test_prompt: _bcoder_data/deepagent_code/py_test.txt

  js:
    ext: .js
    test_ext: .test.js
    test_cmd: "node --test {test_file}"
    implement_prompt: _bcoder_data/deepagent_code/js_implement.txt
    test_prompt: _bcoder_data/deepagent_code/js_test.txt

  go:
    ext: .go
    test_ext: _test.go
    test_cmd: "go test -run {test_name} -v"
    implement_prompt: _bcoder_data/deepagent_code/go_implement.txt
    test_prompt: _bcoder_data/deepagent_code/go_test.txt
```

Criterion: **one file = one test = one command**. Languages requiring a build system (Java/Maven, C/CMake) are not suitable.

Supported: Python, JavaScript, TypeScript, Go, PHP, Ruby, Groovy, Swift.
Not supported: Java, C, C++, Rust (strict compiler makes single-file tests impractical).

---

## 6. What This Is Not

### 6.1 Not a Replacement for a Programmer

The human designs the architecture (levels 0–1), names functions, defines signatures and contracts. The model fills in leaf implementations. This is a **power tool**, not an autonomous agent.

### 6.2 Not deepagent_md

deepagent_md expands text recursively and has no verification. A model can write nonsense about Shakespeare and no test will catch it. deepagent_code has ground truth — the test either passes or fails.

### 6.3 Not Guaranteed to Produce Working Software

Unit tests pass individually but integration can fail. The calendar problem is real. deepagent_code reduces manual coding effort but does not eliminate the need for human review of contracts and integration.

### 6.4 An Honest Simulation Detector

A practical observation: deepagent_code can generate a large volume of code with passing tests, produce impressive git commits, and make it look like real work happened. The yaml state file is the audit trail — it shows how many retries each function took, how many bugs were fixed, and whether integration tests exist at all. Without integration tests, the output is decorative.

---

## 7. Relation to External Supervision

This concept is a direct application of the External Supervision thesis (see `AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md`):

- **Model generates moves** — writes 5–15 lines of a leaf function
- **Automaton tracks the field** — yaml state file, dependency graph, test results
- **Ladder gate = structural verification** — test pass/fail is a milestone, not a model judgment
- **Human holds the plan** — levels 0–1 are human-authored architecture

The key insight from external supervision: small models cannot maintain multi-step task threads. deepagent_code eliminates the need — each model call is stateless, context is one function, verification is deterministic.

---

## 8. MVP: BFS Code Decomposition (deepagent_md adapted for code)

### 8.1 Concept

The MVP reuses the core architecture of deepagent_md — BFS tree expansion through N levels — but produces code files instead of markdown sections. The orchestrator handles numbering and tree traversal. The model handles naming and content.

```
/flow deepagent_code "Budget management CRUD for SQLite" --lang py --depth 2
```

### 8.2 Two Modes Per Level

| Levels 0..N-1 | Level N (leaves) |
|---|---|
| **Decompose**: model writes a skeleton — function signature + sub-function calls with comments | **Implement**: model writes the actual body of the function, 5–15 lines |

The `--depth` flag controls where decomposition stops and implementation begins. No heuristics in MVP — explicit depth only.

### 8.3 Decompose Step (levels 0..N-1)

Orchestrator sends to model:

```
System: You are decomposing a function into sub-functions.
        Write a Python function skeleton. The body should contain ONLY
        calls to sub-functions, each with a comment describing what it does.
        Do not implement the sub-functions — just call them.

User:   Function: read_budgets(db_path: str) -> list[dict]
        Task: Read all budgets from SQLite database and return as list of dicts.
```

Model returns:

```python
def read_budgets(db_path: str) -> list[dict]:
    conn = connect_to_db(db_path)       # establish SQLite connection
    query = prepare_query("budgets")     # build SELECT query for budgets table
    data = fetch_data(conn, query)       # execute query and return rows as dicts
    conn.close()
    return data
```

Orchestrator parses this, extracts 3 sub-function calls with names and descriptions, creates 3 child nodes in the tree.

### 8.4 Implement Step (level N)

Orchestrator sends to model:

```
System: {lang_implement_prompt}
        Write ONLY the function body. No explanations outside the code fence.

User:   Implement: connect_to_db(db_path: str) -> sqlite3.Connection
        Description: establish SQLite connection
        Parent context: called by read_budgets() which reads budget data
```

Model returns code. Orchestrator extracts code fence → saves as file. Text outside code fence → saved as `.md` file (corner cases, notes — useful for retry context).

### 8.5 File Naming

Orchestrator assigns tree position, model provides function name:

```
output/
  1-budget_system.py                    ← level 0, skeleton
  1-1-read_budgets.py                   ← level 1, skeleton
  1-2-create_budget.py                  ← level 1, skeleton
  1-3-delete_budget.py                  ← level 1, skeleton
  1-1-1-connect_to_db.py               ← level 2, implemented
  1-1-2-prepare_query.py               ← level 2, implemented
  1-1-3-fetch_data.py                   ← level 2, implemented
  1-2-1-connect_to_db.py               ← level 2, implemented (duplicate, OK in MVP)
  1-2-2-validate_budget.py             ← level 2, implemented
  1-2-3-insert_row.py                   ← level 2, implemented
  ...
```

Duplicate function names across branches (e.g., two `connect_to_db`) are allowed in MVP. Each branch is self-contained. Deduplication is a post-MVP concern.

### 8.6 Context Passed to Each Model Call

On every call, the model sees:
- **Its own node**: name, signature, description (from parent's comment)
- **Parent skeleton**: the decompose output that created this node — so the model knows calling context
- **Language system prompt**: conventions, imports, style for the target language
- **Previous attempt + error** (on retry): the code that failed + traceback

The model never sees the full tree. Maximum context per call: ~500 tokens.

### 8.7 Output Artifacts

After completion, the output directory contains:

```
output/
  index.md                              ← table of contents (generated by orchestrator)
  1-budget_system.py
  1-1-read_budgets.py
  ...
  1-1-3-fetch_data.py
  1-1-3-fetch_data.md                   ← model's notes (if any text outside code fence)
  joined.py                             ← all code joined into one file (optional)
```

#### index.md — The Tree Map

Generated automatically by the orchestrator after all levels are complete:

```markdown
# Budget management CRUD for SQLite

## Tree

- `1-budget_system.py` — top-level orchestrator
  - `1-1-read_budgets.py` — read all budgets from database
    - `1-1-1-connect_to_db.py` — establish SQLite connection
    - `1-1-2-prepare_query.py` — build SELECT query
    - `1-1-3-fetch_data.py` — execute query, return rows
  - `1-2-create_budget.py` — create new budget entry
    - `1-2-1-connect_to_db.py` — establish SQLite connection
    - `1-2-2-validate_budget.py` — check business rules
    - `1-2-3-insert_row.py` — INSERT into database
  - `1-3-delete_budget.py` — delete budget by ID
    - ...

## Stats

Total files: 12
Depth: 2
Leaf functions: 9
Language: Python
```

This serves the same purpose as the table of contents in deepagent_md — a navigable map of what was generated.

#### joined.py — Assembled Program

Optional `--join` flag produces a single file with all functions concatenated in dependency order (leaves first, root last):

```python
# === 1-1-1-connect_to_db.py ===
def connect_to_db(db_path: str):
    import sqlite3
    return sqlite3.connect(db_path)

# === 1-1-2-prepare_query.py ===
def prepare_query(table: str) -> str:
    return f"SELECT * FROM {table}"

# === 1-1-3-fetch_data.py ===
def fetch_data(conn, query: str) -> list[dict]:
    cursor = conn.execute(query)
    ...

# === 1-1-read_budgets.py ===
def read_budgets(db_path: str) -> list[dict]:
    conn = connect_to_db(db_path)
    query = prepare_query("budgets")
    data = fetch_data(conn, query)
    conn.close()
    return data

# === 1-budget_system.py ===
...
```

This file will likely **not run** without manual editing — signatures may not match, imports may be missing, shared functions are duplicated. But it gives the human a starting point: one file to read, fix, and refactor. Finding a bug in `1-2-3-insert_row.py` (15 lines) is easier than finding it at line 2453 of a monolith.

### 8.8 Comparison with deepagent_md

| Aspect | deepagent_md | deepagent_code (MVP) |
|---|---|---|
| Input | task description | task description + `--lang` |
| BFS expansion | sections of text | function skeletons |
| Leaf output | paragraph of text | 5–15 lines of code |
| Naming | `### 1.2.3 Section Title` | `1-2-3-function_name.py` |
| Numbering | markdown headings | orchestrator (filenames) |
| Composition | `--compose flat/linked/html` | `index.md` + optional `joined.py` |
| Verification | none | post-MVP: unit tests + ladder |
| Context per call | parent section text | parent skeleton + signature + description |
| Parallel | `--profile` workers expand sections | same — workers implement leaf functions |

### 8.9 What Exists vs What's Needed

| Component | Exists in 1bcoder | Needed for MVP |
|---|---|---|
| `/flow` infrastructure | yes | — |
| BFS tree expansion loop | yes (deepagent_md) | adapt: code extraction instead of markdown |
| `/save file.ext code` fence extraction | yes | reuse in orchestrator |
| `/parallel` for multiple workers | yes | reuse for leaf implementation |
| Language system prompts | no | new: `_bcoder_data/deepagent_code/{lang}_decompose.txt`, `{lang}_implement.txt` |
| Skeleton parser (extract sub-function calls from model output) | no | new: ~80 lines — regex for `name(args)  # comment` |
| `index.md` generator | no | new: ~40 lines — walk tree, write markdown |
| `joined.py` assembler | no | new: ~30 lines — concatenate in reverse level order |
| File naming logic (`1-2-3-name.py`) | no | new: ~20 lines |

Estimated: **~300 lines** of new Python flow code + 2 prompt templates per language.

The bulk of the work — BFS loop, parallel dispatch, progress display — is already implemented in deepagent_md.

---

## 9. Post-MVP Roadmap

### 9.1 Unit Tests (Phase 2)

After MVP produces files, add test generation and ladder verification:
- Separate model call per leaf: "write test for this function"
- `pytest {test_file}` via ladder gate
- Retry loop on failure (max 3, with traceback in context)
- Status tracking in `deepagent_code.yaml`

### 9.2 Shared Function Detection (Phase 3)

When two branches generate functions with the same name:
- Detect duplicates by name after decompose step
- Prompt: "Two callers need `connect_to_db`. Caller A passes (db_path: str), caller B passes (host, port). Write one function that serves both."
- Single file, referenced by both branches

### 9.3 Integration Tests (Phase 4)

Bottom-up: when all children of a node pass their unit tests, generate an integration test for the parent. This is where the calendar problem surfaces — and where human review becomes mandatory.

### 9.4 Ensemble Implementation (Phase 4)

Use `/parallel` to generate N implementations of the same leaf function with different models. Run tests on all. Pick the one that passes. If multiple pass — pick shortest.

---

## 11. Context Enrichment via Library Routing

During the `--think` step, the model produces a plan that typically lists the libraries
and symbols it intends to use. Before the code generation step begins, deepagent_code
can intercept these mentions and automatically enrich the context with real documentation
or local code examples.

### The routing rule

For each library or symbol mentioned in the think output:

```
svitovyd find <symbol>
    → results found  →  internal code  →  RAG via simargl
    → empty result   →  external lib   →  DDG web search
```

The check is a single `svitovyd find` call. If the symbol appears in the local codebase
(as a file name, class name, or identifier), it is treated as internal and fed to the
RAG pipeline. If svitovyd returns nothing, the symbol is external and a web search is
performed instead.

This avoids maintaining any whitelist of public packages. The question is not
"is this on PyPI?" but simply "is this in our code?". Standard library modules
(`socket`, `datetime`, `os`) and popular packages (`requests`, `psutil`) will return
empty from svitovyd and go to DDG. A private class like `MyExceptionHandler` or
`ProjectDbConnector` will be found by svitovyd and go to RAG.

### Why this ordering matters

The enrichment happens before generation, not during. Each function implementation
receives pre-fetched context as additional input. The model does not decide when to
search — the orchestrator does, based on the think output alone.

### Practical limits

Web search results are unpredictable in length and relevance. A reasonable strategy
is to take only the first DDG result, extract the first code block found on the page,
and inject that. For RAG, a top-3 chunk retrieval from simargl is sufficient.

The total injected context per function should be kept under ~500 tokens to avoid
overwhelming small models (1B–4B). If multiple external libraries are found, search
them all but concatenate only the most relevant snippet per library.

---

## 10. Open Questions

1. **Skeleton parsing reliability**: can a small model consistently produce parseable skeletons (`name(args)  # comment`)? If not, need a more structured output format (YAML? JSON?).

2. **Depth 3+**: at depth 3 with branching factor 5, we get 125 leaf files. Is this practical? Probably yes for generation, but human review becomes infeasible. Maybe cap at depth 2 for real use, allow depth 3+ only for demonstration/benchmarking.

3. **Function name collisions**: `1-1-1-connect_to_db.py` and `1-2-1-connect_to_db.py` — in `joined.py` these produce two functions with the same name. The assembler needs to either deduplicate or rename (`connect_to_db_1`, `connect_to_db_2`). MVP: just keep duplicates, warn in `index.md`.

4. **Cross-language feasibility**: the decompose step is language-agnostic (function signatures are universal). The implement step is language-specific (imports, idioms, test framework). How many languages can we realistically support with quality prompts? Start with Python, add JS/Go if demand exists.
