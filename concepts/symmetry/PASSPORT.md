# Object Passport — Technical Vocabulary via 1B Classification

---

## 1. Problem

Semantic search retrieves files by task-text similarity. This fails when:
1. Task uses domain vocabulary unknown to the model (PPCon, EBU, PnP)
2. Bug is syntactic (division by zero, deprecated annotation) not semantic
3. Required file is infrastructure (repo, config) with low task-text similarity

**Solution**: pre-compute a technical vocabulary for each file via YES/NO classification.

---

## 2. YES/NO Classification — Verified Properties

**Tested**: gemma3:1b, 4s/file, temperature=0.

```python
PASSPORT_QUESTIONS = {
    # operations
    "ops:div":         "Does this code contain arithmetic division (/ as operator)? YES/NO",
    "ops:db":          "Does this code execute SQL or connect to a database? YES/NO",
    "ops:file":        "Does this code read or write to files? YES/NO",
    "ops:http":        "Does this code make HTTP or network requests? YES/NO",
    # state
    "flags:deprecated":"Does this code contain @Deprecated annotation or deprecated usage? YES/NO",
    "flags:nullable":  "Does this code handle null values or Optional fields? YES/NO",
    # layer
    "layer:svc":       "Is this a service class with business logic (not UI, not DB)? YES/NO",
    "layer:repo":      "Is this a repository, DAO, or data access class? YES/NO",
    "layer:ctrl":      "Is this a controller, handler, or API endpoint class? YES/NO",
}
```

**Verified behavior**:
- `int result = total / count;` → ops:div = YES
- `String path = "/usr/local/bin"; // comment with /` → ops:div = NO
- Model correctly distinguishes `/` as operator vs path vs comment (context-based, not token-based).

---

## 3. Runtime Cost

```
4s per question per file

Top-500 files × 9 questions = 4,500 calls = 5h (overnight, sequential)
Top-500 files × 9 questions = ~35 min (parallel, 8 workers)

Full 12,532 files × 9 questions = 125h sequential → not feasible
Strategy: index top-500 by task frequency first, expand incrementally.
```

**Priority**: files that appear most often across historical tasks (from ragmcp task index).

---

## 4. map.txt Integration

Extended map.txt entry format (backward-compatible, new lines appended):

```
# Before passport
CoverageComputer.java
  defines : computeForBranch(ln:45), computeLineCoverage(ln:78)
  links  → BranchRepository.java (import:BranchRepository, call:findById)

# After passport (new lines added)
CoverageComputer.java
  defines : computeForBranch(ln:45), computeLineCoverage(ln:78)
  links  → BranchRepository.java (import:BranchRepository, call:findById)
  ops    : db, div
  layer  : svc
  flags  : nullable
```

**Searchable immediately** via existing `/map find`:
```
/map find \ops:div              → all files with division
/map find \flags:deprecated     → all files with deprecated usage
/map find \layer:svc \ops:db   → service classes that access DB
/map find \layer:repo \ops:div  → repos with arithmetic (potential DivisionByZero)
```

No changes to `map_query.py` required. The `\term` syntax already searches child lines.

---

## 5. Chunk-Level vs File-Level

Current ragmcp: one embedding per file (average of all chunks).
→ Good for: task → file retrieval.
→ Bad for: finding a specific code pattern (e.g., division) inside a large file.

**For pattern search**: chunk-level indexing using natural boundaries from map_index.py.

```python
# map_index.py already extracts:
# CoverageComputer.java: defines: computeForBranch(ln:45), computeLineCoverage(ln:78)

# Use these line numbers as chunk boundaries:
chunks = [
    {"file": "CoverageComputer.java", "func": "computeForBranch", "lines": (45, 77)},
    {"file": "CoverageComputer.java", "func": "computeLineCoverage", "lines": (78, EOF)},
]

# Index each chunk separately in pgvector:
for chunk in chunks:
    code = read_lines(chunk["file"], chunk["lines"])
    vec = embed(code)
    pg.insert(file=chunk["file"], func=chunk["func"],
              start_line=chunk["lines"][0], embedding=vec)
```

Result: search returns `CoverageComputer.java:computeLineCoverage(ln:78)` not just `CoverageComputer.java`.

---

## 6. Exception Tracking

Exceptions are a valid passport dimension. They indicate what external resources a file touches.

| Exception Class | Indicates | Passport Key |
|---|---|---|
| `ArithmeticException`, `ZeroDivisionError` | arithmetic division | `ops:div` |
| `SQLException`, `PSQLException` | database access | `ops:db` |
| `FileNotFoundException`, `IOError` | file I/O | `ops:file` |
| `ConnectionException`, `SocketTimeoutException` | network | `ops:http` |
| `NullPointerException`, `AttributeError` | nullable data | `flags:nullable` |
| `DeprecationWarning` | deprecated usage | `flags:deprecated` |

**Two use cases for exception tracking**:
1. **Runtime traceback** (error already happened) → traceback is sufficient, no static analysis needed.
2. **Static search** (find where X could happen before it does) → passport `ops:div` → candidates.

For SIMARGL retrieval (task text → files to change), use case 2 applies. No traceback exists at task time.

---

## 7. Implementation: index_passport.py

```bash
# Run overnight
python index_passport.py \
    --files top500.txt \           # list of file paths to process
    --questions passport.json \    # question definitions
    --model gemma3:1b \
    --workers 8 \
    --append .1bcoder/map.txt      # add passport lines to existing map.txt

# Output appended to map.txt:
# CoverageComputer.java
#   ops    : db, div
#   layer  : svc
#   flags  : nullable
```

**File size**: ~80 lines Python.
**Dependencies**: `requests` (Ollama API), file I/O.
**Idempotent**: checks if passport lines already exist before re-scanning.

---

## 8. Two-Vocabulary Problem

See `KEYWORD_INDEXING.md §Domain Vocabulary`.

The Object Passport provides **technical vocabulary** (extractable automatically).
Domain vocabulary (PPCon, EBU, PnP) requires **human experience** (6+ months on a project).

Technical passport enables navigation for:
- New team members who don't know domain vocabulary
- 1B models with no domain context
- Cross-project search (same technical patterns, different domain names)

Technical passport does NOT replace domain vocabulary for experienced users. It provides an alternative path to the same files.
