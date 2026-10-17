# SYMMETRY_INDEX

**Scope**: Symmetry as a formal framework for code navigation, structural analysis, and retrieval.
**Created**: 2026-03-08
**Status**: Concepts confirmed, implementations planned.

---

## 1. Core Principle

Every software artifact has a symmetric counterpart. Broken symmetry = bug class. Measurable symmetry = navigable structure.

| Symmetry Pair | Broken → | Bug Class |
|---|---|---|
| `def` ↔ `call` | call without def | NameError / linker error |
| `def` ↔ `call` (arity) | argument count mismatch | TypeError |
| `interface` ↔ `implementor` | missing method | AbstractMethodError |
| `DB field` ↔ `model field` | schema drift | AttributeError |
| `open` ↔ `close` | unclosed resource | ResourceLeak |
| `lock` ↔ `unlock` | deadlock | ConcurrencyError |
| `serialize` ↔ `deserialize` | version mismatch | DataCorruption |
| `import X` ↔ `X defined in target` | phantom import | ImportError |
| `defines X` ↔ `X called anywhere` | dead code | TechnicalDebt |

Full table: `symmetry/METRICS.md §1`.

---

## 2. Metrics

| Metric | Formula | Scope | Detail |
|---|---|---|---|
| `ASYMMETRY_SCORE` | `orphans / total_defines` ∈ [0,1] | project-wide | `symmetry/METRICS.md §2` |
| `ΔASYMMETRY` | `COUNT(after) - COUNT(before)` per edit | per commit | `symmetry/METRICS.md §3` |
| `HARMONY@K` | `mean(intra_fraction)` top-K by out-degree | project-wide | `symmetry/METRICS.md §4` |
| `SYMMETRY@K` | fraction of BFS-depth-K nodes with matching define | per `/map trace` | `symmetry/METRICS.md §5` |
| `IRR` | Information Retention Rate ∈ [0,1] | per edge type | `symmetry/EMBEDDING.md §2` |

**Key finding**: Use `ΔASYMMETRY`, not absolute `ASYMMETRY_SCORE`. Stable orphans → `ΔASYMMETRY=0` → no alert. New orphans → `ΔASYMMETRY>0` → alert. Solves 90% false-positive problem (SonarQube pattern).

`ΔASYMMETRY` operationalizes `SDR` (Structural Drift Rate) from `INDEX.md §6b` at per-edit resolution.

**2×2 operational matrix:**

```
                    HARMONY HIGH     HARMONY LOW
ΔASYMMETRY < 0     HEALING          CLEANUP NEEDED
ΔASYMMETRY = 0     EVOLUTION        STAGNATION
ΔASYMMETRY > 0     DISRUPTION       DECAY
```

---

## 3. Transformation Types

Not all identifier relationships are symmetric. Classification determines which metrics and tools apply.

| Type | Example | Invertible | IRR | Tool |
|---|---|---|---|---|
| Rename | `SALARY1` → `salary` | Yes | ~1.0 | Rotation matrix |
| Layer translation | `salary` → `send(salary)` | Approximate | ~0.9 | Rotation matrix |
| Computation (1:1) | `salary` → `after_tax` | Partial | ~0.7 | Additive composition |
| Aggregation (N:1) | `km_day[]` → `annual_mileage` | No | ~0.1 | Explicit `aggregate:` edge |
| Complex fold (N:1) | `km + hours` → `avg_speed` | No | ~0.0 | Formula tracking |

Noether theorem applies to types 1-2 only. Types 4-5 follow information-theoretic rules (entropy increases by design — not a bug). Detail: `symmetry/EMBEDDING.md §2`.

---

## 4. Retrieval Application

`SYMMETRY@K` as retrieval metric: of top-K files returned by semantic search, how many have their symmetric partner also in top-K.

**High SYMMETRY@K** = search returned a complete change cluster (all layers present).
**Low SYMMETRY@K** = search returned isolated files, likely missing related files.

**Hypothesis generation pipeline** (solves semantic search mismatch):

```
task_text → 1B model → [unfixed_code_h1, h2, h3, h4]
                              ↓ embed each (bge-small-en)
                        search RAG × N
                              ↓ merge by frequency
                        ranked candidates
```

**Tautology criterion**: `embed(AI_hypothesis) ≈ embed(real_file_chunk)` → found target.

The 1B model generates what UNFIXED code looks like, not the fix. Detail: `symmetry/RETRIEVAL.md`.

---

## 5. Object Passport

A per-file technical vocabulary built overnight by YES/NO classification via 1B model.

**Questions answered per file (~9 total):**

| Key | Question |
|---|---|
| `ops:div` | Does this code contain arithmetic division? |
| `ops:db` | Does this code execute SQL or connect to DB? |
| `ops:file` | Does this code read or write files? |
| `ops:http` | Does this code make HTTP requests? |
| `flags:deprecated` | Does this code contain @Deprecated usage? |
| `layer:svc` | Is this a service class? |
| `layer:repo` | Is this a repository or DAO? |
| `layer:ctrl` | Is this a controller or API endpoint? |

Result stored as extended `map.txt` attributes → searchable via `/map find \flags:deprecated`.

**Performance**: ~4s per file per question (gemma3:1b, tested). Top-500 files × 9 questions = ~5h overnight.

Detail: `symmetry/PASSPORT.md`.

---

## 6. Tools for 1bcoder

### Tool 1: `/map sym` *(planned — map_query.py)*

**Solves**: Detect structural health of project without reading files.

```
/map sym
```

Output: `ASYMMETRY_SCORE`, list of top orphans, `HARMONY@K`.

### Tool 2: `/map idiff` enhanced *(planned — map_query.py)*

**Solves**: Detect structural degradation after AI edits.

```
/map idiff
```

Current output + `ΔASYMMETRY`, new orphans list, healed orphans list.

### Tool 3: `/map trace` annotated *(planned — map_query.py)*

**Solves**: Find where in a call chain symmetry breaks.

```
/map trace validate_token
```

Each hop annotated: `[sym:OK]`, `[ORPHAN]`, `[GHOST]`, `[cross⚠]`.

### Tool 4: `search.py` *(planned — new CLI)*

**Solves**: Find relevant files when semantic search returns wrong results (doc files instead of code).

```bash
python search.py "coverage ignored for pull requests" --domain java --n 4 --top 5
```

Output: ranked `file:function(ln:N)` list. Use via `/run` in 1bcoder.

### Tool 5: `ask.py` *(planned — new CLI)*

**Solves**: Level-2 filter on 40-file candidate list. YES/NO per file, language-agnostic.

```bash
python ask.py CoverageComputer.java "does this skip processing for pull requests?"
# → YES  (4s)
```

Use via `/run` in 1bcoder after `search.py` narrows candidates.

### Tool 6: `index_passport.py` *(planned — batch overnight)*

**Solves**: Build technical vocabulary for entire project, enabling `/map find \ops:db`.

```bash
python index_passport.py --files top500.txt --append .1bcoder/map.txt
```

Runs once (overnight). Updates map.txt with `ops:`, `layer:`, `flags:` attributes.

---

## 7. Cascade Architecture

```
Level 1: embedding search   (0.01s/file)  12,532 → 40 files   [ragmcp search_files]
Level 2: ask.py YES/NO      (4s/file)     40     → 5-10 files  [Tool 5]
Level 3: /read + /map trace (manual)      5-10   → 1-2 files   [1bcoder built-in]
```

Each level uses output of previous as input. Total cost: <5 min for full pipeline.

---

## 8. Mathematical Grounding

| Theorem | Application |
|---|---|
| Curry-Howard correspondence | Type = proposition, program = proof. Type errors = failed symmetry. |
| Dyck language | Balanced brackets = free group. Parser = Dyck membership check. |
| Galois connection (Cousot) | Abstract interpretation = formal symmetry between concrete/abstract semantics. |
| Alpha-equivalence | Variable rename invariance = gauge invariance. Refactoring = gauge transformation. |
| Myhill-Nerode | Minimal DFA = canonical symmetric form. Dead states = dead code. |
| Procrustes problem | Cross-layer embedding alignment. SVD gives optimal rotation matrix W. |

Detail: `symmetry/EMBEDDING.md §1`.

---

## 9. Sub-documents

| File | Contents |
|---|---|
| `symmetry/METRICS.md` | Formal definitions, formulas, examples for all 5 metrics |
| `symmetry/EMBEDDING.md` | Calibration invariance, rotation matrices, IRR, aggregation |
| `symmetry/RETRIEVAL.md` | SYMMETRY@K for retrieval, hypothesis generation, tautology criterion |
| `symmetry/PASSPORT.md` | Object Passport, YES/NO classification, map.txt integration |
| `symmetry/JEV.md` | Passport as a jev rubric; symmetric questionnaire (same questions of the task and of the code); jev in the cascade, domain hint, aligned pairs for rotation matrices |

**Related existing documents:**
- `INDEX.md §6b` — SDR, OCS (ΔASYMMETRY operationalizes SDR)
- `COMPOSITIONAL_CODE_EMBEDDINGS.md` — additive/multiplicative composition (applies to IRR ~0.7 cases)
- `KEYWORD_INDEXING.md §Domain Vocabulary` — two-vocabulary problem (technical vs domain)
- `1BCODER.md §2.5` — `/map` system as OKG seed

---

**Version**: 1.0
**Next**: Implement `ΔASYMMETRY` output in `/map idiff` (map_query.py ~30 lines).
