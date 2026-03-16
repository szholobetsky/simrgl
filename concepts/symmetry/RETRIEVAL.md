# Symmetry in Retrieval — Hypothesis Generation

---

## 1. Problem

Semantic search (task text → file embeddings) fails when task language and code language diverge.

**Example** (TaskExample.txt):
```
Task: "RuleFinder was deprecated in 5.1, because loading all server side rules
       on scanner side was too costly. The deprecation is misleading."

search_files(task) → adding-coding-rules.md, python.md, webpack.config.js
                   → WRONG (documentation, not code)

Correct target: RuleFinder.java with @Deprecated annotation
```

Cause: `embed(task_text)` is in NL space. `embed(RuleFinder.java)` is in code space. Gap between spaces.

---

## 2. SYMMETRY@K as Retrieval Metric

Given top-K files returned by semantic search, measure how many have their symmetric counterpart also in top-K.

```
Known symmetry pairs: (TABLE, MODEL), (MODEL, SERVICE), (SERVICE, CONTROLLER), (DEF, CALL)

SYMMETRY@K = |{fᵢ ∈ top-K : ∃ fⱼ ∈ top-K such that (fᵢ, fⱼ) is a known pair}| / K
             ∈ [0, 1]
```

**High SYMMETRY@K**: retrieved files form a complete change cluster (all architectural layers present).
**Low SYMMETRY@K**: retrieved files are isolated — search likely missed related files.

Use as a retrieval quality signal. If `SYMMETRY@10 < 0.3` → trigger hypothesis search (§3).

---

## 3. Hypothesis Generation Pipeline

**Core idea**: Generate code that looks like the UNFIXED state, embed it, search code index.

1B model generates unfixed-code fragments (NOT the fix). Search finds files matching those fragments.

```
task_text
    ↓
gemma3:1b
"In a {domain} codebase, this bug exists: {task}
 Write {N} short code fragments showing BUGGY code BEFORE fix.
 Only code, no explanation."
    ↓
[hypothesis_1, hypothesis_2, hypothesis_3, hypothesis_4]
    ↓ embed each with same model as index (bge-small-en-v1.5)
[vec_1, vec_2, vec_3, vec_4]
    ↓ search_files(vec_i, top_k=5) × N
[hits_1, hits_2, hits_3, hits_4]
    ↓ merge, score = sum of similarity across hypotheses
{RuleFinder.java: 2.4, DefaultRuleFinder.java: 1.8, webpack.config.js: 0.3}
    ↓ rank by score
top candidates (cross-hypothesis agreement = higher confidence)
```

**Why unfixed code**: the file with the bug currently EXISTS in the index. A fragment describing how it looks NOW (with the bug) will match it. A fragment describing the fix will not match (the fix doesn't exist yet).

---

## 4. Tautology Criterion

Every retrieval task reduces to:

```
AI_hypothesis(form_k of unfixed code) ≈ REAL_code(file_j)
```

When `cosine(embed(hypothesis), embed(file_chunk)) > threshold` → file_j is the target.

The fix is the transformation that makes the tautology hold: `hypothesis = real_code` before and after.

---

## 5. Multi-Hypothesis Rationale

One task → multiple hypothesis forms because the same bug appears in different syntactic forms:

**Task**: "RuleFinder deprecated misleadingly"

| Hypothesis | Form | Target |
|---|---|---|
| `@Deprecated(since="5.1") public class RuleFinder` | Java annotation | RuleFinder.java |
| `deprecated = true // since 5.1` | boolean field | config.properties |
| `// @deprecated since 5.1` | comment | RuleFinder.java |
| `rulefinder.deprecated=true` | config key | sonar.properties |

Each hypothesis searches a different region of the code index. Union covers more ground than single query. File appearing in multiple searches scores highest.

---

## 6. Domain Hint Requirement

Tested: gemma3:1b without domain context → garbage output.
Tested: gemma3:1b with domain context ("Java coverage, branch type, isPullRequest") → usable output.

**Two-input hypothesis generation**:
```
input_1: task_text          (what to find)
input_2: domain_hint        (what kind of code to generate)
```

Source of `domain_hint`:
- Option A: user provides manually
- Option B: first call `search_modules(task)` → module name → domain_hint

---

## 7. Cascade Architecture

Three-level pipeline. Each level filters for the next.

```
Level 1  search_files(task_text)          0.01s/file   12,532 → 40 files
Level 2  search_files(hypothesis_i)       0.01s/file   40     → 5-10 files
         merged by cross-hypothesis score
Level 3  ask.py YES/NO per file           4s/file      5-10   → 1-3 files
Level 4  /read + /map trace               manual       1-3    → exact location
```

Level 1: existing ragmcp `search_files`.
Level 2: `search.py` (Tool 4 in SYMMETRY_INDEX.md).
Level 3: `ask.py` (Tool 5 in SYMMETRY_INDEX.md).
Level 4: 1bcoder built-in.

Total automated cost: Level 1 + 2 ~1s, Level 3 ~20-40s. Total <1 min.

---

## 8. Implementation: search.py

**File**: `1bcoder/search.py` or standalone CLI.
**Dependencies**: `requests`, `sentence_transformers`, `psycopg2` (existing ragmcp DB).
**Size**: ~60 lines.

```bash
# Usage
python search.py "coverage ignored for pull requests" --domain java --n 4 --top 5

# Output
2.847  sonar-scanner-engine/CoverageComputer.java:computeForBranch(ln:45)
2.103  sonar-ce/BranchAnalysis.java:collectMeasures(ln:112)
1.891  sonar-scanner-engine/AbstractCoverageIT.java:setUp(ln:23)

# In 1bcoder
> /run python search.py "coverage ignored for pull requests" --domain java
> /read sonar-scanner-engine/CoverageComputer.java 40-60
```

**Implementation notes**:
- Embed hypotheses with `BAAI/bge-small-en-v1.5` (same model as ragmcp index)
- Use existing pgvector DB from ragmcp
- Parse `[code_block]` delimiters from 1B output
- Score = sum of per-hypothesis cosine similarities for same file
