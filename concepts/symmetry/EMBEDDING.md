# Symmetry in Embedding Space — Calibration Invariance

---

## 1. Mathematical Grounding

The same concept appears under different names across code layers. Embedding must be invariant to naming.

| Theorem | Application to Code |
|---|---|
| **Alpha-equivalence** (λ-calculus) | Variable rename preserves semantics. Refactoring = gauge transformation. Valid refactoring = element of the symmetry group. |
| **Gauge invariance** (physics) | Naming convention = gauge. `SALARY1`, `main_salary`, `salary` = same concept under different gauges. |
| **Procrustes problem** | Given aligned pairs (X_src, X_tgt), find rotation W minimizing `‖WX - Y‖_F` subject to `W^T W = I`. Solved via SVD: `W = V U^T` where `X^T Y = U Σ V^T`. |
| **Curry-Howard** | Type = proposition, program = proof. Type symmetry = logical consistency. Type error = broken symmetry. |
| **Myhill-Nerode** | Minimal DFA = canonical symmetric form. Dead states (unreachable + non-accepting) = dead code. |
| **Dyck language** | Balanced brackets = free group with generators + inverses. Parser = Dyck membership check. Unbalanced = symmetry violation → syntax error. |
| **Galois connection** (Cousot) | Abstract interpretation = formal symmetry between concrete and abstract semantics. Static analyzer = Galois connection. |

---

## 2. Transformation Types and IRR

Full table in `METRICS.md §6`. Summary:

```
Rename       IRR=1.0  → rotation matrix applies  → Noether framework applies
Aggregation  IRR=0.1  → projection, not rotation  → thermodynamic framework applies
```

**Aggregation is not broken symmetry — it is intentional information reduction.**

```
km_day₁, km_day₂, ..., km_day₃₆₅  →  annual_mileage = SUM(km_day)
```

- Cannot recover `km_day_183` from `annual_mileage`. Expected.
- Bug: `km_day` defined but not aggregated anywhere → ORPHAN.
- Bug: `annual_mileage` formula references undefined `km_day` → GHOST.
- Bug: formula aggregates `km_day` but not `engine_hours` when both are needed → INCOMPLETE_AGGREGATE.

New map edge type needed: `aggregate:` (directed, N:1, lossy). Distinct from `call:` (directed, 1:1 or 1:N, lossless reference).

---

## 3. Three Approaches to Embedding Invariance

**Problem**: `embed(SALARY1) ≠ embed(main_salary) ≠ embed(salary)` with naive embedding.

### Approach A — Tokenization (simplest, no model needed)

Split identifiers into tokens:
```
SALARY1      → ["salary"]              strip digits, lowercase
main_salary  → ["main", "salary"]      split on _
salaryAmount → ["salary", "amount"]    split camelCase
```

Embed token list. Common tokens → automatic similarity.

**Limitation**: `main_salary`, `tax_salary`, `max_salary` all → `["*", "salary"]`. Cannot distinguish.

### Approach B — Context Window (resolves ambiguity)

Embed identifier + surrounding code context:
```
"SALARY1 DECIMAL NOT NULL"                       → DB layer, numeric, required
"main_salary: float  # User's base salary"       → model layer, float, user domain
"def send(user_id: int, salary: Decimal)"        → service layer, parameter
```

Same concept, similar context → similar embedding even with different names.
Different concepts with same token (`Index` in project vs `pandas.Index`) → different context → different embedding.

**Practical**: sufficient for cross-layer fuzzy search. What exp3 RAG already does for tasks.

### Approach C — Rotation Matrices (cross-layer alignment)

Learn W_db→model such that `W @ embed(SALARY1) ≈ embed(main_salary)`.

```python
# Training: aligned pairs (X_db, X_model)
# Source: ORM field mappings, git co-changes, naming convention extraction
X = stack([embed(src) for src, _ in pairs])  # shape (N, 384)
Y = stack([embed(tgt) for _, tgt in pairs])  # shape (N, 384)

U, S, Vt = np.linalg.svd(Y.T @ X)
W = U @ Vt  # optimal rotation matrix, shape (384, 384)
```

**Application** (cross-layer concept tracing):
```
embed(SALARY1)          → search DB layer     → SALARY1 confirmed
W_db→model @ embed(...)  → search MODEL layer  → main_salary found
W_model→svc @ embed(...)  → search SERVICE layer → salary parameter found
W_svc→api @ embed(...)   → search API layer    → salaryAmount in JSON found
```

If search returns similarity < threshold at any step → concept lost between layers → structural bug.

**Training data sources**:
1. ORM field declarations: `Column("SALARY1") → salary` in SQLAlchemy
2. Git co-changes: files changed in same commit → implied correspondence
3. Naming convention normalization: `SALARY1` → strip digits → `salary` → match

Naming normalization alone gives ~70% aligned pairs without any ML.

---

## 4. Cross-Layer Concept Tracking

Given rotation matrices W_db→model, W_model→svc, etc.:

```
Step 1: embed(source_artifact)
Step 2: apply W_src→tgt
Step 3: cosine search in target layer index
Step 4: if similarity > threshold → concept found → continue
         if similarity < threshold → concept lost → structural alert
```

**Detection criterion**: concept disappears between layers → either:
- File at target layer is missing (schema drift, missing migration)
- Naming diverged beyond rotation correction (deliberate renaming without updating all layers)

This is the embedding analog of ORPHAN detection for cross-layer relationships.

---

## 5. What Already Exists vs What to Build

| Component | Status | Location |
|---|---|---|
| Tokenization (camelCase/snake split) | ✅ exists | exp5, Word2Vec training |
| Context window embedding | ✅ exists | exp3 (sentence-transformers) |
| Cross-layer aligned pairs | ❌ not built | needs ORM parser or git co-change extractor |
| Rotation matrices W | ❌ not built | ~20 lines numpy after pairs exist |
| Concept flow trace | ❌ not built | ~50 lines on top of rotation matrices |
