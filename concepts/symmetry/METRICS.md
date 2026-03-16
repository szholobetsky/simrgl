# Symmetry Metrics — Formal Definitions

---

## 1. Symmetry Type Table (Noether Analog)

For each symmetry pair, one broken invariant → one bug class.

| Symmetry | Invariant | Violation Name | Bug Class |
|---|---|---|---|
| `def F` ↔ `call F` | def exists for every call | GHOST | NameError, linker error |
| `def F` ↔ `call F` | call exists for every def | ORPHAN | Dead code |
| `def F(n params)` ↔ `call F(n args)` | arity preserved | ARITY_MISMATCH | TypeError |
| `interface M` ↔ `class implements M` | all methods implemented | MISSING_IMPL | AbstractMethodError |
| `DB field X` ↔ `model field X` | schema = model | SCHEMA_DRIFT | AttributeError |
| `import X from F` ↔ `X defined in F` | import target exists | PHANTOM | ImportError |
| `open R` ↔ `close R` | every open has close | LEAK | ResourceLeak |
| `lock L` ↔ `unlock L` | every lock has unlock | DEADLOCK | ConcurrencyError |
| `@Deprecated` ↔ `migration path` | deprecated has alternative | MISLEADING_DEP | confusion, misuse |

**Note**: In `map_index.py`, `links →` are built only for names in `global_index` (project-defined names). External libraries (`import pandas`, `import java.util`) are not in `global_index` and do not generate ghost/phantom entries. False positives from external imports do not occur in the current scanner.

---

## 2. ASYMMETRY_SCORE

Global health ratio. No ranking required.

```
D = set of all defined identifiers in project
C = set of D where at least one internal caller exists

ASYMMETRY_SCORE = |D \ C| / |D|   ∈ [0, 1]

0 → all defines are called (perfect symmetry)
1 → no define has any caller (complete dead code)
```

**Source**: `map.txt` — `defines:` vs `links →` entries.

**Example**:
```
total defines: 147
orphans (no callers): 16
ASYMMETRY_SCORE = 16 / 147 = 0.109
```

---

## 3. ΔASYMMETRY

Per-edit delta. Primary alert metric. Replaces absolute counts.

```
ASYMMETRY_COUNT(t) = number of symmetry violations at snapshot t

ΔASYMMETRY(edit) = ASYMMETRY_COUNT(after) - ASYMMETRY_COUNT(before)
```

| Value | Meaning | Action |
|---|---|---|
| `< 0` | violations removed (healing) | log as improvement |
| `= 0` | no change in violation count | silent |
| `> 0` | new violations introduced (degradation) | alert |

**Relation to SDR** (`INDEX.md §6b`):
```
SDR = mean(ΔASYMMETRY) over last N edits
SDR ≤ 0.0 → target (maintain or improve)
```

**Why ΔASYMMETRY, not ASYMMETRY_SCORE**: Stable historical violations produce `ΔASYMMETRY = 0` → no alert. Only new violations alert. Eliminates alert fatigue (SonarQube 90% false positive problem).

**Source**: diff of two consecutive `map.txt` snapshots (`map.prev.txt` → `map.txt`).

---

## 4. HARMONY@K

Structural cohesion of top-K identifiers. Ranked list required.

```
Rank all identifiers by out-degree (number of links they participate in).
top(D, K) = top-K identifiers by out-degree.

L = all internal caller edges (caller_file, target_file, name)
L_intra = L where module(caller_file) == module(target_file)

HARMONY@K = |{(c,t,n) ∈ L_intra : n ∈ top(D,K)}| / |{(c,t,n) ∈ L : n ∈ top(D,K)}|
            ∈ [0, 1]
```

| Value | Meaning |
|---|---|
| `HARMONY@5 = 1.0` | top-5 most connected identifiers are called only within their module |
| `HARMONY@50 = 0.6` | 40% of calls to top-50 identifiers are cross-module |

**Module** = directory of the file (e.g., `auth/routes.py` → module `auth`).

---

## 5. SYMMETRY@K (in `/map trace`)

Per-trace metric. K = BFS depth.

```
For a trace starting from identifier X:
nodes_at_depth(K) = all nodes reachable from X in ≤ K BFS steps

SYMMETRY@K = |{n ∈ nodes_at_depth(K) : n has matching define in map}| / |nodes_at_depth(K)|
             ∈ [0, 1]
```

Degrades as K increases (deeper chain = more likely to hit undefined references).

**Example**:
```
trace: validate_token

K=1: auth/routes.py ← call:validate_token   [define found ✓]   SYMMETRY@1 = 1.0
K=2: main.py ← import:init                  [define found ✓]   SYMMETRY@2 = 1.0
K=3: deploy.sh ← run:main                   [NO define ✗]      SYMMETRY@3 = 0.67
K=4: ci.yml ← refs:deploy                   [NO define ✗]      SYMMETRY@4 = 0.50
```

---

## 6. IRR — Information Retention Rate

Per-edge-type classification. Determines which tools apply.

```
IRR(edge) ∈ [0, 1]
1.0 → lossless (rename, layer translation)
0.0 → complete information loss (complex aggregation)
```

| Edge Type | Example | IRR | Invertible | Metric Applicable |
|---|---|---|---|---|
| Rename | `SALARY1` → `salary` | ~1.0 | Yes | ASYMMETRY_SCORE |
| Layer translation | `salary` → `send(salary)` | ~0.9 | Approximate | ASYMMETRY_SCORE |
| Computation (1:1) | `salary` → `after_tax` | ~0.7 | Partial | ASYMMETRY_SCORE |
| Aggregation (N:1) | `km_day[]` → `annual_mileage` | ~0.1 | No | explicit `aggregate:` edge |
| Complex fold | `km + hours` → `avg_speed` | ~0.0 | No | formula tracking |

Noether theorem applies only to IRR ≥ 0.7. Aggregations (IRR < 0.3) follow thermodynamic rules: entropy increase is by design, not a bug. The bug is an orphaned source with no aggregate consumer.
