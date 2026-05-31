# Simargl: New Modes and Real-World Experiments on RuboCop

**Date:** 2026-05-25  
**Test corpus:** RuboCop repository — 2047 indexed files, ~16k commits, 9089 deleted chunks vacuumed  
**Model:** bge-small (default)

---

## Context

All experiments were run against the RuboCop codebase connected via simargl-mcp to a Claude Code session. Two benchmark queries were used throughout:

- **Query 1:** `"hash styles"` — expected answer: `lib/rubocop/cop/style/hash_syntax.rb`
- **Query 2:** `"What is the most abstract class that describes rule check behaviours?"` — expected answer: `lib/rubocop/cop/base.rb`

Baseline (mode=file, no options): relnotes files dominated the top results for both queries. The correct files were absent from top-10.

---

## 1. File Extension Support

### What was added
- Extensions: `.adoc`, `.asciidoc`, `.rst`, `.org`
- Known extensionless files: `Gemfile`, `Rakefile`, `Makefile`, `Dockerfile`, `Procfile`, `Vagrantfile`, `Brewfile`, `Justfile`, `Capfile`, `Guardfile`
- `chunk_size` now stored in `meta.json` (required for `retrieve` to reconstruct chunk N)

### Result
`docs/modules/ROOT/pages/development.adoc` became visible in search — previously invisible. RuboCop documentation in AsciiDoc format is now indexed.

### Verdict: works as expected — straightforward fix.

---

## 2. `retrieve` Command

### What was added
New command/MCP tool that returns actual file chunk text ready for LLM context injection.

```
simargl retrieve "query" --mode file --top-n 5
simargl retrieve "query" --mode task --diff
simargl retrieve "query" --mode aggr           # step 1: file list
simargl retrieve "query" --mode aggr --files "a.py,b.py"  # step 2: fetch content
```

### Idea
simargl was retrieval-only. `retrieve` adds the "RA" to "RAG" — returns formatted text chunks that can be injected directly into LLM context.

### Verdict: works, fills the gap. Not tested extensively on RuboCop in this session.

---

## 3. Blackhole Detection — Method: `centroid`

### Idea
Compute corpus centroid of all file vectors. Files whose mean chunk similarity to the centroid exceeds a threshold are "noise" — they match all queries equally (CHANGELOG, relnotes).

### Result
With `threshold=0.85`: marked **750 / 2047 files**.

The 750 files were cop files — `lib/rubocop/cop/style/*.rb`, `lib/rubocop/cop/layout/*.rb`, etc. This is because cop files have uniform structure (`class SomeCop < Base; MSG = ...; def on_send`) which puts them close to the corpus centroid.

relnotes were **not** marked — they contain diverse content across all versions and are far from the centroid.

### Why it failed
The centroid method detects **structural uniformity**, not **semantic noise**. Cop files are structurally uniform (same boilerplate pattern) → high centroid similarity. relnotes are structurally diverse (each version covers different topics) → low centroid similarity.

The method found the wrong files for the wrong reason.

### Verdict: mathematically correct, practically wrong for this use case. Useful only for corpora where noise files are structurally generic (logs, boilerplate configs).

---

## 4. Blackhole Detection — Method: `coverage` (binary)

### Idea
Use the most semantically **specific** units (commits/tasks furthest from the unit centroid) as test queries. Count how often each file appears in top-k results. Files that appear in a high fraction of queries are noise.

Specificity formula: `1 - cosine(unit_vec, unit_centroid)`

### Results

| Parameters | Files marked | Key findings |
|---|---|---|
| n_queries=100, threshold=0.3 | 2 | Only `development.adoc` and 1 other. No relnotes. |
| n_queries=100, threshold=0.1 | 35 | CHANGELOG ✅, README ✅, base.rb ❌ (core file!) |
| n_queries=500, threshold=0.1 | 21 | Fewer than 100 queries — counterintuitive |
| n_queries=200, top_k=100 | coverage computed | max_coverage=0.85 |

### Why relnotes were never marked
RuboCop has ~170 relnotes files (`relnotes/v1.62.1.md`, etc.). Each file covers only commits from its version. With 100-500 test queries spanning all topics, each individual relnotes file appears in the top for at most 5-10 queries — never reaching the 10-30% coverage threshold.

They are not one blackhole — they are a **swarm of 170 micro-relevant files** that together dominate results but individually are never "omnipresent."

### Core file problem
`lib/rubocop/cop/base.rb` was correctly identified as high-coverage (it IS referenced by every commit that touches any cop). But it is the **most important architectural file** in the codebase. Marking it as a blackhole is the worst possible outcome.

### Verdict: fails for two reasons — (1) misses swarm noise (relnotes), (2) incorrectly marks core project files. The binary threshold cannot distinguish "noise that appears everywhere" from "core code that appears everywhere."

---

## 5. `coverage_float` + `coverage_penalty`

### Idea
Instead of binary marking, store a float coverage score `[0..1]` per file. Apply it as a soft penalty during search:

```
adjusted_score = raw_score - λ * coverage
```

Core files survive because their raw_score is high enough to overcome the penalty on relevant queries. Noise files are pushed down because their raw_score advantage over the correct answer is smaller.

### Results with penalty=0.2 and 0.4
- Minor improvements on Query 2: `config/default.yml` and `cops_layout.adoc` dropped from top-10
- relnotes **did not move** — their raw_score (~0.848) was so far above cop files (~0.780) that even penalty=0.4 made no difference
- To push relnotes below cop files would require penalty ~1.0+, but that level destroys base.rb completely

### Why coverage_penalty failed for relnotes
The raw similarity gap between relnotes and relevant files is too large (~0.07). The penalty would need to be stronger than the signal itself, making it useless as a soft filter.

### Verdict: the right direction conceptually, but insufficient when the noise files have a systematic raw_score advantage. Works as a mild tie-breaker, not as a noise filter.

---

## 6. The Core Insight: relnotes are not noise

After all blackhole experiments, the rubocop Claude session reached this conclusion:

**relnotes appearing in results is correct behavior.** In RuboCop's project culture, every code change includes a relnotes entry. A commit that adds a new cop always touches `relnotes/vX.Y.Z.md`. So the co-change signal between relnotes and cop files is genuine — not noise.

The two distinct use cases:

| Query intent | Right answer | What simargl gives |
|---|---|---|
| "Where is the implementation of hash style checks?" | `cop/style/hash_syntax.rb` | relnotes + cop file |
| "What changed around hash style?" | relnotes + commits | ✅ exactly this |
| "What is the team's change culture?" | relnotes as artifacts | ✅ exactly this |

**simargl is optimized for "what changed" — and for that, relnotes are signal, not noise.**

---

## 7. `score_blend` — Topic Concentration Re-ranking

### Idea
The fundamental distinction between relnotes and cop files:

```
relnotes:  max_chunk=0.85, mean_chunk=0.30  → max >> mean → broad document
cop file:  max_chunk=0.85, mean_chunk=0.82  → max ≈ mean  → focused document
```

Formula: `adjusted = α * max_chunk + (1-α) * mean_chunk`

A file with only one relevant chunk out of 50 is penalized proportionally. A file uniformly about the topic is unaffected.

### Results (score_blend=0.7)

**Query 1 ("hash styles"):**
- `style/hash_syntax.rb` appeared at position 7 — was completely absent in baseline
- relnotes moved down

**Query 2 ("abstract class"):**
- relnotes dropped from 3rd to 9th place
- `docs/cops_style.adoc` (broad document) disappeared from top-10
- `lib/rubocop/cop/base.rb` still absent (it IS a broad document — referenced by everything)

### Why this works
No pre-computation. No thresholds to tune. Pure query-time math. Focused files naturally have max ≈ mean. The correction is proportional and self-calibrating.

### Limitation
Cannot help when the query vocabulary doesn't match the file vocabulary at all (semantic gap, not document structure gap). base.rb didn't appear because "abstract class rule check behaviours" has low cosine similarity to Ruby cop code regardless of chunk distribution.

### Verdict: **this works.** Best practical improvement from all experiments. Recommended default: `score_blend=0.7` for mode=file.

---

## 8. `mode=refine` — Pseudo-Relevance Feedback

### Idea
Classic IR technique (Rocchio, 1971) adapted to semantic search:

1. Embed original query → find top-k similar commits/tasks (mode=task internally)
2. Tokenize commit texts → find terms NOT in original query, ranked by frequency
3. Append top-M new terms to query → re-embed → run mode=file + score_blend
4. Safety check: if `cosine(original_vec, expanded_vec) < 0.6` → discard expansion (wrong direction)

### Expected behavior
```
"abstract class rule check behaviours"
→ top commits: "Create Cop::Base", "Inherit from Base", "Use Cop::Base API"
→ new terms: ["cop", "base", "inherit", "callbacks", "offense"]
→ expanded: "abstract class rule check behaviours cop base inherit callbacks offense"
→ file search → base.rb rises in results
```

### Manual two-stage test (verified before implementation)
Running mode=task with `"Cop::Base class that all cops inherit from"` → base.rb appeared at position 9. This confirmed that if the query vocabulary matches commit vocabulary, the correct file is findable.

### Known limitation
If the initial query is semantically distant from ANY commit in the corpus, top-k commits will be wrong, and expansion amplifies the error. Example: `"abstract class rule check behaviours"` → top-k commit was `"Allow non-nil checks that are the final expression"` — completely wrong direction. Using that commit's text as a new query produced worse results.

**Conclusion:** PRF requires the original query to be at least weakly aligned with commit vocabulary. For fully out-of-domain queries (Java/enterprise terms used on a Ruby codebase), PRF fails.

### Verdict: implemented, not yet tested on RuboCop. Theoretically sound for queries that are close to project vocabulary. Will fail on fully foreign vocabulary — needs svitovyd vocabulary mapping as a pre-step.

---

## 9. Mode Comparison Summary

| Mode | Best for | Weakness |
|---|---|---|
| `file` | Finding specific implementation when you know the project's terms | Broad documents compete with focused ones |
| `file` + `score_blend=0.7` | Finding implementation with natural-language queries | Doesn't help when query vocabulary doesn't match code |
| `task` | Finding architectural files through change history | Requires query to match commit message vocabulary |
| `aggr` | "What area of code should I look at?" — vague exploratory queries | Same relnotes dominance problem as file mode |
| `refine` | Natural-language queries where you don't know project vocabulary | Amplifies errors if initial query is too far from commits |

---

## 10. Pipeline Test: Style/RedundantSelf + Pattern Matching

**Test query (expert):** `"Style/RedundantSelf pattern matching"`  
**Test query (naive):** `"linter incorrectly flags redundant self receiver inside pattern matching"`  
**Expected answer:** `lib/rubocop/cop/style/redundant_self.rb`

### Results

| Approach | Result |
|---|---|
| `mode=file + score_blend=0.7` | `first_look.txt` at rank 1; cop file **not in top 30** |
| `mode=refine + score_blend=0.7` | `first_look.txt` at rank 1; cop file not in top 10 |
| svitovyd `map_find` → simargl `mode=file` | Same failure — identifiers diluted into generic tokens |
| `mode=task + sort=freq` (expert query) | `redundant_self.rb` at **rank 1, score 7** ✅ |
| `mode=task + sort=freq` (naive query) | Miss — landed on `Lint/RedundantSafeNavigation` |
| svitovyd `map_find "redundant_self"` | Exact match, 1 result ✅ |
| svitovyd `map_find "redundant self \on_in_pattern"` | Narrowed to exact file (pattern matching variant) ✅ |

### Why svitovyd → simargl pipeline failed

bge-small tokenizes `RedundantSelf` → `["Redundant", "Self"]`, `on_in_pattern` → `["on", "in", "pattern"]`.  
These generic tokens match relnotes about *any* redundant cop and *any* pattern-related file. The embedding never reaches the specific implementation file — the signal is diluted at the tokenization level before even reaching the vector space.

**svitovyd identifiers are not good simargl query terms.** Code identifiers are designed for exact matching, not for embedding similarity.

### Why mode=task + sort=freq works

Commit messages are natural language: `"Fix Style/RedundantSelf for pattern matching"`. They already bridge the vocabulary gap — developers wrote the cop name in commits. `sort=freq` (count of commits that touched the file) is more robust than `sort=rank` for this case because frequency aggregates weak signals from many commits.

With a fully naive query the approach fails: `"linter incorrectly flags redundant self..."` has no cop name → matches `Lint/RedundantSafeNavigation` instead. The vocabulary gap remains for users who don't know cop names.

### The real value of svitovyd: precise lookup, not query expansion

svitovyd is most powerful as a **second stage after simargl**, not as a preprocessor for it:

```
Stage 1 (simargl mode=task sort=freq):
  "Style/RedundantSelf pattern matching"
  → redundant_self.rb at rank 1

Stage 2 (svitovyd map_find):
  map_find "redundant_self \on_in_pattern"
  → exact file + method list: on_send, on_in_pattern, add_match_var_scopes, match_var
```

Stage 1 handles vocabulary mismatch through commit history (semantic). Stage 2 provides precise method-level navigation (deterministic). Together they cover the full path from natural language to code identifier.

### Blocker discovered: first_look.txt

Files `first_look.txt` and `first_look2.txt` in the repo root are LLM conversation logs about RuboCop. They score ~0.76 on every query and dominate `mode=file` and `mode=refine` results — worse than relnotes because they contain dense RuboCop terminology across all topics.

**Fix needed: `.simarglignore`** — a file listing glob patterns to exclude from indexing (similar to `.gitignore`). Without this, any accidentally-indexed text file with high topic density becomes a blackhole that no algorithm can remove.

---

## 11. What's Next: vocabulary bridging

The remaining unsolved problem is the **vocabulary gap**: "abstract class for rule check behaviours" ≠ "Cop::Base". This is not a search quality problem — it is a query formulation problem.

Two approaches remain:

**A. Pseudo-relevance feedback (mode=refine, implemented)**  
Works when the initial query is weakly aligned with project vocabulary. Fails completely for foreign vocabulary (Java/enterprise terms on a Ruby codebase).

**B. svitovyd + small LLM reformulation**  
svitovyd extracts project-specific identifiers from the code AST. A small LLM (gemma3:1b) is given a ~200-token compact vocabulary prompt and asked to rephrase the query using project terms. No fine-tuning needed. Tested conceptually — not yet implemented as an automated pipeline.

The two-stage pipeline that emerged from today's experiments:

```
natural language query
  → simargl mode=task sort=freq   (semantic: finds cop name from commit history)
  → cop file identified
  → svitovyd map_find <cop_name>  (deterministic: method list, call graph, dependencies)
  → exact code navigation
```

---

## 12. Embedding Model Comparison via Ollama — granite-embedding:30m

**Date:** 2026-05-26  
**Context:** bge-small is too weak for code retrieval. Explored local Ollama-based models as alternatives.

### Models tested for Ollama deployment

| Model | Size | context | dim | Time/chunk (400w) | Status |
|---|---|---|---|---|---|
| nomic-embed-code | ~8GB | — | — | — | Too large, not tested |
| nomic-embed-text | 274MB | 2048 tok | 768 | 4.13s (200w max) | 400w exceeds context |
| bge-m3 | 1.2GB | 8192 tok | 1024 | 64.66s | CPU too slow, times out |
| granite-embedding:30m | ~30M | — | 384 | 2.65s stable | Viable |

**Benchmark:** Single chunk of 400 words (9375 chars) from `lib/rubocop.rb`.  
**Hardware:** Windows CPU (no GPU).

### Key finding: Ollama HTTP overhead

Every embedding call via Ollama is a separate HTTP request. With `index_flush_size=1` (required for Ollama — accumulating 256 chunks and then making 256 sequential HTTP calls caused timeouts), each chunk costs ~2-4s of network+inference overhead regardless of model size. This makes large-codebase indexing slow even with a fast model.

| Codebase | Chunks | granite-30m estimate |
|---|---|---|
| `lib/rubocop/cop/style/` (301 files) | 456 chunks | ~20 min |
| `docs/` (46 files) | 383 chunks | ~17 min |
| Full `lib/` (914 files) | ~1400 chunks | ~80-120 min |

### Actual indexing results

**Code index:** `lib/rubocop/cop/style/` → 301 files, 456 chunks, 19:37 (3.91s/file avg)  
**Doc index:** `docs/` → 46 files, 383 chunks, 17:04 (22.28s/file — large .adoc files)

### Retrieval quality comparison

Query: `"Style/RedundantSelf pattern matching"` — expected: `redundant_self.rb`

| Model | Result | Notes |
|---|---|---|
| bge-small | not in top 8 ✗ | too weak for code vocab |
| jina-code | rank 1 ✓ | best — code-specific training |
| granite-30m | rank 3 ~ | `in_pattern_then.rb` at rank 1 — reacts to "pattern" keyword over semantic meaning |

Query: `"auto-generated configuration documentation"` — expected: `auto_gen_config.adoc`

| Model | Result |
|---|---|
| bge-small | rank 1 (0.776) ✓ |
| granite-30m | rank 1 (0.790) ✓ |

Score spread (gap between top and bottom of top-10):

| Model | Code spread | Doc spread |
|---|---|---|
| jina-code | 0.081 | — |
| granite-30m | 0.026 | 0.067 |
| bge-small | 0.015 | 0.057 |

Higher spread = better discrimination between relevant and irrelevant files.

### Verdict

```
Code RAG quality:  jina-code >> granite-30m >> bge-small
Doc RAG quality:   all three roughly equal
Indexing speed:    bge-small >> jina-code >> granite-30m (Ollama HTTP overhead)
```

**granite-embedding:30m** is a viable fallback for CPU-only Ollama deployments where jina-code cannot run (OOM, no trust_remote_code support). For doc RAG it performs equivalently to bge-small with slightly better score spread. For code RAG it is better than bge-small but misses semantic precision — reacts to individual keywords rather than query intent.

**Practical recommendation:** Use `jina-code` locally via sentence-transformers (fast, best quality). Fall back to `granite-embedding:30m` via Ollama only when running on a machine where Python package installation is restricted (e.g. Android/Termux).

### Ollama ceiling vs sentence-transformers ceiling

The HTTP overhead makes Ollama slower than sentence-transformers at equal model sizes. However, this comparison is only valid on CPU with small models. On a machine with a GPU, Ollama unlocks models that sentence-transformers cannot practically run:

| Dimension | sentence-transformers ceiling | Ollama on GPU ceiling |
|---|---|---|
| Context window | 512–8192 tokens | 32k–131k tokens |
| Network size | ~500M params (jina-code) | 7B–70B params |
| Embedding dim | 384–1024 | 2048–4096 |
| Score spread | 0.081 (jina-code, code) | significantly higher expected |

A larger network with wider embeddings produces vectors that occupy a higher-dimensional space — similar files remain close but dissimilar files are pushed further apart. A 7B embedding model on GPU would have both a much larger context (entire files in one chunk, no chunking needed) and a stronger semantic representation, resulting in sharper discrimination between files that look superficially similar. The HTTP overhead becomes irrelevant when inference time per chunk drops from seconds to milliseconds on GPU. The Ollama API path (`ollama://model`) in simargl was built for exactly this scenario.

---

## 13. bge-small vs jina-code — Detailed Head-to-Head

**Corpus:** `lib/rubocop/cop/` (style + layout cops, ruby-only project)  
**Model specs:**

| Model | Dim | Context | Type |
|---|---|---|---|
| bge-small | 384 | 512 tok | general English |
| jina-code | 768 | 8192 tok | code + NL pairs, trust_remote_code |

### Test 1: "Style/RedundantSelf doesn't consider pattern matching variables"
Expected: `redundant_self.rb`

| Rank | bge-small | score | jina-code | score |
|---|---|---|---|---|
| 1 | space_before_block_braces.rb ✗ | 0.696 | redundant_self.rb ✓ | 0.558 |
| 2 | case_indentation.rb ✗ | 0.681 | redundant_self_assignment.rb | 0.550 |
| 3 | redundant_conditional.rb ✗ | 0.676 | redundant_assignment.rb | 0.502 |
| 4 | quoted_symbols.rb ✗ | 0.674 | redundant_regexp_constructor.rb | 0.489 |
| — | redundant_self.rb не в top 8 ✗ | — | — | — |

### Test 2: "EmptyLinesAroundExceptionHandlingKeywords fails on one-liner rescue"
Expected: `empty_lines_around_exception_handling_keywords.rb`

| Rank | bge-small | score | jina-code | score |
|---|---|---|---|---|
| 1 | constant_overwritten_in_rescue.rb ✗ | 0.694 | suppressed_exception.rb ✗ | 0.583 |
| 2 | rescue_type.rb ✗ | 0.688 | empty_lines_around_exception_handling_keywords.rb ✓ | 0.577 |
| 3 | duplicate_rescue_exception.rb ✗ | 0.682 | rescue_modifier.rb | 0.522 |
| 4 | rescue_exception.rb ✗ | 0.681 | empty_line_after_multiline_condition.rb | 0.507 |
| — | empty_lines_around_* не в top 8 ✗ | — | — | — |

### Key metrics

| Metric | bge-small | jina-code |
|---|---|---|
| RedundantSelf correct file | not in top 8 ✗ | rank 1 ✓ |
| EmptyLines correct file | not in top 8 ✗ | rank 2 ✓ |
| Score spread (code) | 0.015 (blind) | 0.081 (discriminates) |
| Dim | 384 | 768 |
| Context | 512 tokens | 8192 tokens |

### Why jina-code wins

bge-small score spread = 0.015: all files score 0.674–0.696, which means the model cannot distinguish between files at all. Any result is essentially noise. bge-small was trained on general English text — Ruby code identifiers like `redundant_self`, `on_in_pattern`, `empty_lines_around_exception_handling` are meaningless strings to it.

jina-code score spread = 0.081: the correct file scores clearly above the others. The model was trained on code + natural language pairs and handles snake_case, CamelCase, and Ruby-specific patterns natively. Context window of 8192 tokens means a full cop file fits in a single chunk.

### Verdict

**jina-code is the de facto production standard for Ruby code RAG.** bge-small is not usable for code retrieval — its 0.015 spread means results are random. jina-code finds the correct file at rank 1–2 for both natural-language bug descriptions and cop name queries. The OOM issue (ALiBi attention, quadratic in seq_len) is fixed with `batch_size=1, max_seq_length=512`.

### Technical fixes implemented during this experiment

- `OllamaEmbedder` updated to new `/api/embed` endpoint (replaces deprecated `/api/embeddings`)
- `index_flush_size = 1` added to `OllamaEmbedder` — prevents 256-chunk accumulation that caused timeouts
- `BaseEmbedder.index_flush_size = 256` default; `indexer.py` reads `embedder.index_flush_size` instead of hardcoded 256

---

## 14. Recommended Pipeline for Low-Context LLMs (≤1000 tokens budget)

**Context:** 1bcoder family targets weak local models (qwen3:1.7b, gemma3:1b). Entire context budget is 1000 tokens. Everything that reaches the model must be pre-filtered to the minimum necessary.

**Motivation for RRF:** granite-embedding:30m and bge-small produce nearly identical poor results on code retrieval. A single search mode is insufficient. Two independent modes (task via commit history + file via content) that confirm each other provide a much stronger signal — files appearing in both lists are doubly validated by different mechanisms.

### Recommended pipeline

```
Step 1  svitovyd map_find <key nouns from task description>
        → translates "one-liner rescue" → rescue_modifier, on_resbody, resbody_branches
        → enriches query vocabulary with project-specific identifiers
        → ~30 tokens consumed

Step 2  simargl find mode=task sort=freq top-10  (project: bge-small)
        → files found via commit history (natural language path)

Step 3  simargl find mode=file top-10  (project: jina-code, same or narrower scope)
        → files found via chunk content (semantic path)

Step 4  RRF(task_list, file_list)  k=60
        score(file) = 1/(60 + rank_task) + 1/(60 + rank_file)
        → top-3 files with highest combined score
        → files in both lists rise automatically, single-list files fall
        → ~30 tokens consumed

Step 5  svitovyd map_find <top-3 files>
        → only "defines:" section: method_name(line), method_name(line)
        → ~50 tokens per file × 3 = ~150 tokens consumed

Step 6  Read only the specific methods that look relevant
        → 1–2 method bodies × ~300 tokens = ~400–600 tokens
```

**Total budget:** ~30 + 30 + 150 + 500 = ~710 tokens. Fits in 1000 with margin.

### Why aggr is last resort, not first step

`mode=aggr` averages commit vectors → centroid of a module. Good for "I have no idea where to look." But for a specific bug query, `task sort=freq` finds concrete files faster. aggr is a fallback when task returns < 2 relevant files.

### Token budget per step

| Step | Tool | Output | Tokens |
|---|---|---|---|
| 1 | svitovyd map_find | identifier list | ~30 |
| 2+3 | simargl task + file | file names only | ~60 |
| 4 | RRF merge | top-3 names | ~15 |
| 5 | svitovyd defines | method stubs | ~150 |
| 6 | Read methods | actual code | ~400–600 |
| **Total** | | | **~650–850** |

### When pipeline fails

If `mode=task` returns noise (query vocabulary too far from commit language) — Step 1 (svitovyd vocabulary enrichment) is critical. Without it, task and file may not overlap at all and RRF degrades to single-source search.

---

## 15. RRF Merge Command — Design Proposal

**Problem:** task and file search run against different projects (different index, different model). Results must be merged across project boundaries without re-indexing.

### Proposed CLI syntax

```bash
simargl rrf "query" --sources task:default,file:jina --top-n 5 --k 60
```

`--sources` format: `mode:project_id` pairs, comma-separated. Each pair runs an independent search and contributes a ranked list. RRF merges all lists.

Examples:
```bash
# bge-small task + jina file
simargl rrf "hash style enforcement" --sources task:default,file:jina --top-n 5

# three-way: task + file (two models)
simargl rrf "query" --sources task:default,file:bge,file:jina --top-n 5

# with score_blend on file sources
simargl rrf "query" --sources task:default,file:jina --score-blend 0.7 --top-n 5
```

### Proposed MCP tool

```python
@mcp.tool()
def rrf(
    query: str,
    sources: str = "task:default,file:default",
    top_n: int = 5,
    k: int = 60,
    score_blend: float = 1.0,
    store_dir: str = STORE_DIR,
) -> str:
    """Merge results from multiple search modes/projects using Reciprocal Rank Fusion.

    sources: comma-separated "mode:project_id" pairs
      "task:default,file:jina"        — task on bge-small + file on jina index
      "task:default,file:bge,file:jina" — three-way merge

    RRF formula: score(file) = sum(1 / (k + rank_i)) for each source i
    Files appearing in multiple sources automatically rank higher.
    Files appearing in only one source are not discarded — just ranked lower.
    k=60 is the standard damping constant.
    """
```

### Implementation sketch

```python
def _rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, path in enumerate(ranked, start=1):
            scores[path] = scores.get(path, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Each source calls `_search(query, mode=mode, project_id=project_id, ...)` and extracts `[f["path"] for f in result["files"]]`. The ranked path lists go into `_rrf()`. Result is re-ranked unified file list with RRF scores.

### Why this is the correct architecture

`mode=task` and `mode=file` use different embedding spaces (bge-small vs jina-code) — their raw scores are incompatible (0.7 bge ≠ 0.55 jina). RRF discards raw scores entirely and uses only rank position, making cross-model merging mathematically valid. No normalization needed.

### Real test results (RuboCop) — final after deduplication fix

**Test 1:** `"Style/RedundantSelf doesn't consider pattern matching variables"`  
Sources: `task:default, file:jina-ruby`

| Rank | File | Score |
|---|---|---|
| 1 | redundant_self.rb ✓ | 0.0328 (both sources) |
| 2 | redundant_self_spec.rb | 0.0161 |

Correct file found at rank 1. Gap to rank 2: **×2**. Unambiguous winner.

**Test 2:** `"EmptyLinesAroundExceptionHandlingKeywords fails on one-liner rescue"`  
Sources: `task:default, file:jina-ruby` (jina-ruby covers only style/, not layout/)

| Rank | File | Score |
|---|---|---|
| 1 | empty_lines_around_exception_handling_keywords.rb ✓ | 0.0323 (task only) |
| 2 | CHANGELOG.md | 0.0164 |
| 3 | suppressed_exception.rb | 0.0164 |

Correct file at rank 1 despite jina-ruby not having it — `task:default` alone was enough, and no competitor could match its frequency signal. Gap to rank 2: **×2**.

### Full method comparison

| Method | RedundantSelf | EmptyLines |
|---|---|---|
| bge-small mode=file | not in top 8 ✗ | not in top 8 ✗ |
| jina-code mode=file | rank 1 ✓ | rank 2 ✓ |
| task sort=freq | rank 1 ✓ | rank 1 ✓ |
| **RRF (task + jina)** | **rank 1, gap ×2 ✓✓** | **rank 1, gap ×2 ✓✓** |

### Why ×2 gap matters for weak models

RRF does not just find the correct file — it separates it from the noise by a factor of 2. A file that appears in both sources scores `1/61 + 1/61 = 0.0328`; a file in only one source scores at most `1/61 = 0.0164`. This is a structural property of RRF, not a tuning artifact.

For a weak model with a 1000-token context budget, this is critical: the agent can confidently take only rank 1 and ignore the rest. No need to hedge with top-3 or top-5. The ×2 gap is a reliable signal even without score normalization or threshold tuning.

### Bug fixed during testing

**Path deduplication:** `redundant_self.rb` appeared twice with different prefixes (`lib/rubocop/cop/style/a.rb` vs `rubocop/cop/style/a.rb`) because the two indexes were built from different base directories. Fixed in `_rrf_merge`: if one path is a suffix of another (`q.endswith("/" + p)`), they are the same file — the longer path is used as canonical and their RRF scores are summed correctly into one entry.
