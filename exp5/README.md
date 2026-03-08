# Experiment 5: Cross-Vocabulary Symbol Grounding for Precise File and Line Localization

> **Goal**: Bridge natural language task descriptions and code identifiers to produce exact (file, line_range) coordinates, not just ranked file lists.

---

## The Core Problem

Experiments 0–4 share a common ceiling: they rank files by vector similarity and report MAP/MRR. But MAP@10 = 3.5% at file level (exp3) means that for a typical task the correct file is *not* in the top 10. The practical output — a ranked list of 50 candidate files — is still too large for a developer (or a 1B LLM) to process.

The bottleneck is not model quality. It is **vocabulary mismatch** and **popularity bias**:

1. **Popularity bias.** 70% of changes in the SonarQube dataset occur in the `server/` module. A centroid embedding for any file in `server/` is pulled toward the dominant patterns of that module. Every new task query lands near `server/` simply because it is statistically omnipresent. `pom.xml`, `static/` images, and `UserService` appear in nearly every result set regardless of task content.

2. **Vocabulary mismatch.** A Jira task says *"add label to bubble chart"*. The code contains `BubbleChartDecorator.addLabel(ln:256)`. The word "label" does not appear in the embedding vocabulary as a code identifier. The embedding maps the task to a semantic neighborhood — but within that neighborhood there is no direct path from the word *label* to the method `addLabel`. Term overlap is zero; semantic proximity is weak.

3. **Context explosion.** Taking all files modified by the 10 most semantically similar historical tasks yields 150+ distinct files — far beyond any LLM context window. Intersection with embedding top-50 may *hide* the correct file if it was absent from either set.

**The proposed solution**: use the embedding for coarse module-level direction, then apply identifier-space TF-IDF and cross-vocabulary co-occurrence to ground task terms to exact identifiers, files, and line numbers.

---

## Central Hypothesis

> A two-stage pipeline — (1) embedding narrows candidates to module level; (2) identifier-space TF-IDF + cross-vocabulary co-occurrence maps task terms to discriminative identifiers → exact (file, line_range) — achieves higher Recall@10 and Precision@5 at file level than embedding similarity alone, and produces actionable line-level coordinates that a 1B LLM can verify in a single `/read` call.

---

## Research Questions

### RQ1 — Identifier-Space Filtering
**Does TF-IDF computed over code identifiers (not task words) effectively remove popularity noise from embedding candidates?**

*Motivation*: `pom.xml` and `UserService` have no discriminative identifiers for a bubble chart task. A file whose identifiers score high on identifier-TF-IDF relative to the candidate set should be a better candidate than one that scores high only on embedding cosine similarity.

*Test*: Take top-50 embedding candidates for 200 test tasks. Compute identifier-TF-IDF across those 50 files using map_index output. Re-rank. Compare Recall@5 / Recall@10 / MAP of (a) embedding ranking vs (b) identifier-TF-IDF re-ranking.

*Falsification*: If identifier-TF-IDF re-ranking achieves the same or worse Recall@10 than the original embedding ranking, the identifier space adds no signal beyond what the embedding already captures.

---

### RQ2 — Cross-Vocabulary Bridge
**Can a co-occurrence matrix built from 9,799 historical (task_text, changed_files) pairs reliably map business-language terms to code identifiers?**

*Motivation*: The task says *"expeditor"*. The code contains `expedit2_id`. These share a character-level root but no embedding-space proximity. A co-occurrence matrix that counts how often *expeditor* (from task text) co-appears with `expedit2_id` (from changed file identifiers) across the training set can build an explicit bridge that embeddings cannot.

*The matrix*: For each training task, extract key terms from task description (using exp1's HHI-filtered vocabulary) and extract identifiers from changed files (using map_index). Build a sparse matrix: `M[term, identifier] = number of tasks where term appears in text AND identifier appears in changed files`.

*Test*: For each test task, take top-5 key terms → look up top-3 identifiers per term in matrix M → search map_index for those identifiers → check if ground-truth file appears in results. Report: Cross-Vocabulary Recall@10 (fraction of test tasks where ground-truth file is found via the identifier lookup chain).

*Falsification*: If Cross-Vocabulary Recall@10 is not significantly better than random retrieval from the embedding top-50, the co-occurrence matrix provides no useful signal. This would indicate that the historical dataset is too sparse for reliable term-identifier pairing (< 5 co-occurrences per pair on average).

---

### RQ3 — Combined Pipeline: Semantic Direction + Identifier Grounding
**Does the two-stage pipeline (embedding → module direction; identifier TF-IDF + co-occurrence → exact file + line) outperform embedding alone at file level?**

*Pipeline*:
```
Stage 1: embed(task) → top-50 candidate files   [embedding, existing infrastructure]
Stage 2a: map_index(candidate files) → identifier inventory
Stage 2b: identifier-TF-IDF(inventory) → discriminative identifiers per file
Stage 2c: co-occurrence lookup(task_key_terms) → predicted identifiers
Stage 2d: intersect(discriminative, predicted) → high-confidence identifiers
Stage 2e: map_query find(identifiers) → (file, line_number) coordinates
Output: top-5 (file, line_range) pairs with confidence score
```

*Primary metric*: **Grounding Recall@5** — fraction of test tasks where at least one ground-truth changed file appears in the top-5 grounded results.

*Secondary metrics*:
- **Line Accuracy** — fraction of grounded (file, line_range) tuples where the ground-truth change falls within the grounded range (±20 lines)
- **Noise Reduction Rate** — fraction of embedding candidates (pom.xml, static/, test files) eliminated by identifier-TF-IDF filtering

*Falsification*: If Grounding Recall@5 ≤ embedding Recall@5 for the same candidate set, the added identifier grounding does not help. If Line Accuracy < 20%, the system cannot produce useful coordinates even when it finds the right file.

---

### RQ4 — LLM Verification on Grounded Context
**Can a 1B–7B LLM reliably confirm or reject a candidate (file, line_range) given only the grounded ~30 lines as context?**

*Motivation*: If Stage 2 produces 5 (file, line_range) pairs with avg 30 lines each, a 1B model can read all 5 in a single context window (~150 lines) and answer: *"which of these contains the implementation relevant to the task?"* This is a binary classification per candidate — the task the model is well-suited for (see 1bcoder design philosophy).

*Test*: For each (task, grounded_candidates) pair, inject candidates into a 1B and 7B model with the prompt: *"Which file and line range is most relevant to this task? Output only: FILENAME ln:START-END. No explanation."* Measure:
- **Verification Precision** — fraction of LLM-selected candidates that are ground-truth files
- **Verification Recall** — fraction of ground-truth files selected at least once across the 5 candidates
- Compare 1B vs 3B vs 7B on this task

*Falsification*: If Verification Precision for 1B is ≤ random selection from 5 candidates (0.20), the model cannot distinguish relevant from irrelevant at this granularity. If 7B is not significantly better than 1B, the task does not require scale — it is a pattern-matching task that prompt engineering alone solves.

---

## Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Grounding Recall@K** | Fraction of test tasks where ≥1 ground-truth file appears in top-K grounded results | ≥ 0.40 at K=5 |
| **Line Accuracy** | Fraction of (file, range) pairs where ground-truth change is within ±20 lines of grounded range | ≥ 0.25 |
| **Noise Reduction Rate** | Fraction of non-relevant files removed by identifier-TF-IDF from embedding top-50 | ≥ 0.60 |
| **Cross-Vocab Recall@10** | Fraction of test tasks where ground-truth file reachable via co-occurrence lookup | ≥ 0.30 |
| **Verification Precision (1B)** | Fraction of 1B LLM-selected candidates that match ground truth | ≥ 0.40 |
| **Verification Precision (7B)** | Same for 7B model | ≥ 0.60 |
| **MHR** (Memoization Hit Rate) | Fraction of task key terms found in co-occurrence matrix with ≥3 supporting tasks | ≥ 0.70 |

Standard IR metrics (MAP, MRR, P@K, R@K) are reported for comparison with exp3 baseline.

---

## Dataset

Primary: **SonarQube** (`sonar.db`) — 9,799 tasks, 12,532 files, 27 modules.

The SonarQube dataset is ideal for this experiment because:
- It has a known popularity imbalance (`server/` = 70% of changes) that embedding-only approaches cannot overcome
- The codebase contains domain-specific identifiers (e.g., `BubbleChartDecorator`, `CoveragePerLine`, `ShortLivedBranchAnalyzer`) that are invisible to general-purpose embeddings but highly discriminative via map_index
- map_index has already been applied to SonarQube (via ragmcp infrastructure)

Secondary (validation): **Kafka** or **Spark** from exp4 — to test whether the cross-vocabulary bridge generalises across domains.

---

## Pipeline Architecture

```
INPUT: Jira task text
  │
  ▼
[STAGE 1 — Semantic Direction]
  embed(task) → cosine similarity → top-50 candidate files
  (uses existing ragmcp PostgreSQL + pgvector infrastructure)
  │
  ▼
[STAGE 2a — Identifier Extraction]
  map_index(candidate_files) → identifier inventory
  {file: [identifier, line_number, type], ...}
  │
  ▼
[STAGE 2b — Identifier-Space TF-IDF]
  TF  = count(identifier in file) / total_identifiers(file)
  IDF = log(N_candidates / files_containing_identifier)
  score(identifier, file) = TF × IDF
  → per-file discriminative identifier set (top-10 per file)
  → filter: remove files with no discriminative identifier (noise elimination)
  │
  ▼
[STAGE 2c — Cross-Vocabulary Co-occurrence Lookup]
  task_key_terms = HHI-filtered terms from task text  (exp1 pipeline)
  for each key_term:
      predicted_identifiers = top-K from M[term, :] (co-occurrence matrix)
  │
  ▼
[STAGE 2d — Confidence Scoring]
  for each candidate file:
      conf(file) = Σ cosine_weight(file)
               + α × max_TF-IDF_score(file)
               + β × co-occurrence_matches(file, predicted_identifiers)
  → re-rank candidates by conf
  → retain top-5
  │
  ▼
[STAGE 2e — Line Localization]
  map_query find(predicted_identifiers) → (file, line_number)
  → for each top-5 file: extract lines [ln-15 : ln+15] around matched identifier
  │
  ▼
OUTPUT: [(file1, ln:240-270), (file2, ln:100-130), ...]

  ▼  [OPTIONAL STAGE 3]
[LLM Verification — 1B / 7B]
  /read file1 240-270
  /read file2 100-130
  → "Which contains the fix for: {task}? Output only FILENAME ln:START-END"
  → verified (file, line_range)
```

---

## Data Requirements

### New: Cross-Vocabulary Matrix (M)

Build from training split of `sonar.db`:

```python
# For each training task:
#   1. Extract key terms from TITLE + DESCRIPTION using exp1 HHI filter
#   2. Extract identifiers from changed files using map_index output
#   3. Increment M[term][identifier] for each (term, identifier) pair

M = defaultdict(lambda: defaultdict(int))
for task_id, task_text, changed_files in training_data:
    key_terms = extract_key_terms(task_text)         # exp1 pipeline
    identifiers = get_identifiers(changed_files)      # map_index output
    for term in key_terms:
        for identifier in identifiers:
            M[term][identifier] += 1
```

Matrix size estimate: ~50K unique terms × ~200K unique identifiers = sparse, storable in SQLite or JSON.

### New: map_index output for SonarQube

The ragmcp system (PostgreSQL + pgvector) already has file-level embeddings but NOT identifier-level map_index output. This experiment requires running `map_index.py` on the SonarQube source tree and storing the result in a queryable form (SQLite table or map.txt).

Required schema:
```sql
CREATE TABLE MAP_IDENTIFIER (
    file_path   TEXT,
    identifier  TEXT,
    line_number INTEGER,
    type        TEXT    -- 'define', 'var', 'link'
);
```

---

## Comparison Baselines

| System | Description |
|--------|-------------|
| **exp3-bge-large** | Embedding-only, top-50 file ranking (MAP@10 = 3.5%) |
| **exp5-stage1-only** | Same embedding, but using only top-5 instead of top-50 |
| **exp5-identifier-TF-IDF** | Stage 1 + Stage 2b only (no co-occurrence) |
| **exp5-cooccurrence-only** | Co-occurrence lookup without embedding pre-filter |
| **exp5-full** | Complete two-stage pipeline (embedding + TF-IDF + co-occurrence) |
| **exp5-full+LLM-1B** | Full pipeline + 1B verification step |
| **exp5-full+LLM-7B** | Full pipeline + 7B verification step |

---

## Hypothesised Outcome

| Stage | Expected improvement over embedding baseline |
|-------|---------------------------------------------|
| Identifier TF-IDF re-ranking alone | +5–15% Recall@10, noise reduction ≥ 60% |
| Cross-vocab co-occurrence alone | +10–20% Cross-Vocab Recall@10 |
| Full pipeline (no LLM) | Grounding Recall@5 ≥ 0.40 vs embedding Recall@5 ≈ 0.15 |
| Full pipeline + 1B LLM | Verification Precision ≥ 0.40 |
| Full pipeline + 7B LLM | Verification Precision ≥ 0.60 |

The expected failure mode: tasks touching files with no distinctive identifiers (migration scripts, config-only changes, test fixtures) will not be groundable. These cases are expected to fall back to the embedding ranking unchanged.

---

## Falsification Conditions

The experiment should be considered **falsified** (and the hypothesis rejected) if:

1. **Identifier-TF-IDF adds no signal**: Recall@10 after TF-IDF re-ranking ≤ embedding Recall@10 for the same candidate set. Interpretation: discriminative identifiers do not correlate with relevance.

2. **Cross-vocabulary matrix is too sparse**: MHR (fraction of task terms with ≥3 co-occurrence supports) < 0.40. Interpretation: 9,799 tasks are insufficient to build a reliable term-identifier bridge for a codebase of this size.

3. **Line Accuracy is too low**: < 10% of grounded (file, range) pairs contain the actual change within ±20 lines. Interpretation: identifier line numbers from map_index do not predict change location.

4. **1B LLM verification is random**: Verification Precision ≤ 0.20 (chance level for 5 candidates). Interpretation: 30-line context is insufficient for even a small model to distinguish relevant from irrelevant code.

If conditions 1 and 2 both hold, the entire exp5 approach should be abandoned in favour of alternative grounding strategies (e.g., AST-based Object Passports, graph-based structural search).

---

## Relation to Prior Experiments

| Dimension | exp0–3 | exp4 | exp5 |
|-----------|--------|------|------|
| Unit of retrieval | File / Module | File / Module | File + **line range** |
| Vocabulary space | Task text | Task text | Task text **+ code identifiers** |
| Model type | Embedding | Embedding (LLM-scale) | Embedding + TF-IDF + co-occurrence |
| Historical data use | Centroid of task embeddings per file | Same | **Explicit (term, identifier) co-occurrence matrix** |
| LLM role | None | None | Optional verification on grounded context |
| Output | Ranked file list | Ranked file list | **(file, line_range) coordinates** |

exp5 does not replace exp3/exp4 — it uses their top-K output as Stage 1 input. It is an additional grounding layer on top of the embedding infrastructure already built.

---

## Connection to Anthill Architecture

This experiment directly instantiates three Anthill concepts described in `../concepts/ANTHILL_DISTRIBUTED_COGNITIVE_OS.md`:

- **Symbol Grounding** (Phenomenological Grounding): mapping business terms → code identifiers → (file, line). The co-occurrence matrix M is the grounding dictionary.
- **Object Passport** (primitive): the (identifier, line_number, type) triples from map_index are a flat-file precursor to the YAML Object Passport.
- **Reduction Ladder** (Level 6→5): `/read file 240-270` (Level 6, raw code) is replaced by a passport-like grounded identifier context (Level 5). The LLM operates on Level 5 input, not Level 6.

The 1bcoder tool is the execution layer for Stage 3: `--planapply GroundingVerify.txt` drives the LLM verification loop across all (file, line_range) candidates automatically.

---

## Scientific Literature

### Feature and Bug Localization
- Ye, X., Fang, R., & Bunescu, R. (2016). *Learning to rank relevant files for bug reports using domain knowledge*. FSE 2016. — File ranking from bug reports; closest prior work to this experiment.
- Wong, W. E., Debroy, V., Golden, R., Xu, X., & Thuraisingham, B. (2012). *Effective software fault localization using an RBF neural network*. IEEE Trans. Reliability. — Spectrum-based fault localization; conceptual predecessor to line-level grounding.
- Lam, A. N., Nguyen, A. T., Nguyen, H. A., & Nguyen, T. N. (2017). *Bug localization with combination of deep learning and information retrieval*. ICPC 2017. — Neural + IR hybrid for bug localization; directly comparable setup.
- Saha, R. K., Lease, M., Khurshid, S., & Perry, D. E. (2013). *Improving bug localization using structured information retrieval*. ASE 2013. — Structured code elements for IR; precursor to identifier-space search.

### Information Retrieval and TF-IDF in Code Search
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. — Foundational TF-IDF theory. Chapter 6 (scoring and ranking) directly applicable.
- Lv, F., Zhang, H., Lou, J.-G., Wang, S., Zhang, D., & Zhao, J. (2015). *CodeHow: Effective code search based on API understanding and extended Boolean model*. ASE 2015. — API-enriched code search; combines text and structural signals.
- Allamanis, M., Barr, E. T., Devanbu, P., & Sutton, C. (2018). *A survey of machine learning for big code and naturalness*. ACM Computing Surveys. — Overview of ML approaches to code; Section 4 (code retrieval) covers identifier-aware models.

### Symbol Grounding and Vocabulary Mismatch
- Harnad, S. (1990). *The symbol grounding problem*. Physica D: Nonlinear Phenomena, 42(1–3), 335–346. — Original formulation of symbol grounding; theoretical foundation for the term→identifier bridge.
- Gärdenfors, P. (2000). *Conceptual Spaces: The Geometry of Thought*. MIT Press. — Conceptual spaces framework; embedding space as geometric grounding mechanism.
- Ko, A. J., & Myers, B. A. (2006). *Finding causes of program output with the Java whyline*. CHI 2006. — Causal tracing in code; related to "why does this output occur?" queries.

### Cross-Vocabulary / Ontology Alignment
- Maedche, A., & Staab, S. (2001). *Ontology learning for the Semantic Web*. IEEE Intelligent Systems. — Automatic ontology construction from text; related to building term-identifier correspondence.
- Euzenat, J., & Shvaiko, P. (2013). *Ontology Matching* (2nd ed.). Springer. — Formal treatment of vocabulary alignment; Chapter 3 (element-level matching) covers string and context-based bridging.

### Statistical Term Analysis (exp1 lineage)
- Bradford, S. C. (1934). *Sources of information on specific subjects*. Engineering, 137, 85–86. — Bradford's Law; theoretical basis for exp1's zone classification.
- Hirschman, A. O. (1964). *The paternity of an index*. American Economic Review. — Herfindahl-Hirschman Index; used in exp1 for term concentration measurement, applicable here for identifier discrimination.

### LLM Verification and Constrained Generation
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). *ReAct: Synergizing reasoning and acting in language models*. ICLR 2023. — Tool-use framework; relevant to the LLM verification stage as a constrained action.
- Wei, J., et al. (2022). *Chain-of-thought prompting elicits reasoning in large language models*. NeurIPS 2022. — Reasoning prompts; constraining to "FILENAME ln:START-END" format is the inverse — suppressing chain-of-thought for classification.

---

## Implementation Roadmap

### Phase A — Data Preparation (prerequisite)
1. Run `map_index.py` on SonarQube source tree → store in `MAP_IDENTIFIER` SQLite table
2. Build cross-vocabulary matrix M from training split using exp1 key-term pipeline
3. Validate: check MHR — if < 0.40, adjust HHI threshold for key term extraction

### Phase B — Baseline Comparison (RQ1)
4. Implement identifier-TF-IDF re-ranking of embedding top-50
5. Evaluate: Recall@10 vs embedding baseline for 200 test tasks
6. Measure noise reduction: count pom.xml / static / test files eliminated

### Phase C — Co-occurrence Bridge (RQ2)
7. Implement co-occurrence lookup: task terms → predicted identifiers → file search
8. Evaluate: Cross-Vocabulary Recall@10 standalone
9. Sparsity analysis: plot distribution of co-occurrence support per (term, identifier) pair

### Phase D — Full Pipeline (RQ3)
10. Combine Stage 1 + Stage 2 (TF-IDF + co-occurrence scoring)
11. Add line localization via map_query
12. Evaluate: Grounding Recall@5, Line Accuracy, MAP for comparison with exp3

### Phase E — LLM Verification (RQ4)
13. Generate verification plan files using `--planapply` (1bcoder)
14. Run 1B and 7B models on grounded context
15. Measure Verification Precision and Recall across model scales

---

**Document Version**: 1.0
**Created**: 2026-03-08
**Project**: SIMARGL / codeXplorer Research Program
**Relation to**:
- `../exp3/README.md` — Stage 1 infrastructure (embeddings, pgvector)
- `../exp4/README.md` — LLM-scale embeddings (may replace Stage 1)
- `../exp1/README.md` — HHI term extraction (key term pipeline for Stage 2c)
- `../concepts/1BCODER.md` — execution layer for Stage 3 (LLM verification)
- `../concepts/ANTHILL_DISTRIBUTED_COGNITIVE_OS.md` — theoretical grounding
- `../concepts/PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md` — symbol grounding metrics
