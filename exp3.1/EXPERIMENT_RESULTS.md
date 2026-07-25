# Experiment 3 — Results and Research Question Answers

**Dataset**: SonarQube (sonar.db) — ~9,799 tasks, 12,532 files, 27 modules, Java
**Models tested**: bge-small-en-v1.5, bge-large-en-v1.5, bge-m3 (partial)
**Raw data**: `../info/LAST/comprehensive_results.csv`

---

## Experimental Design

Each experiment is identified by the combination of:

| Variable | Values | Description |
|----------|--------|-------------|
| `model` | bge-small, bge-large, bge-m3 | Embedding model |
| `source` | title, desc, comments | What text is embedded as the query |
| `target` | module, file | Retrieval unit (module = directory, file = individual file) |
| `window` | w100, w1000, all | Knowledge base temporal window (days) |
| `split_strategy` | recent, modn | How test tasks are selected |

**Split strategies:**
- `recent`: test set = the 200 most recent tasks. Knowledge base = tasks preceding them. Creates temporal proximity — may inflate scores.
- `modn`: test set = uniformly sampled every N-th task from full history. More honest evaluation of general prediction ability.

**Aggregation**: Average aggregation used throughout (exp2 proved all strategies produce identical results).

**Test size**: 200 tasks per experiment.

---

## Summary Results Table — Module Retrieval (Primary Target)

### modn split (honest evaluation)

| Model | Source | Window | MAP | MRR | P@1 | P@5 | R@10 |
|-------|--------|--------|-----|-----|-----|-----|------|
| **bge-large** | comments | w1000 | **0.4590** | **0.5093** | 0.265 | 0.200 | 0.856 |
| **bge-large** | desc | w1000 | **0.4522** | **0.5028** | **0.275** | 0.198 | 0.863 |
| bge-large | comments | all | 0.3666 | 0.4199 | 0.240 | 0.175 | 0.780 |
| bge-large | title | w1000 | 0.4118 | 0.4580 | 0.195 | 0.192 | 0.834 |
| bge-large | desc | all | 0.3392 | 0.3775 | 0.200 | 0.170 | 0.774 |
| bge-large | title | all | 0.3250 | 0.3692 | 0.200 | 0.161 | 0.774 |
| bge-large | desc | w100 | 0.2402 | 0.2975 | 0.115 | 0.202 | 0.811 |
| bge-small | comments | w1000 | 0.4441 | 0.4919 | 0.245 | 0.193 | 0.869 |
| bge-small | desc | w1000 | 0.4341 | 0.4870 | 0.255 | 0.190 | 0.880 |
| bge-small | comments | all | 0.3356 | 0.3787 | 0.200 | 0.164 | 0.750 |
| bge-small | title | w1000 | 0.4087 | 0.4505 | 0.210 | 0.184 | 0.824 |
| bge-small | desc | all | 0.3168 | 0.3590 | 0.180 | 0.163 | 0.738 |
| bge-m3 | title | w1000 | 0.4079 | 0.4501 | 0.195 | 0.188 | 0.823 |
| bge-m3 | title | all | 0.3233 | 0.3691 | 0.185 | 0.169 | 0.771 |

**bge-m3 note**: Only `title` source was completed. `desc` and `comments` runs were cut short by GPU OOM (6GB VRAM, batch_size=32 too large). Comparison with bge-large is therefore incomplete and unfair.

### recent split (for reference — likely inflated)

| Model | Source | Window | MAP | MRR | P@1 | P@5 | R@10 |
|-------|--------|--------|-----|-----|-----|-----|------|
| **bge-large** | desc | w100 | **0.7978** | **0.8747** | **0.830** | **0.242** | 0.975 |
| bge-large | title | w100 | 0.7896 | 0.8692 | 0.825 | 0.208 | 0.976 |
| bge-large | comments | w100 | 0.7886 | 0.8691 | 0.820 | 0.206 | 0.977 |
| bge-small | desc | w100 | 0.7858 | 0.8610 | 0.805 | 0.243 | 0.977 |
| bge-small | title | w100 | 0.7845 | 0.8589 | 0.805 | 0.241 | 0.977 |
| bge-m3 | title | w100 | 0.7862 | 0.8654 | 0.830 | 0.244 | 0.925 |
| bge-large | desc | w1000 | 0.3529 | 0.3614 | 0.070 | 0.230 | 0.928 |
| bge-small | desc | w1000 | 0.4663 | 0.4836 | 0.070 | 0.229 | 0.937 |

---

## Summary Results Table — File Retrieval

File retrieval is substantially harder. Best results (modn split):

| Model | Source | Window | MAP | MRR | P@1 | R@10 |
|-------|--------|--------|-----|-----|-----|------|
| **bge-large** | comments | all | **0.0745** | **0.1838** | **0.105** | 0.146 |
| bge-large | desc | all | 0.0632 | 0.1722 | 0.110 | 0.133 |
| bge-m3 | title | all | 0.0552 | 0.1517 | 0.110 | 0.092 |
| bge-small | comments | all | 0.0525 | 0.1617 | 0.100 | 0.117 |
| bge-small | desc | all | 0.0501 | 0.1703 | 0.120 | 0.111 |
| bge-large | comments | w1000 | 0.0101 | 0.0597 | 0.035 | 0.023 |
| bge-small | desc | w1000 | 0.0049 | 0.0412 | 0.030 | 0.008 |

---

## Research Question Answers

### RQ1: Granularity Impact
> **How does the granularity of the target (File vs. Module) affect accuracy?**

**Answer: Module retrieval is dramatically more accurate than file retrieval.**

| Configuration | Module MAP | File MAP | Ratio |
|--------------|-----------|---------|-------|
| bge-large, desc, w1000, modn | 0.452 | 0.007 | 65× |
| bge-large, desc, all, modn | 0.339 | 0.063 | 5× |
| bge-large, comments, all, modn | 0.367 | 0.075 | 5× |

Module-level Recall@10 consistently reaches **0.86–0.88** on the honest (modn) split — meaning the correct module is within the top-10 recommendations 86–88% of the time.

File-level Recall@10 reaches at most **0.146** — only 1-in-7 correct files appear in the top-10.

**Interpretation**: Module-level retrieval is tractable and practically useful. File-level retrieval is very difficult because: (1) individual files have sparse change histories, making embeddings noisy; (2) there are many more files than modules (~12,500 vs ~27), creating a much harder ranking problem; (3) a task typically touches 2–5 files across 1–3 modules, so module-level queries have a higher signal-to-noise ratio.

**Hypothesis verdict**: Partially confirmed. Module retrieval does achieve higher Recall@k. The magnitude of the gap (5–65×) was not anticipated.

---

### RQ2: Semantic Density — Title vs Description
> **Does the high "semantic density" of Task Titles outperform Task Descriptions?**

**Answer: No. Description consistently outperforms Title.**

| Model | Window | Split | MAP (title) | MAP (desc) | Winner |
|-------|--------|-------|-------------|------------|--------|
| bge-large | w1000 | modn | 0.4118 | 0.4522 | desc (+9.8%) |
| bge-large | all | modn | 0.3250 | 0.3392 | desc (+4.4%) |
| bge-large | w100 | modn | 0.2453 | 0.2402 | title (+2.1%) |
| bge-small | w1000 | modn | 0.4087 | 0.4341 | desc (+6.2%) |
| bge-small | all | modn | 0.3033 | 0.3168 | desc (+4.5%) |
| bge-large | w100 | recent | 0.7896 | 0.7978 | desc (+1.0%) |

Description wins in all but one configuration (w100 modn — where both scores are low due to the narrow window). The advantage of description is most pronounced at the optimal window (w1000).

**Hypothesis verdict: REJECTED.** Titles do not outperform descriptions. Descriptions contain more useful semantic content that helps locate the correct module. The "semantic density" hypothesis was wrong: descriptions provide richer, more complete signal rather than just noise.

---

### RQ3: Noise Tolerance — Impact of Comments
> **Does including Task Comments degrade retrieval performance?**

**Answer: No. Comments slightly improve performance, not degrade it.**

| Model | Window | Split | MAP (desc) | MAP (comments) | Winner |
|-------|--------|-------|------------|----------------|--------|
| bge-large | w1000 | modn | 0.4522 | **0.4590** | comments (+1.5%) |
| bge-large | all | modn | 0.3392 | **0.3666** | comments (+8.1%) |
| bge-large | w100 | modn | 0.2402 | 0.2444 | comments (+1.7%) |
| bge-small | w1000 | modn | 0.4341 | **0.4441** | comments (+2.3%) |
| bge-small | all | modn | 0.3168 | **0.3356** | comments (+5.9%) |
| bge-large | w1000 | recent | 0.3529 | 0.3565 | comments (+1.0%) |

Comments consistently produce equal or better results than description alone. The improvement is particularly visible with the `all` window (longer history), which suggests that discussion in comments provides additional domain-relevant vocabulary that helps match module patterns.

**Hypothesis verdict: REJECTED.** Adding comments does not degrade performance. Comments act as additional signal rather than noise — at least for module-level retrieval. The noise-to-signal ratio concern was not borne out by the data.

---

### RQ4: Temporal Dynamics — Knowledge Base Window
> **Does limiting the knowledge base to recent tasks improve prediction accuracy?**

**Answer: Medium window (w1000, ~1000 days) is optimal. Narrow (w100) and full history (all) both underperform.**

Module MAP by window (bge-large, desc, modn split):

| Window | MAP | MRR | P@1 | Interpretation |
|--------|-----|-----|-----|----------------|
| w100 | 0.2402 | 0.2975 | 0.115 | Too narrow — excludes most tasks, sparse module representations |
| **w1000** | **0.4522** | **0.5028** | **0.275** | Optimal — enough history without noise from obsolete code |
| all | 0.3392 | 0.3775 | 0.200 | Too broad — old task-module associations pollute module embeddings |

**Critical finding about the `recent` split**: The `recent` split strategy (test tasks = most recent 200) combined with `w100` window gives MAP=0.7978 — but this is an **evaluation artifact**, not true model performance. Because:
- Test tasks are very recent
- The knowledge base (w100) is built from tasks immediately preceding them
- These tasks touch the same modules (temporal locality)
- The embedding model does not need to "understand" the semantic content — it simply retrieves modules recently touched by similar tasks

The `modn` split breaks this temporal locality by distributing test tasks across the full history. It is the honest evaluation: MAP=0.4522 (desc, w1000, bge-large) vs 0.7978 for `recent`.

**Hypothesis verdict: PARTIALLY CONFIRMED with nuance.** Limiting to recent tasks does help compared to using all history — but only up to a point. The optimal window is `w1000` (~2.7 years), not the minimum window. Very narrow windows (w100 = ~3 months) lose too much historical context.

---

## Model Comparison

### Best configurations per model (modn split, module target)

| Model | Best Source | Best Window | MAP | MRR | P@1 | Notes |
|-------|------------|------------|-----|-----|-----|-------|
| **bge-large** | comments | w1000 | **0.459** | **0.509** | 0.265 | Full results |
| bge-large | desc | w1000 | 0.452 | 0.503 | 0.275 | Full results |
| bge-small | comments | w1000 | 0.444 | 0.492 | 0.245 | Full results |
| bge-small | desc | w1000 | 0.434 | 0.487 | 0.255 | Full results |
| bge-m3 | title | w1000 | 0.408 | 0.450 | 0.195 | **title only** — OOM cut run short |

**bge-large > bge-small** across all configurations (~4–5% MAP advantage).

**bge-m3 vs bge-large**: The comparison is currently impossible to make fairly. bge-m3 only has `title` results due to OOM during `desc` embedding (batch_size=32 on 6GB VRAM). bge-large with `title` scores 0.412, which is actually slightly above bge-m3's 0.408. Whether bge-m3's `desc` results would exceed bge-large cannot be determined from current data.

**Fix for bge-m3**: Set `batch_size=4` in config and rerun.

---

## Effect Size Summary

### Source variant effect (desc vs title, best window, modn):

| Model | Δ MAP (desc over title) |
|-------|------------------------|
| bge-large, w1000 | +9.8% relative |
| bge-small, w1000 | +6.2% relative |

### Window effect (w1000 vs all, desc, modn):

| Model | MAP (w1000) | MAP (all) | Δ |
|-------|------------|----------|---|
| bge-large | 0.452 | 0.339 | +33% relative |
| bge-small | 0.434 | 0.317 | +37% relative |

### Model effect (bge-large vs bge-small, desc, w1000, modn):

| Metric | bge-large | bge-small | Δ |
|--------|----------|----------|---|
| MAP | 0.452 | 0.434 | +4.1% relative |
| MRR | 0.503 | 0.487 | +3.3% relative |
| P@1 | 0.275 | 0.255 | +7.8% relative |

The **window choice has a larger effect than model choice**. Getting the right window (w1000 vs all) matters more than upgrading from bge-small to bge-large.

---

## Key Findings for Publication

1. **Module-level retrieval is feasible** with dense embeddings: MAP=0.45, R@10=0.86 (honest modn evaluation on SonarQube). The correct module is found in top-5 recommendations 19.8% of the time (P@5) and in top-10 86.3% of the time.

2. **File-level retrieval remains very hard**: MAP=0.07 at best. Significantly more work needed for direct file prediction.

3. **Description outperforms Title** (RQ2 hypothesis rejected): Richer context in task descriptions provides better semantic signal than the concentrated-but-short titles.

4. **Comments improve, not hurt** (RQ3 hypothesis rejected): Including discussion comments adds domain vocabulary that helps module matching, particularly for older tasks.

5. **Temporal window matters more than model choice**: w1000 gives +33% MAP over `all` history; bge-large vs bge-small gives only +4%.

6. **Recent split inflates results by ~77%**: `recent` (MAP=0.80) vs `modn` (MAP=0.45) at best configurations. Papers reporting only `recent` results may significantly overstate system performance.

7. **bge-m3 remains untested**: The larger bge-m3 model was cut short by OOM. Fix: `batch_size=4`. Expected to slightly outperform bge-large given its higher MTEB retrieval score (54.6 vs 39.0).

---

## Recommended Configuration for Future Work

Based on these results, the optimal configuration for exp4 comparisons is:

```
split_strategy = modn          (honest evaluation)
source         = desc          (best source; comments nearly equal)
target         = module        (tractable; file too hard currently)
window         = w1000         (optimal temporal range)
model          = bge-large     (best available with complete results)
```

This gives: **MAP=0.452, MRR=0.503, P@1=0.275, R@10=0.863** as the baseline for exp4.
