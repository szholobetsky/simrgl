# exp3.1 Portfolio Results Summary

Analysis of the full 10-project sweep (`experiment_results/all_projects_results.csv`,
1333 completed variant-runs — see `EXPERIMENT_RESTRICTIONS.md` for the one
known gap: 35 `bge-large`/`vscode/commit` variants missing due to an OOM
kill, immaterial to the conclusions below). Generated 2026-07-31.

Projects: `agilebill`, `celery`, `flink`, `hadoop`, `kubernetes`, `pulumi`,
`rubocop`, `sonar`, `spark`, `vscode`. Dimensions: `task_unit`
(ticket/commit) × `model` (bge-small/bge-large) × `split_strategy`
(recent/modn) × `source` × `target` (file/module) × `window` (w100/w1000/all).

---

## 1. Locked exp3 (sonar-only) findings — confirmed at portfolio scale

- **Target granularity (file vs module)**: dramatic and consistent across
  every project — **MAP file=0.129 vs module=0.589**. Fully confirmed, now
  on a far larger sample than the original sonar-only result.
- **Split strategy**: `recent` systematically inflates MAP (0.389) vs. the
  honest `modn` baseline (0.329) — confirmed portfolio-wide, not just on
  sonar.

## 2. New finding: `window` preference is project-specific, not universal

This is the most notable result. The original sonar-only claim ("w1000 best
for modn, w100 best for recent") **holds when checked on sonar alone**
(`modn`: w1000=0.212 vs all=0.211 — near tie, edge to w1000; `recent`:
w100=0.469 — clear max). But it **does not generalize** across the
portfolio:

| Project | Best window (modn) | Best window (recent) |
|---|---|---|
| sonar | w1000 | w100 |
| celery | w1000 | w1000 |
| kubernetes | w100 | w1000 |
| hadoop, flink, spark, agilebill | **all** | mixed |
| pulumi, rubocop, vscode | w100 | w100 |

No single window dominates across all projects — the optimal `window`
appears to depend on project-specific dynamics (release cadence, history
age, team size), not a universal property of the method. **The AGENTS.md
locked finding on window preference should be scoped as sonar-specific, not
stated as a general rule**, pending further investigation.

## 3. Source variant comparison

- **Ticket mode**: `comments` (0.352) ≈ `desc` (0.350) > `title` (0.339) —
  mild edge to richer context, consistent with the prior "desc best"
  finding, though the gap is narrower at portfolio scale.
- **Commit mode**: `desc` (0.376) > `diff` (0.371) > `title` (0.365) — raw
  diff text is slightly weaker than the structured commit message, but all
  three are close.

## 4. Model: bge-large vs bge-small

Minimal difference (0.363 vs 0.355 mean MAP). bge-large has a slight edge
but not a dramatic one — for most practical purposes bge-small (smaller,
faster) is a reasonable choice; retrieval quality here appears bounded by
the nature of the task→code problem itself, not embedding-model capacity.

**Paired analysis (2026-07-31)**: comparing the two models on the exact same
649 variants (identical project/task_unit/split/source/target/window, model
as the only difference — the 35 `bge-large`/`vscode/commit` variants missing
per `EXPERIMENT_RESTRICTIONS.md` are naturally excluded from the pairing):

- Mean paired diff (bge-large − bge-small) = **+0.0121 MAP**, median +0.0104
- bge-large wins **77.2%** of pairs, bge-small wins 22.7%
- Paired t-test p = 5×10⁻³⁷; Wilcoxon signed-rank p = 1×10⁻⁴⁴ — the edge is
  **statistically real and consistent**, not noise, and holds across nearly
  every project (spark +0.021, celery +0.021, vscode +0.018 down to hadoop
  +0.007, the weakest).
- But the effect size is small in absolute terms (~1 MAP point) and not
  monotonic per-configuration — some individual variants favor bge-large by
  a lot (agilebill +0.10, sonar/hadoop +0.08), others favor bge-small by a
  lot (sonar/ticket/desc/w1000: **−0.11**). So it's "slightly better on
  average with real scatter," not "always better."

**Cost side**: bge-large's 1024-dim vectors are ~2.7× the storage/memory of
bge-small's 384-dim, and bge-large was the direct trigger (combined with the
large `vscode.db` load) of the OOM crash documented in
`EXPERIMENT_RESTRICTIONS.md` — a real operational risk on this hardware (16
GB RAM / 6 GB VRAM), not just a theoretical cost.

**Conclusion**: the ~+0.01 MAP gain does not justify the 2.7× overhead and
demonstrated OOM risk on this hardware. **bge-small should be the default
model for the full grid going forward**; bge-large is worth spot-checking
only on a final best-configuration candidate, not running across the whole
combinatorial sweep.

## 5. Ticket vs. commit (`task_unit`)

Commit mode scored somewhat higher on average (0.370 vs 0.347 MAP) —
notable since raw commits are usually assumed noisier than curated tickets.
Possibly commits carry a more direct text↔file signal (a commit message
literally describes the change made, whereas a ticket only describes
intent).

## 6. Per-project ranking (module-target MAP)

| Easiest | Hardest |
|---|---|
| vscode 0.752, celery 0.735 | flink 0.426, sonar 0.446 |
| hadoop 0.656, rubocop 0.653 | kubernetes 0.528 |

**Caveat**: hadoop peaks at **0.9325** (near ceiling) — worth checking
hadoop's module count; if it has very few top-level modules, MAP inflates
naturally from a small label space rather than genuine retrieval quality.
Conversely, `sonar/ticket/.../all` bottoms out at **0.046** — expected
(locked finding: `all` window + 14 years of history = concept drift).

## 7. Best / worst single configurations

- **Best**: `hadoop/ticket/w100` — MAP up to 0.93 (several near-identical
  top variants).
- **Worst**: `sonar/ticket/.../all` — MAP down to 0.046.

---

## Open questions for follow-up

1. Check module-count distribution per project to rule out label-space-size
   artifacts inflating hadoop's (and possibly others') module-target MAP.
2. Investigate *why* window preference varies by project — candidate
   explanatory variables: commit/task volume, project age, release cadence.
3. Rerun `vscode/commit`/`bge-large` (35 missing variants, see
   `EXPERIMENT_RESTRICTIONS.md`) once the memory issue is addressed, to
   complete the grid.
