# exp3.2 — Results: Cross-Vocabulary Retrieval

**Design**: index built from commit-message text (`train_source`:
title/desc/diff), queried with the held-out Jira ticket test set
(`query_source`: title/desc/comments) — full 3×3 Cartesian product, not
matched pairs. See `README.md` for the full rationale.

**Projects**: 9 (`celery`, `flink`, `hadoop`, `kubernetes`, `pulumi`,
`rubocop`, `sonar`, `spark`, `vscode` — `agilebill` excluded, no `TASK`
rows). **Model**: `bge-small` only (see `EXPERIMENT_PLAN.md`'s model-scope
note). **Grid**: 3 train_source × 3 query_source × 2 targets × 3 windows ×
2 splits = 108 rows/project, **972 rows total, 0 failures**.

**Raw data**: `experiment_results/all_projects_results.csv`.

---

## Headline result: cross-vocabulary retrieval is competitive with matched-vocabulary retrieval

The question this whole experiment exists to answer: given a task
description written in **business language, before any code change
exists**, how well can an index built purely from **commit-message
vocabulary** (technical, post-hoc) be searched? Compared against
`exp3.1`'s two matched-vocabulary baselines — `ticket` (index and query
both ticket text) and `commit` (index and query both commit text,
methodologically self-referential — see `README.md`'s "Problem" section):

| project | **cross** | ticket (matched, 3.1) | commit (matched, 3.1) |
|---|---|---|---|
| celery | 0.521 | 0.525 | 0.490 |
| flink | 0.233 | 0.241 | 0.249 |
| hadoop | 0.362 | 0.374 | 0.384 |
| **kubernetes** | **0.317** | 0.259 | 0.346 |
| pulumi | 0.307 | 0.323 | 0.318 |
| **rubocop** | **0.465** | 0.418 | 0.411 |
| **sonar** | **0.217** | 0.200 | 0.328 |
| spark | 0.293 | 0.336 | 0.400 |
| vscode | 0.359 | 0.447 | 0.404 |

(Mean MAP per project, pooled across the full grid — coarse but the
clearest single number per project. `cross` values are this experiment's
own `all_projects_results.csv`; `ticket`/`commit` are `exp3.1`'s.)

**Cross-vocabulary retrieval beats the matched `ticket` baseline outright
in 3 of 9 projects** (kubernetes +22%, rubocop +11%, sonar +9%), and stays
within ~13% of it in 4 more (celery, flink, hadoop, pulumi). Only `vscode`
shows a real gap (0.359 vs 0.447, −20%) and `spark` a moderate one (−13%).
An index containing **zero words from any ticket** performs, on average,
almost as well as an index built from the ticket text itself — this is a
genuinely non-obvious result: it means business-language queries can find
their way to a technical-vocabulary index nearly as reliably as they find
their way to an index built from the same vocabulary they're written in.

Against matched `commit` (the self-referential baseline), `cross` wins
outright in 2 projects (celery, rubocop) and loses more clearly in others
— most sharply in `sonar` (−34%) and `spark` (−27%). This lines up exactly
with a head-to-head `ticket`-vs-`commit` comparison run on `exp3.1`'s own
data earlier this session (matched `title`/`desc` variants, 48 comparable
rows per project): `sonar` had `commit` winning **44/48** rows (mean
MAP delta +0.137) and `spark` **42/48** (+0.069) — the two most lopsided
projects in that whole comparison. In other words, `sonar`/`spark` are
exactly the two projects where matched `commit`-mode's numbers were most
inflated by vocabulary tautology to begin with, and `cross` mode —
structurally built to prevent that tautology — naturally can't replicate
that inflation there.

---

## Q1: Does `diff` content help the index?

Marginal mean MAP by `train_source` (pooled across everything else):

| train_source | mean MAP |
|---|---|
| **diff** | **0.348** |
| desc | 0.340 |
| title | 0.337 |

**Yes, fairly consistently.** `diff` wins outright in **6 of 9 projects**
(celery, flink, hadoop, pulumi, sonar, spark); `desc` wins in 2
(kubernetes, vscode); `title` wins in 1 (rubocop). This is the cleanest,
most directional finding in the whole grid — commit diff content (the
actual changed code, capped at 4000 chars) measurably strengthens the
index beyond what the commit message alone provides, in most projects.

---

## Q2: Do `comments` help the query?

Marginal mean MAP by `query_source` (pooled): `desc` 0.347, `comments`
0.340, `title` 0.338 — looks like `desc` wins clearly. **It doesn't, once
you check per-project sign-count instead of the pooled mean:**

| query_source | projects where it wins |
|---|---|
| **title** | **5** (celery, kubernetes, pulumi, rubocop, vscode) |
| comments | 3 (flink, sonar, spark) |
| desc | 1 (hadoop) |

This is a Simpson's-paradox situation — the pooled mean is dominated by a
handful of projects (`hadoop` especially) where `desc` does well, while
the *majority* of projects by count actually favor the shortest option,
`title`. **No clean universal winner on the query side**, in sharp
contrast to Q1's clean `diff`-helps-the-index result and to `exp3.1`'s own
matched-`ticket` finding (`desc` > `title` > `comments`, stated in
`AGENTS.md`'s locked findings — that finding does not carry over to the
cross-vocabulary setting).

**Interpretation**: a plausible explanation is vocabulary alignment, not
information content. Ticket titles tend to use short, direct,
implementation-adjacent phrasing ("Fix NPE in consumer poll loop");
descriptions and especially comments introduce more ticket-specific
narrative language (discussion, workarounds, "see also" references) that
doesn't share tokens with commit-message phrasing the way a terse title
does. More words in the query isn't free here — it can dilute the signal
that actually bridges to commit vocabulary.

---

## Universal patterns (cross-checked against `exp3.1`'s locked findings)

| axis | pooled mean | sign-count across 9 projects | matches `exp3.1`? |
|---|---|---|---|
| **target** | module 0.555 vs file 0.128 | module wins **9/9** | Yes — exactly matches "file retrieval much harder" |
| **window** | all 0.364 > w100 0.333 > w1000 0.328 | `all` wins **6/9** | Directionally new — `exp3.1` favored `w1000` for `modn`; here more training data wins more often |
| **split** | recent 0.353 vs modn 0.330 | recent wins **5/9**, modn **4/9** | **Diverges** — `exp3.1` found `recent` inflates MAP almost universally (14/15 rows); here it's nearly a coin flip |

The **split divergence is itself a finding worth flagging**: `exp3.1`'s
"recent split inflates MAP via temporal proximity" locked finding assumed
query and index share vocabulary, so a `recent`-split test task's
neighbors in the training window are *literally* worded similarly to it.
In `cross` mode the query (ticket) and index (commit) never share that
vocabulary, so the temporal-proximity shortcut that inflated `recent`
elsewhere has much less to grab onto — consistent with the "cross mode
answers a genuinely different, less leaky question" framing from
`README.md`.

**Sanity check** — the closest thing to a "diagonal" comparison against
`exp3.1`'s matched `commit` mode (module target, `modn` split): `cross`'s
`title`→`title` = 0.526 and `desc`→`desc` = 0.549, vs. `exp3.1` matched
`commit`-mode's `title` = 0.572 and `desc` = 0.577. Same order of
magnitude, `cross` running a few points below as expected (it lacks the
self-referential vocabulary match), not a wild divergence that would
suggest a bug.

---

## Best configurations (module target, `modn` split — the honest slice)

| train_source | query_source | window | MAP |
|---|---|---|---|
| desc | comments | w100 | 0.592 |
| desc | desc | w100 | 0.592 |
| diff | desc | w100 | 0.582 |
| title | desc | w100 | 0.579 |
| diff | comments | w100 | 0.576 |

Best single project/row overall: **`rubocop`, `title`→`title`, `w100`,
`recent` split: MAP 0.793**, MRR 0.883 — module target. `rubocop` is the
strongest project for cross-vocabulary retrieval across nearly every
configuration tested, `vscode` and `sonar` the weakest.

---

## Key findings

1. **Cross-vocabulary retrieval is a viable substitute for matched-ticket
   retrieval**, not just a diagnostic curiosity — beats it outright in 1/3
   of projects tested, stays within ~13% in most of the rest. This is the
   empirical case for the `codeXport` framing in `AGENTS.md` (code as the
   business's "digital emanation," findable without shared vocabulary).
2. **`diff` content reliably strengthens the index** (6/9 projects) — the
   one clean, directional result in this grid.
3. **No query-side variant reliably wins** — `title` wins on sign-count
   despite `desc` winning on pooled mean, a genuine Simpson's-paradox
   trap. Report sign-count, not pooled mean, for this axis specifically.
4. **The `recent`-split-inflation finding from `exp3.1` does not
   generalize to cross-vocabulary retrieval** — it's a property of
   matched-vocabulary temporal leakage, not a property of the retrieval
   task in general.
5. **Module >> file holds universally** (9/9 projects), same as every
   prior experiment in this line of work.

## Caveats

- **`bge-small` only** — no cross-check yet against `bge-large` or other
  architectures for this specific grid (deliberately deferred, see
  `EXPERIMENT_PLAN.md`'s model-scope note).
- **Pooled means can mislead** (see Q2) — sign-count across the 9 projects
  is the more trustworthy summary wherever the two disagree.
- **`agilebill` untested** here (no ticket data) — this experiment says
  nothing about the pure-git, no-tracker case.
