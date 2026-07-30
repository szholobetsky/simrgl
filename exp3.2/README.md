# exp3.2: Cross-vocabulary retrieval — commit-built index, ticket-language queries

`exp3.1` added a `commit` task-unit criterion so tracker-less/weakly-linked
projects could still be evaluated, by treating one commit as one synthetic
"task" (query = commit message, ground truth = that commit's own changed
files). That mode is methodologically self-referential: the query text
(commit message) and the index text (other commits' messages) describe the
*same kind of already-happened change*, so a match is closer to "this
commit's wording resembles other commits that touched this file" than to
"predict which files a not-yet-written change will touch." It has some
value as a differential/diagnostic signal (see `exp3.1/PROGRESS_LOG.md`),
but it does not answer the actual product question: *given a task
description written **before** the code change exists, in business
language, which files should change?*

`exp3.2` answers that question directly by **decoupling which text builds
the retrieval index from which text encodes the query**:

- **Index (target) side**: per-file / per-module centroid vectors built
  from **commit messages** — the code-side, technical vocabulary. Same
  construction as `exp3.1`'s `commit` mode (`_build_commit_tasks()`,
  noise-filtered: merges / >50-file mass-diffs / <8-char subjects dropped).
- **Query side**: the same 200 held-out **Jira tickets** `exp3.1`'s
  `ticket` mode already uses — business-language `TITLE`/`DESCRIPTION`/
  `COMMENTS`, written before any of the linked commits existed.
- **Ground truth**: unchanged — all files across every commit linked to
  that ticket via `TASK_NAME` (same as `exp3.1` ticket mode).

This is a genuine train→predict split in vocabulary as well as in time,
which the `commit`-mode self-match wasn't.

## Cartesian grid, not matched pairs

`exp3.1` only ever ran matched source pairs (`title`↔`title`, `desc`↔`desc`,
`comments`↔`diff` as the "noisy extra context" slot on each side). That
conflates two separate questions into one number. `exp3.2` splits the
`source` axis into two independent axes and runs the full 3×3 product:

|                      | query: `title` | query: `desc` | query: `comments` |
|----------------------|:--:|:--:|:--:|
| **train: `title`**   | • | • | • |
| **train: `desc`**    | • | • | • |
| **train: `diff`**    | • | • | • |

- **`TRAIN_SOURCE`** (index side, commit vocabulary): `title` / `desc`
  (title+message) / `diff` (title+message+diff content, capped at 4000
  chars — same as `exp3.1`'s commit-mode `diff` variant).
- **`QUERY_SOURCE`** (query side, ticket vocabulary): `title` / `desc`
  (title+description) / `comments` (title+description+comments).

This isolates the two effects `exp3.1` couldn't separate: does adding
`diff` content to the **index** help, independent of how the query is
phrased? Does adding `comments` to the **query** help, independent of what
built the index? The diagonal (`title`-`title`, `desc`-`desc`,
`diff`-`comments`) reproduces something close to `exp3.1`'s old commit-mode
numbers as a sanity check; the off-diagonal cells are the new information.

## Full grid

```
TRAIN_SOURCE × QUERY_SOURCE × TARGET × WINDOW × SPLIT × MODEL
     3       ×      3       ×   2    ×   3    ×   2   ×   2   = 216 variants/project
```

- **`TARGET`**: `file` / `module` (unchanged from `exp3.1`).
- **`WINDOW`**: `w100` / `w1000` / `all` (unchanged).
- **`SPLIT`**: `recent` / `modn` (unchanged).
- **`MODEL`**: `bge-small` / `bge-large` (unchanged — still the only two
  that fit this GPU's 6GB VRAM).

No subsetting on any axis — same "full grid, every project" principle
`exp3.1`'s `PROGRESS_LOG.md` (2026-07-25 entry) settled on after an earlier
attempt to trim it turned out to rest on an unstated, unverified
assumption.

**Cost note**: index-side embedding work depends only on `TRAIN_SOURCE`
(not `QUERY_SOURCE`), so the actual GPU-heavy step — embedding training
texts, aggregating to centroids, upserting — is the *same* 72
`(train_source, target, window, split, model)` builds `exp3.1`'s
`commit`-mode grid already did per project. What triples is the **eval**
step (encode 200 query texts, vector search, compute metrics) — cheap
relative to embedding generation. Expect total wall-clock closer to
`exp3.1`'s commit-mode per-project time plus a moderate eval-time
increment, not a flat 3×. Worth confirming with a small smoke test before
committing to the full run (see `EXPERIMENT_PLAN.md` §7).

## Projects

Same 9 projects `exp3.1` uses for the ticket/commit comparison — every
project where `db/project_stats.csv` marks **both**
`ticket_criterion_usable` and `commit_criterion_usable` as `True`:
`celery`, `flink`, `hadoop`, `kubernetes`, `pulumi`, `rubocop`, `sonar`,
`spark`, `vscode`. `exp3.2` needs both sides (tickets for queries, commits
for the index), so this is the natural project set — no new inventory
work needed, `exp3.1/db/project_stats.csv` already answers it.

**`agilebill` is excluded** — it has zero `TASK` rows, so there is no
query side at all. This isn't a shortcut inferred from another project;
it's a hard fact about `agilebill` (`ticket_criterion_usable=False`).

## Dependency on `exp3.1` (and why it's shallow)

`exp3.2` needs the per-project ticket-mode test set — 200 held-out tasks
with `TITLE`/`DESCRIPTION`/`COMMENTS` and ground-truth files — the same
artifact `exp3.1`'s `run_comprehensive_experiments.py` already writes to
`exp3.1/experiment_results/<project>/ticket/test_set_<strategy>_<model>.json`.

Rather than reading that file across the `exp3.1`/`exp3.2` folder
boundary, `exp3.2` **regenerates it locally** (cheap — `create_split()` +
`save_test_set()`, no embedding involved) so the experiment is
self-contained and doesn't depend on `exp3.1`'s run having reached that
project yet. The split is a deterministic function of `(project, split
strategy, test_size)`, so the regenerated file is byte-identical to
`exp3.1`'s — this is a convenience decision, not a methodological one.

**Operationally**, `exp3.2` should still be launched only after `exp3.1`'s
run finishes (per the plan discussed in this session) — both share the
one GPU and the one Postgres instance, and running them concurrently would
just mean context-switching the same hardware, not real parallelism.
Collection names are namespaced by `task_unit='cross'`, so there's no
naming collision risk either way — this is about resource contention, not
correctness.

## What this will tell us

- **Marginal effect of `diff` on the index**, holding query fixed: compare
  `title→title` vs `desc→title` vs `diff→title` (and same for the other
  two query columns). Answers "does technical diff content in the index
  help a business-language query find the right files?" — independent of
  `exp3.1`'s conflated commit-mode result.
- **Marginal effect of `comments` on the query**, holding index fixed:
  compare `desc→title` vs `desc→desc` vs `desc→comments` (and same for the
  other two index rows). Answers "does a richer ticket query help against
  a fixed commit-message index?"
- **Cross-vocabulary retrieval quality itself**: does a business-language
  query ever find a technical-vocabulary index at all competitively with
  `exp3.1`'s ticket-mode numbers (where index and query are both built
  from ticket text)? This is the number that actually matters for the
  product use case in `AGENTS.md`'s `codeXport` concept (business terms ↔
  code identifiers).

See `EXPERIMENT_PLAN.md` for execution mechanics (directory layout,
checkpointing, resume protocol, aggregation) and `IMPLEMENTATION_PLAN.md`
for what gets copied from `exp3.1` and what code actually needs to change.
