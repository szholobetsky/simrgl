# exp3.2 Execution Plan

Mechanics of running the cross-vocabulary grid described in `README.md`:
loop order, directory layout, checkpointing, resume protocol, and how the
results answer the two marginal questions (diff-on-index, comments-on-query).
Written to survive being picked up cold in a brand new session, the same
way `exp3.1/EXPERIMENT_PLAN.md` was — every fact needed to resume lives on
disk, not in a running process or in conversation history.

**Implemented and unit-verified** (Phase A, this session — see `PROGRESS_LOG.md`
once it exists for the full account). Not yet run against real data: that's
Phase B, gated on `exp3.1` finishing (now done — see §8) and the
model-scope decision below.

**Model scope (decided after `exp3.1` finished):** `bge-small` only, not
`bge-small` + `bge-large`. `exp3.1`'s completed run found `bge-large`'s
MAP gain over `bge-small` statistically significant but small on every
project except `vscode` — which OOM'd on `bge-large` entirely, so no
`bge-large` comparison exists there at all. Given that, doubling `exp3.2`'s
already-new 3×3 grid to also sweep a second model — one whose payoff was
already shown to be marginal — isn't worth it for this first pass. Other
architectures (`nomic-embed-text`, `exp4`'s model families) are explicitly
deferred rather than folded in now: the point of this round is to isolate
the cross-vocabulary question on its own, not stack multiple unproven
variables into one run.

## 1. Execution order

```
for project in [celery, rubocop, pulumi, sonar, flink, hadoop, spark, kubernetes, vscode]:
    for split_strategy in [recent, modn]:
        model = bge-small  # see model-scope note above
        for window in [w100, w1000, all]:
            for train_source in [title, desc, diff]:
                # index build — happens ONCE per (project,split,model,window,train_source,target)
                for target in [file, module]:
                    run ETL -> build collection from commit-message centroids

                # eval — fans out over query_source AFTER the collection exists
                for target in [file, module]:
                    for query_source in [title, desc, comments]:
                        run eval -> MAP/MRR/P@k/R@k, query = ticket test set

                # cleanup — after every query_source has evaluated this target
                for target in [file, module]:
                    delete collection
```

Same project ordering as `exp3.1` (smallest task-count first: `celery`
→ `vscode`), same rationale — an interruption early leaves the broadest
partial dataset instead of stalling mid-`vscode`.

**Critical difference from `exp3.1`'s loop**: the ETL step (embedding +
aggregate + upsert) must complete for **all three `query_source` evals**
before the collection gets cleaned up. `exp3.1`'s
`_process_source_window()` builds a collection, evaluates it once, and
deletes it immediately (`_cleanup_collection()` right after
`run_experiment_variant()`) — reusing that structure unmodified for
`exp3.2` would rebuild the same collection three times, once per
`query_source`, silently defeating the whole "eval is cheap, embedding is
expensive" argument in `README.md`. See `IMPLEMENTATION_PLAN.md` for the
exact loop restructuring this requires in
`run_comprehensive_experiments.py`.

Total variant-runs: 9 projects × 2 splits × 1 model × 3 windows × 3
train_source × 3 query_source × 2 targets = **972 eval-runs**, backed by
9 × 2 × 1 × 3 × 3 × 2 = **324 ETL/collection-builds** (half `exp3.1`'s
`commit`-mode ETL work across all 9 projects, since that ran both models —
see §7 for the timing argument).

## 2. Data access

Same principle as `exp3.1` §2: the 9 project `.db` files stay at
`/home/stzh/Projects/db/*.db`, never copied. `exp3.2` opens its own
read-only `sqlite3.connect()` per run.

Two things get read per project, independently of each other:
- **`COMMITS`/`RAWDATA`** table → `_build_commit_tasks()` → index-side
  training texts (same code path as `exp3.1` `commit` mode).
- **`TASK`** table → `create_split()` + `save_test_set()` → query-side test
  set (200 held-out tickets + ground truth). Regenerated locally per
  `(project, split_strategy, model)` rather than read from
  `exp3.1/experiment_results/<project>/ticket/` — see `README.md`
  "Dependency on exp3.1" for why this is safe (deterministic split,
  byte-identical either way) and preferred (keeps `exp3.2` self-contained).

## 3. Directory / file layout

```
exp3.2/experiment_results/
  <project>/
    cross/
      checkpoint.json                        # resume state
      test_set_ticket_recent_bge-small.json  # regenerated locally, ticket-side
      comprehensive_results.csv              # train_source + query_source columns
      run.log
  STATUS.md                                  # regenerated rollup (status.py)
  all_projects_results.csv                   # merged dataset (aggregate_results.py)
```

Directly mirrors `exp3.1`'s `<project>/<task_unit>/` pattern with
`task_unit='cross'` — same isolation guarantee (one project's crash can't
touch another's checkpoint), same `rm -rf experiment_results/<project>/`
reset story.

One naming note: the query-side test-set filename gets a `ticket_` prefix
(`test_set_ticket_<strategy>_<model>.json`) to make explicit — in a
directory literally named `cross` — that this file holds Jira ticket text,
not commit text, even though the collections being queried in that same
directory were built from commits.

## 4. Checkpointing

Reuses `checkpoint_manager.py` **unchanged** — no new methods, no schema
change. The trick: `CheckpointManager`'s `source` parameter is just
interpolated into a string ID, so it doesn't need to know there are now
two source axes:

- **ETL checkpoint** (doesn't depend on `query_source`):
  `mark_etl_completed(model, strategy, train_source, target, window)` —
  called exactly as `exp3.1`'s commit-mode ETL already does, `train_source`
  passed positionally where `source` used to go.
- **Experiment checkpoint** (does depend on both):
  `mark_experiment_completed(model, strategy, f"{train_source}_q{query_source}", target, window)`
  — the composite string (e.g. `desc_qcomments`) becomes the `source` slot,
  giving each of the 9 `(train_source, query_source)` pairs its own unique
  variant ID without touching `CheckpointManager` at all.

```python
CheckpointManager(f"experiment_results/{project}/cross/checkpoint.json")
```

## 5. Resume protocol

Same shape as `exp3.1` §5:

1. Read `PROGRESS_LOG.md` first (once it exists) for the narrative — what
   was tried, decided, and why.
2. Read `experiment_results/STATUS.md` — one row per project, `done/972`
   variant count (or per-project sub-totals if `status.py` is extended to
   show train/query breakdown — TBD, not required for correctness).
3. Resume the first non-`done` project:
   ```bash
   cd exp3.2
   python3.12 run_comprehensive_experiments.py --project <name> \
       --models bge-small --strategies recent modn --task-unit cross
   ```
   Checkpoint skip-logic (`is_etl_completed` / `is_experiment_completed`)
   picks up mid-grid automatically.
4. If `STATUS.md` looks stale, regenerate from checkpoints (source of
   truth, always): `python3.12 status.py`.
5. Append a `PROGRESS_LOG.md` entry before and after each work session.

## 6. Aggregating results — answering the two marginal questions

`aggregate_results.py` walks `experiment_results/*/cross/comprehensive_results.csv`,
adds a `project` column, concatenates into `all_projects_results.csv`. Same
script `exp3.1` uses, same idempotent/safe-to-run-partial behavior — the
glob pattern doesn't hardcode `task_unit` names, so no change should be
needed (verify when implementing — see `IMPLEMENTATION_PLAN.md` §4).

With `train_source` and `query_source` as separate columns, the two
questions from `README.md` become simple `groupby` operations:

```python
# Does diff help the index, holding query fixed?
df.groupby(['query_source', 'train_source'])['MAP'].mean().unstack()
# read down each column: title -> desc -> diff, same query_source

# Does comments help the query, holding index fixed?
df.groupby(['train_source', 'query_source'])['MAP'].mean().unstack()
# read across each row: title -> desc -> comments, same train_source
```

As with `exp3.1`'s ticket-vs-commit comparison, report **sign-count across
the 9 projects**, not just a pooled mean — `kubernetes`/`vscode` have far
more commits/tasks than `celery`/`rubocop` and would dominate a naive
average. Also worth checking the **diagonal cells**
(`title`-`title`, `desc`-`desc`, `diff`-`comments`) against `exp3.1`'s old
`commit`-mode numbers as a sanity cross-check — they're not the same
measurement (different query text, ticket vs. commit), so don't expect
exact agreement, but a wildly different order of magnitude would flag a
bug before trusting the rest of the grid.

## 7. Timing — real measurement (Phase B smoke test, 2026-07-30)

Measured directly: `celery` (cheapest project), `bge-small`, `recent`
split, `w100` window, one `train_source` at a time, all 3 `query_source`
values, both targets (`file`+`module`) — i.e. one full `(train_source,
window, split, model)` cell = 1 embedding pass + 2 ETL builds + 6 evals.

- **`train_source=desc` cell, cold (no checkpoint), end to end: 40.4s.**
  Confirmed exactly **1** `"Generating embeddings for source variant"`
  call across both targets and all 3 query_sources — the target-sharing
  and query-fan-out caching both hold as designed (see §1's critical
  loop-order requirement).
- **`train_source=title` cell, same shape, run in two parts** (interrupted
  after 2 of 3 `file`-target evals via `timeout 22s`, then resumed):
  first part did ETL(file) + 2 evals in ~8s before being killed; resumed
  run correctly `[SKIP]`ped the completed ETL (5-part id, no `_q` suffix)
  and the 2 completed evals (6-part composite id), re-ran only the
  interrupted 3rd eval, then did ETL(module) + all 3 module evals — total
  resumed-run wall time 18.4s, no wasted recomputation, no double-counted
  checkpoint entries, `failed_etl`/`failed_experiments` both empty
  afterward.
- **Real cost breakdown, not previously visible from `exp3.1`'s numbers**:
  the query-encoder model (`SentenceTransformer`) reloads fresh **once per
  eval** — 6 reloads per cell here, ~2.3s each ≈ 13.8s, roughly a third of
  the 40.4s total for `w100` (where embedding-generation itself is nearly
  free — only ~100 training texts). This didn't exist as a cost in
  `exp3.1`'s matched-pair design (1 source → 1 eval, not 3), so it's new
  to `exp3.2` specifically. Flagged, not fixed (see `IMPLEMENTATION_PLAN.md`'s
  risk list) — worth revisiting if it turns out to dominate on the larger
  windows/projects where it won't be diluted by a bigger embedding step.

**Extrapolation, with an explicit caveat**: at `w100`, one full
`(train_source, split)` pass costs ~40s on `celery`, so one full `(split,
window=w100)` slice (3 train_sources) ≈ 2 min, and both splits ≈ 4 min —
for `w100` only. `w1000` and `all` will cost more (embedding-generation
time grows with training-set size — `all` is celery's full ~2,400-task
train set, `w1000` is capped at 1,000 — while the eval-side model-reload
+200-query-encode cost stays fixed regardless of window), so this is not
a flat multiply-by-9-projects estimate. `celery` is also the smallest
project by task count, so every other project will take longer in
absolute terms. Treat "a few minutes per project at w100, more at
w1000/all" as the honest current estimate — a real number for `w1000`/`all`
needs its own timed run before the full 9-project launch, not extrapolated
from `w100` alone.

## 8. Relationship to `exp3.1`

- **`exp3.1`'s full grid has finished.** `vscode`/`bge-large` failed with
  an out-of-memory error and was not completed — every other
  `(project, task_unit, model)` cell finished. This is now the basis for
  the model-scope decision above, not a blocker for `exp3.2` starting.
- Ran **after** `exp3.1`'s full grid finished, by design — shared GPU,
  shared Postgres instance, no benefit to interleaving.
- Shares the same Postgres container (`semantic_vectors_db`) and the same
  `switch_project.sh` between-project cleanup pattern — collections are
  namespaced by `task_unit='cross'` in `config.collection_name()`, so there
  is no naming collision with anything `exp3.1` built or will build, even
  if runs somehow overlapped.
- Does not read or write anything under `exp3.1/experiment_results/` — the
  ticket-side test set is regenerated locally (§2), not read across the
  folder boundary.
