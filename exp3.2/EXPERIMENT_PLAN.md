# exp3.2 Execution Plan

Mechanics of running the cross-vocabulary grid described in `README.md`:
loop order, directory layout, checkpointing, resume protocol, and how the
results answer the two marginal questions (diff-on-index, comments-on-query).
Written to survive being picked up cold in a brand new session, the same
way `exp3.1/EXPERIMENT_PLAN.md` was — every fact needed to resume lives on
disk, not in a running process or in conversation history.

**Not yet implemented** — this is the plan; `IMPLEMENTATION_PLAN.md` covers
what code needs to exist before any of this can run.

## 1. Execution order

```
for project in [celery, rubocop, pulumi, sonar, flink, hadoop, spark, kubernetes, vscode]:
    for split_strategy in [recent, modn]:
        for model in [bge-small, bge-large]:
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

Total variant-runs: 9 projects × 2 splits × 2 models × 3 windows × 3
train_source × 3 query_source × 2 targets = **1,944 eval-runs**, backed by
9 × 2 × 2 × 3 × 3 × 2 = **648 ETL/collection-builds** (same order of
magnitude as `exp3.1`'s `commit`-mode ETL work across all 9 projects — see
§7 for the timing argument).

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
2. Read `experiment_results/STATUS.md` — one row per project, `done/1944`
   variant count (or per-project sub-totals if `status.py` is extended to
   show train/query breakdown — TBD, not required for correctness).
3. Resume the first non-`done` project:
   ```bash
   cd exp3.2
   python3.12 run_comprehensive_experiments.py --project <name> \
       --models bge-small bge-large --strategies recent modn
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

## 7. Timing estimate

No direct measurement yet (nothing built). Reasoning from `exp3.1`'s
already-observed numbers:

- **ETL side**: `exp3.2`'s embedding workload per project
  (`train_source × target × window × split × model` = 3×2×3×2×2 = 72
  collection-builds) is the *same size* as `exp3.1`'s `commit`-mode ETL
  grid, which has already run to completion for most of the 9 projects
  (see `exp3.1/PROGRESS_LOG.md` — e.g. `pulumi`'s full 72-variant sweep
  took roughly ~2h on this GPU). So the ETL portion of `exp3.2` should cost
  roughly the same wall-clock as `exp3.1`'s `commit`-mode run already did,
  summed across the 9 projects.
- **Eval side**: triples relative to `exp3.1`'s commit-mode (216 evals per
  project instead of 72), but each eval is "encode 200 short ticket texts
  + vector search + metric computation" — small next to embedding an
  entire project's training set. Not measured directly in this codebase
  yet.

**Recommended before launching the full run**: a smoke test — one project
(`celery`, cheapest), one model, one split, one window, one `train_source`,
all three `query_source` values — timed directly, mirroring how `exp3.1`
verified its embedding-caching fix before trusting the projected numbers
in its own `EXPERIMENT_PLAN.md` §6a. Put a real number in this section
once that's run instead of the qualitative argument above.

## 8. Relationship to `exp3.1`

- Runs **after** `exp3.1`'s full grid finishes — shared GPU, shared
  Postgres instance, no benefit to interleaving.
- Shares the same Postgres container (`semantic_vectors_db`) and the same
  `switch_project.sh` between-project cleanup pattern — collections are
  namespaced by `task_unit='cross'` in `config.collection_name()`, so there
  is no naming collision with anything `exp3.1` built or will build, even
  if runs somehow overlapped.
- Does not read or write anything under `exp3.1/experiment_results/` — the
  ticket-side test set is regenerated locally (§2), not read across the
  folder boundary.
