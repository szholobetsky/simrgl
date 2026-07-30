# exp3.2 Implementation Plan

What to copy from `exp3.1`, what to change, and why. Nothing in this file
has been executed yet — this is the plan, written for review before any
code is touched. File/line references point at `exp3.1`'s current state
(as of this session).

`exp3.1` accumulated a lot of pre-multi-project legacy (`.bat` files, old
single-DB scripts predating `--project`/`--task-unit`, Windows-only
duplicates). This plan only lists what's actually load-bearing for the
active pipeline (per `exp3.1/PROGRESS_LOG.md` and `EXPERIMENT_PLAN.md` §7)
— everything else is skipped, listed explicitly in §3 so it's a checked
decision, not an oversight.

## 1. Copy verbatim, no changes

| File | Why unchanged |
|---|---|
| `checkpoint_manager.py` | `source` param is opaque, just interpolated into a string ID — the composite `train_source_qquery_source` trick (see `EXPERIMENT_PLAN.md` §4) needs zero changes here. |
| `vector_backends.py` | Search/upsert/collection-create logic doesn't know or care what built the text that got embedded. |
| `gpu_utils.py` | Model-cleanup/memory utilities, no source-variant awareness. |
| `status.py` | Globs `experiment_results/*/*/checkpoint.json` generically — doesn't hardcode `ticket`/`commit`, so `cross` should just work. **Verify this assumption when copying** — read it once before trusting it. |
| `aggregate_results.py` | Same generic-glob argument as `status.py`. Output columns come from whatever's in each `comprehensive_results.csv`, so it'll pick up `train_source`/`query_source` automatically once `utils.format_metrics_row` emits them (§2). **Verify, don't assume.** |
| `run_postgres.sh` | Idempotent Postgres-container bring-up, no experiment-logic coupling. |
| `switch_project.sh` | Between-project Postgres schema cleanup — still needed since `exp3.2` also loops over 9 projects sequentially. No source-variant awareness to break. |
| `postgres-compose.yml`, `requirements.txt`, `requirements_postgres.txt`, `.gitignore` | Infra/env, unrelated to the source-axis split. |
| `monitor.sh`, `monitor_con.sh`, `monitor_dashboard.py`, `monitor_dashboard.sh` | Log/checkpoint-scraping dashboards. `monitor_dashboard.py`'s Level-2 progress bar (72-variant grid) will need its denominator updated for the 216-variant grid — flagged in §2, everything else copies clean. |

## 2. Copy and modify

### `config.py`

- Add `'cross'` to `TASK_UNITS` (currently `['ticket', 'commit']`,
  `config.py:21`). `TASK_UNIT`/`--task-unit` plumbing elsewhere doesn't
  need further changes — it's already just a string used for path/
  collection namespacing.
- Split `SOURCE_VARIANTS` (`config.py:146-173`) into two dicts:
  - `TRAIN_SOURCE_VARIANTS` = today's `title`/`desc`/`diff` (the
    commit-mode-relevant subset — drop `comments`, which is always `''`
    for a raw commit).
  - `QUERY_SOURCE_VARIANTS` = today's `title`/`desc`/`comments` (the
    ticket-mode-relevant subset — drop `diff`, which doesn't exist for a
    ticket).
  - Keep the dict *values* (field lists, names, descriptions) identical to
    today's `SOURCE_VARIANTS` entries — only the grouping changes.
- `collection_name()` (`config.py:39-65`) — **no signature change**. Call
  sites pass `source=train_source, task_unit='cross'`; the 63-byte
  truncation-safety logic already handles arbitrarily-shaped names.

### `etl_pipeline.py`

- `generate_embeddings()` (`etl_pipeline.py:333-374`) currently reads
  `config.SOURCE_VARIANTS[source_variant]` (`etl_pipeline.py:350`) — point
  this at `config.TRAIN_SOURCE_VARIANTS` instead. This is the only change
  needed in this file; `load_data()` / `_build_commit_tasks()` /
  `create_split()` / `apply_time_window()` / `aggregate_by_target()` are
  reused **unmodified** by calling them exactly as `exp3.1`'s `commit` mode
  already does (`task_unit='commit'` internally for the index-build half —
  `exp3.2` never needs a `task_unit='cross'` branch inside `etl_pipeline.py`
  itself, only in the orchestrator's collection-naming, since the
  synthetic-commit-task machinery is identical either way).
- Ticket-side test-set generation (`load_data()` with `task_unit='ticket'`,
  `create_split()`, `save_test_set()`) is also called unmodified — just
  from the new orchestrator, with `task_unit` forced to `'ticket'`
  regardless of the outer `cross` context (see `EXPERIMENT_PLAN.md` §2).

### `run_experiments.py` (`ExperimentRunner`)

This is where the real signature change lives — today one `source_variant`
string drives both collection lookup and query encoding.

- `encode_queries()` (`run_experiments.py:94-124`) — reads
  `config.SOURCE_VARIANTS[source_variant]` (`run_experiments.py:109`) —
  point at `config.QUERY_SOURCE_VARIANTS` instead.
- `evaluate_experiment()` / `run_single_experiment()`
  (`run_experiments.py:158-267`, `269-368`) — add a `query_source`
  parameter alongside the existing `source_variant` (which becomes, in
  effect, `train_source`):
  - `collection_name` is still built from `train_source` only (unchanged
    call shape: `config.collection_name(train_source, target, window,
    strategy, model_key=...)`).
  - `encode_queries(self.test_set, query_source)` — now driven by the new
    param, not `source_variant`.
  - `experiment_id` needs both: e.g.
    `f"{train_source}_q{query_source}_{target}_{window}_{strategy}"`.
  - Result row needs both columns — depends on the `utils.format_metrics_row`
    change below.

### `run_comprehensive_experiments.py` (`ComprehensiveExperimentRunner`)

The most involved change — the loop restructuring described in
`EXPERIMENT_PLAN.md` §1.

- Constructor: replace `self.sources` with `self.train_sources` (default
  `list(config.TRAIN_SOURCE_VARIANTS.keys())`) and add
  `self.query_sources` (default `list(config.QUERY_SOURCE_VARIANTS.keys())`).
- `get_total_variants()` (`run_comprehensive_experiments.py:116-124`) —
  multiply by both `len(self.train_sources)` and `len(self.query_sources)`.
- `_process_source_window()` (`run_comprehensive_experiments.py:184-254`)
  is today: `for target: etl(target) -> eval(target) -> cleanup(target)`.
  Restructure to:
  ```
  for target in self.targets:
      etl_success = _run_etl_for_target(..., train_source, target, window, ...)
      if not etl_success: continue

      for query_source in self.query_sources:
          run_experiment_variant(..., train_source, query_source, target, window, ...)

      _cleanup_collection(..., train_source, target, window, ...)
  ```
  i.e. push the `query_source` loop *inside* the per-target block, and move
  `_cleanup_collection()` to run once after all three query_sources are
  done, not after each one. This is the change that keeps the "eval is
  cheap, embedding is expensive" property from `README.md` actually true.
- `run_experiment_variant()` (`run_comprehensive_experiments.py:256-347`) —
  add `query_source` param, thread through to `ExperimentRunner.run_single_experiment()`
  (§2 above), and to checkpoint calls:
  `mark_experiment_completed(model_key, strategy, f"{train_source}_q{query_source}", target, window)`.
- `run_all()` (`run_comprehensive_experiments.py:384-465`) — the top of the
  loop currently does `pipeline.load_data()` once per `(model, strategy)`
  and immediately `save_test_set()` using that same `pipeline`'s
  `task_unit` (`run_comprehensive_experiments.py:424-428`). For `exp3.2`
  this needs **two** separate `load_data()` calls per `(model, strategy)`:
  one with `task_unit='commit'` (→ `train_tasks_all`, feeds the embedding/
  aggregation side, unchanged shape) and one with `task_unit='ticket'`
  (→ query-side `test_tasks`, feeds `save_test_set()` only). Both share the
  same `ETLPipeline` class, just instantiated/called twice with different
  `task_unit`.
- `save_results()` (`run_comprehensive_experiments.py:467-498`) — column
  order list needs `train_source`/`query_source` in place of the single
  `source` column.

### `utils.py`

- `format_metrics_row()` (`utils.py:175-213`) — replace the single
  `source: str` param with `train_source: str, query_source: str`; emit
  both as separate columns in the returned row dict instead of one
  `source` key.
- Everything else (`combine_text_fields`, `calculate_metrics_for_query`,
  `aggregate_metrics`, `extract_file_path`, `extract_module_path`,
  `get_commits_table_name`) copies unmodified — none of them know about
  the source axis at all.

### New driver script: `run_cross_experiment.sh`

Adapted from `exp3.1/run_full_experiment.sh`: same per-project loop
(`run_postgres.sh` sanity check → `switch_project.sh` → orchestrator →
`status.py`/`aggregate_results.py`), same `flock`-based single-instance
lock (`exp3.1/PROGRESS_LOG.md`, 2026-07-28 entry — the double-launch
incident, worth inheriting the fix preemptively rather than rediscovering
it), but:
- 9 projects only, **no `ticket`/`commit` alternation** — every project
  runs once, since `task_unit='cross'` internally touches both `TASK` and
  `COMMITS` tables in a single pass.
- No `agilebill` step at all (not even a commit-only one — `exp3.2` has no
  reduced mode for it, since it categorically lacks a query side).

## 3. Explicitly not copied (legacy, dead since the exp3→exp3.1 port)

`BIG_EXPERIMENT.sh`, all `backup_*`/`restore_*`/`clear_postgres_vectors*`
scripts, `check_task_collections.bat`, `create_missing_task_embeddings.*`,
`create_tasks*.{bat,sh}`, `create_task_collection.py`, `docker-compose.yml`
(Qdrant — `exp3.1` runs on the Postgres backend), all `.bat` files
(Windows-only, this environment is Linux), `quick_start.*`,
`recreate_file_collection.*`, `rerun_experiments.bat`, `resume_etl.py`,
`run_comprehensive_experiment.sh` (old singular-project predecessor of
`run_full_experiment.sh`), `run_etl_*.{bat,sh}` (pre-`--project` era),
`run_subset_experiment.sh`, `start*.{bat,sh}`, `test_postgres.py`. Also
skipping the narrative/reference docs that don't describe active code:
`COMPREHENSIVE_EXPERIMENT_GUIDE.md`, `EXPERIMENT_RESULTS.md`,
`README_EXPERIMENTS.md`, `TASK_EMBEDDINGS.md`, `VENV_SETUP.md`,
`POSTGRES_SETUP.md`, `research_questions.md`, `walkthrough.md`,
`implementation_plan.md` (the old pre-`exp3.1` one — superseded by this
file for `exp3.2`'s purposes).

## 4. Verification plan before the full launch

Mirrors how `exp3.1` verified its embedding-caching fix
(`PROGRESS_LOG.md`, 2026-07-25 "later still" entry) before trusting
projected timings — small, targeted checks, not the full grid:

1. **Ticket test-set regeneration matches `exp3.1`'s**: run `exp3.2`'s
   local `create_split()`+`save_test_set()` for one project/strategy/model
   already completed in `exp3.1`, diff the resulting JSON against
   `exp3.1/experiment_results/<project>/ticket/test_set_<strategy>_<model>.json`.
   Should be byte-identical (deterministic split) — if not, something in
   the copy diverged and needs finding before trusting anything downstream.
2. **Collection built once, queried three times**: run one
   `(project=celery, model=bge-small, strategy=recent, window=w100,
   train_source=desc)` cell with all three `query_source` values, grep the
   log for `"Generating embeddings for source variant"` — expect exactly
   **one** occurrence, not three (same check `exp3.1` used to verify its
   own caching fix).
3. **Checkpoint resume across the composite ID**: kill the process
   mid-way through the three `query_source` evals for one target, restart,
   confirm the completed ones `[SKIP]` and only the remaining ones re-run.
4. **Results shape**: confirm `comprehensive_results.csv` has both
   `train_source` and `query_source` columns, 9 rows for that one
   `(model,strategy,window,target)` cell (3×3), and that `status.py`/
   `aggregate_results.py` don't choke on the new column shape.
5. **Timing**: time the smoke-test cell for real, use it to replace the
   qualitative estimate in `EXPERIMENT_PLAN.md` §7 with a real number
   before committing to the full 9-project run.

## 5. Open decisions before writing code

- Exact CLI shape for `run_comprehensive_experiments.py --train-sources
  ... --query-sources ...` vs. keeping a single `--sources` flag that
  populates both when unspecified — minor, but affects the smoke-test
  commands in `EXPERIMENT_PLAN.md` §5/§7.
- Whether `monitor_dashboard.py`'s Level-2 progress bar should show the
  216-variant total per project as one number, or break it out as
  `train_source × query_source` sub-progress the way it currently parses
  "Processing variant" log lines — cosmetic, doesn't block correctness,
  can be decided while implementing.
