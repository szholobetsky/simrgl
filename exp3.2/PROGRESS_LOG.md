# exp3.2 Progress Log

Chronological, append-only record of what was done, why, and what's next.
`EXPERIMENT_PLAN.md` is the stable design doc; `STATUS.md` (once it
exists) is an auto-generated numeric rollup from checkpoints; **this file
is the narrative** — read it to understand what's been tried and decided,
not just what's finished. Newest entries at the bottom.

---

## 2026-07-29 — Phase A implementation complete, verified end-to-end (no GPU/Postgres)

**Context**: `exp3.1`'s `commit` task-unit mode turned out to be
methodologically self-referential — query text and index text were both
commit messages, so a "match" was closer to stylistic resemblance between
commits than genuine prediction. `exp3.2` decouples index-building text
(commit messages) from query text (the same 200 held-out Jira tickets
`exp3.1`'s `ticket` mode uses), and runs the full 3×3 Cartesian product of
`train_source` × `query_source` instead of matched pairs. Design fully
worked out and approved across `README.md`/`EXPERIMENT_PLAN.md`/
`IMPLEMENTATION_PLAN.md` in the same session, then implemented via an
ordered, independently-verifiable plan (see
`/home/stzh/.claude/plans/binary-plotting-sky.md`).

**Done** (all of Phase A — pure code, no GPU/Postgres needed):
- Copied 14 files verbatim from `exp3.1` (`checkpoint_manager.py`,
  `vector_backends.py`, `gpu_utils.py`, `status.py`,
  `aggregate_results.py`, `run_postgres.sh`, `switch_project.sh`,
  `postgres-compose.yml`, `requirements*.txt`, `.gitignore`, `monitor*.sh`,
  `monitor_dashboard.py`) — verified via `diff`, zero drift.
- `config.py`: added `'cross'` to `TASK_UNITS`; split `SOURCE_VARIANTS`
  into `TRAIN_SOURCE_VARIANTS` (title/desc/diff, commit vocabulary) and
  `QUERY_SOURCE_VARIANTS` (title/desc/comments, ticket vocabulary).
- **Critical gate passed**: regenerated the ticket-mode test set locally
  (`ETLPipeline(task_unit='ticket').load_data()` → `create_split()` →
  `save_test_set()`) for two independent `(project, strategy)` pairs
  (`sonar`/recent, `pulumi`/modn) already completed in `exp3.1`, diffed
  against `exp3.1`'s own files — byte-identical both times. This confirms
  `exp3.2` can regenerate the query-side test set locally instead of
  reading across the `exp3.1`/`exp3.2` folder boundary, with zero risk of
  divergence.
- `etl_pipeline.py`, `utils.py`, `run_experiments.py`: wired to
  `TRAIN_SOURCE_VARIANTS`/`QUERY_SOURCE_VARIANTS` respectively;
  `run_experiments.py`'s `evaluate_experiment()`/`run_single_experiment()`
  now take separate `train_source`/`query_source` params.
- `run_comprehensive_experiments.py` — the load-bearing restructuring:
  `_process_source_window()` now builds the index collection once per
  `(train_source, target, window)` and fans out over all 3 `query_source`
  evals *before* cleaning up the collection (moved `_cleanup_collection()`
  out of the old per-eval-immediate position) — otherwise the same
  collection would've been rebuilt 3×, defeating the entire point of
  decoupling index cost from query cost. `run_all()` now does two
  `load_data()` calls per `(model, strategy)`: `task_unit='commit'` for
  the index side, `task_unit='ticket'` for the query side — independent of
  `self.task_unit` (which only scopes the results-dir/checkpoint path,
  always `'cross'` in practice). Checkpoint granularity: ETL keyed on
  `train_source` alone, experiment completion keyed on a composite
  `f"{train_source}_q{query_source}"` string passed through
  `CheckpointManager`'s existing opaque `source` slot — zero changes
  needed to `checkpoint_manager.py` itself.
- New `run_cross_experiment.sh` driver (9 projects, no `agilebill` — it
  has zero `TASK` rows, so no query side at all — separate `flock` lock
  file from `exp3.1`'s). Written, **not executed**.
- Verified via a synthetic fixture (hand-built `checkpoint.json` +
  `comprehensive_results.csv` with composite IDs) that `status.py` and
  `aggregate_results.py` handle the new shape correctly with zero code
  changes — both already glob generically on the `task_unit` path
  segment.

**Real gaps found during implementation, beyond what the three design
docs anticipated** (all fixed):
- `run_comprehensive_experiments.py`'s `--sources` argparse flag would
  have crashed immediately (`config.SOURCE_VARIANTS` no longer exists)
  had it not been split into `--train-sources`/`--query-sources` — this
  was flagged as "minor" in `IMPLEMENTATION_PLAN.md` §5 but was actually
  load-bearing, not cosmetic.
- `etl_pipeline.py` and `run_experiments.py` each had two more
  `config.SOURCE_VARIANTS` references in their own standalone
  `main()`/argparse blocks (not called by the orchestrator, but would
  break `--help` and standalone debugging) — fixed alongside the primary
  change in each file.
- `monitor.sh` had `run_full_experiment.sh`/`logs/full_run_*.log`
  hardcoded — would have silently reported "NOT RUNNING" during an active
  `exp3.2` run and found no log to tail. Fixed to
  `run_cross_experiment.sh`/`logs/cross_run_*.log`.
- `monitor_dashboard.py`'s `VARIANTS_PER_STEP`/`TOTAL_STEPS` constants
  (72/19) were `exp3.1`-specific; fixed to 216/9 at the time (later
  revised again to 108/9 — see the model-scope entry below). Its
  `LOG_GLOB` had the same `full_run_*`/`cross_run_*` mismatch as
  `monitor.sh`, fixed the same way.
- Checked, but turned out to be a non-issue: the design review worried
  `get_current_variant()`'s 5-part `"Processing variant: ..."` line
  parser would break on the 6-part composite `train_source_qquery_source`
  ID. It doesn't — the actual implementation logs `"Processing variant:
  ..."` once per `target`, at `train_source` granularity, *before* the
  `query_source` fan-out begins, so that particular log line stays
  5-part. The composite ID only ever appears in checkpoint keys and
  `[EVAL] ...`/`[SKIP] ...` log lines, which nothing parses positionally.

**State left behind**: no `experiment_results/` or `logs/` directories
exist yet under `exp3.2/` — confirms nothing has actually executed against
real data. All verification this session was `diff`, `python3.12 -c`
assertions, `--help` smoke tests, and one synthetic fixture — all
CPU-only, no GPU/Postgres touched.

**Next**: Phase B (real GPU/Postgres smoke test: one project, confirm
single embedding call across the 3 query_sources, checkpoint-resume
across a kill/restart, results-shape check, real timing) — explicitly
deferred until `exp3.1`'s entire run finishes, per the user's decision
this session (not even a short contention window).

---

## 2026-07-29 (later same day) — `exp3.1` finished; model scope narrowed to `bge-small` only

**`exp3.1`'s full run completed.** Every `(project, task_unit, model)`
cell finished except `vscode`/`bge-large`, which ran out of GPU memory and
was not completed. Across every project that *did* finish both models,
`bge-large`'s MAP gain over `bge-small` was statistically significant but
small.

**Decision**: `exp3.2`'s Phase B run will use `bge-small` only, not both
models. Reasoning (user's call): `bge-large`'s payoff was already shown to
be marginal, and `exp3.2` is already introducing one new, unproven
variable (the cross-vocabulary grid) — stacking a second model sweep on
top of that would double the compute cost without a strong prior that it
matters, and would make it harder to attribute any interesting result to
the actual variable under test. Other architectures (`nomic-embed-text`,
`exp4`'s model families) are wanted eventually but deliberately deferred —
one new axis at a time, not everything folded into one run.

**Updated to reflect this**:
- `run_cross_experiment.sh`: `MODELS="bge-small bge-large"` →
  `MODELS="bge-small"`, header comment updated with the reasoning.
- `EXPERIMENT_PLAN.md`: grid math recomputed (216 → 108 eval-runs/project,
  72 → 36 ETL-builds/project; 1,944 → 972 total eval-runs, 648 → 324 total
  ETL-builds across 9 projects), §1 pseudocode's model loop collapsed to a
  single assignment, §5's resume command dropped `--models bge-small
  bge-large` → `--models bge-small --task-unit cross`, §7's timing
  reasoning halved accordingly, §8 updated to record that `exp3.1` has
  actually finished (was written prospectively before).
- `README.md`: grid math and `MODEL` bullet updated with the same
  reasoning; "Operationally" paragraph in the `exp3.1` dependency section
  updated from future tense ("should be launched only after `exp3.1`'s run
  finishes") to reflect that the gate has now actually cleared.
- `monitor_dashboard.py`'s `VARIANTS_PER_STEP` updated 216 → 108
  (`1 * 2 * 3 * 3 * 2 * 3`) to match the single-model grid.

**Next**: Phase B can begin — precondition check that `exp3.1`'s process
has actually exited (it has finished, but re-verify no stray process is
still running before touching GPU/Postgres), then the smoke test sequence
from `EXPERIMENT_PLAN.md`/`IMPLEMENTATION_PLAN.md` §4, scoped to
`bge-small` only.

---

## 2026-07-30 — Phase B smoke test: all checks pass, real timing recorded

**Precondition check**: `ps aux | grep -E "run_full_experiment|run_comprehensive_experiments"`
found no matching process (the earlier `pgrep -af` "hit" was a false
positive — it matched its own invoking shell command, which contained the
search string as literal text). `nvidia-smi`: 0% utilization, 1 MiB / 6080
MiB used — GPU genuinely idle. Postgres (`semantic_vectors_db`) up and
healthy. Clear to proceed.

**Ran all four Phase B checks from `IMPLEMENTATION_PLAN.md` §4** against
`celery` (cheapest project), `bge-small`, `recent` split, `w100` window —
all passed:

1. **Single-embedding-call check**: `train_source=desc`, both targets, all
   3 `query_source`s, cold start — exactly 1 `"Generating embeddings for
   source variant"` log line, confirming target-sharing and
   query-source-fan-out caching both hold. 6/6 evals completed, 0 failed,
   40.4s wall time.
2. **Checkpoint-resume check**: started `train_source=title` under
   `timeout 22s`, killed mid-way through the 3rd of 3 `file`-target evals
   (2 already completed). Resumed identical command: `[SKIP] ETL already
   completed` on the 5-part id (no `_q` suffix, confirming ETL genuinely
   doesn't depend on `query_source`), `[SKIP] Experiment already
   completed` on the 2 finished 6-part composite ids, the interrupted 3rd
   eval correctly re-ran (never reached `completed_experiments`), then
   `module`-target ETL+3 evals proceeded normally. No duplicate
   embedding-generation call in the resumed run, `failed_etl`/
   `failed_experiments` both empty at the end.
3. **Results-shape check**: `comprehensive_results.csv` has separate
   `train_source`/`query_source` columns (12 rows total across the two
   `train_source` cells run), no merged `source` column anywhere.
   Re-ran `status.py`/`aggregate_results.py` against the real (not
   synthetic) `experiment_results/` tree — both produced correct output
   (`STATUS.md` shows `celery | cross | 12 | 0 | ... | in_progress`;
   `all_projects_results.csv` has `project`/`task_unit` columns correctly
   inserted alongside `train_source`/`query_source`).
4. **Timing**: real numbers now in `EXPERIMENT_PLAN.md` §7, replacing the
   qualitative estimate. Headline finding not visible from `exp3.1`'s own
   numbers: the eval-side query-encoder model reloads fresh once per
   `query_source` (3× per cell, not 1×, since `exp3.1`'s matched-pair
   design never had more than one eval per built collection) — at `~2.3s`
   per reload this is roughly a third of the 40.4s `w100` cell's total
   time. Not fixed, just measured and flagged — `w1000`/`all` will dilute
   this cost relative to a bigger embedding step, but it's worth watching
   if per-project totals come in higher than the `w100`-based
   extrapolation suggests.

**Interesting early numbers, not a finding** (2 `train_source` × 3
`query_source` × 2 targets = 12 rows, `celery` only, `w100` only —
explicitly a smoke-test slice, not a real sweep, flagged the same way
`exp3.1`'s early `commit`-mode numbers were): cross-vocabulary MAP on
`celery` came out surprisingly high — `file` target 0.45–0.58,
`module` target 0.73–0.80, with `comments` as the query source
consistently *lowest* within each `train_source` row (0.45 and 0.73 vs.
`title`/`desc` queries' 0.54–0.58 and 0.75–0.80). Worth watching whether
that comments-hurts-cross-queries pattern holds once the real sweep runs,
since it would be a genuinely new result — `exp3.1`'s ticket-mode data
suggested `comments` usually *helped* when index and query were both
ticket-text.

**State left behind**: `experiment_results/celery/cross/` now holds a
real but partial checkpoint (2 of 3 `train_source`s × both targets × all
3 `query_source`s, `w100`/`recent` only — 12 of the eventual 108 rows for
this project). `logs/smoke_b1.log`, `logs/smoke_b2_part1.log`,
`logs/smoke_b2_part2.log` hold the raw smoke-test output. This will be
picked up and extended by the real run's checkpoint-resume, not
overwritten from scratch, when `run_cross_experiment.sh` actually
launches.

**Next**: full launch — `nohup ./run_cross_experiment.sh > /dev/null 2>&1
& disown`, per the same detached-process pattern `exp3.1` used. Not yet
done — handed to the user to start themselves, per the established
convention in this project of keeping multi-day runs out of any single
chat session.

---

## 2026-07-31 — launch/stop tooling ported from exp3.1

Added `run.sh` and `STOP_PROCESS.sh`, adapted from `exp3.1`'s versions
(process/log names swapped from `run_full_experiment.sh`/`logs/full_run_*`
to `run_cross_experiment.sh`/`logs/cross_run_*`, `--task-unit ticket|commit`
alternation dropped since `exp3.2` has none). `run_postgres.sh` (the
reboot-recovery Postgres check) and `monitor_dashboard.sh` were already in
place from Phase A (copied verbatim / already fixed for exp3.2's naming) -
no new work needed there, just confirmed still correct. `STOP_PROCESS.sh`
verified live: ran it with nothing active, correctly reported "Nothing
running - already stopped."

Full toolset now in place for an unattended run: `./run_postgres.sh`
(pre-flight, also auto-run as `run_cross_experiment.sh`'s first step) →
`./run.sh` (launch, detached) → `./monitor.sh` / `./monitor_dashboard.sh`
(watch) → `./STOP_PROCESS.sh` (pause safely, resumable via `./run.sh`
again).
