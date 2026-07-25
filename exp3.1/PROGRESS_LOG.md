# exp3.1 Progress Log

Chronological, append-only record of what was done, why, and what's next.
`EXPERIMENT_PLAN.md` is the stable design doc; `STATUS.md` (once it exists)
is an auto-generated numeric rollup from checkpoints; **this file is the
narrative** — read it to understand what's been tried and decided, not just
what's finished. Newest entries at the bottom.

---

## 2026-07-25

**Done:**
- Diagnosed that `exp3` was sonar-only by construction (`config.DB_PATH`
  hardcoded to `../data/sonar.db`), and that the real blocker for scaling
  to 10 projects isn't code, it's record-keeping culture: linkage between
  commits and tracker tickets varies wildly, and `agilebill` has no
  tracker at all.
- Corrected data location: real per-project databases are at
  `/home/stzh/Projects/db/*.db` (10 projects, ~29GB total), not this
  repo's `data/` folder.
- Built `db/collect_db_stats.py` — scans all project DBs, reports commit
  counts, tracker-linkage rate, and feasibility of two possible
  "task unit" criteria (`ticket` vs `commit`). Ran it against all 10 DBs;
  results in `db/project_stats.csv` (mirrored in `exp3.1/project_stats.csv`).
- Key finding: **9 of 10 projects are usable, `agilebill` is not**
  (only 120 total commits — below the ≥600 threshold for either
  criterion). `hadoop` has the best tracker discipline (92.6% linkage);
  `celery` (25.2%) and `kubernetes` (43.0%) are the messiest and most
  interesting test cases for whether the `commit` criterion helps.
- Discovered the two DB generations use different commits-table names
  (`RAWDATA` in the old `data/sonar.db`, `COMMITS` in all 10 databases
  under `/home/stzh/Projects/db/`) — same columns either way.
- Copied all 70 git-tracked source files from `exp3` into `exp3.1`
  (kept `exp3.1`'s own `README.md`, written earlier this session).
- Added `utils.get_commits_table_name()` and wired it into
  `etl_pipeline.py`'s `load_data()` — auto-detects `RAWDATA` vs `COMMITS`.
  Verified against both a `RAWDATA` db (`data/sonar.db`) and a `COMMITS`
  db (`rubocop.db`).
- Added `config.PROJECTS_DIR` / `config.PROJECT` + `--project <name>` on
  `etl_pipeline.py` and `create_task_collection.py`, so any of the 10 DBs
  can be selected without editing `config.py`.
- Established that **indexes on `COMMITS`/`RAWDATA` would not speed up
  the experiment itself** — `load_data()` always does an unconditional
  full-table `SELECT`, so there's no `WHERE`/`ORDER BY`/`COUNT(DISTINCT)`
  for an index to help with. Indexes would only help the one-off
  `collect_db_stats.py` diagnostics, not the ETL path.
- Found the actual win instead: `SELECT *` was pulling `DIFF` (full commit
  diffs), unused anywhere in the codebase (confirmed via `grep`) and
  ~82.5% of `DIFF+MESSAGE+PATH` bytes on `rubocop.db`. Narrowed the query
  to `SHA, PATH, MESSAGE, CMT_DATE, TASK_NAME`. Verified on `rubocop.db`
  (79,731 rows load correctly with 5 columns instead of 9).
- Wrote `exp3.1/EXPERIMENT_PLAN.md` — multi-project execution plan:
  read-only access to the 10 DBs in place (no copying), per-
  `(project, task_unit)` directory layout under `experiment_results/`,
  reuse of `checkpoint_manager.py` keyed per project+criterion, a planned
  `status.py` / `aggregate_results.py`, and a sign-count (not pooled-mean)
  comparison for the ticket-vs-commit question across the 9-project
  portfolio.

**Not yet done (see EXPERIMENT_PLAN.md §7 for the full list):**
- Project outer loop in `run_comprehensive_experiments.py`
- `status.py`, `aggregate_results.py`
- `COLLECTION_PREFIX` project-namespacing (Postgres collections from
  different projects would currently collide)
- Pointing at `exp3`'s existing `venv_py312` instead of building a new one
- `TASK_UNIT='commit'` ETL support (the actual Phase 2 payload — synthetic
  per-commit task table + noise filtering for merges/trivial/mass-diff
  commits)

**Next:** implement the mechanical pieces (project loop, `status.py`,
`aggregate_results.py`, collection namespacing) so Phase 1 (ticket
criterion, all 9 usable projects, `bge-small` + `recent` split only) can
actually run end-to-end.

---

## 2026-07-25 (later same session) — switching mechanics + reality check on timing

**Done:**
- Designed and built `switch_project.sh`: checkpoint-completeness check on
  the previous project → optional size-gated `pg_dump` backup (skipped by
  default; vectors are reproducible from source `.db`, not worth the disk
  by default) → non-interactive `DROP/CREATE SCHEMA vectors` → verify
  empty. Confirmed Postgres per-variant collections are *already*
  auto-deleted after each eval inside a project run
  (`_cleanup_collection()`), so this script only needed to handle the
  coarser between-project transition, not per-variant cleanup.
- Found the existing Postgres container (`semantic_vectors_db`, volume
  `exp3_postgres_data`) is stopped and ~6 months stale — holds old
  sonar-only `exp3` vectors, not anything from this session.
- **Checked GPU availability: broken.** `nvidia-smi` sees a card, but
  `torch.cuda.is_available()` is `False` — driver (470.256.02, CUDA 11.4)
  doesn't match the installed `torch` build. Nothing fixed yet; flagged as
  the single highest-leverage blocker in `EXPERIMENT_PLAN.md` §6a/§7 (item 0a).
- **Ran a real CPU benchmark**: `bge-small`, 301 actual `sonar` task texts
  (title+description) → 3.6 texts/sec on this machine's 8 cores.
- **Found a 6x redundancy bug**: `run_comprehensive_experiments.py`'s
  `run_etl_variant()` recomputes embeddings for every `(source, target,
  window)` triple, even though embeddings only depend on `source` —
  `target`/`window` just change downstream aggregation. Not fixed yet;
  flagged as item 0b, independent of the GPU issue.
- Extrapolated Phase 1 timing (9 projects, 225,116 total `TASK` rows,
  `bge-small` + `bge-large`, ticket criterion) — full table in
  `EXPERIMENT_PLAN.md` §6a. Headline numbers: **~313 hours on CPU with
  today's code** (redundancy bug included) vs. **~52 hours on CPU with the
  redundancy fix** vs. **~8-15 hours on GPU with the fix** (GPU number is
  a rough scale-up from `exp3`'s old P106-100 figures, not verified this
  session since CUDA is currently unusable here).
- User specified model scope: **`bge-small` + `bge-large` only** — the
  other embedding models (`bge-m3`, `gte-qwen2`, etc.) ran out of memory
  previously, so they're out of scope for this round regardless of the
  GPU/CPU question.

**Conclusion:** Phase 1 across all 9 projects is **not feasible in a
reasonable timeframe** until at least one of (GPU driver fix, embedding
caching fix) lands — this is now the actual next decision, ahead of the
mechanical items (project loop, `status.py`, `aggregate_results.py`) from
the previous entry.

**Next:** get direction on whether to (a) attempt fixing the `torch`/CUDA
driver mismatch, (b) implement the embedding-caching fix regardless of
GPU status (worthwhile either way), or (c) both. Nothing else in §7 should
be built out further until this is resolved, since it changes the shape
of everything downstream (how many projects/sources are even worth
attempting in the first pass).

---

## 2026-07-25 (later still) — GPU fixed system-wide; §7 fully implemented and verified

**GPU resolved without a separate session.** Root cause was never the
driver - it was testing through the wrong Python (`python3` = 3.14, user
site-packages had `torch==2.9.1+cu128`, incompatible with driver
470.256.02). `exp3/venv_py312` already had a working `torch==2.7.1+cu118`.
Installed that same stack (`torch`/`torchvision`/`torchaudio` cu118 +
`sentence-transformers`/`transformers`/`pandas`/`numpy`/`qdrant-client`/
`psycopg2-binary`, versions matched to `venv_py312` exactly) at user level
for **`python3.12`** (bootstrapped `pip` for it first via `ensurepip`).
Verified from `/tmp` (arbitrary directory, no venv activation): CUDA
available, real `sentence-transformers` encode works. System default
`python3`/`python` (3.14) deliberately left untouched - no cu118 wheel
exists for it upstream, and repointing the system `python3` symlink risks
breaking Fedora system tooling. **All exp3.1 commands must use `python3.12`
explicitly.**

**Implemented all of §7 + the caching fix**, in this order (plan written
directly, not via a Plan subagent - see [[feedback_no_plan_agent]]):
1. `config.collection_name()` / `config.task_collection_name()` - single
   source of truth for the project+task_unit-namespaced naming pattern
   that used to be duplicated (and drifting) across 5 call sites
   (`etl_pipeline.py`, `run_comprehensive_experiments.py` x3,
   `run_experiments.py`; `create_task_collection.py` got its own smaller
   variant). Left two legacy/non-critical-path files
   (`resume_etl.py`, `experiment_ui.py`) un-namespaced - out of the
   approved plan's scope.
2. `config.TASK_UNIT`/`TASK_UNITS` + `--task-unit` CLI flag on
   `etl_pipeline.py` and `run_comprehensive_experiments.py`.
3. `TASK_UNIT='commit'` mode: `ETLPipeline._build_commit_tasks()` builds a
   synthetic per-commit task table (title=subject line, description=full
   message, no comments), with noise filtering (drop merges, >50-file
   mass-diffs, <8-char subjects). Verified against `celery.db`: 11,692 of
   13,009 commits survived filtering; all 200 held-out test tasks got
   non-empty ground truth.
4. **Embedding-caching fix**: restructured `run_comprehensive_experiments.py`'s
   loop from `source→target→window` to `window→source→target`, computing
   embeddings once per `(source,window)` on the *already-windowed* train
   subset and sharing across both `target`s (previously: full unfiltered
   train set re-embedded 6x per source). Verified directly: a 2-target run
   logged exactly 1 "Generating embeddings" call, not 2. Also fixed a
   latent crash-at-the-end bug (`experiment_count`/`experiment_failed`
   referenced but never defined in the old `run_all()`'s final summary).
5. `--project` + per-`(project,task_unit)` isolation: `results_dir` and
   `CheckpointManager` now scoped to `experiment_results/<project>/<task_unit>/`.
6. `status.py` - scans `experiment_results/*/*/checkpoint.json`, writes
   `STATUS.md`.
7. `aggregate_results.py` - merges all `comprehensive_results.csv` into
   `all_projects_results.csv` with `project`/`task_unit` columns.

**Verified end-to-end** (real runs against Postgres, `python3.12`, tiny
grids - not full sweeps):
- `switch_project.sh` used for real for the first time: cleared Postgres
  before `sonar`, correctly read `sonar`'s checkpoint (2 done, 0 failed)
  before switching to `celery`.
- `sonar`/ticket, `title`/`file`/`w100`: MAP=0.0133 (recent), MAP=0.0004
  (modn) - both plausible, matches previously-documented typical ranges.
- Same grid re-run with `--no-resume` omitted (piped `y`): both variants
  correctly `[SKIP]`ped via checkpoint, no recompute.
- `sonar`/ticket, `title`/`{file,module}`/`w100`: `file` MAP=0.01330
  matches the earlier single-target run's 0.01331 almost exactly (float
  noise only) - confirms the restructuring didn't change results, only
  computed them more efficiently. `module` MAP=0.784 - much higher, as
  RQ1's hypothesis predicted (coarser granularity → easier retrieval).
- `celery`/**commit**, `title`/`file`/`w100`: MAP=0.264, MRR=0.321 - far
  higher than sonar/ticket's 0.013. Interesting substantive early signal
  (commit-message titles may correlate more tightly with touched files
  than Jira ticket titles do) but this is one tiny smoke-test grid, not a
  real comparison yet - noted here so it isn't mistaken for a finding.
- Collection names confirmed correctly namespaced in logs:
  `rag_exp_sonar_ticket_title_file_w100_modn_bge-small` vs
  `rag_exp_celery_commit_title_file_w100_recent_bge-small`.
- `status.py` / `aggregate_results.py` both ran correctly against the
  smoke-test data above, producing `STATUS.md` and
  `all_projects_results.csv` with correct `project`/`task_unit` columns.

**State left behind:** `experiment_results/sonar/ticket/` and
`experiment_results/celery/commit/` contain only the tiny smoke-test grids
above (1-2 source/target/window combos each), not full 18-variant sweeps -
don't mistake these for completed Phase 1 runs. Postgres container
(`semantic_vectors_db`) is running and schema is currently empty (last
`switch_project.sh` call cleared it for `celery`).

**Next:** run the real Phase 1 sweep - all 9 projects x `bge-small` +
`bge-large` x ticket criterion (commit criterion as a second pass) - using
`run_comprehensive_experiments.py --project <name> --task-unit <unit>`
plus `switch_project.sh` between each. `--strategies recent` only (per
earlier scope decision - `modn` doubles cost for an already-answered
question). Full 18-variant grid this time, not the smoke-test subset used
for verification above.

---

## 2026-07-25 (later still) — Phase 1 runner script + unattended-run fix

User wants the real Phase 1 run launched as its own detached process in
its own terminal session (nohup/tmux), not as a subprocess of a chat
session - sound call, since the estimated runtime (~35-40h) far exceeds
any single conversation. Also asked to confirm commit-mode works for
`sonar` specifically (yes - already proven via `celery` above, same code
path, no project-specific logic).

**Fixed a blocker for unattended operation**: `ComprehensiveExperimentRunner`'s
resume path called `input("Resume from checkpoint? (y/n): ")` - would hang
or raise `EOFError` with no attached TTY (exactly the nohup/background
case). Added `--yes` / `auto_resume` to skip the prompt and resume
automatically when a checkpoint exists.

**Built `run_phase1_all.sh`**: loops over 9 projects (ticket) + `sonar`
(commit) again at the end, ordered smallest-task-count-first
(celery→rubocop→pulumi→sonar→flink→hadoop→spark→kubernetes→vscode, then
sonar/commit), calling `switch_project.sh` then
`run_comprehensive_experiments.py --yes` for the full 18-variant grid
(both models) at each step, running `status.py`/`aggregate_results.py`
after every step so `STATUS.md`/`all_projects_results.csv` stay current
throughout rather than only at the very end. Logs to `logs/phase1_<timestamp>.log`
via `tee`; writes `experiment_results/RUNNING.txt` with the current step
so progress is checkable without touching the running process. No `set -e`
around the per-project step - one project crashing doesn't block the rest.

Not yet launched - handed to the user to start themselves
(`nohup ./run_phase1_all.sh > /dev/null 2>&1 & disown`), per their request
to keep it out of this session. Existing smoke-test data in
`experiment_results/sonar/ticket/` and `experiment_results/celery/commit/`
(1-2 variants each, from earlier verification) will be picked up and
extended by the real run's checkpoint-resume, not overwritten from scratch.

**Next:** once launched, monitor via `experiment_results/RUNNING.txt`,
`tail -f logs/phase1_*.log`, and `experiment_results/STATUS.md` when asked -
don't proactively poll a multi-day background process.

---

## 2026-07-25 (later still) — corrected: no more "Phase 1", full grid, all 10 projects

**User caught a real mistake**: the "Phase 1" plan trimmed model choice
and split-strategy to `bge-small`/`recent`-only, reasoning those were
"already-answered questions" - based on a *single* project (`sonar`) run
in the old single-project `exp3`. That's not statistically valid - the
whole point of the 9(10)-project portfolio is to check whether
conclusions hold *across* projects, not inherit them from one and skip
re-checking. Corrected: **no subsetting from unstated assumptions,
full combinatorial grid, every axis, every project.** Renamed
`run_phase1_all.sh` → `run_full_experiment.sh` (overwrote the stale
single-project script `exp3` had under that name - no longer applicable,
predates `python3.12`/multi-project/checkpoint-scoping work).

**Design review with the user surfaced things I'd missed or under-specified:**
- `target` (file/module) dimension - already in the grid mechanically
  (`--targets file module`), just hadn't been called out explicitly when
  re-describing the plan. Confirmed important (RQ1).
- **`agilebill` (10th project) - user wants it run too**, deliberately,
  specifically *to see how bad the numbers get* at that volume - that's a
  data point, not something to hide by excluding it. Two real constraints
  it has, both now handled explicitly:
  - Zero `TASK` rows (no tracker at all) → **commit-mode only**, ticket
    mode isn't just weak there, it's impossible.
  - Only 120 total commits → default `--test-size 200` would leave
    nothing to train on. Added `--test-size` as a CLI flag on
    `run_comprehensive_experiments.py` (threaded through to
    `ETLPipeline`); `agilebill` uses `--test-size 20`. Verified: 103/120
    commits survive noise filtering, 83 train / 20 test, all 20 test
    tasks get non-empty ground truth.
- **New idea from the user**: for `commit` mode's third/"noisy" source
  variant, use the commit's **diff** content instead of a `comments`
  duplicate (`COMMENTS` is always `''` for a raw commit, so `comments`
  would be byte-identical to `desc` there - pure waste). Implemented:
  - `load_data()` now conditionally fetches `DIFF` from
    `RAWDATA`/`COMMITS` **only** when `task_unit='commit'` - ticket-mode
    queries stay exactly as lean as before.
  - `_build_commit_tasks()` aggregates all of a commit's per-file diffs
    (one `SHA` can span many rows) into one text blob per task, capped at
    `MAX_DIFF_CHARS=4000` (embedding models truncate long inputs anyway -
    the cap costs nothing a real embedding call would have used).
  - Hit and fixed a real bug during verification: `groupby().first()`
    already carried a (meaningless, single-arbitrary-row) `DIFF` column
    into `one_per_sha`, so merging in the real aggregated version silently
    produced `DIFF_x`/`DIFF_y` instead of `DIFF` (pandas' merge-collision
    suffixing) - fixed by dropping the stale column before merging.
  - Added `config.SOURCE_VARIANTS['diff']` (`TITLE+DESCRIPTION+DIFF`).
  - Verified end-to-end via the real orchestrator against `celery`:
    MAP progresses `title` 0.266 → `desc` 0.284 → `diff` 0.302 - diff
    content measurably helps, a genuinely informative result (not
    something a `comments` duplicate could ever have shown).

**Final `run_full_experiment.sh`**: 19 steps (9 projects × {ticket,commit}
+ `agilebill`×commit-only), source variants conditional on `task_unit`
(`title desc comments` for ticket, `title desc diff` for commit),
`--test-size` passed per-step (200 default, 20 for `agilebill`). All
verified working (`celery`/commit/diff and `agilebill`/commit/20 both ran
real ETL+eval through the full orchestrator, no errors).

**Not yet launched** - still handed to the user to start themselves, per
their explicit request to keep the multi-day run out of any single chat
session (`nohup ./run_full_experiment.sh > /dev/null 2>&1 & disown`).
