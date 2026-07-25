# Multi-Project Experiment Plan (exp3.1)

This supersedes the single-project plan `exp3` was built with. Scope now:
run the task→code retrieval experiment across the **9 usable projects**
(all except `agilebill` — see `project_stats.csv`), under the **ticket**
criterion now and the **commit** criterion once it's implemented, and merge
everything into one comparable dataset.

The plan is written to survive being interrupted at any point — including
across days, in a brand new Claude Code session with no memory of this
conversation. Every fact needed to resume lives in a file on disk, never
only in a running process or in my head.

## 1. Execution order

**Correction (2026-07-25):** an earlier version of this plan trimmed the
run to `bge-small`/`recent`-only "for now," reasoning that model choice
and split-strategy were "already-answered questions" — based on a single
project (`sonar`) run in the old, single-project `exp3`. That's not a
statistically valid basis for anything: the entire point of assembling a
9-project portfolio was to check whether conclusions hold *across*
projects, not to inherit conclusions from one project and skip re-checking
them. There is no shortcut here — the full grid runs, on every project,
under both criteria. The only two exclusions that stand are hardware/data
facts, not inferences from another project's results (see below).

```
for project in [celery, rubocop, pulumi, sonar, flink, hadoop, spark, kubernetes, vscode]:
    for task_unit in [ticket, commit]:
        for split_strategy in [recent, modn]:
            for model in [bge-small, bge-large]:
                for source in [title, desc, comments]:
                    for target in [file, module]:
                        for window in [w100, w1000, all]:
                            run ETL  -> build vectors for this variant
                            run eval -> MAP/MRR/P@k/R@k against the 200-task test set
```

- **Models**: `bge-small` + `bge-large` only — not a scope-trimming choice,
  a hardware fact: the other embedding models OOM on this GPU's 6GB VRAM.
- **Everything else runs in full**: both task-unit criteria, both split
  strategies, all 9 usable projects, the complete 18-variant
  (source×target×window) grid.
- That's 9 projects × 2 task_units × 2 strategies × 2 models × 18 = **1,296
  variant-runs total**. Expected to take a long time (plausibly days) — that's
  accepted; the machine runs it, not a person watching it.
- **`agilebill` is excluded** — not inferred from another project, a fact
  about `agilebill` itself: only 120 total commits, below the ~600-unit
  floor needed for a 200-task held-out test set under *either* criterion
  (`db/project_stats.csv`: `ticket_criterion_usable=False`,
  `commit_criterion_usable=False`). Flagged explicitly so it's a checkable
  claim, not a silent assumption.
- Projects ordered smallest-task-count-first (celery, rubocop, pulumi,
  sonar, flink, hadoop, spark, kubernetes, vscode — see
  `db/project_stats.csv` for exact counts) so an interruption early on
  still leaves the broadest possible partial dataset rather than being
  stuck mid-way through `vscode` (largest). `ticket`/`commit` run
  back-to-back per project so each project's full ticket-vs-commit
  comparison lands together.
- Run via `./run_full_experiment.sh` (see its header for full documentation) —
  launched by the user as its own detached process (`nohup ... & disown`
  or tmux), not as a subprocess of a chat session, specifically so it
  survives independently of any single conversation (including
  unpredictable real-world interruptions — war-time power/connectivity
  loss is a real, named constraint here, not hypothetical). Fully
  resumable from a cold start at any point: re-running the exact same
  command re-verifies already-done steps quickly via checkpoint lookups
  and continues real work exactly where it stopped — no manual
  bookkeeping, no need to remember which step was in flight.

## 2. Data access: databases are never copied

The 9 project `.db` files live at `/home/stzh/Projects/db/*.db` and stay
there — total ~29GB, no reason to duplicate any of it into `exp3.1`.

- `config.PROJECTS_DIR` + `--project <name>` (already implemented) point
  `etl_pipeline.py` / `create_task_collection.py` directly at the source
  file. Every run opens its own read-only-in-practice `sqlite3.connect()`
  and only ever issues `SELECT` — nothing in this pipeline writes to a
  source database.
- Because these are read-only reads, running ETL for two different
  projects concurrently is safe at the SQLite level (no lock contention
  across separate files). Default is still sequential, since everything
  shares one GPU for embeddings.
- The one artifact that *is* generated and kept locally per project is the
  **test set JSON** (200 held-out tasks + ground truth files) — that's
  derived data, not a copy of the source DB, and is what the eval phase
  reads instead of re-querying SQLite each time.

## 3. Directory / file layout

```
exp3.1/experiment_results/
  <project>/
    <task_unit>/                        # 'ticket' | 'commit'
      checkpoint.json                   # resume state for this project+criterion
      test_set_recent_bge-small.json    # 200 held-out tasks + ground truth
      comprehensive_results.csv         # this project+criterion's results
      run.log                           # this project+criterion's full log
  orchestrator.log                      # one line per project/criterion start+end+pass/fail
  STATUS.md                             # regenerated rollup table (see §5)
  all_projects_results.csv              # final merged dataset (see §6)
```

Rationale: today's `exp3` bakes model/strategy/source/target/window into
filenames but keeps everything in one flat `experiment_results/` — fine
for one project. Adding `project` and `task_unit` as **path segments**
instead of more filename suffixes keeps each project+criterion's state
fully isolated: a crash mid-`kubernetes` run can't corrupt `sonar`'s
checkpoint, and deleting/retrying one project's results is `rm -rf
experiment_results/kubernetes/` without touching anything else.

## 4. Checkpointing (resumable mid-project, not just mid-portfolio)

Reuse `checkpoint_manager.py` as-is (`variant_id = model_strategy_source_
target_window` already uniquely identifies a run *within* a project+
criterion) — just instantiate it against the per-project path:

```python
CheckpointManager(f"experiment_results/{project}/{task_unit}/checkpoint.json")
```

`run_comprehensive_experiments.py` needs one new outer loop (over
`project`, later also `task_unit`) wrapping its existing model/strategy/
source/target/window loops — everything inside stays as-is, since
`is_etl_completed()` / `is_experiment_completed()` already skip finished
variants. **Not yet implemented** — mechanical change, not started.

This gives two independent resume granularities for free: interrupted
mid-project → the existing checkpoint skip-logic resumes mid-grid;
interrupted between projects → the orchestrator loop just starts the next
project not yet marked done in `STATUS.md`.

## 5. Resume protocol — what to do in a fresh session

1. Read **`PROGRESS_LOG.md`** first — narrative record of what's been
   done, decided, and tried so far (append-only, newest entries at the
   bottom). This is the file to read to understand *why* things are in
   whatever state they're in, not just what's finished.
2. Read `exp3.1/experiment_results/STATUS.md` next. One row per
   (project, task_unit): `variants_done/18`, `last_updated`, `status`
   (`done` / `in_progress` / `not_started` / `failed: <n> variants`).
3. Pick the first non-`done` row and resume it:
   ```bash
   cd exp3.1
   python run_comprehensive_experiments.py --project <name> --task-unit <unit> \
       --models bge-small --strategies recent
   ```
   The per-project checkpoint makes this pick up mid-grid automatically —
   no need to know exactly which of the 18 variants was last completed.
4. If `STATUS.md` is missing or looks stale, regenerate it from the
   checkpoints themselves (source of truth is always the `checkpoint.json`
   files, never `STATUS.md`):
   ```bash
   python status.py   # scans experiment_results/*/*/checkpoint.json, rewrites STATUS.md
   ```
5. Before doing anything new, append a `PROGRESS_LOG.md` entry — what's
   about to be done and why — and add another when it's done (or when
   interrupted partway). This is what step 1 of the *next* resume reads.
6. I'll save a project memory pointing at `PROGRESS_LOG.md` and the resume
   command so a session on a different day starts here instead of
   re-deriving the plan.

`status.py` and the project-loop change in `run_comprehensive_experiments.py`
are both **not yet implemented** — small, mechanical additions, listed in §7.

## 6. Aggregating everything into one CSV

New script `aggregate_results.py` (not yet implemented): walks
`experiment_results/*/*/comprehensive_results.csv`, adds `project` and
`task_unit` columns parsed from the path, concatenates into
`experiment_results/all_projects_results.csv`. Safe to run at any time —
not just once everything finishes — so partial cross-project comparisons
are available from whatever's done so far.

How that CSV answers the actual research question (does the approach
generalize across a messy portfolio, not just `sonar`):

- **RQ1/RQ2/RQ4 (existing)**: group by `(source, target, window)` and
  compare `MAP` *across* `project` — consistent direction across 9 very
  different codebases is a much stronger claim than one project's numbers.
- **Ticket vs. commit criterion (the new question)**: pivot to one row per
  `(project, source, target, window)` with `ticket_MAP` and `commit_MAP`
  side by side, then look at the **sign** (how many of the 9 projects favor
  ticket vs. commit) and the **magnitude** of the delta — not just a pooled
  mean across all rows. Pooling would let `kubernetes`/`vscode` (10x more
  commits than `rubocop`/`celery`) dominate the average and mask
  per-project disagreement. A simple sign-count / paired comparison is more
  honest given only 9 data points and very unequal project sizes.

## 6a. Time estimate (historical — see §7 for current status; both fixes below are now done)

**GPU is not currently usable in this environment.** `nvidia-smi` sees a
card, but the installed `torch` reports `cuda.is_available() == False` —
driver (470.256.02 / CUDA 11.4) doesn't match what the installed `torch`
build expects. Needs a matching `torch` wheel (or driver update) before any
GPU number below is real; until then, everything runs on CPU (8 cores here).

**Measured (this session, CPU, 8 cores):** `bge-small`, 301 real `sonar`
task texts (title+description, avg 1536 chars) → **3.6 texts/sec**.

**Projected task volume, all 9 usable projects (`TASK` row counts):**
celery 2,998 · rubocop 6,990 · pulumi 9,531 · sonar 9,799 · flink 17,073 ·
hadoop 24,737 · spark 30,478 · kubernetes 57,964 · vscode 65,546 →
**225,116 tasks total**.

**Critical inefficiency found:** `run_comprehensive_experiments.py`'s
`run_etl_variant()` calls `generate_embeddings()` fresh for **every**
`(source, target, window)` triple — but embeddings only depend on
`source`; `target`/`window` just change how already-embedded vectors get
aggregated/filtered downstream. That's a **6x** redundant recompute
(2 targets × 3 windows) that should be fixed regardless of hardware.

| Scenario | bge-small only | + bge-large (assume ~5x slower/text, unverified) |
|---|---|---|
| CPU, current code (6x redundant) | 225,116 × 3 sources × 6 / 3.6/s ≈ **313 hours** (~13 days) | ~1,880 hours |
| CPU, **with caching fix** (embed once per source) | 225,116 × 3 / 3.6/s ≈ **52 hours** (~2.2 days) | ~310 hours |
| GPU (assumed working, exp3's old ~5-8 min/variant figure, scaled by task count, 6x redundant) | ≈ **45 hours** | ≈ 70-90 hours (rough) |
| GPU, **with caching fix** | ≈ **8-15 hours** (rough) | ≈ 15-25 hours (rough) |

**Bottom line:** running this on CPU as the code stands today (313+
hours for just one model) is not realistic. Two independent levers, both
worth pulling before a real run:
1. Fix the `torch`/CUDA driver mismatch so the GPU is actually used.
2. Fix the embedding-recompute redundancy (cache per-source embeddings,
   reuse across target/window) — this alone is a 6x win on any hardware.

(Both fixes landed later the same day — GPU works via `python3.12`, and
the embedding-caching fix is in `run_comprehensive_experiments.py`. See §7.)

## 6b. Switching between projects (Postgres cleanup)

Results (`comprehensive_results.csv`, `test_set_*.json`, `checkpoint.json`,
logs) are plain files under `experiment_results/<project>/<task_unit>/` —
never touched by Postgres cleanup, always safe.

Postgres only ever holds **reproducible intermediate vectors** (regenerable
from the source `.db` + ETL, same model/config → same output), so a backup
before clearing is a convenience, not a safety net, and is skipped by
default given limited disk.

`switch_project.sh --from <prev> --to <next> [--task-unit ticket] [--backup] [--backup-max-mb 2000] [--yes]`:
1. Checks the previous project's `checkpoint.json` for completeness /
   failures; asks for confirmation before proceeding if something looks
   unfinished (never silently discards an in-progress run).
2. Optional `--backup`: `pg_dump`s the `vectors` schema into that
   project's `experiment_results/` folder, but only if the DB is under
   `--backup-max-mb` (default 2000MB) — skips automatically otherwise,
   since the data is reproducible anyway.
3. Drops and recreates the `vectors` schema (non-interactive — safe to run
   unattended, unlike the existing interactive `clear_postgres_vectors.py`).
4. Verifies the schema is actually empty before declaring success.

One important thing this *doesn't* need to solve: per-variant Postgres
collections are already deleted right after each variant's eval
(`_cleanup_collection()` in `run_comprehensive_experiments.py`), so
Postgres never holds more than one variant's vectors at a time *within* a
project. `switch_project.sh` only handles the coarser between-*project*
transition (and mops up anything a crash left behind).

## 7. Built vs. still needed

**Everything in this section is now built and verified** (2026-07-25, see
`PROGRESS_LOG.md` for the full account, including verification evidence):

- `config.PROJECTS_DIR` / `--project` switch (`etl_pipeline.py`,
  `create_task_collection.py`, `run_comprehensive_experiments.py`)
- `RAWDATA`/`COMMITS` table-name auto-detection (`utils.get_commits_table_name`)
- Trimmed `SELECT` (drops unused `DIFF`/`AUTHOR_*`/`ID`)
- `db/project_stats.csv` — per-project inventory (commit counts, linkage
  rate, which criteria are usable)
- `switch_project.sh` — checkpoint-aware, size-gated, non-interactive
  Postgres cleanup between projects (§6b) — used for real between the
  `sonar` and `celery` verification runs
- GPU fixed (`python3.12` + matching `torch==2.7.1+cu118` stack installed
  at user level, no separate session needed — see PROGRESS_LOG)
- `config.collection_name()` / `config.task_collection_name()` — the
  namespacing helper, wired into all 5 duplicated call sites
- `config.TASK_UNIT`/`TASK_UNITS` + `--task-unit` CLI flag
- `TASK_UNIT='commit'` mode (`ETLPipeline._build_commit_tasks`) — noise
  filtering (merges, >50-file mass-diffs, <8-char subjects) verified
  against `celery.db`
- Embedding-caching fix — `run_comprehensive_experiments.py` restructured
  to `window→source→target`, embeddings computed once per `(source,window)`
  and shared across targets — verified directly (1 embedding call logged
  for a 2-target run, not 2)
- `--project` + per-`(project,task_unit)` isolation (`experiment_results/<project>/<task_unit>/`)
- `status.py`, `aggregate_results.py`

**Not done / not needed:**
- Pointing at `exp3`'s `venv_py312` — superseded by the system-wide
  `python3.12` fix, which is simpler (no venv activation needed at all).
- `resume_etl.py` / `experiment_ui.py` still build unnamespaced collection
  names — legacy/non-critical-path, deliberately left out of scope.

**Next:** run `./run_full_experiment.sh` — the complete grid, no
subsetting (see §1's correction). The smoke tests that verified this
section only ran 1-2 variants per project, not full sweeps.

**Open question, not yet resolved:** in `commit` mode, `COMMENTS` is
always `''` (no comments concept for raw commits), so the `comments`
source variant (`TITLE+DESCRIPTION+COMMENTS`) is byte-identical to `desc`
(`TITLE+DESCRIPTION`) whenever `task_unit='commit'` — running it computes
and stores a guaranteed duplicate. Options: (a) run it anyway for grid
uniformity (small waste, ~1/18 of the commit-mode grid), (b) skip it for
`commit` runs specifically (saves that compute, avoids a misleading row
that looks like it tested something new but didn't). Not yet decided.
