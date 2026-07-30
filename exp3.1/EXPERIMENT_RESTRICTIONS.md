# Experiment Restrictions — Hardware-Induced Data Gaps

Documents cases where the full combinatorial sweep (`run_full_experiment.sh`)
could not complete on the hardware it ran on, so future researchers rerunning
this experiment on the same or similar hardware know what to expect, and
anyone with more RAM knows exactly what to fix.

---

## Hardware used for this run

| Component | Spec |
|---|---|
| CPU | Intel Core i7-2600 @ 3.40GHz, 4 cores / 8 threads |
| RAM | 16 GB total (`MemTotal: 16044380 kB`), ~12 GB typically available under normal desktop load |
| Swap | 8 GB zram (compressed RAM-backed swap, not disk-backed — doesn't help with sustained large working sets, only bursts) |
| GPU | NVIDIA P106-100, 6 GB VRAM |
| Disk | 476 GB, project DBs on same filesystem as the repo |
| OS | Fedora Linux 44, kernel 7.1.5-200.fc44.x86_64 |
| Python / Torch | python3.12, torch 2.7.1+cu118 |

---

## Gap: `vscode/commit`, `bge-large` model — 35/72 variants missing

**Date:** 2026-07-30, during the `exp3.1` full 19-step / 10-project sweep.

**What's missing:** all 35 `bge-large` variants for `vscode/commit`
(`title`/`desc`/`diff` × `file`/`module` × `w100`/`w1000`/`all` ×
`recent`/`modn`, minus the 1 that completed before the crash). All 36
`bge-small` variants for the same project/task_unit completed fine.

**Root cause: OOM-killed by the Linux kernel, not a code bug.**

```
kernel: Out of memory: Killed process 129545 (python3.12)
  total-vm:35991932kB anon-rss:11591456kB (~11.6 GB resident)
```

confirmed via `journalctl -k`, exact timestamp 2026-07-30 22:51:29, matching
the `run_full_experiment.sh` log's `vscode/commit exited with code 137`
(137 = 128 + SIGKILL).

**Why this specific step, and why only `bge-large`:**

1. `vscode.db` is the **largest project database in the portfolio — 12.6 GB**
   (vs. 8.6 GB for `kubernetes.db`, the next largest). In `commit` mode,
   `ETLPipeline.load_data()` (`etl_pipeline.py:71-72`) adds a `DIFF` column
   to the `RAWDATA` query and pulls the **entire table via
   `pd.read_sql_query`** — no chunking, no streaming. `DIFF` holds every
   changed file's diff text per commit row, which is by far the largest
   column by byte volume. Loading it means materializing gigabytes of raw
   diff text as Python string objects in a single pandas DataFrame, which
   is significantly heavier in RAM than the file's on-disk (SQLite-encoded)
   size.
2. `load_data()` is called **once per `(model, strategy)` pair** — 4 times
   total per project/task_unit (`run_comprehensive_experiments.py:413-424`,
   loop order `model → strategy → window → source`) — so this multi-GB load
   happens 4 separate times per step, each producing its own transient peak.
3. `bge-small` (384-dim embeddings, smaller model weights) got through all
   4 `(strategy, window×source×target)` combinations fine. The crash hit
   directly after switching to `bge-large` (1024-dim embeddings, larger
   model weights + larger activation/output tensors) — the **combination**
   of the already-heavy `load_data()` peak with `bge-large`'s bigger memory
   footprint (model + batch activations + a full 1024-dim embedding matrix
   for the loaded train set) is what crossed the 16 GB ceiling. `bge-small`
   alone, or `bge-large` on a smaller DB, did not reproduce this — every
   other project (including `kubernetes`, whose commit-mode load alone hit
   ~11 GB RSS by itself and still finished, per manual `/proc` inspection
   during that step) completed both models without incident.

**Not the cause (ruled out):**
- GPU/VRAM: irrelevant here — `nvidia-smi` showed 0% utilization at the time
  of the kill; this was a host-RAM OOM, not a CUDA OOM.
- Batch size / `UPSERT_BATCH_SIZE`: these bound per-batch memory, not the
  size of the fully-materialized train DataFrame or embedding matrix.
- Vector storage: Postgres upserts happen after embedding generation and are
  batched (`config.UPSERT_BATCH_SIZE = 100`); they were never reached for
  the missing variants.

**Impact on results:** none beyond the missing rows themselves — every
variant is an independent, atomic row in `comprehensive_results.csv` (see
`PROGRESS_LOG.md` for the general atomicity discussion). No other
project/task_unit/model combination is affected; confirmed via a full
checkpoint diff against the expected 72-variant grid for every
`(project, task_unit)` pair. The portfolio still has 9/10 projects fully
complete for both models, and `vscode` is fully complete for `bge-small` —
sufficient breadth for cross-project analysis without `vscode/bge-large`.

**How to reproduce / how to fix, for anyone revisiting this on similar
hardware:**
- Rerun `sh run.sh` — checkpoint resume will skip the 37 completed variants
  and retry exactly the 35 missing `bge-large` ones. Whether this succeeds
  depends on how much RAM is free at the time; it is not guaranteed to avoid
  the same OOM if run under similar memory pressure.
- Actual fixes, not attempted here (out of scope for the dissertation's
  current phase, left for future work or researchers with more RAM):
  - Stream/chunk the `commit`-mode `RAWDATA` load (`etl_pipeline.py:72`)
    instead of a single `pd.read_sql_query` for large DBs — e.g. read in
    SHA-batches and aggregate incrementally rather than materializing the
    full table at once.
  - Drop the `DIFF` column from memory immediately after the per-SHA
    aggregation step (`etl_pipeline.py:142-158`) instead of keeping the full
    per-row `DIFF` text around for the rest of the DataFrame's lifetime.
  - Run on a machine with more RAM (32 GB+ comfortably clears this), or move
    `bge-large` runs to a separate pass per project so peak memory from
    `load_data()` and the embedding model don't need to coexist.
