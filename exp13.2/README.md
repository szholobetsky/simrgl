# exp13.2 — where will a task create files? Rules "anchor file → (folder, extension)"

**Question.** In exp13.1, 43 of the 181 files the open pull requests of sqlfluff change (24 %) are files
the PR **creates**. No retrieval can return them — they do not exist. But their *place* is
predictable: a change to `dialect_mysql.py` comes with a new pair
`test/fixtures/dialects/mysql/<x>.sql` + `.yml`. This is a relation, like the ones gamayun's health
report measures (novelty: is the relation new; structurality: does it stay inside one module) — a new
fixture next to its dialect is a new link of high structurality, the EVOLUTION zone. exp13.2 asks:
predicted from history, how often is the folder (and type) of a new file right?

## Method (`spawn.py`)

**Learning** — every task of the project's history. `git log --name-status` (first parent, no rename
detection) gives each commit's added (A) and modified (M) files; the database's SHA → TASK_NAME
groups them by task. For a task, every modified file *m* and every added file *a* give a rule
*m → slot(a)*, with `conf = tasks(m, slot) / tasks(m)`; rules seen in fewer than 2 tasks are dropped.
Two slot levels:

- `dir` — (folder of *a*, extension of *a*): `test/fixtures/dialects/mysql`, `.sql`
- `parent` — (the folder's parent, extension): a file in a new folder
  (`autofix/ansi/035_long_order_by/before.sql`) still lands in a known parent (`autofix/ansi`).

**Prediction** for a future task (exp13.1: open PR closing an issue, the issue as the query): the
anchors are the top 10 files a component returned for the task, weighted 1/rank;
`score(slot) = Σ weight · conf(anchor → slot)`. Anchor sets:

| anchors | what it is |
|---|---|
| `flat`, `simargl_units` | what a user would have: exp13.1's fused list, or simargl alone |
| `oracle` | the PR's true modified files — the rules alone, retrieval perfect |
| `prior` | no anchor: the slots most often created in the history |

**Score** — on the future tasks that create files: `hit@k` (a true slot among the top k) and slot
`recall@k`. The name of the new file is not predicted, only where and what kind.

```
python spawn.py          # reads ../exp13.1/projects.csv, out/<p>/future.json, runs.json
```

-> `out/<p>/rules_dir.json`, `rules_parent.json` (the strongest rules), `predictions_*.json`,
`scores.json`, `results.csv`.

## Results — sqlfluff

History: 7 073 commits, 4 346 tasks, **1 703 tasks (39 %) create files**; 2 128 rules at `dir`
level, 2 387 at `parent`. The strongest rules are the dialect → fixture ones:
`dialect_snowflake.py → test/fixtures/dialects/snowflake .sql/.yml` (136 tasks, conf 0.34),
`dialect_snowflake_keywords.py → …/snowflake` (79 tasks, conf 0.61).

Of exp13.1's 29 future tasks, **10 create files**.

**slot = dir** (folder + extension — the informative level)

| anchors | hit@1 | hit@3 | hit@5 | hit@10 | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|---|---|---|
| flat | 0.40 | 0.50 | 0.50 | 0.60 | 0.16 | 0.38 | 0.43 | 0.52 |
| simargl_units | **0.50** | 0.50 | 0.50 | **0.70** | 0.19 | 0.38 | 0.40 | 0.50 |
| oracle | 0.40 | 0.50 | 0.50 | 0.50 | 0.16 | **0.42** | 0.42 | 0.47 |
| prior | 0.10 | 0.10 | 0.20 | 0.50 | 0.05 | 0.10 | 0.13 | 0.28 |

**slot = parent**: every anchor set, the prior included, has hit@1 0.6–0.7 — in sqlfluff almost every
new file is a fixture under `test/fixtures/dialects/` or `test/fixtures/linter/`, so the parent level
is too coarse to tell anything apart (`scores.json`).

Examples (flat anchors, rank of the true slots in the top 10): PR 8693 "MySQL/MariaDB: support the
SOURCE command" → `test/fixtures/dialects/mariadb .sql/.yml` at ranks 1, 2; PR 8620 "BigQuery — WITH
expressions" → `…/bigquery` at 1, 2; missed: PR 8687 (a new `autofix/ansi/035_…` folder), PR 8663 (a
Rust test), PR 8635 and 8448 (a Postgres / generic issue whose fixture went to `ansi`).

## Conclusions (one project, 10 tasks — a sign, not a result)

1. **The place of a new file is predictable from history.** With the anchors a user would have,
   the right folder and type is first for 4–5 of 10 tasks that create files, against 1 of 10 for the
   prior — the relation "changed file → where new files appear beside it" carries the information,
   not the popularity of the folder.
2. **Retrieval quality is not the bottleneck here**: the oracle anchors (the PR's true modified
   files) do no better than flat or simargl_units. What misses is the rules themselves — new folders
   (`autofix/ansi/035_…`), a fixture put in `ansi` though the issue names Postgres, a Rust test. A
   rule needs a level between `dir` (too fine for new folders) and `parent` (too coarse): e.g. the
   folder pattern `autofix/ansi/<n>_<name>/{before,after}.sql`.
3. **What it would add to the fused list**: the 43 created files are 24 % of the PR's changes that
   no component can return. A recommendation "and a new `.sql` + `.yml` in
   `test/fixtures/dialects/mariadb/`" turns part of that quarter into an answer.
4. Whether a task creates files at all is not predicted here (scored only on the 10 that do; 39 % of
   history tasks create files). A real component needs that gate, or shows the slot as a
   suggestion with its confidence.

## What it needs in the tools

- **mokosh** now stores `CHANGE_TYPE` (A / M / D / R) per commit-file row (2026-10-10); a database
  made before gets it filled from `git log --name-status` on the next `mokosh ingest`. exp13.2 reads
  git directly and does not depend on it.
- **gamayun** could add a predicate `spawns_in` (file → folder where new files appear with it),
  beside `co_changes_with`, and show it in search and in the health report.

## Next

- More projects (each needs exp13.1's `future.json` + `runs.json`).
- A folder-pattern level for new folders; the "creates files?" gate.
