# exp13 — which component finds the code best, and on what kind of project

**Question.** exp10 and exp11 compared simargl, gamayun and svitovyd on two projects (sonar, flink).
There the components came out about equal (sonar: simargl = gamayun, hit@10 0.69) or simargl ahead
(flink: 0.40 vs 0.30). But two projects cannot separate *size* from everything else that differs
between them. exp13 runs the same comparison on ~10 projects, with every index each tool can build,
and puts the scores next to a profile of the project.

**Hypotheses** (written before the runs):
1. The bigger the history, the more simargl (embeddings) gains on gamayun (ontology): an ontology
   spreads its significance over more files, a vector neighbourhood does not.
2. Where task texts name the code directly (identifiers, class names), svitovyd and gamayun win —
   structure beats meaning when the words already point at the files.
3. Fusing all components beats the best single one on every project, and a fusion that fuses each
   tool's two indexes first (family) beats a flat fusion of all five (two simargl indexes on the
   same model are not two independent votes).

The answer to any of them being "no" is a result too.

## Projects

`projects.csv` — one row per project, three columns:

```
name,repo,db
sonar,C:\Projects\Data\sonarqube,C:\Projects\Data\sonar.db
```

`repo` is a git clone at today's code, `db` the task/commit database mokosh wrote (TASK/TASKS +
COMMIT/COMMITS, read through `mokosh.schema`).

## Protocol

- **Held-out tasks** — the 30 tasks of a project whose last commit is the latest (dates in UTC), as in
  exp11. Every task-history index is built from `past.db`, a copy of the database without them, so
  no index has seen the answer. The code indexes see **today's** code — the way the tools are used on
  a new task; this is the same choice as exp11 and its limit is the same: today's code already
  contains what the held-out tasks changed.
- **Truth** — the files a held-out task changed, read as today's paths (a moved file → its one
  namesake in today's tree). A task none of whose files exists today is left out of the scores.
- **Defaults only.** Every tool runs as a user would run it: gamayun's default parameters, simargl's
  default model (`ollama://granite-embedding:30m` since 2026-10-09), svitovyd's one mode. Tuning per
  project is the next step, not this one.

## The six indexes per project (`build.py`)

| step | command | from |
|---|---|---|
| `svitovyd` | `svitovyd index <repo> --out …/map.txt` | today's code |
| `keywords` | `svitovyd keywords --map … index` | the map |
| `gamayun_db` | `gamayun index --db past.db --repo <repo> --project exp13-<p>-db` | past tasks |
| `gamayun_files` | `gamayun index --files <repo> --project exp13-<p>-files` | today's code |
| `simargl_units` | `simargl index units past.db --project units` | past tasks (embeddings) |
| `simargl_files` | `simargl index files <repo> --project files --chunk-size 200` | today's code (embeddings) |

Everything goes to `out/<project>/` (`--store-dir`, `--root`, `--out`); nothing is written into the
repositories. The tools still register what they build machine-wide (`~/.gamayun/projects.json`,
`~/.svitovyd/maps.json`, `~/.simargl/stores.json`), so the UIs list the exp13 indexes too; gamayun
registers by name, hence `exp13-<project>-db` / `-files`. Each step is timed into `out/<project>/build.json` — the cost side of the comparison.
A built step is skipped on the next run (`--force` to rebuild). The embedding steps are the slow ones
(hours on this CPU for a big project); `--only` picks steps:

```
python build.py                                   # all projects, all steps
python build.py flink --only split,svitovyd,keywords,gamayun_db,gamayun_files   # no embeddings
python build.py flink --only simargl_units        # later, when the machine is free
```

## Components and fusions (`run.py`)

Five answers per task — svitovyd's keyword index is built (and timed) but `files_for_text` reads the
map, so the six indexes make five components:

| component | answer |
|---|---|
| `simargl_units` | the past tasks most like this one vote for their files (mode task, sort freq) |
| `simargl_files` | the code chunks most like the task text (mode file) |
| `gamayun_db` | the ontology of the past tasks |
| `gamayun_files` | the ontology of today's code |
| `svitovyd` | `files_for_text` on today's map |
| `tri` | RRF of simargl_units + gamayun_db + svitovyd — simargl's Triangulation page |
| `flat` | RRF of all five |
| `family` | RRF inside each tool first, then RRF of the three tools |

Scores: hit@1/5/10, recall@10, MRR, MAP (over the top 50) → `out/<project>/scores.json` and
`results.csv` (all projects), with each component's seconds per task.

## The project profile (`profile.py`)

What a score may depend on → `out/<project>/profile.json`, `profiles.csv`: tasks, commit rows,
history files, years; files per task (mean, median); task words (median); today's files and the top
extensions; hubs (files changed by ≥ 1 % of tasks); `names_code` — the share of the latest 500 past
tasks whose text names an identifier defined in the code.

```
python build.py && python run.py && python profile.py
```

## Next: a component that recommends the setup for a project

With `results.csv` and `profiles.csv` side by side for ~10 projects, the question becomes: can the
profile — cheap to compute, no embeddings — predict which components (and which fusion) to use on a
new project? If it can, that is a tool: it reads a repository and its database, profiles them, maybe
builds the cheap indexes and scores them on the project's own latest tasks (exactly this experiment,
in minutes without embeddings), and recommends what to build and how to fuse. Ten projects are too
few to fit a model; they are enough to see whether the hypotheses above hold and which profile
numbers move the scores.

## State

- 2026-10-09: scripts written, smoke-run on sqlfluff without the embedding steps. The sqlfluff
  database then had no task texts at all (`.mokosh/project.yaml` named the GitHub repository wrong;
  fixed in mokosh — empty tracker answers are no longer stored, and `mokosh init` asks for the
  repository URL). Re-ingested 2026-10-10: 4 344 of 4 346 tasks have a title.
- 2026-10-10: **sqlfluff done** (below). sonar and flink: nothing built yet.

### How sqlfluff was built

The three code indexes were the user's own, built on 2026-10-09 in the repository and copied into
`out/sqlfluff/` (`build.json` says `adopted`): simargl `sqlfluff-simargl-code` (granite, chunk 200 —
exp13's defaults), gamayun `sqlfluff-code`, svitovyd `map.txt` + `keyword.txt`. They hold no tasks.
The two task-history indexes were rebuilt from `past.db` (the user's were built from the whole
database, held-out tasks included): gamayun_db 12 s, simargl_units 939 s (4 316 tasks, granite).

`run.py` gives a second score, **old**: the truth without the files a held-out task created itself
(`git --diff-filter=A` of its commits). Today's code already holds those files, so a code index can
find them from the task's own words. sqlfluff: 37 of 338 truth files (11 %).

### Results — sqlfluff, 30 latest tasks (all reachable)

| component | hit@1 | hit@10 | R@10 | MRR | MAP | MAP old |
|---|---|---|---|---|---|---|
| simargl_units | 0.700 | 0.900 | **0.538** | 0.764 | 0.405 | **0.476** |
| simargl_files | **0.767** | 0.900 | 0.452 | **0.813** | **0.420** | 0.413 |
| gamayun_db | 0.200 | 0.500 | 0.244 | 0.325 | 0.174 | 0.192 |
| gamayun_files | 0.067 | 0.667 | 0.188 | 0.237 | 0.097 | 0.124 |
| svitovyd | 0.267 | 0.700 | 0.249 | 0.424 | 0.155 | 0.203 |
| tri | 0.600 | **0.933** | 0.415 | 0.759 | 0.369 | 0.455 |
| flat | 0.633 | 0.900 | 0.437 | 0.755 | 0.354 | 0.435 |
| family | 0.567 | **0.933** | 0.419 | 0.678 | 0.295 | 0.377 |

Profile (`profile.json`): 4 316 tasks over 7.9 years, 51 929 commit rows, 6 058 files today (.sql
2 668, .yml 2 562, .py 475 — mostly test fixtures), 3 files per task (median), task texts 168 words
(median), 59 hubs.

### Conclusions (one project — read as signs, not results)

1. **Embeddings win by far on sqlfluff**: both simargl indexes reach MAP ≈ 0.40, the ontologies and
   svitovyd 0.10–0.17. sqlfluff is one file per SQL dialect (`dialect_duckdb.py`, …) and the task
   says the dialect ("DuckDB: Parse …"); a vector neighbourhood finds that file first.
2. **Fusion does not beat the best single component here** (hypothesis 3 — no): it raises hit@10
   (0.93) and lowers MAP — the three weak components pull the two strong ones down. `family` is
   below `flat` (the second half of hypothesis 3 — no).
3. **The protocol flatters the code indexes.** The task texts are pull-request descriptions written
   after the change, and today's code already contains the change. Leaving out the files a task
   created costs simargl_files a little (hit@1 0.77 → 0.63 old), simargl_units nothing. exp13.1
   measures this properly — on open pull requests, whose change is not in the code yet: there
   simargl_files falls to hit@1 0.38 and fusion beats the best single component on MAP and hit@1. **Read exp13's
   numbers for the code indexes as an upper bound.**
4. The texts of a GitHub project are PR descriptions ("Brief summary of the change made"), not
   problem statements as in Jira (sonar, flink): `mokosh` links commits to PR numbers. Comparing
   projects needs this noted in the profile.
