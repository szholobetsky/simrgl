# exp13.1 — future tasks: open pull requests that close an issue

**Question.** exp13 scores the components on a project's latest *done* tasks. Two things flatter it
there: the task text is often written after the change (a GitHub project's tasks are pull-request
descriptions), and today's code already contains the change, so a code index meets the task's own
new words in the code. exp13.1 asks the same components about tasks that are **not done yet**:
the project's open pull requests that close an issue. The change is not in the code; the query is
the **issue** — the problem as its reporter wrote it; the truth is the files the PR changes.

It is exp13's measurement (same components, fusions, metrics — imported from `../exp13`) on a
different set of tasks. Kept apart so that each protocol can be rerun on its own.

## Protocol

- `fetch.py` — the open PRs of the project (GitHub API, `GITHUB_TOKEN`) whose description closes an
  issue (`Fixes #123`, `closes https://github.com/o/r/issues/123`; a "fix" pointing at another PR is
  skipped). Per PR: the issues' titles and bodies (the query, ≤ 2 000 characters as in exp13), the
  PR's own title + description (`pr_text`, for the control below), its changed files with their
  status, draft or not. → `out/<p>/future.json`
- **Indexes** — the project's own, built from the whole history (`projects.csv` names them): none of
  the future tasks is in the database yet, so nothing is held out. Checked for sqlfluff: no issue
  number and no PR number of the 29 is a task in `units.db`.
- **Truth** — the PR's changed files that exist today (modified / removed / renamed, read as today's
  paths through exp13's resolver). Files the PR **adds** do not exist yet — no index can know them —
  and are counted apart, not scored.
- **Scores** — all future tasks, and `ready` (drafts left out).
- **Control** — `future.py --text pr` asks with the PR's own text instead of the issue's. exp13 vs
  this control isolates "the code already holds the change"; the control vs the issue run isolates
  "the text describes the done change vs the problem".

```
python fetch.py               # out/<p>/future.json
python future.py              # query = the issue          -> runs.json, scores.json
python future.py --text pr    # query = the PR description -> runs_pr.json, scores_pr.json
                              # results.csv: every project, both queries, all / ready
```

A pull request is a moving truth: it can still change, and a draft more so. The fetch date is in
`future.json`; rerunning `fetch.py` later gives a different set.

## Projects

`projects.csv`: `name, repo, db, github` (owner/repo) and where the five indexes are —
`store` (simargl store) with `units` / `files` (simargl project names), `onto_db` / `onto_files`
(gamayun project folders), `map` (svitovyd).

## Results — sqlfluff, fetched 2026-10-10 22:45

75 open PRs, **29 close an issue** (2 drafts). The issues date from 2021 to 2026 (20 of them 2026).
181 changed files: 43 added by the PR (not scored), 4 not in today's tree, the rest scored; all 29
tasks reachable. Indexes: simargl `sqlfluff` (units, granite, 4 346 tasks) and
`sqlfluff-simargl-code` (files, granite, chunk 200), gamayun `sqlfluff` and `sqlfluff-code`,
svitovyd map — all the user's, from 2026-10-09/10.

**Query = the issue** (the real use: a new problem, code not changed yet)

| component | hit@1 | hit@10 | R@10 | MRR | MAP |
|---|---|---|---|---|---|
| simargl_units | 0.517 | **0.897** | **0.568** | 0.665 | 0.354 |
| simargl_files | 0.345 | 0.690 | 0.425 | 0.470 | 0.302 |
| gamayun_db | 0.103 | 0.414 | 0.231 | 0.239 | 0.131 |
| gamayun_files | 0.138 | 0.621 | 0.320 | 0.271 | 0.170 |
| svitovyd | 0.138 | 0.655 | 0.309 | 0.324 | 0.178 |
| tri | **0.586** | 0.862 | 0.514 | **0.684** | 0.388 |
| flat | 0.552 | **0.897** | 0.558 | 0.681 | **0.408** |
| family | 0.483 | 0.828 | 0.502 | 0.603 | 0.357 |

`ready` (27 tasks without the drafts) differs by ≤ 0.05 everywhere — `scores.json`.

**Control: query = the PR description** (exp13's kind of text, code still not changed)

| component | hit@1 | hit@10 | R@10 | MRR | MAP |
|---|---|---|---|---|---|
| simargl_units | **0.759** | 0.862 | 0.509 | 0.780 | 0.425 |
| simargl_files | 0.379 | 0.759 | 0.478 | 0.526 | 0.313 |
| gamayun_db | 0.069 | 0.379 | 0.224 | 0.168 | 0.121 |
| gamayun_files | 0.172 | 0.724 | 0.397 | 0.338 | 0.214 |
| svitovyd | 0.345 | 0.759 | 0.415 | 0.486 | 0.231 |
| tri | 0.586 | 0.931 | 0.552 | 0.704 | 0.413 |
| flat | 0.724 | **0.966** | **0.639** | **0.792** | **0.494** |
| family | 0.690 | **0.966** | 0.577 | 0.755 | 0.436 |

Seconds per task: simargl 2.3–2.4 (an Ollama embedding per query), gamayun 0.04–0.11, svitovyd 0.05.

## Conclusions (one project, 29 tasks — a hit@1 of 0.5 is ± 0.18 at 95 %; signs, not results)

1. **The code index was flattered in exp13.** With the PR text as the query in both, simargl_files
   falls from hit@1 0.77 / MAP 0.42 (exp13, change already in the code) to 0.38 / 0.31 (code not
   changed yet). simargl_units, which reads only the history, does not fall (0.70 → 0.76). For a
   *new* task, the history of similar tasks is the stronger embedding index.
2. **The issue is harder than the PR text** — for every component that matches words: simargl_units
   hit@1 0.76 → 0.52, svitovyd 0.35 → 0.14, flat MAP 0.49 → 0.41. A PR description names what was
   changed (dialect, rule code, file); an issue names the symptom. The real use is the issue, so
   **the issue run is the number to quote** for sqlfluff; the PR run is an upper bound.
3. **Fusion pays off on future tasks** (hypothesis 3, first half — yes here, unlike exp13): with the
   issue as the query, `tri` has the best hit@1 (0.59 vs 0.52) and `flat` the best MAP (0.41 vs 0.35),
   and fusion matches the best hit@10. The weak components are wrong in different places from
   simargl, which is what fusion needs. `family` is again below `flat` (second half — no).
4. **simargl_units is the single component to keep** on sqlfluff: best or near best on every metric,
   with either text. The two gamayun indexes and svitovyd are weak alone (MAP 0.13–0.18) but are
   what lifts the fusion.
5. 43 of 181 changed files (24 %) are files the PR creates — a quarter of every future change is out
   of reach of any index by construction. exp13 counts them (after the fact they exist).

## Next

- The same on more GitHub projects of the ten (each needs only `fetch.py` + its indexes).
- For Jira projects (sonar, flink) there are no open PRs with the change: the counterpart is the
  issues fixed after the code snapshot, with the code checked out at the snapshot.
