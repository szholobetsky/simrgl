# exp10 — triangulation: simargl, gamayun and svitovyd on the same tasks

Three tools of the toolkit look at the same code base from three sides:

| tool | evidence | in one sentence |
|---|---|---|
| **simargl** | meaning — vectors of past tasks | "a task like this one was done before, and it changed these files" |
| **gamayun** | statistics — significant links word ↔ file, co-change (G²) | "this word almost always leads here" |
| **svitovyd** | structure — definitions and import / call / ref links of the code | "this identifier is defined here, and these files use it" |

**Question.** Do they find the same files, or different ones — and does a fusion of them find more
than the best of them alone?

**Data.** SonarQube, the exp7 held-out tasks: 200 test tasks (the exp3 `modn` split) and 200 dev
tasks, which alone choose any weight. No tool sees the answers of a held-out task:
simargl's index answers only from the other tasks (dev and test tasks are cut out of its similar
tasks), gamayun uses the exp7 ontology built without them, svitovyd reads code, not history.

**How each tool ranks files** (`run.py`):

- simargl — the 10 most similar past tasks (bge-small, the existing index) vote for the files they
  changed; `rank` = the best similarity of a task that changed the file (simargl's default), `freq` =
  how many of the 10 changed it.
- gamayun — `search` over the exp7 ontology with the parameters tuned on dev (exp9).
- svitovyd — identifier-shaped tokens of the task text (a capital inside, an underscore or a digit:
  `RuleUpdate`, `getKey`, `sonar_core`) that the code defines, and the files that define them; a
  rarer name weighs more (log N / defining files).

Metrics as in exp3/exp7 (`gamayun.evaluation.metrics_for_query`): MAP, MRR, P@1, R@10, hit@10.

## Part 1 — svitovyd on today's code

### Each tool alone (test, 200 tasks)

| tool | MAP | MRR | P@1 | R@10 | hit@10 |
|---|---|---|---|---|---|
| simargl (rank) | 0.146 | 0.288 | 0.230 | 0.200 | 0.58 |
| simargl (freq) | **0.153** | **0.333** | **0.255** | **0.209** | 0.615 |
| gamayun | 0.129 | 0.287 | 0.215 | 0.174 | **0.62** |
| svitovyd | 0.009 | 0.019 | 0.015 | 0.010 | 0.03 |

Modules (the first folder): simargl MAP 0.715, gamayun's own module ranking 0.759 (hit@10 1.00).

### They find different files

| | |
|---|---|
| top-10 overlap, simargl ∩ gamayun (mean Jaccard) | **0.067** |
| right files in the top-10 of simargl only | 103 |
| right files in the top-10 of gamayun only | 81 |
| right files in the top-10 of all three | 0 |
| recall of the union of top-10s: simargl / gamayun / both | 0.200 / 0.174 / **0.264** |

simargl and gamayun share only one file in fifteen of their top-10s, and the union of their
top-10s holds 32 % more of the right files than the better of them.

### Fusion (test; weights and k chosen on dev only)

| fusion | MAP | MRR | P@1 | R@10 | hit@10 |
|---|---|---|---|---|---|
| best single tool (simargl freq) | 0.153 | 0.333 | 0.255 | 0.209 | 0.615 |
| RRF, equal weights, k = 60 | **0.168** | **0.357** | **0.280** | **0.221** | **0.755** |
| RRF, weights tuned on dev (1.5 / 2 / 0.5, k = 10) | 0.163 | 0.354 | 0.275 | 0.212 | 0.755 |
| RRF simargl + gamayun | 0.162 | 0.345 | 0.270 | 0.213 | 0.755 |

Reciprocal rank fusion with equal weights — no tuning at all — beats the best tool on every
measure: MAP +10 %, and **hit@10 from 0.62 to 0.755**: in three tasks out of four a right file is in
the top 10. Weights tuned on dev do not carry over to test (they lose to equal weights).

### Why svitovyd is near zero here — the code changed

Only **19.9 %** of the files the test tasks changed exist in today's code: the tasks span
2010–2024, and most of those files have since been deleted, moved or renamed. svitovyd reads the code
as it is today, so it cannot name files that are gone. In only 12 of the 200 test tasks do all the
right files still exist. The structural widening of the fused list (neighbours of its top files on
today's map) lowered hit@10 (0.755 → 0.705) for the same reason.

A first run also took plain words that the code happens to define (`fails`, `users`, `System`) as
identifiers; only identifier-shaped tokens count now.

## Part 2 — svitovyd on the code as it was before each task

`snap.py`: one worktree, the test tasks in date order, for each the code at the parent of the
task's first commit, svitovyd's map rebuilt incrementally — svitovyd then ranks against the code the
developer actually had in front of them, and the neighbours of the top files are taken from it.

Stopped after **33 of the 200 test tasks** (2010-10 … 2014-03, ~30 s a task): the direction was
clear and the snapshots are slow. `snap.py` resumes where it stopped. 33 tasks are a direction, not a
proof — one or two lucky hits move the means by a few points.

| | |
|---|---|
| right files that existed before their task | **39 %** (today's code: 20 %) |
| tasks whose right files all existed before them | 11 of 33 |

**Each tool alone (33 tasks)**

| tool | MAP | MRR | P@1 | R@10 | hit@10 |
|---|---|---|---|---|---|
| simargl (rank) | 0.102 | 0.195 | 0.121 | 0.157 | 0.42 |
| gamayun | 0.051 | 0.154 | 0.061 | 0.098 | 0.52 |
| svitovyd, today's code | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 |
| svitovyd, code before the task | 0.054 | 0.125 | 0.121 | 0.081 | 0.15 |

On the 11 tasks whose files all existed before them svitovyd ranks best of the three
(MAP 0.158 against 0.134 simargl and 0.083 gamayun; P@1 0.27) — eleven tasks, an anecdote.

**Fusion (equal-weight RRF, k = 60; 33 tasks)**

| fusion | MAP | MRR | P@1 | R@10 | hit@10 |
|---|---|---|---|---|---|
| simargl + gamayun | 0.079 | 0.183 | 0.091 | 0.145 | 0.64 |
| + svitovyd on today's code | 0.079 | 0.183 | 0.091 | 0.145 | 0.64 |
| + svitovyd on the code before the task | **0.103** | **0.227** | **0.121** | **0.185** | **0.73** |

**Structural widening** (import/call/ref neighbours of the fused top-10, on the snapshot): no weight
helped (MAP 0.079 → 0.078 … 0.045); only 4 right files were among the top-20 neighbours that neither
simargl nor gamayun returned.

## Conclusions

1. **simargl and gamayun find different files.** Their top-10s share one file in fifteen (Jaccard
   0.067); the union of their top-10s holds 32 % more of the right files than either (200 tasks).
2. **Fusion needs no tuning.** Equal-weight reciprocal rank fusion beats the best single tool on every
   measure (MAP 0.153 → 0.168, hit@10 0.62 → 0.755, 200 tasks); weights tuned on dev did worse on test.
3. **svitovyd helps only on the code of its time.** On today's code it adds nothing (80 % of the
   files the tasks changed no longer exist); on the code before each task it joins the fusion with
   +30 % MAP and hit@10 0.64 → 0.73 (33 tasks). For a live task this is no restriction — the code
   in front of the developer *is* the code of its time — but every retrospective evaluation of a
   structural tool must rebuild the code as it was.
4. **Widening by structure does not pay** in this form: the neighbours of the top files are many,
   and the right ones among them few.

For a product this means a triangulation page needs only rank fusion of the three answers on the
current code; for the dissertation, (1)–(3) on 200 tasks would need the snapshot run finished.

## Files

| file | |
|---|---|
| `run.py` | each tool ranks files for the 399 held-out tasks → `out/runs.json` |
| `score.py` | metrics, reach, overlap, fusion (dev-tuned), structural widening → `out/score.json` |
| `snap.py` | svitovyd on per-task snapshots of the code → `out/snap.json` |
