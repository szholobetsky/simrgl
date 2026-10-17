# exp13.3 — the identifiers a task touches and the new links it makes

**Question.** exp13–13.2 predict files. A fix is finer than a file: it touches some identifiers and
sometimes connects two that were not connected (*"table User + service Warehouse → a new link
User–Warehouse"*). The new link is what gamayun's health report measures after the fact (novelty;
structurality: inside one module or across). Can the identifiers, and the new links, be predicted
from the task text before the change?

Tasks: exp13.1's future tasks of sqlfluff — 29 open PRs that close an issue; the issue is the query.
The PRs are not merged, so today's svitovyd map is the code **before** each task: "already linked"
and "new" can be told apart on it. This needs the DIFF of the history the right way round
(mokosh fixed and the databases rewritten on 2026-10-10, see mokosh README).

## Method (`ids.py`)

**Identifiers K** — names the map defines in code files (`.py`, `.rs`), 3+ characters, defined in at
most 10 files, and shaped like an identifier (an upper-case letter, `_` or a digit: `FluffConfig`,
`recursive_crawl`, `CURSOR`). A first run with plain lower-case words too was dominated by comments
and docstrings on the changed lines (`only`, `which`, `must`) and was dropped. sqlfluff: 4 521.

**Truth** — from the PR's patches (GitHub `/pulls/N/files`, code files only):

| | what |
|---|---|
| `touched` | identifiers of K on the changed lines (+ and −) |
| `new_refs` | identifiers of K a file starts to use: on a `+` line of file F, not defined in F, not among the names F already links to in the map (any kind — `expr` is a bare word match, so "not linked" means F never mentioned it) |
| `new_links` | the pairs (F, b) of `new_refs` with F an existing file |
| `new_defs` | names a `+` line defines that K does not know (counted only) |

**Identifier rankings**

| ranking | how |
|---|---|
| `simargl_units` … `flat` | the top 20 files of an exp13.1 component; each file's identifiers, weighted 1/rank and by how many of the task's words they contain |
| `history` | the 10 past tasks simargl finds most like the issue (mode task); the identifiers **their own DIFF** changed, weighted by similarity × idf |
| `ids_flat` | RRF of all of the above |

**New links** — pairs (F, b): F from flat's top 10 files (or the PR's true existing files: oracle),
b from an identifier ranking's top 50, b not defined in F and not linked from F;
score 1/rank(F) · 1/rank(b).

```
python ids.py     # reads ../exp13.1 (projects.csv, out/<p>/future.json, runs.json); fetches the patches once
```

-> `out/<p>/patches.json`, `truth.json`, `rankings.json`, `scores.json`, `results.csv`

## Results — sqlfluff, 29 future tasks

Truth: 28 tasks touch identifiers of K (median 10 per task); 23 make new references (180 in all),
19 new links from an existing file (142); 190 new names are defined. The most touched are the
framework's: `BaseSegment`, `FluffConfig`, `Linter`, `CodeSegment`, `parse_string`, `Delimited`,
`OneOf`.

**touched** (28 tasks)

| ranking | hit@1 | hit@10 | R@10 | R@50 | R@100 | AP@100 |
|---|---|---|---|---|---|---|
| simargl_units (files) | 0.04 | 0.29 | 0.04 | 0.07 | 0.11 | 0.02 |
| simargl_files (files) | 0.07 | 0.21 | 0.03 | 0.06 | 0.09 | 0.02 |
| gamayun_db (files) | 0.00 | 0.07 | 0.01 | 0.05 | 0.07 | 0.01 |
| gamayun_files (files) | 0.00 | 0.11 | 0.02 | 0.06 | 0.08 | 0.01 |
| svitovyd (files) | 0.04 | 0.14 | 0.04 | 0.08 | 0.10 | 0.04 |
| flat (files) | 0.04 | 0.21 | 0.04 | 0.09 | 0.11 | 0.02 |
| **history** | **0.32** | **0.71** | **0.20** | **0.41** | **0.50** | **0.14** |
| ids_flat | 0.07 | 0.29 | 0.05 | 0.12 | 0.19 | 0.04 |

**new_refs** (23 tasks)

| ranking | hit@1 | hit@10 | R@10 | R@50 | R@100 | AP@100 |
|---|---|---|---|---|---|---|
| best of the file-based | 0.00 | 0.17 | 0.04 | 0.08 | 0.10 | 0.02 |
| **history** | **0.17** | **0.35** | **0.12** | **0.30** | **0.33** | **0.09** |
| ids_flat | 0.00 | 0.04 | 0.01 | 0.09 | 0.12 | 0.01 |

**new_links** (19 tasks)

| pairs | hit@1 | hit@10 | R@50 | R@100 | AP@100 |
|---|---|---|---|---|---|
| flat files × ids_flat | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 |
| flat files × history | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| oracle files × ids_flat | 0.00 | 0.00 | 0.06 | 0.09 | 0.00 |
| oracle files × history | 0.05 | 0.16 | 0.08 | 0.17 | 0.01 |

Example, PR 8679 (Snowflake storage lifecycle policy): of `history`'s top 10, six are touched —
`Delimited`, `EqualsSegment`, `OneOf`, `ExpressionSegment`, `ObjectReferenceSegment`, `BaseSegment`.
New links look like `dialect_mysql.py → FileSegment`, `mysql_test.py → FluffConfig`.

## Conclusions (one project, 29 tasks — signs, not results)

1. **The identifiers of a fix are predictable — from history, not from the files.** The identifiers
   the most similar past tasks changed (`history`) find a touched identifier in the top 10 for 71 %
   of the tasks and half of them in the top 100. Taking the identifiers *defined* in the found
   files does far worse (hit@10 ≤ 0.29), even though those files are mostly right (exp13.1): a fix
   *uses* identifiers defined elsewhere — grammar combinators, framework classes — and "what similar
   changes touched" carries that, "what the found file defines" does not.
2. **Do not fuse the identifier rankings equally**: `ids_flat` is far below `history` alone — five
   weak file-based lists outvote the one that works.
3. **New references are partly predictable** (`history`: a newly used identifier in the top 10 for
   35 % of the tasks), **new links (file × identifier) are not yet**: ≈ 0 with the files a user would
   have; even with the true files, hit@10 0.16. The pair space is the product of two uncertain
   rankings; precision on both sides is needed.
4. **sqlfluff's new links are not the "User + Warehouse" kind.** They are a dialect starting to use
   a framework class (`FileSegment`, `WhitespaceSegment`), a test starting to use `FluffConfig` /
   `lint_string` — plumbing, not two domain concepts meeting. A SQL linter has little domain; the
   question belongs to a business application (sonar is closer; flink too).
5. **New names are frequent**: 190 identifiers defined by 29 PRs. Their names cannot be predicted;
   their links to existing identifiers can be, which is where (3) has to improve.

## Next

- The same on sonar / flink (Java, more domain): needs a code map at the snapshot before the tasks
  (no open PRs there).
- Link prediction from history directly: the past tasks that changed F — which identifiers did they
  newly bring into F? (needs the map before each past task, or the DIFF's `+` lines against F's text
  before it).
- Split test and source code in the truth: much of `touched` / `new_links` is test plumbing.
