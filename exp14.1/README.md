# exp14.1 — business terms from glossary: three glossaries of sqlfluff (docs, code, tasks)

**Question.** exp14 (concepts/HEALTH_EVALUATION.md, part C) compares the task vocabulary with the code's,
and its weak point is telling a term of the problem from an ordinary word: a hand-made stop list
never covers English (how, want, provide, …), PR templates (brief, checklist, side effects) or
stack traces. Here a local LLM decides what a term is — 1bcoder's `/flow glossary` — on three
sources of the same project, each its own glossary:

| glossary | source | files | chunks (≈) |
|---|---|---|---|
| `sqlfluff-docs` | `docsv/**/*.md` (the VitePress documentation; site scripts left out) | 39 | ~90 |
| `sqlfluff-code` | `.py` / `.rs` of `src/sqlfluff`, `plugins`, `sqlfluffrs` — test and fixture folders left out | 377 | ~1 900 |
| `sqlfluff-tasks` | `mokosh export units.db --task --level desc` — title + description, one file per task | 4 344 | ~4 400 |

The copies are in `data/` (same relative paths as in the repository); the glossaries in
`.1bcoder/glossary/<project>/` (one `<term>.md` per term + `glossary.md`).

**Terms only.** A full glossary run asks the model three times per term (terms, definition, facts) and
once more per code chunk (think-step) — ~20 calls a chunk, weeks for this repository on this CPU.
For "which words are terms, and how often" one call per chunk is enough: `--terms-only`, added to
`glossary.py` for this (1bcoder, `_bcoder_data/flows/glossary.py`, test
`tests/test_glossary_terms_only.py`). Every chunk a term is found in goes to its `SEEN:` list, with
the file and line — the term's support. A later full run on the same project adds definitions and
facts to the same files.

The flow used is the copy in `.1bcoder/flows/glossary.py` (1bcoder takes a project's own flow first;
`~/.1bcoder/flows/glossary.py` is an older copy without `--terms-only`).

## Run

```
cd exp14.1
1bcoder --scriptapply "C:\Projects\simrgl\exp14.1\glossary_runs.txt" --host http://panteon.local:11434 --model gemma3:4b-it-q4_K_M >> glossary_run.log 2>&1
1bcoder --scriptapply "C:\Projects\simrgl\exp14.1\glossary_runs_small.txt" --host http://panteon.local:11434 --model gemma3:1b >> glossary_run_small.log 2>&1
```

**Models.** Terms: `gemma3:4b-it-q4_K_M` on panteon. Tried on the same two chunks (`linter.py` and the
newest task), panteon: gemma3:4b 45 s / 18 s — terse lists with multi-word terms (*dbt project, dbt
templater, dbt manifest*); qwen3:4b-instruct 57 / 20 s and ministral-3:3b 53 / 18 s — long lists
(ministral with stray backticks); qwen2.5-coder:7b 60 / 30 s. On this machine qwen2.5-coder:7b took
~12 s a chunk, but the machine ran out of memory (the first run was stopped at 12:56 after 8 of 39 docs
files; it is resumed, not repeated — see "incremental" in `concepts/GLOSSARY.md`).

Decision model (`--jev-model`): `tev1:4b`. On a test sentence (*LT05, max_line_length, snowflake,
lexer, checklist, Please*) `tev1:0.8b` called every word *technology*, *Please* included, and took an
invented fact ("was written in Rust") as stated; `tev1:4b` called *Please* noise (0.77), *checklist*
borderline (technology 0.46) and dropped the invented fact. In a developer tool like sqlfluff the
problem's own terms (rules, dialects) are technical, so *technology* vs *domain* will say less here than
in a business project — *noise* is the useful split.

The script path must be absolute: 1bcoder looks for a bare script name in `.1bcoder/scripts/`.
Run it from your own terminal: Claude Code stops the background shells it started when memory runs
short (it did, three times). Chunk 1 000 tokens, overlap 50. Log: `glossary_run.log` (start / end time).

## Analysis (after the run)

- **Support thresholds:** a term counts from k chunks (`SEEN:`) and m tasks (a plain match in the
  task texts); k, m swept.
- **The three vocabularies against each other:** terms of the tasks that the code glossary also has
  (the problem's words the code names), code terms the tasks never use (speculative generality or
  infrastructure), docs as a third view (what the project tells its users).
- **exp14 part C again with glossary terms instead of the stop list:** tasks → code, code → tasks,
  spread over units — does the noise (templates, plain English) go away?
- **Without an LLM:** do gamayun's statistics (G², lift, concentration — `concepts/GAMAYUN.md`) single
  out the glossary's task terms? If they do, glossary is the annotator and the filter needs no model.
- Multi-word terms (`quality gate`) against CamelCase identifiers (`QualityGate`).

## Results (docs glossary complete: 39 files, 678 terms; `python compare.py`)

**B — the health measures of `HEALTH_CONCEPT.md` §8 on the glossary terms** (638 after dropping terms made
only of stop words or words of ≤ 2 letters; k = support, the chunks a term was seen in):

| | k ≥ 1 (638) | k ≥ 2 (153) | k ≥ 3 (65) | exp14, stop list |
|---|---|---|---|---|
| tasks → code (of the terms tasks use, the code names) | 0.52 | 0.72 | 0.81 | 0.65 |
| code → tasks (of the terms the code names, tasks use) | 0.98 | 0.99 | 1.00 | 0.93 |
| a term in one unit only | 0.46 | 0.40 | 0.33 | 0.33 |
| a term over 5+ units | 0.12 | 0.20 | 0.29 | 0.11 |

- **code → tasks ≈ 1:** the code names nothing the tasks never talk about — no speculative vocabulary.
- **tasks → code 0.52:** the gap is not missing domain concepts. Tasks-only terms are the *process*
  (`pull request`, `issues`, `git`, `tox`, `review`, `ruff`, `pre-commit`, `test suite`) and *naming
  mismatches* (`t-sql` vs `tsql`, the CLI commands `sqlfluff lint|fix`). Matching artifacts lower it too:
  plurals (`issues` vs `issue`) and class names the glossary wrote as one word (`lintresult`,
  `datatypesegment` — not split, so not found in `LintResult`).
- **Spread:** rare terms live in one unit; the core ones (`rule`, `parser`, `dialect`, `segment`) cross
  5+ units — for a linter that is the design (every dialect, every rule its own module), not decay.
- **Docs only:** `bison grammar`, `main phase`, `templater section`, the SQL style guides, `adoption tiers`
  — what the project tells its users and neither code nor tasks name.

**C — the glossary against a stop-list frequency list** (the head of the Zipf curve, top 678 1–3-word
n-grams seen ≥ 2 times):

| | glossary | frequency list |
|---|---|---|
| terms in common | 186 of 678 | 186 of 678 |
| multi-word share | 0.54 | 0.14 |
| seen < 2 times in the docs | 0.35 | 0 (by construction) |

Frequency list top, not in the glossary: `fluff`, `com`, `files`, `code`, `docs`, `run`, `set`, `like`.
Glossary only: `dbt labs sql style guide`, `structure column order`, `ambiguous column count`,
`aggregate order by`, `array slices`. Open: a fairer baseline (Luhn's middle zone, keyness / TF-IDF) and
precision by a person — blind sample `out/label_sample.csv` → `python compare.py --score`.

## State

- 2026-10-11 12:37: first run (local qwen2.5-coder:7b) — stopped by the low-memory guard at 12:56 after
  8 of 39 docs files (255 terms).
- 2026-10-11: `glossary.py` got `_index.json` (incremental / resume), `--jev`, `--jev-model`; run restarted
  on panteon (gemma3:4b + tev1:4b), docs → code → tasks.
