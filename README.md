# SIMARGL — research repository

Experiments, concepts and data of the PhD research «Інтелектуальна технологія підтримки розробки та
супроводу програмних продуктів» (Intelligent Technology for Software Development and Maintenance Support).

**Research question.** Given a task description in natural language (Jira, GitHub Issues), can we predict
which modules, files and identifiers of the code have to change — and do it with small local models,
on a CPU, without sending the code anywhere?

The research grew into ten tools. Each lives in its own repository; this repository holds the experiments
that measure them, the concept documents behind them, and the dataset lists.

---

## Tools

| Tool | Role |
|---|---|
| **[simargl](https://github.com/szholobetsky/simargl)** | Task-to-code retrieval — given a task description, finds which files and modules are likely affected, using semantic similarity over git history |
| **[svitovyd](https://github.com/szholobetsky/svitovyd)** | Project map — scans any codebase and produces a structural map of definitions and cross-file dependencies; queried from the CLI, a web UI with a layered map, or an MCP server |
| **[1bcoder](https://github.com/szholobetsky/1bcoder)** | AI coding assistant for small local models — surgical context management, agents, parallel inference, proc scripts |
| **[yasna](https://github.com/szholobetsky/yasna)** | Session memory — indexes conversations from all AI agents so you can find what was discussed, when, and where |
| **[radogast](https://github.com/szholobetsky/radogast)** | Context drift monitor — measures how far an AI agent's conversation has drifted from the original task |
| **[vyrii](https://github.com/szholobetsky/vyrii)** | Local AI web UI — chat, translate, web research, RAG and file management in the browser (Flask; Gradio optional); powered by Ollama or any OpenAI-compatible backend |
| **[syryn](https://github.com/szholobetsky/syryn)** | Bluetooth identity beacon for headless devices — the device's IP is its Bluetooth name; hostname, mDNS and network interfaces on request |
| **[alkonost](https://github.com/szholobetsky/alkonost)** | Browser cockpit for 1bcoder's flows — the `deepagent_*` spec pipeline (plan tree → specs → task board) and the glossary (build, browse, search), tuned for small local models |
| **[gamayun](https://github.com/szholobetsky/gamayun)** | Statistical ontology without an LLM — builds triples of terms, identifiers, files and modules from task history or a folder of documents and code; every triple carries its significance test |
| **[mokosh](https://github.com/szholobetsky/mokosh)** | Dataset gathering — commits of a git repository and the tasks of its tracker (Jira, GitHub, GitLab, YouTrack) in one SQLite database that the other tools read |

- **simargl** finds the code related to a task by vector search over the task history (embeddings).
- **svitovyd** answers: *how is the code structured and what depends on what?*
- **1bcoder** answers: *how do I work with local models efficiently?*
- **yasna** answers: *where did I already discuss this?*
- **radogast** answers: *is the AI agent still on track toward the goal?*
- **vyrii** answers: *how do I access all of this through a browser?*
- **syryn** answers: *what is the address of this headless device?*
- **alkonost** answers: *how do I turn one task description into a reviewed, working plan — and see the project's glossary — without a big model or a big context window?*
- **gamayun** finds the code related to a task through an ontology of significant links between words, files and modules (no LLM).
- **mokosh** prepares the initial dataset from version control and the issue tracker.

---

## Experiments

Each folder has its own README with the method, the numbers and how to rerun it. MAP is file-level unless
said otherwise; `modn` is the honest split (every n-th task held out), `recent` (the latest tasks held
out) flatters the scores.

### Retrieval baselines (2024–2025)

| Exp | What | Result |
|---|---|---|
| [exp0](exp0/) | TF-IDF of task words over code modules. | Failed: MAP 0.5–1.5 % and 4–48 hours of computation; kept for reference only. |
| [exp1](exp1/) | Statistical analysis of the task and commit data (Bradford ranking, HHI). | Term distributions and module concentration; descriptive, not predictive. |
| [exp2](exp2/) | Embedding models compared at module level: Word2Vec, FastText, GloVe, BERT, CodeBERT, BGE. | BGE-large best (MAP 0.37 sonar, 0.36 flask); CodeBERT worst — tasks are natural language, not code. |
| [exp3](exp3/) | Full RAG pipeline: sentence-transformers + pgvector/Qdrant, splits, windows, sources. | Module MAP ≈ 0.45 on `modn` (≈ 0.80 on `recent`), file MAP < 0.03; title+description is the best source. |
| [exp3.1](exp3.1/) | The task unit: a tracker ticket vs a commit message, 10 projects. | The exp3 findings hold on 10 projects (module 0.59 vs file 0.13 MAP); the best history window is project-specific. |
| [exp3.2](exp3.2/) | Cross-vocabulary: an index of commit messages searched with ticket text, 9 projects. | Competitive with matched vocabularies (e.g. sonar 0.217 vs 0.200, rubocop 0.465 vs 0.418). |
| [exp4](exp4/) | Architecture families: ModernBERT, ColBERT, Qwen3 / jina / e5 embeddings. | Planned. |
| [exp5](exp5/) | Symbol grounding: task words → code identifiers → file and line. | Planned. |
| [exp6](exp6/) | simargl's retrieval modes `task`, `file`, `aggr` and their combinations. | Evaluation code and plan. |

### Ontology and triangulation (2026)

| Exp | What | Result |
|---|---|---|
| [exp7](exp7/) | gamayun: an ontology of tracker + git history with no LLM, every triple with a significance test. | Module MAP 0.752 with naive Bayes over the triples (vs 0.693 for the most frequent modules). |
| [exp8](exp8/) | Synergy of simargl, gamayun and svitovyd on sonar and flink. | Each finds about a third of the right files alone; fusion adds 13–16 % MAP over the best single tool. |
| [exp8.1](exp8.1/) | Does an ontology of the code help the ontology of the task history? | No: the code ontology is five times weaker (MAP 0.024 vs 0.116) and any share of it lowers the score. |
| [exp10](exp10/) | Triangulation: simargl + gamayun + svitovyd, equal-weight reciprocal rank fusion, 200 `modn` tasks. | Fusion beats the best single tool on every measure (hit@10 0.62 → 0.755); tuned weights did worse. |
| [exp11](exp11/) | Triangulation on the 20 latest tasks against today's code. | Fusion ≥ the best tool; a file named by two or three tools is right about four times as often as one named by one. |
| [exp12](exp12/) | Embedding models for finding past tasks on a CPU. | Interim: granite-embedding:30m best measured (MAP 0.200, bge-small 0.171). |
| [exp13](exp13/) | Six indexes per project (svitovyd, simargl units/files, gamayun db/files), latest tasks held out, sqlfluff. | Embeddings win (MAP ≈ 0.40); fusion is not better than the best single; the code indexes are flattered. |
| [exp13.1](exp13.1/) | Future tasks: open pull requests that close an issue, the issue as the query. | The code index drops (hit@1 0.77 → 0.38); fusion beats the best single (MAP 0.41 vs 0.35). |
| [exp13.2](exp13.2/) | Where a task creates files: rules "anchor file → (folder, extension)". | The right folder first for 4–5 tasks of 10, against 1 of 10 for the prior. |
| [exp13.3](exp13.3/) | The identifiers a task touches and the new links it makes. | Identifiers of the most similar past tasks: hit@10 0.71; new links are not predictable yet. |

### Agents with small models (2026)

| Exp | What | Result |
|---|---|---|
| [exp9](exp9/) | A Haiku agent with the tools' rankings vs plain code search. | Without reading code it missed 4 of 5 tasks; the comparison was unfair and exp9.1 repeats it. |
| [exp9.1](exp9.1/) | The same, fair: code checked out before the fix, both groups may read code. | Reading code is the main factor (MAP ~0.25 → ~0.8); the indexes help where the task has no literal anchor. |
| [exp9.2](exp9.2/) | Small local models (1–9b, 4k context, CPU) with the indexes as tools. | Small models do not call the indexes: to use a tool a model needs an intent, which is what it lacks. |
| [exp9.3](exp9.3/) | Passive search: the indexes search, the small model judges. | It gives a small model a chance on hard tasks (the best was qwen3:1.7b), but the candidates must be checked. |
| [exp9.4](exp9.4/) | Passive search + companions: 3 models × 10 tasks, with time and tokens. | gemma4:e4b with companions is the first small model to beat the indexes alone (MAP 0.409 vs 0.345). |
| [exp9.5](exp9.5/) | `gamayun compass` on tasks the rule never saw. | P@1 drops from 0.70 to 0.24–0.28: the first ten tasks were unusually easy. |
| [exp9.6](exp9.6/) | Haiku with compass and companions vs Haiku of exp9.1. | Not a killer feature: MAP 0.688 vs 0.797; a wrong compass sends companions the wrong way. |
| [exp9.7](exp9.7/) | Searching for "deep" tasks, where the needle is not where the task's words point. | Rare: 5 of ~400 tasks (≈ 1 %); gamayun has the path to the needle but does not rank it high. |
| [exp9.8](exp9.8/) | Deep tasks: Haiku with code search vs Haiku with the indexes. | Needles reachable only through "changes together" links were found only with the indexes, at lower cost. |

### Health and glossary (2026)

| Exp | What | Result |
|---|---|---|
| [exp14](exp14/) | What the Novelty × Structurality square measures, on sqlfluff, sonar and flink ([concept](concepts/HEALTH_EVALUATION.md)). | A unit's zone follows its place in the code, not its state; new files are 0.20–0.35 of the pairs. |
| [exp14.1](exp14.1/) | Business terms from a glossary of the sqlfluff docs (1bcoder `/flow glossary`), big vs small models. | Running. On the docs glossary: of the terms the tasks use, the code names 0.52 (0.81 for terms seen 3+ times). |

### Locked findings

1. The aggregation strategy (avg, sum, median, weighted, cluster) has no effect on MAP — use `avg`.
2. The split matters: `recent` inflates module MAP to ~0.80; `modn` (~0.45) is the honest baseline.
3. The description beats the title and the comments as the query source; title + description is the best.
4. History window: w1000 best for `modn`, w100 for `recent` on sonar; on other projects it varies (exp3.1).
5. General text models beat code models (CodeBERT is the worst): tasks are written in natural language.
6. File-level retrieval is far harder than module-level (MAP < 0.03 vs ~0.45 on sonar).

---

## Repository layout

```
exp*/        experiments — scripts, README, results (rebuildable indexes are git-ignored)
concepts/    research concept documents (INDEX.md indexes them)
datasets/    project lists: projects.csv (infrastructure / dev tools), bizprojects.csv (domain-rich)
ragmcp/      the first MCP server + local Ollama agent over the exp3 index
info/        results, leaderboard, planning notes
images/      logos and illustrations
1bcoder/ vyrii/ simargl/ radogast/ yasna/ alkonost/ svitovyd/
             git submodules — pinned snapshots; each tool is developed in its own repository
```

Datasets are SQLite databases with a task table and a commit table (one row per commit and file),
gathered by [mokosh](https://github.com/szholobetsky/mokosh). `AGENTS.md` is the working context for AI
assistants in this repository.

---

## About

(c) 2026 Stanislav Zholobetskyi, Oleh Andriichuk  
Institute for Information Recording, National Academy of Sciences of Ukraine, Kyiv  
PhD research: «Intelligent Technology for Software Development and Maintenance Support»

Academic research code; use for educational and research purposes.
