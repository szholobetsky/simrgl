# 1bcoder — Video Course Plan

**Format**: 9 lectures, each 20–40 minutes
**Target audience**: Developers who want to use small local LLMs (1B–7B) for real coding tasks — offline, no cloud, no subscription
**Prerequisite**: Ollama installed, at least one model pulled (`qwen2.5-coder:1.5b` recommended to start)

---

## Why this course exists

Most AI coding tools assume you have a fast internet connection, a paid subscription, and a large model. 1bcoder assumes the opposite: you have a laptop, maybe 4–8 GB of RAM, and a model that hallucinates if you look at it funny.

The insight behind 1bcoder is simple: **small models fail not because they're stupid, but because they're overwhelmed**. Give a 1B model a 600-line file and ask it to "fix the bug" — it panics and invents something. Give it 15 lines and ask it to output exactly `LINE N: corrected content` — it frequently succeeds.

This course is about learning to work *with* that constraint rather than against it.

## The autonomy tradeoff — why 1bcoder is different

Most agent frameworks are built around one goal: maximum autonomy. The agent reads the task, explores the repo, writes the code, runs the tests, ships — with no human in the loop. That goal is real and worth pursuing. It also requires the largest possible models. Below GPT-4 class or 70B+ locally, fully autonomous agents fail in ways that are hard to predict: wrong file, wrong assumption, subtle logic error, confident and silent.

1bcoder makes the opposite bet.

We accept partial autonomy as the design, not the limitation. With a 4B model you get a reliable read-only research agent (`/agent ask`). With a 1.5B model you get an agent that can execute bounded, well-defined loops. With a 0.5B or even a 350M model there is no agent loop at all — but the model still earns its place: explain this function, name this pattern, write this boilerplate given these 15 lines. That is useful. That is enough.

**The programmer stays in the loop.** Confirming actions. Choosing files. Recognizing when the model is confused and narrowing the question. This is not a bug to fix. It is the architecture. Small models are precise tools, not general reasoners. A precise tool in a skilled hand beats a confused agent every time.

The payoff is everything this course is about: offline, private, on your own hardware, with a model that costs nothing to run and is available right now.

## Privacy — your code stays on your machine

Every prompt you send to a cloud AI assistant leaves your machine. Your code, your internal API names, your database schemas, your business logic — it all travels to a third-party server, is logged, and is subject to a data retention policy you don't control.

For a personal side project this may be acceptable. For professional work it rarely is. Most employment contracts prohibit sending proprietary code to external services. In finance, healthcare, government, and defense the constraints are regulatory, not just contractual. And even where there is no rule written down, handing your internal architecture to a vendor is a security risk that most teams would refuse from any other tool.

1bcoder runs entirely on your hardware. No prompt leaves your machine. No API key, no telemetry, no network connection required. The model that runs in your terminal knows only what you show it — and forgets everything when the session ends.

This is not a niche concern. For many developers it is the only reason they can use AI assistance at work at all.

---

## Lecture 1 — The Model, the Context, and How 1bcoder Thinks

**Core concept**: The context window is not just a technical limit — it's the working memory of the model. Everything the model "knows" during a conversation fits inside it. When it fills up, the model starts forgetting. When it overflows, quality collapses.

**What you'll learn**:
- Starting 1bcoder, selecting a model
- What the status line tells you: `qwen2.5-coder:1.5b [1.1G Q4_K 32K] │ ctx 312 / 8192 (4%)`
- `/ctx` — checking fill percentage at any moment
- `/ctx compact` — asking the AI to summarize and compress the conversation when context is getting full
- `/ctx compact N` — compact only the last N messages in place (keep the rest)
- `/ctx cut` — mechanical trim of oldest messages
- `/ctx clear` — fresh start for a new task
- `/model` and `/host` — switching models and providers at runtime
- `/param temperature 0.1` — why lower temperature helps with code tasks
- `/config save` / `/config auto on` — persisting session settings across restarts

**Narrative**: Before touching any files, you need to understand what you're working with. A 1B model with 8K context and 30% fill is a different tool than the same model at 90% fill. The status line is your dashboard — check it like a fuel gauge.

**Key insight**: 1bcoder shows context fill after every message because it matters. When you hit 70–80%, stop and compact. A fresh, compressed context outperforms a bloated one every time.

**Example session**:
```
> /ctx
  messages: 12   tokens: 5842 / 8192 (71%)
> /ctx compact
[AI summarizes the session into ~400 tokens]
  tokens: 412 / 8192 (5%)
> /ctx compact 1
[compact last 1 message — that 900-token essay the model wrote]
  1 message(s) → 1 summary
```

---

## Lecture 2 — Navigating a Codebase

**Core concept**: Before you can fix something, you need to find it. 1bcoder has three levels of navigation — directory structure, file content search, and structural map. Each serves a different question.

**What you'll learn**:
- `/init` — creating the `.1bcoder/` workspace in your project
- `/tree` — visualizing directory structure, injecting it into context
- `/find` — searching filenames and content with regex; flags `-f` (files only), `-c` (content), `-i` (case-insensitive)
- `/map index` — scanning the entire project into a searchable structural index
- `/map find` — querying the map: which files define or call a given identifier?
- `/map trace` — BFS backwards: who depends on this function?
- `/map idiff` — re-index after an edit, see what structurally changed

**Narrative**: Imagine you've just cloned a 50,000-line Java project and need to find where user authentication happens. You could read files randomly, or you could: `/find auth -c` to find files mentioning auth, then `/map find \UserService` to see what depends on it, then `/map trace validateToken` to trace the call chain. In three commands you've mapped the relevant slice of the codebase — without reading a single file in full.

**Example**:
```
> /map index .
[map] indexed 127 files → .1bcoder/map.txt

> /map find \validateToken
auth/TokenValidator.java
  defines: validateToken(ln:34)
  links → auth/UserService.java (call:validateToken)
  links → tests/TokenTest.java (call:validateToken)
```

---

## Lecture 3 — Semantic Search with simargl (/mcp)

**Core concept**: `/map find` answers a structural question — "where is this identifier defined?" simargl answers a semantic question — "given this task description, which files are likely to change?" The two tools are complementary: map gives you the call graph, simargl gives you the blast radius.

**What you'll learn**:
- What simargl is: a task-to-code retrieval index built from git history and Jira tasks
- `/mcp connect simargl simargl-mcp --cwd <store-dir>` — connecting the server from within 1bcoder
- `/mcp simargl find -t sort rank <query>` — retrieve files ranked by relevance to the most similar tasks
- `/mcp simargl find -t sort freq <query>` — retrieve files ranked by how often they co-appear across tasks
- `/mcp simargl find -f <query>` — direct semantic search in the file index (no task layer)
- `/mcp simargl find -a <query>` — aggregated: average the top-K task vectors, then search files
- `rank` vs `freq`: rank surfaces files specific to the most similar tasks; freq surfaces files that change across many tasks (pom.xml, build.gradle, entry points)
- `/mcp simargl status` — check what is indexed, model used, file and task counts
- `simargl index files <path>` and `simargl index tasks <db.sqlite>` — building the index from CLI

**Narrative**: You receive a Jira ticket: "add sonar.buildString to api/project_analyses/search." You have no idea which files that touches. `/map find buildString` finds nothing — it's not in the codebase yet. `/mcp simargl find -t sort rank add sonar.buildString to api/project_analyses/search` returns the files changed in the 20 most similar historical tasks. You now have a ranked list of files to load — before reading a single line of code.

**rank vs freq in practice**:
- Use `rank` when you want precision — files specific to tasks most similar to yours
- Use `freq` when you want coverage — entry points and infrastructure touched by many tasks
- Run both and compare: `rank` gives the surgical cut, `freq` gives the shared scaffolding

**The two-map pattern**: start every unfamiliar task with both tools:
```
> /map find <identifier>          # structural: call graph, definition sites
> /mcp simargl find -t sort rank <task description>   # semantic: historical blast radius
```
Intersect the two lists. Files appearing in both are your highest-confidence targets.

**Example**:
```
> /mcp connect simargl simargl-mcp --cwd C:/data/sonar-index

> /mcp simargl find -t sort rank add sonar.buildString to api/project_analyses/search
files:
  1. server/sonar-webserver-api/src/main/java/org/sonar/server/project/Project.java  [0.87]
  2. server/sonar-webserver-es/src/main/java/org/sonar/server/issue/index/IssueDoc.java  [0.81]
  3. server/sonar-webserver-webapi/src/main/java/org/sonar/server/projectanalysis/ws/SearchAction.java  [0.79]
modules:
  1. server/sonar-webserver-webapi  [3 files]
  2. server/sonar-webserver-api     [2 files]

> /mcp simargl find -t sort freq add sonar.buildString to api/project_analyses/search
files:
  1. pom.xml  [18 tasks]
  2. server/sonar-webserver-webapi/src/main/java/.../SearchAction.java  [12 tasks]
  3. server/sonar-webserver-api/src/main/java/.../Project.java  [9 tasks]
```

**Key insight**: simargl does not read your code — it reads your history. The index is built from tasks that were already completed, embedding task descriptions alongside the files that were changed. The model learns "tasks like this touch files like those" purely from git history and issue tracker data.

**Connecting to /map**: after simargl gives you the file list, hand it to `/map trace` to extend along the call graph:
```
> /map trace SearchAction        # who calls this? who is called?
> /readln .../SearchAction.java 1-50
```

---

## Lecture 4 — Reading and Editing Code

**Core concept**: The model should never see more than it needs. `/read` is not just file loading — it's context sculpting. The smaller and more precise the context, the better the output.

**What you'll learn**:
- `/read file.py` — inject full file (use sparingly)
- `/read file.py 10-30` — inject only lines 10–30 (preferred)
- `/readln file.py 10-30` — same but with line numbers (use before `/fix` or `/patch`)
- `/fix file.py 25-25 wrong operator` — AI proposes `LINE N: corrected content`, shows diff, asks to apply
- `/patch file.py 20-40 fix the loop logic` — AI proposes SEARCH/REPLACE block
- `/patch file.py code` — apply SEARCH/REPLACE from last AI reply without a new LLM call
- `/edit file.py code` — apply full code block from last AI reply
- `/diff file.py file.py.bkup` — compare two files side by side
- `/bkup save file.py` / `/bkup restore file.py` — snapshot before risky edits

**Narrative**: The difference between `/fix` and `/patch` is the difference between surgery and construction. `/fix` is for 1B models — one line, no freeform. `/patch` is for 7B+ models that can write a precise SEARCH/REPLACE block. Know which model you're using, and pick the right tool.

**The golden rule**: always use `/readln` before `/fix` or `/patch`. Line numbers in context let the model reference exact locations.

**Example**:
```
> /readln calc.py 1-20
> what is wrong with the divide function?
AI: The function divides by b without checking if b is zero.
> /fix calc.py 8-8 divide by zero
[fix] LINE 8: return a / b if b != 0 else None
Apply? [Y/n]: y
[ok] calc.py patched
```

---

## Lecture 5 — Scripts: Automating Repetitive Work

**Core concept**: A script is a sequence of commands stored in a text file. It turns a workflow you figured out once into something you can repeat in one command.

**What you'll learn**:
- Script file format: one command per line, `#` for comments, `[v]` for done steps
- `/script create` — start a new empty script
- `/script create ctx` — capture this session's commands into a ready-to-run script automatically
- `/script show N` — display script N from the list with all its steps
- `/script apply` — run step by step with Y/n confirmation
- `/script run <file> [key=value ...]` — run all steps automatically, no confirmation
- `{{param}}` placeholders — reusable parameterized scripts
- `/script run fix.txt file=calc.py range=1-10`
- Global script library — DevOps, research, and workflow templates shipped with 1bcoder

**Narrative**: You just fixed a tricky authentication bug. It took you 20 minutes of `/read`, `/find`, `/fix`, `/run` back-and-forth. Now imagine the same bug pattern appears in 5 other files. Instead of repeating the work, run `/script create ctx` — 1bcoder captures every command you typed into a reusable script. Next time: `/script run fix-auth.txt file=payments.py`.

**Example script file** (`.1bcoder/scripts/fix-null-check.txt`):
```
# Add null check to a function
/readln {{file}} {{range}}
what is wrong with the null handling here?
/fix {{file}} {{range}} missing null check
/run python -m pytest tests/ -x -q
```

```
> /script run fix-null-check.txt file=users.py range=45-60
```

---

## Lecture 6 — Agent Mode: Autonomous Task Execution

**Core concept**: Agent mode lets the model decide which tools to use. You give it a task — it reads, edits, runs, and iterates. You watch and confirm (or let it run automatically with `-y`).

**What you'll learn**:
- `/agent ask <question>` — read-only research agent: explores with `/tree`, `/find`, `/map`; never edits files; best for 4B+ models
- `/agent advance <task>` — full toolset for 7B+ models: reads, patches, runs tests
- `/agent <task>` — default agent (basic toolset including tree and find)
- `/agent -y <task>` — fully autonomous, no confirmations
- `/agent -t 5 <task>` — limit to N turns
- Named agents — `.1bcoder/agents/<name>.txt` defines system prompt, tools, max_turns
- `agent.txt` — configuring default agent behavior
- When to use agent vs. script (agent = exploration; script = known procedure)
- `/map idiff` after agent runs — checking what actually changed

**`/agent ask` — the 4B workhorse**:
- Uses `/tree` first to understand structure
- Then `/map find` and `/find` to locate relevant code
- Then `/read` to examine it
- Returns a plain-text answer or plan — never edits anything
- Ideal for: "where is the payment logic?", "explain the auth flow", "which files handle ISBN?"

**Narrative**: For 1B models, give a single-file task with clear success criteria: "read utils.py and add type annotations to all functions." For 4B with `/agent ask`: "where is the session timeout configured?" For 7B+ with `/agent advance`: "find the authentication timeout bug and fix it." Don't ask a 1B agent to explore — it will try, get confused, and produce noise. Use `/agent ask` for exploration even on small models when the model is 4B+.

**The confirmation gate**: by default, every action pauses and shows `[Y/n/e/f/q]`. This is intentional — you're the auditor. The model proposes, you approve. The `-y` flag removes the gate for tasks you trust completely.

**Example**:
```
> /agent ask where is ISBN validation implemented?
[agent] turn 1/15
ACTION: /tree
[tool result] ...
ACTION: /find isbn -c
[tool result] bookcrossing/models.py:45: isbn = ...
AI: ISBN validation is in bookcrossing/models.py at line 45,
    called from views.py before saving a new book.
```

---

## Lecture 7 — Project Context: /proj and /ctx compose

**Core concept**: Across multiple sessions on the same project, you accumulate context files — saved conversations that loaded specific files and built up understanding. The challenge: how do you reuse that understanding without re-running every command from scratch?

**Two tools**: `/proj` organizes context files per project ticket or task. `/ctx compose` merges selected context files into one — deduplicating the shared root (tree output, system description) and combining the unique branches (the actual files you loaded).

**What you'll learn — `/proj`**:
- `/proj set ABC-123` — activate a project, creates `.1bcoder/projects/ABC-123/`
- `/proj save session.txt` — save current context to project folder
- `/proj load session.txt` — load it back (fuzzy filename match)
- `/proj show` — list saved contexts in current project
- `/proj find <term> [-f|-c]` — search all projects; results are numbered for compose
- `/proj keyword add isbn, book, legacy` — annotate project for future search
- `/proj file add models.py, views.py` — track which files belong to this project
- `/proj index` — auto-extract file paths from saved ctx files (regex on `/read`/`/edit` commands)
- `/config save` — persist active project across restarts

**What you'll learn — `/ctx compose`**:
- Pre-building "knowledge path" ctx files: `/tree -d 1` → description → `/tree module/` → `/read file.py`
- `/ctx compose add <file>` — add to compose queue
- `/ctx compose add 1,3` — add by number from last `/proj find` results
- `/ctx compose list` — review queue with sizes and running total
- `/ctx compose run task.ctx` — merge with content-level dedup
- `/ctx load task.ctx` — load the merged context — LLM wakes up knowing all branches

**The kung-fu moment**: a change touching `models.py`, `views.py`, and `book.html` normally requires three separate context-loading sessions. With compose, you build three thin knowledge-path ctx files once, then merge them: the shared `/tree` output appears once, the three file reads appear in sequence. The LLM opens its eyes already knowing the full picture.

**Example workflow**:
```
# Build knowledge paths (done once, saved to project)
/tree -d 1
This is a Flask book management app
/tree bookcrossing -d 1
/read bookcrossing/models.py
> /proj save ctx-models.txt

# Later: compose for a multi-file task
/proj find isbn
  1. ABC-123/ctx-models.txt — files: models.py
  2. ABC-123/ctx-views.txt  — files: views.py
  3. ctx/ctx-book-html.txt  — ctx: [3]ctx-book-html.txt

/ctx compose add 1,2,3
/ctx compose run isbn-task.ctx
/ctx load isbn-task.ctx
# LLM knows models.py, views.py, book.html — one tree root, three branches
```

---

## Lecture 8 — /proc: Post-processing and the Map-Reduce Pattern

**Core concept**: Small models "think while they write" — their reasoning is embedded in the text they generate. If you constrain their output length, you constrain their thinking. The solution: let them generate freely, then extract structure programmatically.

**What you'll learn**:
- The map-reduce pattern: free generation → structured extraction
- `/proc run <name>` — one-shot post-processor on last reply
- `/proc on <name>` — persistent processor after every reply
- `/proc off` / `/proc off <name>` — stop one or all
- Built-in processors: `extract-files`, `extract-code`, `extract-list`, `grounding-check`, `collect-files`
- `regexp-extract <pattern> [-i] [-u] [-g N]` — the universal extractor
- Writing your own processor (stdin/stdout protocol)

**Narrative**: You ask a 1B model "what is 11 × 11?" and tell it to respond with exactly one number. It writes two tokens and those two tokens are its entire reasoning. Wrong answer guaranteed. Instead: let it write freely, then run `/proc run regexp-extract \b[0-9]{3}\b` — search the output for all three-digit numbers. The model gets to think. You get structure.

**`grounding-check` in persistent mode**:
```
> /proc on grounding-check
> /read auth/session.py 10-40
> how should I fix the token expiry logic?
AI: You should update the TokenManager class...
[proc:grounding-check] identifiers: TokenManager(✓) expiry(✗) update(✗)
  score: 1/3 (33%) ← WARNING: low grounding
```

The model mentioned identifiers that don't exist in your codebase. Now you know before applying anything.

---

## Lecture 9 — /parallel and /team: Many Models, One Task

**Core concept**: Instead of one model doing everything, send the same question to several models simultaneously and compare. Or split the task into specialized sub-tasks and run them in parallel with different models.

**What you'll learn**:
- `/parallel "prompt" host|model|file ...` — same prompt, multiple models, simultaneous
- Context control: `--ctx` (full), `--last` (last message only), `--no-ctx` (prompt only)
- Output routing: `ctx` output feeds back into main context
- Saved profiles for reuse
- `/team run <name>` — yaml-defined workers, each with own model and script
- Built-in team scripts: tree-worker, search-worker, map-worker, summarize
- `collect:` output type — aggregate all worker results

**Narrative**: Three models look at the same function and you ask "what could go wrong here?" One mentions a race condition. One mentions a null pointer. One misses both and talks about style. The human reads three short answers and synthesizes. This is not three times smarter — it's three independent perspectives. Diversity of failure modes matters more than average quality.

**The specialization pattern**:
```yaml
# .1bcoder/teams/code-analysis.yaml
workers:
  - host: localhost:11434
    model: qwen2.5-coder:1.5b
    plan: team-tree-worker.txt      # where does auth live structurally?
  - host: localhost:11434
    model: qwen2.5-coder:1.5b
    plan: team-search-worker.txt    # which functions implement it?
  - host: localhost:11434
    model: nemotron-3-nano:4b
    plan: team-map-worker.txt       # what depends on it?
```

```
> /team run code-analysis --param keyword=authentication
[team] spawning 3 workers...
[team] all workers finished
```

---

## Bonus — Phone Farm: Old Miners Are Gold. 10 × 1B vs 1 × 10B

**Core concept**: A single 10B model running on a gaming laptop costs $1,500 and runs one query at a time. Ten old Android phones running 1B models via Termux cost $50 each on the used market and run ten queries simultaneously. This lecture is about when the farm wins — and why the answer is not obvious.

**What you'll learn**:
- Installing Ollama on Android via Termux (no root required)
- Pulling `qwen2.5-coder:1.5b` — the phone's sweet spot: 1GB, fits in 3GB RAM, runs at 5–10 tok/s
- Connecting each phone to 1bcoder: `/host http://192.168.1.101:11434` or via `/parallel` host list
- The phone farm topology: coordinator laptop + N worker phones on a local WiFi network
- `/parallel` and `/team` with remote hosts — distributing work across the farm
- When 10 × 1B beats 1 × 10B — and when it doesn't

**Setting up a phone worker**:
```bash
# On Android phone, in Termux:
pkg update && pkg install curl
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5-coder:1.5b

# Expose to LAN (Termux, not a service — just the shell):
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

From the coordinator laptop:
```
> /host http://192.168.1.101:11434
> /model qwen2.5-coder:1.5b
> what is 11 × 11?
```

**The 10 × 1B vs 1 × 10B argument**:

A 10B model is smarter per query. But "smarter" is not always what you need.

| Task type | Winner | Why |
|---|---|---|
| Single complex reasoning | 10B | More parameters = better chain of thought |
| 10 independent file reviews | 10 × 1B | Parallelism — all done simultaneously |
| Exploring 10 hypotheses at once | 10 × 1B | Diversity of failure modes |
| `/agent advance` on large codebase | 10B | Sequential reasoning, long context |
| `/parallel` — same question, N perspectives | 10 × 1B | Each phone is independent, no shared state |
| Code review across 10 modules | 10 × 1B | One phone per module, same latency as one |

**The parallelism argument**: a 10B model is ~10× slower than a 1B model per token (roughly). If you have 10 tasks and run them sequentially on the 10B model, that is 100× the latency of running all 10 tasks simultaneously on 10 phones. For embarrassingly parallel workloads — independent file reviews, multi-hypothesis exploration, team workers — the farm wins on wall-clock time even if each phone is weaker.

**The diversity argument**: the deeper insight is not speed but independence. Ten 1B models sampling from slightly different temperature runs are more likely to surface different failure modes than one 10B model run once. In `/parallel` mode, you're not looking for the best answer from one model — you're looking for the answer that three independent models agree on. Consensus under weak models is often more reliable than confidence from one strong model.

**The cost argument**: the total VRAM of ten phones (3GB × 10 = 30GB) costs ~$500 used. A single GPU with 30GB VRAM costs $3,000+. For inference-only workloads at local scale, the phone farm is the budget research cluster.

**What doesn't work on a phone**:
- Models above ~3B (insufficient RAM, thermal throttling under sustained load)
- `/agent advance` with 20+ turns (too slow per token, context management gets messy)
- Tasks requiring long context (phones typically run 4K–8K ctx reliably)

**The right mental model**: think of each phone as a specialized worker, not a general-purpose assistant. A phone running `qwen2.5-coder:1.5b` is excellent at `/fix` (one line), extracting structure (`/proc run extract-files`), and answering narrow questions about a single function. It is not a replacement for a 7B model on complex reasoning — it is a replacement for human effort on repetitive, parallelizable micro-tasks.

**Example — farm-powered code review**:
```yaml
# .1bcoder/teams/farm-review.yaml
workers:
  - host: http://192.168.1.101:11434   # phone 1
    model: qwen2.5-coder:1.5b
    plan: team-search-worker.txt
  - host: http://192.168.1.102:11434   # phone 2
    model: qwen2.5-coder:1.5b
    plan: team-tree-worker.txt
  - host: http://192.168.1.103:11434   # phone 3
    model: qwen2.5-coder:1.5b
    plan: team-map-worker.txt
  - host: http://192.168.1.104:11434   # phone 4
    model: qwen2.5-coder:1.5b
    plan: team-summarize.txt
```

```
> /team run farm-review --param keyword=authentication
[team] spawning 4 workers across 4 hosts...
[team] all workers finished  ← same latency as one phone doing one task
```

**The punchline**: your old Xiaomi from 2019 that you were about to throw away runs a 1B coding model at 6 tokens per second. Ten of those in a drawer is a local inference cluster that costs nothing, requires no cloud account, and runs offline. The miner is old. The gold is real.

---

## Appendix: Choosing Your Model

| Model | Size | Best for |
|---|---|---|
| `qwen2.5-coder:0.6b` | 400MB | Autocomplete, single-line fixes, fast iteration |
| `qwen2.5-coder:1.5b` | 1GB | Standard 1bcoder work — the default recommendation |
| `nemotron-mini:4b` | 2.5GB | Code quality tasks, SEARCH/REPLACE blocks, `/agent ask` |
| `qwen3:1.7b` | 1.1GB | Reasoning tasks — set `/param think_exclude false` + `/config save` |
| `qwen3:4b` | 2.5GB | Reasoning + code, good for `/agent ask` with thinking |
| `qwen2.5-coder:7b` | 4.5GB | `/agent advance`, complex multi-file tasks |

**Rule of thumb**: use the smallest model that reliably produces the output format you need. For `/fix` — 1.5B is enough. For `/patch` with complex logic — 4B+. For `/agent advance` — 7B minimum.

**Thinking models** (qwen3 family): enable with `/param think_exclude false`. The model reasons inside `<think>` blocks before answering. Useful for planning and debugging tasks. Save the setting per project with `/config save` — different projects may prefer different think settings.

---

*1bcoder is part of the SIMARGL research project on intelligent software development support.*
*(c) 2026 Stanislav Zholobetskyi, Institute for Information Recording, National Academy of Sciences of Ukraine*
