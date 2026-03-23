# 1bcoder — Video Course Plan

**Format**: 4–7 lectures, each 20–40 minutes
**Target audience**: Developers who want to use small local LLMs (1B–7B) for real coding tasks — offline, no cloud, no subscription
**Prerequisite**: Ollama installed, at least one model pulled (`qwen2.5-coder:1.5b` recommended to start)

---

## Why this course exists

Most AI coding tools assume you have a fast internet connection, a paid subscription, and a large model. 1bcoder assumes the opposite: you have a laptop, maybe 4–8 GB of RAM, and a model that hallucinates if you look at it funny.

The insight behind 1bcoder is simple: **small models fail not because they're stupid, but because they're overwhelmed**. Give a 1B model a 600-line file and ask it to "fix the bug" — it panics and invents something. Give it 15 lines and ask it to output exactly `LINE N: corrected content` — it frequently succeeds.

This course is about learning to work *with* that constraint rather than against it.

---

## Lecture 1 — The Model, the Context, and How 1bcoder Thinks

**Core concept**: The context window is not just a technical limit — it's the working memory of the model. Everything the model "knows" during a conversation fits inside it. When it fills up, the model starts forgetting. When it overflows, quality collapses.

**What you'll learn**:
- Starting 1bcoder, selecting a model
- What the status line tells you: `qwen2.5-coder:1.5b [1.1G Q4_K 32K] │ ctx 312 / 8192 (4%)`
- `/ctx` — checking fill percentage at any moment
- `/ctx compact` — asking the AI to summarize and compress the conversation when context is getting full
- `/ctx cut` — mechanical trim of oldest messages
- `/ctx clear` — fresh start for a new task
- `/model` and `/host` — switching models and providers at runtime
- `/param temperature 0.1` — why lower temperature helps with code tasks

**Narrative**: Before touching any files, you need to understand what you're working with. A 1B model with 8K context and 30% fill is a different tool than the same model at 90% fill. The status line is your dashboard — check it like a fuel gauge.

**Key insight**: 1bcoder shows context fill after every message because it matters. When you hit 70–80%, stop and compact. A fresh, compressed context outperforms a bloated one every time.

**Example session**:
```
> /ctx
  messages: 12   tokens: 5842 / 8192 (71%)
> /ctx compact
[AI summarizes the session into ~400 tokens]
  tokens: 412 / 8192 (5%)
```

---

## Lecture 2 — Navigating a Codebase

**Core concept**: Before you can fix something, you need to find it. 1bcoder has three levels of navigation — directory structure, file content search, and structural map. Each serves a different question.

**What you'll learn**:
- `/init` — creating the `.1bcoder/` workspace in your project
- `/tree` — visualizing directory structure; injecting it into context with `ctx`
- `/find` — searching filenames and content with regex; flags `-f` (files only), `-c` (content only), `-i` (case-insensitive)
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

Add to context? [Y/n]: y
```

---

## Lecture 3 — Reading and Editing Code

**Core concept**: The model should never see more than it needs. `/read` is not just file loading — it's context sculpting. The smaller and more precise the context, the better the output.

**What you'll learn**:
- `/read file.py` — inject full file (use sparingly)
- `/read file.py 10-30` — inject only lines 10–30 (preferred)
- `/read file.py file2.py` — inject multiple files in one command
- `/readln file.py 10-30` — same but with line numbers (use before `/fix` or `/patch`)
- `/fix file.py 25-25 wrong operator` — AI proposes `LINE N: corrected content`, shows diff, asks to apply
- `/patch file.py 20-40 fix the loop logic` — AI proposes SEARCH/REPLACE block
- `/patch file.py code` — apply SEARCH/REPLACE from last AI reply without a new LLM call
- `/edit file.py code` — apply full code block from last AI reply
- `/diff file.py file.py.bkup` — compare two files side by side
- `/bkup save file.py` / `/bkup restore file.py` — snapshot before risky edits

**Narrative**: The difference between `/fix` and `/patch` is the difference between surgery and construction. `/fix` is for 1B models — one line, no freeform. `/patch` is for 7B+ models that can write a precise SEARCH/REPLACE block. Know which model you're using, and pick the right tool.

**The golden rule**: always use `/readln` before `/fix` or `/patch`. Line numbers in context let the model reference exact locations. Then when you apply the patch, the file is read without line numbers — so the SEARCH block matches the real content.

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

## Lecture 4 — Plans: Automating Repetitive Work

**Core concept**: A plan is a sequence of commands stored in a text file. It turns a workflow you figured out once into something you can repeat in one command — or run headlessly overnight.

**What you'll learn**:
- Plan file format: one command per line, `#` for comments, `[v]` for done steps
- `/plan create` — start a new empty plan
- `/plan create ctx` — capture this session's commands into a ready-to-run plan automatically
- `/plan show` / `/plan apply` — view and execute step by step
- `/plan apply -y` — run all pending steps automatically
- `{{param}}` placeholders — reusable parameterized plans
- `/plan apply fix.txt file=calc.py range=1-10`
- Global plan library — DevOps, research, and workflow templates shipped with 1bcoder
- `--planapply` — headless execution from the CLI

**Narrative**: You just fixed a tricky authentication bug. It took you 20 minutes of `/read`, `/find`, `/fix`, `/run` back-and-forth. Now imagine the same bug pattern appears in 5 other files. Instead of repeating the work, run `/plan create ctx` — 1bcoder captures every command you typed into a reusable plan. Next time: `1bcoder --planapply fix-auth.txt --param file=payments.py`.

**Example plan file** (`.1bcoder/plans/fix-null-check.txt`):
```
# Add null check to a function
/readln {{file}} {{range}}
what is wrong with the null handling here?
/fix {{file}} {{range}} missing null check
/run python -m pytest tests/ -x -q
```

```bash
1bcoder --planapply fix-null-check.txt --param file=users.py --param range=45-60
```

---

## Lecture 5 — Agent Mode: Autonomous Task Execution

**Core concept**: Agent mode lets the model decide which tools to use. You give it a task — it reads, edits, runs, and iterates. You watch and confirm (or let it run automatically with `-y`).

**What you'll learn**:
- `/agent <task>` — basic agent loop (small model toolset)
- `/agent advance <task>` — full toolset for 7B+ models
- `/agent -y <task>` — fully autonomous, no confirmations
- `/agent -t 5 <task>` — limit to N turns
- `/agent continue` — resume a stopped agent session
- `agent.txt` — configuring max_turns, tools whitelist
- When to use agent vs. plan (agent = exploration; plan = known procedure)
- `/map idiff` after agent runs — checking what actually changed

**Narrative**: Agent mode is not magic. It works well when the task is bounded and the model is large enough. For 1B models, give it a single-file task with clear success criteria: "read utils.py and add type annotations to all functions." For 7B+ with `/agent advance`: "find the authentication timeout bug and fix it." Don't ask a 1B agent to refactor across files — it will try, get confused, and produce noise.

**The confirmation gate**: by default, every action pauses and shows `[Y/n/e/f/q]`. This is intentional — you're the auditor. The model proposes, you approve. The `-y` flag removes the gate for tasks you trust completely.

**Example**:
```
> /agent -t 3 read main.py and explain the startup sequence
[agent] turn 1/3
ACTION: /read main.py 1-50
[tool result] ...

[agent] turn 2/3
ACTION: /read main.py 51-100
[tool result] ...

[agent] turn 3/3
AI: The startup sequence initializes the database connection pool first,
then registers middleware, then starts the HTTP listener...
[agent] task complete
```

---

## Lecture 6 — /proc: Post-processing and the Map-Reduce Pattern

**Core concept**: Small models "think while they write" — their reasoning is embedded in the text they generate. If you constrain their output length, you constrain their thinking. The solution: let them generate freely, then extract structure programmatically.

**What you'll learn**:
- The map-reduce pattern: free generation → structured extraction
- `/proc run <name>` — one-shot post-processor on last reply
- `/proc on <name>` — persistent processor after every reply
- `/proc off` / `/proc off <name>` — stop one or all
- Built-in processors: `extract-files`, `extract-code`, `extract-list`, `grounding-check`, `collect-files`
- `regexp-extract <pattern> [-i] [-u] [-g N]` — the universal extractor
- Writing your own processor (stdin/stdout protocol)

**Narrative**: You ask a 1B model "what is 11 × 11?" and tell it to respond with exactly one number. It writes two tokens and those two tokens are its entire reasoning. Wrong answer guaranteed. Instead: let it write freely, then run `/proc run regexp-extract \b[0-9]{3}\b` — search the output for all three-digit numbers, check which ones are divisible by 11. The model gets to think. You get structure.

This is why `/proc` is philosophically the most important feature for small model workflows.

**`grounding-check` in persistent mode**:
```
> /proc on grounding-check
[proc] persistent: grounding-check (runs after every reply)
> /read auth/session.py 10-40
> how should I fix the token expiry logic?
AI: You should update the TokenManager class...
[proc:grounding-check] identifiers: TokenManager(✓) expiry(✗) update(✗)
  score: 1/3 (33%) ← WARNING: low grounding
```

The model mentioned identifiers that don't exist in your codebase. Now you know before applying anything.

**regexp-extract examples**:
```
/proc run regexp-extract \b[A-Z]\w+Service\b -u    # find all Service classes
/proc run regexp-extract "def (\w+)\(" -g 1 -u     # extract function names
/proc run regexp-extract [\w./\\-]+\.py -u          # collect mentioned .py files
```

---

## Lecture 7 — /parallel and /team: Many Models, One Task

**Core concept**: Instead of one model doing everything, send the same question to several models simultaneously and compare. Or split the task into specialized sub-tasks and run them in parallel with different models.

**What you'll learn**:
- `/parallel "prompt" host|model|file ...` — same prompt, multiple models, simultaneous
- Saved profiles for reuse
- `/team run <name>` — yaml-defined workers, each with own model and plan
- Built-in team plans: tree-worker, search-worker, map-worker, summarize
- `--param` forwarding to all workers
- Reading and aggregating results from `.1bcoder/results/`

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
> /team run code-analysis --param keyword=authentication --param task="login fails silently"
[team] spawning 3 workers...
[team] all workers finished
> /plan apply team-summarize.txt --param keyword=authentication
```

---

## Appendix: Choosing Your Model

| Model | Size | Best for |
|---|---|---|
| `qwen2.5-coder:0.6b` | 400MB | Autocomplete, single-line fixes, fast iteration |
| `qwen2.5-coder:1.5b` | 1GB | Standard 1bcoder work — the default recommendation |
| `nemotron-3-nano:4b` | 2.5GB | Code quality tasks, SEARCH/REPLACE blocks |
| `lfm2.5:1.2b` | 800MB | Fast general responses, planning steps |
| `qwen2.5-coder:7b` | 4.5GB | `/agent advance`, complex multi-file tasks |
| `qwen3:1.7b` | 1.1GB | Reasoning tasks — use `/param enable_thinking true` |

**Rule of thumb**: use the smallest model that reliably produces the output format you need. For `/fix` — 1.5B is enough. For `/patch` with complex logic — 4B+. For `/agent advance` — 7B minimum.

---

*1bcoder is part of the SIMARGL research project on intelligent software development support.*
*(c) 2026 Stanislav Zholobetskyi, Institute for Information Recording, National Academy of Sciences of Ukraine*
