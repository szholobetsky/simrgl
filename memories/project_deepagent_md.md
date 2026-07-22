---
name: project-deepagent-md
description: "deepagent_md flow — recursive markdown document tree generator with parallel workers, web research, and compose modes"
metadata: 
  node_type: memory
  type: project
  originSessionId: 457af3c4-1663-4f2f-a188-730d645cdea6
---

# deepagent_md — Recursive Markdown Tree Generator

**Location:** `C:\Project\1bcoder\_bcoder_data\flows\deepagent_md.py`

**What it does:** Generates a folder of markdown files where each file expands one section of its parent. Results compose into a single navigable document.

## Key Features

- **BFS parallel generation** via `--profile <name>` — distributes nodes at same depth across Ollama workers simultaneously (ThreadPoolExecutor)
- **DFS sequential** (default, no profile) — depth-first, one file at a time
- **--web** — DuckDuckGo search + fetch runs locally, results embedded in prompt sent to remote workers
- **--ctx N** — serialize last N chat messages into prompt for local/index.md generation (default 6)
- **--ctx-worker N** — separate ctx for remote workers (default 0 — only web + parent section)
- **--max_parent_ctx N** — chars of parent section to inject (0 = unlimited)
- **--maxdepth N** — recursion depth (default 3)
- **plan: l1, l2, l3** — focus label per depth level
- **list: a, b, c** — aspects to cover in each section

## File Structure

```
.1bcoder/planMD/plan1/
├── index.md          ← top-level ## 1. sections
├── item_1.md         ← expands section 1
├── item_1.1.md       ← expands subsection 1 of item_1
├── item_1.1.1.md
└── ...
```

## Compose Modes

```
/flow deepagent_md compose plan1                  → flat single md
/flow deepagent_md compose plan1 --mode linked    → hypertext/ folder with md + cross-links
/flow deepagent_md compose plan1 --mode html      → hypertext/ folder with HTML + cross-links
```

Compose uses **depth-first ordering**: after each section of a file, the corresponding child file is inserted recursively. Heading levels shift automatically by depth.

## Workflow Pattern

```
> discuss project details, business needs, user count...
> /flow deepagent_md "Design REST API with PostgreSQL" --ctx 8 --maxdepth 3 --profile phones
> /flow deepagent_md compose plan1 --mode html
```

## Why: Confirmed value
- gemma3:1b generated high-quality thematic content
- Parallel workers = N phones/machines generating simultaneously  
- `--ctx` preserves business context from conversation into all generated files
- Parent section injected into child prompts prevents theme drift
- Plans saved locally in `.1bcoder/planMD/` (project-local, not global home dir)

## Architectural insight: why this works on CPU-only hardware (2026-07-05)
`--max_parent_ctx` and `--ctx` deliberately cap what's injected per node, independent of total tree depth/breadth. This means overall task complexity (architecture, methods, detail level) is unbounded, but the context size of any single LLM call stays small and fixed. This is the recursive generalization of the map-reduce pattern in [[project_ctxtimer|CTXTIMER concept]] (dense orchestrator compresses → MoE reasoner thinks on short input) — applied at every level of the tree instead of once. See [[project_pantheon_hardware]]: a naive agent loop that accumulates full history hits the ctxtimer-measured context ceiling within a few steps on CPU-only hardware; deepagent_md structurally avoids that ceiling by design rather than fighting it, since no single call's context grows with total task size.
