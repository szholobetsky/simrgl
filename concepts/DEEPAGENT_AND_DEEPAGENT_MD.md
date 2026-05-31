# DeepAgent and DeepAgent_MD: Recursive Thought Expansion as a First-Class Operation

## Abstract

This document describes two related tools — `/flow deepagent` and `/flow deepagent_md` — implemented within the 1bcoder CLI. Both tools address the same fundamental limitation of language model interaction: a single prompt produces a single layer of thought. Neither tool is a chat assistant. Both are **cognitive amplifiers** — systems that take a seed idea and recursively expand it into a structured artifact whose depth exceeds what any single prompt could produce. The document describes the architecture, the key design decisions, the philosophical motivation, and the directions for future development. Special attention is given to `deepagent_md`'s parallel execution model, which treats networks of small local models as a distributed cognitive infrastructure — a concrete instantiation of the Anthill OS principles described elsewhere in this concepts directory.

---

## 1. The Problem: Flat Thinking in a Hierarchical World

### 1.1 The Single-Prompt Ceiling

When a user asks a language model to "design a REST API with PostgreSQL and an Android client," the model produces a response. The response is competent. It is also shallow. The model allocates its generative budget across all topics simultaneously: authentication, schema design, endpoint structure, error handling, Android integration — each receives perhaps two paragraphs. Nothing receives genuine depth.

This is not a failure of the model. It is a failure of the **interaction paradigm**. The single-prompt interface enforces breadth at the expense of depth. The model cannot simultaneously hold the full problem space and generate deep analysis of any one part of it. Context is finite. Attention is distributed.

The same limitation appears in human thinking under time pressure. A consultant asked to "just give me the overview" produces something flat. The same consultant given three hours per topic and asked to write a detailed memo per subtopic produces something deep. The difference is not intelligence — it is **structure of engagement**.

### 1.2 The Hierarchical Nature of Knowledge

All serious knowledge domains are hierarchical. A software architecture has layers: business requirements → system design → component design → implementation details. A research question has levels: hypothesis → sub-questions → evidence → interpretation → implications. A technical specification has depth: goals → constraints → design decisions → rationale → edge cases.

A flat prompt cannot navigate this hierarchy. It can describe it, but description is not the same as traversal. To genuinely explore a hierarchical domain, the cognitive process must itself be hierarchical — each node must be expanded in its own right, with its own context, its own focus, and its own output.

This is the core insight behind both deepagent tools.

---

## 2. DeepAgent: Hierarchical Decomposition of a Task

### 2.1 Architecture

`/flow deepagent` implements a tree-structured decomposition of a task into a flat text file using dot-notation (`1.2.3.4`). The root node is the task. Each leaf node is expanded by calling the LLM with a focused prompt asking for 2–5 children that cover distinct aspects of the leaf's text.

The expansion is level-by-level (BFS over levels). Each level is labeled (via `plan:`) to indicate the focus perspective at that depth: `overview → detail → implementation`. Each node can additionally be filtered through a set of lenses (via `list:`): `pro argument`, `counter-argument`, `real example`, `edge case`.

The result is a single file with a complete hierarchical decomposition:

```
1 Design REST API
  1.1 Authentication layer
    1.1.1 OAuth2 flow [pro argument]
    1.1.2 JWT refresh strategy [counter-argument]
    1.1.3 Token storage on Android [real example]
  1.2 Database schema
    ...
```

This file can be directly used as a plan for `/agent file: deepplan.md`.

### 2.2 Key Design Decisions

**Concreteness detection**: nodes that already contain code, formulas, or completion markers are skipped — the tree stops expanding when it reaches genuinely concrete content, not just arbitrarily.

**Cycle detection**: if a node's text appears more than twice in the tree, expansion is stopped — the model is cycling back to already-covered territory.

**Parroting detection**: if the model echoes phrases from the expansion prompt itself, it is flagged — this indicates the context is too thin and the model has nothing to say.

**Resume**: if the output file already exists, the tree is loaded and expansion continues from where it stopped. Long runs survive interruption.

---

## 3. DeepAgent_MD: Documents Instead of Labels

### 3.1 The Core Difference

`deepagent` produces a tree of **labels** — short phrases connected by hierarchical numbering. `deepagent_md` produces a tree of **documents** — each node is a full markdown file with multiple sections, paragraphs, and real content.

The output of a single `deepagent_md` run at depth 3 with 4 sections per file is:
- 1 `index.md` (4 sections)
- 4 `item_N.md` files (4 sections each = 16 files)
- 16 `item_N.M.md` files at depth 3

Total: 21 markdown files, each containing 400–1500 words of specific content. Combined, this is a document of 10,000–30,000 words covering a topic at four levels of specificity.

No single prompt produces this. No single LLM call produces this. This is a **cognitive manufacturing process**.

### 3.2 Context Propagation

The central problem in recursive document generation is **theme drift**: the model generating `item_1.2.3.md` has no memory of the original task or even of the content of `item_1.2.md`. Each generation is a fresh call.

`deepagent_md` addresses this with three complementary mechanisms:

**1. `root_task` in every prompt**: every generation call includes the original task description as an anchor. The model is reminded at every level what it is ultimately contributing to.

**2. Parent section injection (`--max_parent_ctx`)**: before generating `item_1.2.3.md`, the system extracts the content of section 3 from `item_1.2.md` and injects it into the prompt. The model expands exactly that section — not the topic in general, but that specific section's content.

**3. Conversation context (`--ctx N`)**: the user's prior conversation with the model (business requirements, constraints, user count, domain details) is serialized and injected into the generation of `index.md`. This ensures that details discussed interactively are captured in the top-level structure, from which they propagate downward through parent section injection.

The combination creates a cascade: conversation → index.md → item files → sub-item files. Each level inherits context from the level above. The original intent does not dissipate.

### 3.3 Compose: Reassembly of the Tree

After generation, the tree of files can be reassembled into a single document via `/flow deepagent_md compose`. Three modes exist:

**flat**: concatenate all files into a single large markdown document with adjusted heading levels.

**linked**: copy files to a `hypertext/` folder, inserting cross-links after each section pointing to the corresponding child file. Works in any markdown renderer that supports inter-file links (GitHub, Obsidian, VS Code).

**html**: convert to HTML with embedded navigation links. Requires `pip install markdown` for full rendering; falls back to basic regex conversion.

The compose operation uses depth-first ordering: after writing section N of a file, the corresponding child file is inserted immediately, then section N+1 follows. The result is a document that reads as a single coherent piece while respecting the recursive structure of generation.

---

## 4. Parallel Execution: The Anthill Model

### 4.1 From Sequential to Distributed

The default execution of `deepagent_md` is sequential — one file at a time, on the local machine's model. For a depth-3 tree with 4 sections per file, this requires 21 sequential LLM calls. At 30–60 seconds per call on a small local model, this is 10–20 minutes.

The `--profile <name>` flag activates parallel BFS execution. The system loads a named profile from `profiles.txt` — a list of Ollama workers, each specified as `host:port|model|output_file`. All nodes at the same depth level are distributed across available workers using round-robin assignment and executed simultaneously via `ThreadPoolExecutor`.

```
Level 1: [item_1] [item_2] [item_3] [item_4]
         worker1  worker2  worker3  worker1   ← parallel
         
Level 2: [item_1.1] [item_1.2] [item_1.3] [item_2.1] [item_2.2] ...
         worker1    worker2    worker3    worker1    worker2    ← parallel
```

For a 3-worker setup, the 21-file tree requires approximately 7 sequential rounds instead of 21 sequential calls — a 3x speedup that scales linearly with the number of available workers.

### 4.2 Remote Worker Context

Remote workers receive HTTP POST requests directly (Ollama `/api/chat`, `stream: false`). They do not have access to the local chat history. Their prompt contains:

- The root task (`root_task`)
- The focus perspective for their depth level
- The aspects/lenses if specified
- The parent section content (extracted locally, transmitted in the prompt body)
- Web research results, if `--web` is active (fetched locally, transmitted in the prompt body)

The `--ctx-worker N` parameter controls whether any portion of the local conversation history is also transmitted to workers (default: 0, disabled). This allows fine-grained control: the local machine uses full conversation context for `index.md`; workers operate with minimal context but full structural inheritance from their parent section.

### 4.3 Web Research in Distributed Mode

When `--web` is active, all web searches and page fetches execute on the local orchestrating machine. Results are embedded in the prompt text before transmission to the worker. Workers require no internet access — only a running Ollama instance and an open port.

The search query combines the section title with the root task: `"{title} {root_task}"`. This ensures that a section titled "Finite Difference Methods" is searched in the context of "heat equation as Cauchy problem in Java" rather than as an isolated topic.

---

## 5. Use Patterns

### 5.1 Pre-seeded Research

The most powerful pattern is to conduct a focused conversation before launching `deepagent_md`:

```
> My REST API needs to handle 100k concurrent users.
> The Android client uses Retrofit. PostgreSQL 16 with pgvector for semantic search.
> JWT auth with refresh tokens. Rate limiting at the gateway level.
> /flow deepagent_md "Design REST API with PostgreSQL + Android" --ctx 8 --maxdepth 3
```

The `--ctx 8` flag serializes the last 8 conversation turns into the `index.md` generation prompt. The model generates section headers that reflect the specific constraints discussed — not generic REST API advice.

### 5.2 Parallel Multi-Phone Cluster

```
/flow deepagent_md "Cauchy problem for heat equation in Java" \
  --maxdepth 3 \
  --profile phones \
  --web \
  --plan: mathematical foundation, algorithmic design, Java implementation \
  --list: theory, complexity, code example, edge cases
```

With a profile containing 3–4 Android phones running Ollama with `qwen3:1.7b`, a 3-level tree of 21+ files generates in under 5 minutes. Each phone handles a subset of nodes in parallel. The orchestrating machine fetches web content and assembles the final document.

### 5.3 Academic Research Scaffold

```
> I am researching the manifestations of the comic in Mayakovsky's early poetry.
> Key aspects: futurist aesthetics, self-irony, grotesque imagery, political satire.
> /flow deepagent_md "Comic in Mayakovsky's early poetry" \
    --ctx 6 --maxdepth 4 \
    --plan: literary context, poetic devices, thematic analysis, critical reception \
    --list: textual evidence, historical context, comparative analysis, theoretical framing
```

The result is a 50+ file markdown tree that can be composed into a document suitable as a research framework, a literature review scaffold, or a detailed outline for academic writing.

---

## 6. Relationship to Anthill OS

`deepagent_md` is a proof-of-concept instantiation of the Anthill OS principles described in `ANTHILL_DISTRIBUTED_COGNITIVE_OS.md`. The key correspondences:

| Anthill Principle | deepagent_md Instantiation |
|---|---|
| External tape (knowledge graph) | Plan directory with md files as persistent nodes |
| Distributed small models | `--profile` workers, each with a small local model |
| Navigational capacity over brute force | Parent section injection — model sees only its specific subtopic |
| Church-Turing sufficiency | gemma3:1b generates high-quality thematic content per section |
| Qualia Horizon | Creative synthesis and inter-section coherence require the local orchestrator |

The system demonstrates that the limiting factor is not model size but **context quality**. A 1.7B model given a focused 500-token parent section and a 3000-token web research block generates content indistinguishable from what a larger model would produce for the same narrow task.

The Qualia Horizon appears at compose time: the coherence of the final assembled document — the sense that section 1.2 and section 3.1 are part of the same intellectual project — requires either a large model doing the final synthesis pass, or a human editor. This is the irreducible remainder.

---

## 7. Implementation Location

- **deepagent**: `C:\Project\1bcoder\_bcoder_data\flows\deepagent.py`
- **deepagent_md**: `C:\Project\1bcoder\_bcoder_data\flows\deepagent_md.py`
- **Plan output**: `.1bcoder/planMD/planN/` (project-local)
- **Profile format**: `.1bcoder/profiles.txt` — `name: host|model|file host|model|file`

---

## 8. Open Questions and Future Directions

**Cross-file coherence pass**: after generation, run a single LLM pass over the composed document to smooth transitions and flag contradictions between sections.

**Adaptive depth**: stop expanding a branch when the parent section content is already "concrete" (contains code, formulas, specific measurements) rather than always going to `max_depth`.

**Bidirectional context**: currently context flows only downward (parent → child). Allowing child summaries to propagate upward (updating the parent's understanding of a subtopic) would enable iterative refinement.

**Section quality scoring**: after generation, score each section for relevance drift and flag sections that have deviated from the root task — allowing targeted regeneration rather than full reruns.

**Integration with codeXplorer**: use `deepagent_md` to generate technical specification trees for software projects indexed by simargl, then use the generated documents as retrieval targets for task-to-document mapping.
