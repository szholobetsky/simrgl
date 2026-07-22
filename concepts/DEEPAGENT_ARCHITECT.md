# DeepAgent_Architect: Research-Grounded Architecture Decomposition

**Component**: 1bcoder `/flow deepagent_architect`
**Date**: 2026-07-18
**Status**: Concept

---

## 1. Problem

Two deepagent_* flows exist and both work, but they don't touch:

- **deepagent_task** (via deepagent_md/deepagent_spec) decomposes a task into a
  narrative plan tree — sections, specs, backlog rows in `tasks.md`. Pure text,
  no code, no verification.
- **deepagent_code** decomposes a function into sub-functions — skeleton at
  levels 0..N-1, 5-15 line implementations at the leaf. Has verification
  (unit tests, syntax check) but **no notion of interface, abstract class,
  or design approach** — it is flat procedural decomposition, one function
  calling other functions. Its own numbering (`1-2-3-name.py`) is completely
  disconnected from `deepagent_task`'s numbering (`item_<id>.md`).

Nothing in the ecosystem answers "given this backlog story, what should the
actual architecture look like — new class? existing interface? new abstract
base?" — and even if it did, doing it as a single blind LLM completion would
reproduce a failure this project has already hit once: **organic-leaf-detection
in `deepagent_spec` was empirically falsified because a small model cannot
reliably judge something it has no external evidence for** (see
`DEEPAGENT_AND_DEEPAGENT_MD.md`, `project_deepagent_spec.md`). Architecture
decisions have exactly the same shape — "is there already an interface for
this?", "does an abstract base already exist?" — are not answerable from the
model's imagination. They require reading the actual codebase.

`deepagent_code`'s own `--ask` flag already gestures at this (`_enrich_ctx`,
`_is_internal`) but only checks "does a file with this exact stem exist
anywhere?" — a name collision check, not architectural research. It never
looks for existing interfaces, base classes, or conventions before a
decompose/think step runs.

---

## 2. Core Idea

A fixed five-level vertical axis, sitting between `deepagent_task` and
`deepagent_code`:

```
0  interface   — contract: capability, method signatures, one-line purpose
1  abstract    — shared base behavior, template methods (may be skipped)
2  approach    — DECISION node: which concrete strategy satisfies 0+1, and why
                 (not necessarily a file — a design record)
3  class       — concrete shell: fields, constructor, method list
4  method /    — LEAF: handed to the existing deepagent_code engine
   query /
   formula
```

`query`/`formula` as leaf types (not just `method`) matter for this project
specifically — SIMARGL's own domain mixes ordinary methods, SQL/pgvector
queries (`exp3/vector_backends.py`), and mathematical formulae (MAP/MRR,
SES/HES) as equally valid "smallest units of work."

**The one change that actually matters**: at every level, before the model
proposes anything, a bounded **read-only research step** runs first — real
tool calls against the real codebase, not a "think step" that is just another
LLM completion in isolation (which is all `--think` currently is; see
`_THINK_SYSTEM` in `deepagent_code.py` — pure prose reasoning, zero tool
calls). This is the one ingredient the Ralph Wiggum loop has that no
`deepagent_*` flow has had until now: **grounding each step in the state of
the real world, not the model's imagination of it** (see
`concepts/DEEPAGENT_CODE.md` section 11 for the closest existing precedent — web/RAG
enrichment before generation — this generalizes that idea to every level, not
just the leaf).

---

## 3. Closing the deepagent_task and deepagent_code gap

Both trees already use dot-notation ids for hierarchy; `deepagent_code` just
doesn't share the id **space**. The fix is not a new ID scheme — it's using
the same one:

- `deepagent_architect` writes into `.1bcoder/arch/<plan>/item_<id>.md`,
  same convention as `planMD`, same `_collect_node_ids`/`_child_ids` machinery
  from `deepagent_md.py` (reused, not reimplemented).
- The **id stays purely numeric** (`3.2.1`, `3.2.1.1`, ...) — no text segments
  like `.iface` embedded in the id itself. `_shift_id`/`_collect_node_ids`
  assume `int(parts[0])`; breaking that would break every existing merge/
  renumber tool built this session (`deepagent_merge.py`). Instead, **which
  level a given depth represents is carried by a `plan:` preset**, exactly
  the mechanism `deepagent_md` already has ("semantic label per depth
  level" — see `PROMPT_AND_CONTEXT_EVALUATION.md`, DeepAgent Parameters
  table). `deepagent_architect` ships one fixed preset:
  `plan: interface,abstract,approach,class,leaf`.
- A `tasks.md` row (`3.2`, from `deepagent_task`) becomes the **root** handed
  to `deepagent_architect 3.2`. Its children (`3.2.1`, `3.2.2`, ...) are
  architecture nodes. At the leaf level, `deepagent_architect` doesn't write
  code itself — it calls into `deepagent_code`'s existing `_expand()` with
  the class/method contract as `root_task`, writing to
  `.1bcoder/code/3.2.<leaf_id>/`. **No new code-generation logic — full reuse
  of the existing, already-verified `deepagent_code` engine** (`--think`,
  `--test`, `--syntax`, `--ask` all keep working unchanged at this level).
- Optional, not required for v1: extend `deepagent_task.py`'s
  `_collect_ids_from_spec`-style fallback (added this session for the
  planMD-optional fix) to also discover `.1bcoder/arch/<plan>/`, so
  architecture nodes show up as rows in `tasks.md` / Alkonost too, alongside
  plan and spec nodes, without a fourth parallel UI.

---

## 4. The Research Step — a dedicated agent, not a bespoke loop

Built and verified this session as two small, generic files — no changes to
`chat.py`'s agent loop needed:

- **`_bcoder_data/agents/research.txt`** — a new named agent, sibling to
  `ask.txt`/`concepts.txt`. `/ask` stays narrow and local-only (`read, tree,
  find, map`); `research` is the broader, multi-source one: local codebase
  *and* web *and* RAG *and* simargl *and* glossary. Calling it `research`
  rather than folding web/RAG/simargl into `/ask` keeps the naming honest —
  "ask" implies a quick local lookup, "research" implies consulting more
  than one kind of source.
- **`_bcoder_data/proc/tool-allowlist-gate.py`** — a generic PASS/FAIL gate
  proc. **Important finding from this session**: an agent file's `tools =`
  section is informational only — nothing in `chat.py`'s `_run_agent_loop`/
  `_agent_exec` actually checks a proposed `ACTION:` against it before
  running it. `/ask` itself has always relied on the model's obedience, not
  on an enforced boundary. For `deepagent_architect`'s research step — whose
  entire value is "guaranteed no code changes yet" — that's not good enough
  on its own, especially since these agents run with `auto_apply=True` (no
  confirmation prompt). `tool-allowlist-gate` closes that gap for real:
  gate procs run *before* the action-execution loop, so a `FAIL:` here
  rejects the whole turn — nothing in it ever reaches `_agent_exec`.

The research step for a `deepagent_architect` node is then just:

```
/agent research <node's design question>
/proc gate on tool-allowlist-gate   (before running, or baked into a preset)
```

No bespoke Python research-loop code, no new prompt-then-filter machinery —
composition of two existing mechanisms (named agents, gate procs), both
already used elsewhere in this project. This also resolves the earlier
"is one round enough?" open question: since this runs through the normal
`/agent` multi-turn loop (`max_turns = 20` in `research.txt`), the model can
naturally do a second search when the first one turns up something worth
reading in full — no fixed one-shot ceiling.

### Allow-list (enforced by `tool-allowlist-gate`, documented in `research.txt`)

| Command | Scope allowed | Why |
|---|---|---|
| `/read`, `/readln` | full | pure file read |
| `/find` | full | grep-style search, no mutation |
| `/map` | `find`, `trace`, `deps` only | structural/symbol queries; `index` writes `map.txt`, `diff`/`idiff`/`keyword` not needed for this step |
| `/tree` | full, but **default to `/tree -d 1`** | large codebases can flood results/context with irrelevant depth — start shallow, go deeper only if the first pass doesn't answer the question |
| `/web` | `search`, `fetch` | both read-only, external-knowledge fallback (mirrors `deepagent_code`'s existing `--ask`/`_web_research`) |
| `/rag` | `search`, `list`, `status` only | `index`/`add`/`remove`/`init`/`ingest` mutate the RAG store — never allowed here |
| `/flow glossary` | `show`, `extract`, `find`, bare term-lookup form | `index`/`relink` mutate glossary files — never allowed here |
| `/flow simargl_files` | full | already a safe, existing wrapper — internally does `/run simargl search` (read-only in effect) + `/read`, exactly this pattern |

Never allowed, regardless of what the model proposes: `/edit`, `/save`,
`/insert`, bare `/run`, `/script`, `/mcp`, and the mutating subcommands
listed above. `tool-allowlist-gate` enforces exactly this list by default
(verified with 11 test cases: every allowed entry passes silently, every
disallowed one — including `/map index`, `/rag index`, `/flow glossary
index`, `/edit` — fails with a clear message and blocks the whole turn).

### README-first rule

Before running any other search, check whether a `README.md` exists at the
relevant directory level(s) (repo root, and the target module's own
directory) and read it if present — cheapest, highest-value signal of
existing convention. This is step 1 in `research.txt`'s own system prompt,
not something left to the model to remember unprompted.

This whole step replaces today's blind, evidence-free `--think` completion
in `deepagent_code.py` (pure prose reasoning, zero tool calls — see
`_THINK_SYSTEM`). The `_enrich_ctx`/`_is_internal` machinery already in
`deepagent_code.py` is the closest working precedent for the *enrichment*
half of this (its existing `_web_research`/`_rag_research` imports from
`deepagent_md` are reused here too, just reached via `/web`/`/rag` commands
instead of direct function calls) — this generalizes it from "does a file
with this exact stem exist" to genuine architectural research.

### Per-level decompose questions

| Level | Research question | Decompose/decide question |
|---|---|---|
| interface | Does an interface/protocol with this responsibility already exist? | Name, method signatures, one-line contract |
| abstract | Do 2+ existing concrete implementations share duplicated boilerplate? | Is an abstract base warranted at all — may be skipped |
| approach | What concrete strategies/libraries already solve this elsewhere in the repo or ecosystem? | Pick one, record 1-2 rejected alternatives + why |
| class | What's the naming/constructor convention in sibling classes? | Fields, constructor signature, method list (→ children) |
| method/query/formula | (delegated to deepagent_code's own `--ask`) | (delegated to deepagent_code's own `--think`/implement) |

**Levels can be skipped per branch, but not on a vibe** — a branch skips
`abstract` only when the research step finds no evidence of shared
boilerplate to justify one. The *decision to skip* is evidence-based, but
the *set of levels that exist at all* (interface→abstract→approach→class→
leaf, in that order) is fixed and external, never an open-ended
model-judged search for "how many layers do I need" — same reasoning as
`deepagent_code`'s explicit `--depth` (no heuristic leaf-detection).

---

## 5. Command Syntax (sketch)

```
/flow deepagent_architect <plan_name> <task_id> [--lang py] [--skip abstract]
```

Reads `spec_<task_id>.<n>.md` (from `deepagent_spec`) as the root task
description if present, else the raw text argument. `--skip` forces a level
out regardless of research evidence, for cases the human architect already
knows. The research step per node runs the `research` agent (`--research-
turns N` optionally overrides its default `max_turns=20`), gated by
`tool-allowlist-gate` — see section 4.

---

## 6. Relation to the Ralph Wiggum Loop

This is the direct application of the Ralph-loop insight identified in this
project's own brainstorm (see chat log 2026-07-18): Ralph's actual power
isn't the `while` loop — it's that **every iteration does real work against
real state** (compile, test, search) instead of pure generation, with
progress living in files/git, not context.

| Ralph ingredient | deepagent_architect equivalent |
|---|---|
| `fix_plan.md`, external state | `.1bcoder/arch/<plan>/item_<id>.md` + research notes, not context |
| Fresh context every iteration | Each node's `research` agent run and decompose call are independent |
| "Only one thing" per loop | One node (one interface, one class) decided per pass |
| Failures piped back as context | Research findings piped into the decompose prompt, same iteration |
| No LLM-judged stopping | Levels are fixed/external; only the skip-or-not decision is evidence-based |
| Real, enforced boundaries, not just instructions | `tool-allowlist-gate` actually blocks disallowed actions — unlike an agent file's own `tools =`, which nothing enforces |
| Cost is continuous, not free | A full `research` agent run (up to 20 turns) per node costs more than today's one-shot `--think` — an accepted, explicit tradeoff |

Huntley's own explicit caveat — **"no way I'd use Ralph on an existing
codebase"** — is the sharpest open risk for this project specifically, since
1bcoder's real usage is almost always an existing codebase, not greenfield.
Research-grounding narrows this risk (it actively looks for what's already
there instead of ignoring it) but does not eliminate it — see section 8.

---

## 7. Relation to External Supervision

Same thesis as `DEEPAGENT_CODE.md` section 7, one level up: model generates a
proposal, the automaton (research step, fixed level schema, id/file
machinery) tracks the field, verification at the leaf is deterministic
(`deepagent_code`'s existing tests/syntax check), the human holds the plan
(chooses which task gets `deepagent_architect`-ed at all, reviews `approach`
decision records — the one place where rejected alternatives and rationale
are visible before code is written).

---

## 8. What This Is Not — Honest Limitations

- **Does not solve the Calendar Problem for genuinely novel contracts.**
  Research grounding prevents *reinventing what already exists* and keeps
  naming/conventions consistent — it does not give a small model the
  ability to reason about the compatibility of two contracts it is
  inventing for the first time in the same run. That remains architect
  (human) work, exactly as `DEEPAGENT_CODE.md` section 3.4 already concludes for
  levels 0-1.
- **Not for redesigning a large existing system's architecture in one
  pass.** Best fit: a single new backlog story that plugs into an existing,
  already-legible codebase — narrow radius, not "architect the whole
  monolith."
- **The `approach` node is a decision record, not a guarantee.** It documents
  what was considered and why one option won — it does not verify the
  choice was correct. Verification only exists at the leaf (tests/syntax).
- **More expensive than today's flows.** Each node costs a bounded research
  sub-run plus a decompose call — budget accordingly; this is not a drop-in
  replacement for quick, low-stakes `deepagent_code` runs.

---

## 9. Optional Integrations (not required for v1)

- **`radogast-gate.py`** (built this session) could gate the research step
  itself — if the allow-listed tool calls and their results drift off the
  node's actual design question (e.g. wander into unrelated modules), the
  same `drift_status == "critical"` signal already used for regular
  `/agent` runs applies unchanged, since the research step's messages are
  just another message list `analyze()` can read.
- **Alkonost visibility** via extending `deepagent_task`'s
  `_collect_ids_from_spec`-style fallback to also list `.1bcoder/arch/`
  nodes — no new UI, reuses the existing task-board.

---

## 10. What Exists vs What's Needed

| Component | Exists | Needed for v1 |
|---|---|---|
| BFS tree / id machinery | yes (`deepagent_md`) | reuse as-is |
| Leaf code generation, tests, syntax check | yes (`deepagent_code`) | reuse as-is, called with a class/method contract as root_task |
| `plan:` semantic-label-per-depth mechanism | yes (`deepagent_md`) | reuse — ship the `interface,abstract,approach,class,leaf` preset |
| Web/RAG enrichment before generation | yes (`deepagent_code` `_enrich_ctx`/`_is_internal`) | extend from name-collision check to real research (grep/read/map find) |
| Multi-source research agent | **done this session** — `_bcoder_data/agents/research.txt` | reuse as-is |
| Enforced read-only allow-list | **done this session** — `_bcoder_data/proc/tool-allowlist-gate.py`, 11 test cases passing | reuse as-is, pair with `research` agent |
| Per-level decompose/decide prompts (interface/abstract/approach/class) | no | new — 4 prompt templates |
| Shared id bridge to `deepagent_task` rows | no (deepagent_code has its own numbering) | new — small, since both already use dot-notation |

The one real remaining gap is the four per-level decompose/decide prompts —
the research mechanism itself (agent + enforced allow-list) is built and
verified, and everything else is composition of flows and primitives that
already exist and already work.

---

## 11. Open Questions

1. ~~Read-only tool enforcement~~ — **resolved**: `tool-allowlist-gate.py`
   enforces it for real, unlike an agent file's own `tools =` (informational
   only — verified nothing in `chat.py` checks it). Built and tested this
   session (section 4).
2. ~~Is one research round enough?~~ — **resolved by construction**: the
   `research` agent runs through the normal multi-turn `/agent` loop
   (`max_turns=20`), so it can naturally take a second look when the first
   search turns up something worth reading in full. No fixed round count.
3. **Skip-decision reliability** — same class of question as
   `deepagent_spec`'s falsified organic-leaf-detection: can research
   evidence reliably decide "skip abstract" for a small model, or does this
   need the same external/explicit fallback (`--skip`) as the primary path,
   with evidence-based skipping only as an optional override?
4. **Cross-plan interface reuse** — if `deepagent_architect` runs twice for
   two different backlog stories and both propose "we need a
   `PaymentGateway` interface," does anything detect and merge that, or is
   this the same Shared Function Problem `deepagent_code` already punts on
   (section 3.3 of `DEEPAGENT_CODE.md`), just one level up?
