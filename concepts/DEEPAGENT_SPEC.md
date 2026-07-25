# DeepAgent_Spec and DeepAgent_Task: From Deepened Idea to Executable Plan

## Abstract

`/flow deepagent_md` (see [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md)) solves the single-prompt-ceiling problem: it turns a seed task into a tree of deepened prose. It does not solve a different, related problem — turning that same task into something a person or a small local model can *execute*, checkbox by checkbox. A deepened idea and a plan are different genres of text, and conflating them produces documents that read as insight but cannot be acted on.

This document describes two new tools that address the second problem:

- **`deepagent_spec`** — a transform pass over an existing `deepagent_md` tree: finds the tree's real leaves, splits each one's own sections into atomic units, and writes each unit to a standardized template (WHO/WHAT/WHEN, SMART, INVEST, acceptance criteria, boundaries). Decomposed along architectural boundaries, not a fixed planning taxonomy — and, as of §2.5, explicitly *not* equivalent to Agile epic/story levels.
- **`deepagent_task`** — a deterministic compiler, not a second LLM pass, that walks the `deepagent_spec` tree and rolls it up into a flat, nested-checkbox `tasks.md` for day-to-day operational tracking.

Both reuse `deepagent_md`'s tree engine unchanged, generation-time prompt included — depth is controlled the same way `deepagent_code` already controls it, by an external `--maxdepth`, never by the model deciding it has reached something concrete (§2.3 was revised after three real test runs disproved that original assumption). What's new is a second, separate pass that transforms whatever the tree's actual leaves turn out to be into specification units (§2.4), and the template those units are written to (§3) — not the generation machinery itself.

---

## 1. Motivation: A Concrete Failure Case

The dissertation author's own training project, **AnimalAlert** (`C:\Project\codeXplorer\capestone\repository\AnimalAlert\`), produced six LLM-generated planning documents from the same raw task description (`documentation/task.txt`) at varying model strengths: `plan.md`, `plan1.md`, `plan2.md`, `implementation_plan.md`, `impl1.txt`, `impl2.md`. All six are architecturally competent — they cover layers, schema, endpoints, scoring logic, scaling strategy. All six fail as *plans*, for the same reason: their "Roadmap" section compresses multiple independent, differently-scoped actions into a single bullet.

```
Фаза 1 (Backend Core): Налаштування БД, створення сутностей,
реалізація /register та /alert.
```

One bullet, at least three unrelated units of work, each of which itself needs further decomposition (which DB, which entities, `/register` = model + controller + validation...). Reading this text produces the sensation the author named directly: *"я не бачу в ньому Плана... План це коли кожна задача відповідає рівню взяв і зробив"* — a plan is when every task is at the level of "take it and do it."

The contrast case is `documentation/new/myplan.md`, the Tasks.md tool's own demo file: nested `- [ ]` checkboxes, tags (`#work @finance`, `due:`, `_spent:`), binary done/not-done state per line. That's the right *genre*. But Tasks.md itself is a flat kanban tool (lane = directory, card = file) — it was never designed to hold a deep, collapsible task tree, and its demo file only goes one level deep, which is a limitation of the tool's UI model, not evidence that checkbox markdown itself can't nest arbitrarily (it can — that's free in the file format).

The design goal, therefore, is not "generate more planning prose." It's: produce a tree whose **leaves are independently executable**, and a flat view of those leaves that can be checked off like Tasks.md, without inheriting Tasks.md's depth ceiling.

---

## 2. Decomposition Axis: Architecture, Not Agile Ceremony

### 2.1 Why epic/story/task/subtask was rejected

The obvious starting point — Jira's epic → story → task → subtask — was considered and set aside. It's a fixed, mandatory four-level taxonomy oriented around *estimation and reporting*, not around the actual shape of the system being built. It forces every project into the same depth regardless of whether that depth reflects anything real about the target architecture.

### 2.2 What was chosen instead

The decomposition axis should mirror the **actual structure** of the system:

- **Inter-module**: frontend / backend / database — real deployment/runtime boundaries.
- **Intra-module**: e.g. MVC (Model / View / Controller) inside the backend — real logical layering within one boundary.
- **Object level**: a concrete class, method, route, or table — the atomic, directly-implementable unit.

This is not a new invention — it's the **C4 model** (Context → Container → Component → Code), a standard software-architecture documentation framework. Container ≈ inter-module, Component ≈ intra-module, Code ≈ object level. Notably, `mdplanner` (one of two real markdown-native planning tools surveyed for this project — see §7) ships a built-in "C4 architecture" view, which is independent confirmation that pairing this decomposition lens with task/spec tooling is an established pattern, not a novelty invented for this project.

C4's labels are used here as **optional vocabulary**, not a mandatory schema. A given branch of the tree might need only two levels (a `database` container that goes straight to `users` table, no intermediate layer); another might need four (a `backend` container → `controller` component → `AlertController` class → `createAlert` method). Depth is decided per branch by content, not imposed uniformly.

### 2.3 Depth is external, exactly like deepagent_code — not an emergent decision

This section originally claimed `deepagent_md` would stop subdividing organically once a node got concrete enough ("organic leaf detection"). That was tested for real against AnimalAlert's own `task.txt`, not just reasoned about, and the claim turned out to be wrong.

- `plan1`/`plan10` (pre-existing, off-topic test runs, effectively `--maxdepth 1`): stopped at depth 1 even though `item_1.md`'s own content already contained further `## N.` sections that `_parse_sections` would have recognized as children.
- `plan2` (`--maxdepth 2`, no `--plan:` labels): stopped at depth 2 for the identical reason — `item_1.1.md` itself contains 4 clean numbered sections that were never expanded.
- `plan3` (`--maxdepth 4`, C4-style `--plan:` labels): stopped at depth 4 for the identical reason again — even `item_1.1.1.1.md`, the deepest node actually generated, still contains 4 numbered sub-sections. Branching factor measured directly at 3–5 children per node, consistently, at every depth tested. `item_1`'s subtree alone reached 57 files in ~97 minutes before the run was interrupted (see §2.5's use of this same file as a worked example).

The pattern across all three real runs is identical: **the model never stops subdividing on its own.** "No `## N.` sections found → leaf," the mechanism this section originally described, has never actually fired once, at any depth, in any real run. Organic leaf detection is falsified, not merely unverified — this is a correction, not a hedge.

The correct model is the one `deepagent_code` already uses: `is_leaf = (current_depth >= max_depth)`, a human-set ceiling, never a model judgment call. `deepagent_spec` inherits this unchanged: **no new generation-time prompt is needed, and the "leaf-or-decompose" system prompt originally proposed here is dropped.** You run `deepagent_md` (unmodified) to whatever `--maxdepth` you choose, exactly as already documented in `DEEPAGENT_AND_DEEPAGENT_MD.md`, and accept that its deepest nodes will still be composite, not atomic — that gap is closed by a separate pass (§2.4), not by asking the model to self-classify mid-generation.

Because guessing the right `--maxdepth` upfront is expensive to get wrong — each additional level multiplies node count, and therefore generation time, by the observed branching factor (roughly 3–5×; going from `plan2`'s depth 2 to `plan3`'s depth 4 turned ~20 minutes of generation into multiple hours) — `deepagent_md` itself should grow a `continue <plan_dir> plan: <new label>` subcommand: persist the original run's `task`/`plan_labels`/`aspects` into the `plan_dir` on first run, then let a later `continue` call re-seed expansion from the tree's *current* leaves with one additional labeled level, instead of committing to a deep `--maxdepth` blind. That's `deepagent_md`'s own concern, not `deepagent_spec`'s — noted here only because it's the practical answer to "how deep should I go," and it removes the pressure to solve that question inside `deepagent_spec` itself.

### 2.4 The transform pass: from composite leaves to atomic specs

Since generation never produces atomic leaves on its own (§2.3), `deepagent_spec` needs a second, separate pass that runs *after* a `deepagent_md` tree exists, at whatever depth it was cut off:

1. Find the tree's actual leaves — nodes with zero children — via `_collect_node_ids()`/`_child_ids()`, unchanged, the same functions reused everywhere else in this family of tools.
2. For each leaf, don't spec the whole file as one unit. Split it by its own `## N.` sections — reusing `_parse_sections()` again, but repurposed: not "should this recurse further," but "these are the atomic spec units this leaf actually contains."
3. For each section, run one generation call against the WHO/WHAT/WHEN + SMART/INVEST + acceptance-criteria + boundaries template (§3), grounded in that section's own text plus the leaf's title and the persisted root task — not the whole leaf file, and never the whole tree.

Concretely, against the real `plan3` data: `item_1.1.1.1.md` ("Alert Submission Endpoint Design") is a leaf in the generation tree, but its own 4 sections (Endpoint Design and Data Flow / Alert Status and Scoring Mechanism / Periodic Notification Endpoint / Alert Inheritance Rules) are 4 separate, independently-specifiable implementation units — 4 specs, not 1. Spec IDs extend the dot-notation one level further purely for spec-numbering (e.g. `ANIMALALERT-1.1.1.1.1` … `.4`) — that last segment exists only in spec-space; it was never a `deepagent_md` generation node.

This replaces the leaf-or-decompose system prompt originally proposed in §2.3 with something both simpler and empirically grounded: a mechanical split applied to real generated content, instead of a mid-generation judgment call resting on an assumption about model behavior that turned out not to hold.

### 2.5 Non-leaf nodes are architectural groupings, not Agile epics/stories

A real objection, raised directly rather than assumed away: are `item_1.2`/`item_1.2.1`-style non-leaf nodes usable as epic/story levels — the way Agile epics and stories represent independently testable slices of business value? **No, and that needs to be stated plainly, not glossed over.**

Agile epic/story decomposition is a **vertical** slice: "user submits an alert sighting" cuts across the Android UI, the HTTP client call, the backend route, and the database write *simultaneously* — a thin slice through every architectural layer at once, chosen precisely so that finishing it gives a user (or a tester) something to actually exercise end-to-end.

The C4-style decomposition this document uses (§2.2) is a **horizontal** slice: `item_1.2` ("Backend" container → "Controller" component) groups nodes by *where in the architecture the code lives*, not by *what a user can do once it's finished*. Its children (`item_1.2.1`, `item_1.2.2`, ...) stay entirely inside the backend — none of them is a complete, testable feature on its own, and finishing every leaf under `item_1.2` still leaves nothing a user can exercise end-to-end, because the Android side and the database side were decomposed in separate, parallel branches (`item_1.1`, `item_1.3`, ...) that this tree never reconnects to `item_1.2`. This is exactly the concern raised: we never get a closed, partially-testable module this way, because a "module" in the C4 tree was never defined as a testable unit — it was defined as a location in the architecture.

This is not a bug to patch inside the C4 tree — it's a genuinely different, **orthogonal** axis, and forcing one tree to serve both purposes (where the code lives vs. what the user can do) is exactly what produces decomposition schemes that satisfy neither well. The two axes have to stay separate:

- The C4 tree (§2.2–§2.4) answers "where does this code live, and precisely what needs implementing" — what a developer opens to know what to write next.
- A **story index** — not designed here, tracked as open question 6 below — would be a flat or lightly-nested list of user-facing capabilities ("submit an alert sighting", "confirm a nearby alert", "level up in rank"), each tagged with the set of leaf-spec IDs, potentially spanning several different C4 branches (frontend, backend, database all at once), that together implement it. That's closer to what a QA engineer or product owner needs to answer "is this feature done and testable" — and it is **not** derivable from the C4 tree's parent/child structure. It would need its own pass: either an LLM clustering leaf specs by shared business capability, or manual `story:` tagging during spec review.

Until a story index exists, `deepagent_spec`'s output answers "what do I implement" well and "is this feature ready to test end-to-end" not at all. That gap should stay visible — `item_1.2` is a location, not an epic, and calling it one would misrepresent what it actually guarantees.

### 2.6 Update: a real epic → story → e2e tree is achievable — validated against AnimalAlert

§2.5 proposed a story index as future work, undesigned. It's since been tested directly — a genuine, independent `deepagent_md` run, not a derivation from the C4 tree — with a clear failure first, then a fix, both worth recording precisely rather than just reporting the final success.

**First attempt (`plan4`) failed.** Root task left as `"how to implement such system"`, only the depth labels changed to `plan: epic, user story`. Result: depth-1 "epics" were still process/mechanism names ("Data Acquisition and Alert Trigger Mechanism", "Real-Time Notification and Confirmation Workflow") — the word "epic" changed nothing about what got grouped together. One "epic" (System Scalability and Performance Considerations) was a non-functional concern, not a feature, misclassified as a peer of real features. The "As a user..." story template was applied *inconsistently* — present in `item_1.1`/`item_2.1`, completely absent in `item_1.2`, which reverted to pure technical prose. And the same scoring/inheritance rule got independently re-derived at least four times across supposedly-distinct epics — exactly the redundancy problem from the C4 runs, now worse because epics are supposed to be non-overlapping by definition.

**Root cause, found by reading `_build_prompt`, not guessed:** the per-depth `plan:` label is injected as a secondary `"Focus perspective: {label}"` line, *after* the dominant first sentence of the prompt — `"Write a detailed markdown analysis of: {title}... This is part of a larger study on: {root_task}"`. For `index.md`, `title == root_task == "how to implement such system"` — an explicit implementation-analysis instruction, stated first and stated twice, competing directly against a one-line "Focus perspective: epic" that comes after it. The label lost that competition every time. Changing only the depth label while leaving the root task's own framing pointed at "how to implement" was never going to work.

**Second attempt (`plan5`) succeeded, by changing the root task's own framing, not just the labels:**

```
/flow deepagent_md "the feature list for this app's store page, written the way a user
reads it before downloading — never mentioning backend, database, or implementation
details" --ctx 2 --maxdepth 3 plan: <epic-instruction>, <story-instruction>, <e2e-instruction>
```

(full label text in `deepagent_md.py`'s own docstring examples now — each label is a full instruction, not a bare word, e.g. the story label spells out *"written strictly as 'As a [app user] I want [one specific action] so that [one specific benefit]' — describe exactly one thing... never combine multiple actions"*.) Result: 4 depth-1 epics with genuinely distinct, non-overlapping scope (Core Alert Notification / User Alert Submission / Community Score and Reputation / Dynamic Alert Status Display — receiving vs. sending vs. scoring vs. status, not four restatements of the same mechanism); depth-2 stories *consistently* applying "As a [user] I want X so that Y", one action per story, no branch reverting to plain prose; depth-3 leaves naming an actual screen, an actual endpoint (`POST /api/alerts/submit`), actual database fields, and the exact user-visible response — concrete enough to spec directly, no further guessing needed.

**What this resolves and what it doesn't:** the story tree is achievable, but as a **second, independent `deepagent_md` run** over a business-capability-reframed root task — not, as originally floated in §2.5, by clustering the existing C4 tree's leaf specs after the fact. That's simpler than clustering, and now has one real, checked example instead of being a guess. What's still unresolved: linking the two trees together — which `spec_<id>` in the C4 tree actually implements which story in the epic/story tree — remains undesigned (see open question 6, revised).

**A second, unrelated bug surfaced during this validation and was fixed while looking at `plan5`'s output**: `_parse_sections()` requires a leading `## N.` — but the model very consistently writes its *first* real section without the number (`"## Core Alert Notification Capability"`, no "1."), even though the prompt explicitly asks for `"## 1. Title"`. Every real tree examined this session (`plan1` through `plan5`) silently lost its single most foundational item this way — the most basic epic, the most basic story — with no error, no warning, just an absent child node. Fixed in `deepagent_md.py`: `_parse_sections` now treats an unnumbered first `##`-or-deeper heading as implicit section 1, while still correctly ignoring a bare single-`#` heading (the node's own file-title line, present only when re-reading an already-generated file). Verified against `plan5`'s actual content and against the resume-read code path, not just the fresh-generation path — both now produce identical, complete section lists. This matters for `deepagent_spec` specifically because the transform pass (§2.4) depends on `_parse_sections` finding *every* real section — before this fix, the most foundational unit at every level would have been silently missing from the spec tree too.

---

## 3. The Spec Template

Each atomic unit found by the transform pass (§2.4) — one per `## N.` section inside a generation leaf, not one per leaf file — is written to a standardized template rather than free prose. Not every field applies at every level (see §6, open question 1) — this is the full template for one such unit:

```markdown
# ANIMALALERT-1.2.3: Implement POST /alert route

WHO:    Backend developer
WHAT:   Accept a new alert submission and persist it with inherited score
WHEN:   Phase 1 (Backend Core) — blocks 1.2.4 (GET /get_alert)

SMART:
  Specific   — one endpoint, one HTTP method, one table write
  Measurable — 201 on success, 400 on invalid animal_id, row exists in Alerts
  Achievable — depends only on Animals/Users tables already specified in 1.1.x
  Relevant   — required for the core alert-creation flow
  Time-boxed — estimated 0.5 day

INVEST:
  Independent — no dependency on 1.2.4/1.2.5 to be implemented
  Negotiable  — response shape may change pending Android client review
  Valuable    — without this, no alert can ever be created
  Estimable   — yes, see SMART/Time-boxed
  Small       — single route handler, no sub-decomposition needed
  Testable    — see Acceptance Criteria

ACCEPTANCE CRITERIA:
  - POST /alert with valid {animal_id, lat, lng} returns 201 and alert id
  - POST /alert with unknown animal_id returns 400
  - Created row has status = PENDING and score inherited from reporter

BOUNDARIES:
  - Does NOT implement confirmation logic (see ANIMALALERT-1.2.4)
  - Does NOT implement rate limiting (see ANIMALALERT-4.x, Premium phase)

SOURCE: [src: plan.md#3, implementation_plan.md#3]
```

The `SOURCE:` line follows the provenance convention already required for `ENCYCLOPEDIA.md` — a leaf that was grounded in an existing planning document (standalone mode, §4) or an existing `deepagent_md` node (layered mode, §4) should cite it, for the same reason: without it, neither faithful quotation nor later re-verification is possible.

The `ANIMALALERT-1.2.3` identifier is the node's own dot-notation ID, prefixed with the project name — **no separate counter or ID allocator is needed**, since the filename-encoded tree already provides a stable, hierarchical, collision-free identifier for free. This differs from Backlog.md's flat incrementing `TASK-1`/`TASK-2` scheme (see §7) precisely because the tree already carries structure Backlog.md's flat model doesn't have.

---

## 4. Two Modes: Standalone and Layered

The question that prompted this document — "do I need to run `deepagent`/`deepagent_md` before `deepagent_spec`, and if so how do I hand it the whole result?" — turns out not to be an either/or. Both modes are supported by the same engine, and the second one answers the context-handoff question by construction rather than by inventing a summarization step.

**Standalone**: `deepagent_spec` runs its own `_expand_bfs` directly from the raw task description (`task.txt`), with no `deepagent_md` prerequisite. This is the right choice when no prior exploration exists, or when the raw task is already well-understood and doesn't need a deepening pass first.

**Layered**: `deepagent_spec` is pointed at an *existing* `deepagent_md` `plan_dir` (one already containing `item_<id>.md` files from a prior deepening run). For each `item_<id>.md`, it writes a matching `spec_<id>.md`, grounded in **only that node's own `item_<id>.md` plus its parent chain** — the same `_extract_parent_section` scoping `deepagent_md` already uses internally, never the whole tree. This is the concrete mechanism for "how do I give the whole deepagent_md result to the spec writer": you never do; the tree's own filename-encoded structure already tells `deepagent_spec` exactly which single file is relevant to which node.

The dot-notation IDs make the two modes directly comparable: `item_1.2.3.md` (deepened idea) and `spec_1.2.3.md` (executable spec) sit side by side in the same `plan_dir`, same ID, same node — one deep-dive, one execution contract.

---

## 5. `deepagent_task`: Deterministic Rollup, Not a Second Generator

The relationship between `deepagent_spec` and `deepagent_task` mirrors the relationship already established between `term.md` and `glossary.md` in [`GLOSSARY.md`](GLOSSARY.md), and between per-article provenance and the deterministic Change History section in `ENCYCLOPEDIA.md`: the deep artifact is generative (LLM), the flat rollup is a **compiler**, not a second LLM pass.

`deepagent_task` walks the `spec_<id>.md` tree (via the same `_collect_node_ids`/`_child_ids` used everywhere else in this family of tools) and extracts one checkbox line per leaf node:

```markdown
## Backend Core
- [ ] Implement POST /alert route [ANIMALALERT-1.2.3]
    - [ ] Validate animal_id against Animals table [ANIMALALERT-1.2.3.1]
    - [ ] Persist alert with inherited score [ANIMALALERT-1.2.3.2]
- [ ] Implement GET /get_alert route [ANIMALALERT-1.2.4]

Blocked on schema decisions — see ANIMALALERT-1.1.x before starting 1.2.x.
```

Two format decisions directly answer the complaints raised about Tasks.md in §1:

- **Arbitrary depth**: plain nested `- [ ]` markdown already supports unlimited nesting; the flat file itself has no depth ceiling. Collapsing/expanding a subgroup is a *rendering* concern (for a future vyrii viewer, following the same wiki-navigation pattern already planned for the glossary browser), not a file-format limitation.
- **Comments without a checkbox marker**: any line that doesn't start with `- [ ]`/`- [x]` (a bare `-` bullet, or a plain paragraph like the "Blocked on schema decisions..." line above) is passed through as narrative context, never parsed as a task. This requires no new syntax — only that the `tasks.md` renderer/parser we write treats checkbox-prefix as the sole task marker, unlike Tasks.md's tool which (per its lane=directory/card=file model) has no such comment concept at all.

Each checkbox line carries its spec ID as a bracketed tag — the same `[ANIMALALERT-1.2.3]` — enabling a click-through from the flat task view to the full `spec_<id>.md` article, the identical mechanism already planned for glossary's `[term](term.md)` cross-links in the vyrii glossary browser (`purrfect-weaving-lightning` plan). Checking a box in `tasks.md` is a pure UI/status action — it does not regenerate or touch `spec_<id>.md`, keeping the deep artifact and the operational view cleanly separated, same as glossary vs. term files.

---

## 6. Open Questions

1. **Field applicability per level is undesigned.** A Container-level node ("backend") shouldn't need Acceptance Criteria the same way a Code-level node ("POST /alert route") does — SMART/INVEST/acceptance-criteria arguably only make sense once a node is concrete enough to be a leaf. Whether boundary-level nodes get a lighter template (just WHO/WHAT/WHEN + a one-paragraph scope statement) or no template at all (just the `## N.` section list itself, as `deepagent_md` already produces) is not yet decided.

2. **ID stability across re-decomposition is unhandled.** If node `1.2` is re-expanded later with a different set of children (say, a controller gets split differently after a design change), existing `1.2.1`/`1.2.2` IDs may now refer to the wrong thing. `glossary.py`'s `--redefine` flag is the closest existing precedent for "regenerate an existing node deliberately," but it doesn't address renumbering. No policy exists yet.

3. **Cross-cutting, non-tree dependencies are out of scope for v1.** The tree captures parent→child structure cleanly, but real work has edges that aren't structural — "Confirm alert" (1.2.5) depends on "Create alert" (1.2.3) despite being siblings, not parent/child. Backlog.md (see §7) handles this with an explicit `dependencies` field per task, outside the tree; `deepagent_task`'s flat rollup would need the same, e.g. a `depends: ANIMALALERT-1.2.3` tag alongside the ID tag. Not designed yet.

4. **RAG-grounding reliability in standalone mode is unmeasured**, for the same reason already documented in `ENCYCLOPEDIA.md` §"Ризик retrieval": if `deepagent_spec` runs standalone with `--rag` against a corpus (rather than layered over an already-deepened `deepagent_md` tree), the retrieval reliability of term/spec-item-as-query against that corpus has not been benchmarked and should not be assumed from exp0–exp3's unrelated task→module MAP numbers.

5. **Local viewer / MCP exposure** for both `spec_<id>.md` trees and `tasks.md` follows the same pattern already scoped for glossary in the `purrfect-weaving-lightning` plan (Files-tab trigger, browsable tree, wiki-style click-through) and the MCP-exposure pattern both `Backlog.md` and `mdplanner` already ship (see §7) — genuinely future work, not designed here.

6. **A story index — partially resolved.** See §2.6: a real epic → user story → e2e-function tree is achievable, empirically validated (`plan5`), as a **second, independent `deepagent_md` run** with the root task itself reframed toward business capability (not just a different `plan:` label — the root task dominates the label, per §2.6's root-cause finding). What's still undesigned: **linking** the two independently-generated trees — which `spec_<id>` from the C4/architecture tree actually implements which story from the epic/story tree. Not derivable from either tree's own parent/child structure; would need a separate cross-reference pass (LLM matching, or manual `story:` tags on C4 leaf specs) — not designed yet.

---

## 7. Prior Art (from direct research, not assumption)

Three real tools were investigated before finalizing this design, specifically to check whether "tasks.md" and "mdplanner" (named by the author from memory) already solved this problem:

- **[Tasks.md](https://github.com/BaldissaraMatheus/Tasks.md)** — self-hosted markdown Kanban; lane = directory, card = file. No hierarchy, no ID scheme, no comment concept. Source of the checkbox/tag *format* the author liked, not a fit for the tool's *behavior*.
- **[Backlog.md](https://github.com/MrLesk/Backlog.md/)** — the closest real precedent: git-native, AI-agent-oriented, spec-driven workflow (decompose → plan → human-approve → implement → review), one file per task with a flat incrementing ID (`TASK-1`), explicit `dependencies` field, CLI + MCP server + web Kanban. Deliberately flat by philosophy ("one task = one context window = one PR") — does not attempt epic/story/subtask nesting, let alone architecture-lensed nesting.
- **[mdplanner](https://github.com/studiowebux/mdplanner)** — heavier PM tool, one `.md` + YAML frontmatter per entity, 25+ views including a C4 architecture view (the source of the C4 naming decision in §2.2) and a built-in MCP server exposing the whole project to Claude Code.

Neither real tool nests tasks along architectural boundaries — that piece is this project's own contribution, not something adopted wholesale from prior art. The file-per-node + ID-in-path idea, the deterministic rollup, and the checkbox/tag leaf format are each borrowed from one of the above (or from `glossary.py`) rather than invented from scratch.

---

## 8. Implementation Location (not yet built)

- **`deepagent_spec`** (planned): `C:\Project\1bcoder\_bcoder_data\flows\deepagent_spec.py` — imports `deepagent_md.py` via the same `importlib.util.spec_from_file_location` pattern `deepagent_test.py` already uses against `deepagent_code.py`. Runs no generation-time decomposition of its own — depth is controlled entirely by whatever `--maxdepth` the prior `deepagent_md` run used (§2.3). Reuses `_collect_node_ids`/`_child_ids`/`_parse_sections`/`_extract_parent_section`/`_generate_with_retry`/`--profile` parallel workers/`--rag`/`_rag_research` unchanged. New: the leaf-to-sections transform pass (§2.4), the per-unit template (§3).
- **`deepagent_task`** (planned): `C:\Project\1bcoder\_bcoder_data\flows\deepagent_task.py` — no LLM calls in the core rollup path; walks `spec_<id>.md` tree, emits `tasks.md`.
- **Output location**: same `plan_dir` convention as `deepagent_md` — `.1bcoder/planMD/planN/`, `spec_<id>.md` alongside any pre-existing `item_<id>.md`.

---

## 9. Related Documents

- [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md) — the tree engine this document reuses unchanged; read first.
- [`ENCYCLOPEDIA.md`](ENCYCLOPEDIA.md) — sibling design: same provenance discipline (`SOURCE:` line), same deterministic-rollup-over-generative-tree pattern (Change History ↔ `deepagent_task`), same honest-uncertainty framing for unmeasured retrieval risk.
- [`GLOSSARY.md`](GLOSSARY.md) — origin of the deep-file/flat-index split (`term.md`/`glossary.md`) that `spec_<id>.md`/`tasks.md` directly mirrors.
