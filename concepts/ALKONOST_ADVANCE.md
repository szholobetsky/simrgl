# Alkonost, Advanced — A Cockpit for the Sequential Tool Chain, Not a Task Board

## Abstract

`concepts/ALKONOST.md` designed Alkonost as a Trello-style viewer/editor over one artifact: `deepagent_task`'s `tasks.md`. That artifact sits at the very *end* of a long, currently under-designed pipeline — raw research becomes a plan, a plan becomes specs, specs become a task list, tasks become architecture decisions, code, and tests — and every step in between that pipeline is either a fixed, un-steerable script or has no UI at all. This document reframes Alkonost's actual job: **not managing a task list, but putting every launch parameter, every intervention point, and every plan-direction decision of the whole `deepagent_*` chain in front of the user, stage by stage.** The task board from v1 does not go away — it becomes the pipeline's last tab, unlocked only once everything upstream of it has actually been done properly.

---

## 0. This Supersedes the Alkonost/Gamayun Split

Earlier in this same design thread, two directions were explored and are both superseded here:

1. **Decouple Alkonost into a generic, portable, generator-agnostic MD task tracker** — its own `.alkonost/` canonical store, populated by a one-time import from `.1bcoder/`, so any agent (1bcoder, pi, opencode, aider, a human) could write compatible files, with a separate tool ("Гамаюн") absorbing everything that needs live, 1bcoder-specific UI (tree browsing, radogast dashboard, one-click `redo`).
2. **Merge that supervision UI directly into Alkonost**, reusing its existing `planmd.py`/`discovery.py` since they already read the exact on-disk convention needed.

Both took the product-positioning question ("should Alkonost be portable, should it be a separate app from the 1bcoder-specific tooling") as the thing to resolve. It wasn't. The actual gap driving this whole thread is that **the transformation from research plan to spec/task list is itself broken** (§1) — not merely unsupervised, but conceptually incomplete — and fixing that requires Alkonost to be *more* deeply wired into the whole `deepagent_*` chain, not less. Portability is dropped as a near-term goal. A separate Гамаюн is not needed — its entire proposed scope folds into the new tabs below (§3).

---

## 1. The Real Gap: Plan → Spec Follows the Wrong Axis

`deepagent_spec` (see `DEEPAGENT_SPEC.md` §2.4) is, today, one fixed system prompt and one loop: find a `deepagent_md`/`deepagent_tree_md` tree's leaves, split each leaf's own `## N.` sections, spec each section — mechanically, 1:1, no choice of direction. This is not a missing feature, it's a category error, and `DEEPAGENT_SPEC.md` already documented the shape of the problem without resolving it:

- §2.2 chose an **architectural** decomposition axis for the research plan itself (C4-style: inter-module → intra-module → object) — *or*, per §2.6, a **capability** axis (epic → user story → e2e) when the root task is framed that way.
- §2.5 states plainly that these are **orthogonal**: "The C4 tree answers where does this code live... A story index... would be a flat or lightly-nested list of user-facing capabilities... potentially spanning several different C4 branches... That's closer to what a QA engineer or product owner needs... and it is not derivable from the C4 tree's parent/child structure."
- §2.6's own closing line: "linking the two trees together — which `spec_<id>` in the C4 tree actually implements which story in the epic/story tree — remains undesigned (see open question 6, revised)."

That's the actual unfinished work. If `deepagent_tree_md` sliced the research along a user-capability vector ("submit an alert sighting," "confirm a nearby alert"), and the specs that get handed to an implementer need to be sliced along an architecture vector ("AlertController," "scoring service," "Android submission screen"), no amount of UI polish on `tasks.md` fixes that — the wrong document was fed into `deepagent_spec` in the first place. §2 is the resolution.

---

## 2. The Axis Plan: A Second, User-Approved `deepagent` Pass Between Research and Spec

Instead of `deepagent_spec` reading the research plan's own leaves directly, insert one new step:

1. **Generate an axis plan.** Run `deepagent` (the lightweight label-tree engine — not `deepagent_md`'s full documents; this pass only needs to decide groupings and names, not write prose) seeded from the *research plan's own content*, explicitly asked to decompose along whichever axis specs actually need (architecture/component, by default — but nameable per project, same `plan:`/`list:` label mechanism `deepagent`/`deepagent_md` already have). Call this tree the **axis plan** — a working name, open to change.
2. **Force approval.** The axis plan is never auto-consumed. The user must look at it and explicitly approve or edit it before anything downstream runs — the same "Ralph needs a leash" principle from `DEEPAGENT_AND_DEEPAGENT_MD.md` §9, applied one level up: an unreviewed re-axis decision is exactly the kind of consequential, hard-to-verify-automatically step that must not run unattended.
3. **Ground each axis-plan node in the research plan.** Every axis-plan leaf gets linked to the research-plan `item_x.x` node(s) that actually justify it, via:
   - **Radogast similarity search** — reuse the *same* embedding/similarity primitives already doing two other jobs in this family (`--dedup`'s `LivePlanIndex`, `plan_analyzer`'s out-of-scope/coverage scoring): embed the axis-plan node's text, nearest-neighbor search against the research plan's own embedded `item_x.x` nodes, surface the top matches as suggested links.
   - **RAG over the research plan**, as a heavier/more thorough alternative or complement for large trees — the same simargl-backed retrieval `_dedup_context`/glossary's synonym-detection already lean on elsewhere in this codebase.
4. **`deepagent_spec` follows the axis plan, not the research plan.** Each spec unit is now grounded in whichever research-plan sections got linked to its axis-plan node — the axis plan supplies the *shape*, the linked research sections supply the *content*, exactly mirroring how a leaf's own sections supply content today, just one indirection later.

This is not spec-specific. The same mismatch — and therefore the same "re-axis, force approval, link back to source" step — plausibly recurs at other stage boundaries (spec → architect's own grouping, spec → code's file/module boundaries). This document does not resolve those instances; §5 flags them as open.

---

## 3. Tab Structure

Each stage of the chain gets its own tab: a launch-parameter form, a generated-file tree/viewer, and the intervention tools `DEEPAGENT_AND_DEEPAGENT_MD.md` §9 already put at the CLI level (`redo`, `--review`'s live y/n/b/a gate, a final batch `plan_analyzer` pass) surfaced as buttons instead of terminal prompts.

| Tab | Underlying flow(s) | Launch form | Viewer | Intervention | Axis step |
|---|---|---|---|---|---|
| **PLAN** | `deepagent_md` (optional prose-research pass) → `deepagent_tree`/`deepagent_tree_md` | task text, `--maxdepth`, `plan:`/`list:` labels, `--profile` | `index.md` + collapsible `item_x.x` tree, open/edit any node | `redo <node_id>`, toggle `--review`, run a final batch `plan_analyzer` pass over the whole tree | — (this tab *produces* the vector everything downstream reacts to) |
| **SPEC** | axis-plan `deepagent` pass (§2) → `deepagent_spec` | axis-plan seed/labels; per-node link review (accept/reject suggested research links) | axis-plan tree side by side with linked research sections; generated `spec_*.md` | same `redo`/`--review`/batch-analysis triggers, now over spec units (flat, no subtree — `DEEPAGENT_AND_DEEPAGENT_MD.md` §9.4) | **yes — mandatory approval gate, §2** |
| **TASKS** | `deepagent_task` | single **CREATE TASK** button, enabled only once specs exist | exactly `ALKONOST.md` v1's board/backlog | tag/lane editing as already built | — (deterministic compiler, no LLM, nothing to re-axis) |
| **ARCHITECT** | `deepagent_architect` | ticket/backlog selection, `--next`/`--loop` claim parameters | fixed 5-level `interface → abstract → approach → class → method` pipeline per ticket | its own `research`-agent reuse-detection output; no numbered-section parent→child here, so `redo`'s subtree semantics don't apply as-is (open, §5) | possibly — open, §5 |
| **CODE** | `deepagent_code` | task/spec source, `--maxdepth`, `--profile` | generated code tree, `duplicates.md` | `redo`/`--review` (already designed, §9.1 of the deepagent doc — real work, not yet UI) | possibly — open, §5 |
| **TEST** | `deepagent_test` | joined-code path, `--lang`, `--think`/`--syntax`/`--run` | per-function test files, `report.md` | `redo`/`--review` (flat, §9.4 of the deepagent doc) | — (tests follow code's own grouping, no independent axis) |
| **SETTINGS** | none (config only) | model to load, context size, timeout/run_timeout, other `params`; theme picker; (future) language picker | raw `.1bcoder/config.yml` / `~/.1bcoder/config.yml` preview | direct read/merge-write, same convention as `tasks.md`/`colors.md` — never a generator, no LLM call here either | — |

### 3.1 Settings: Two Separate Stores, Not One

Confirmed directly in `chat.py` (not guessed): 1bcoder already has its own config file — `.1bcoder/config.yml` (project, `CONFIG_FILE`) and `~/.1bcoder/config.yml` (machine-wide, `GLOBAL_CONFIG_FILE`) — with a documented schema and its own `/config save|load|show|auto|del` CLI (`chat.py` ~9150–9270): `host`, `model`, `ctx` (→ `num_ctx`), a flat `params` dict (`timeout` — HTTP read timeout; `run_timeout` — separate cap for `/run`, default 30; `temperature`, `num_predict`, whatever else `chat.params` currently holds), plus `vars`/`procs`/`hooks`/`active_project` and an `auto: on/off` flag for whether it loads automatically at session start.

This settles what the SETTINGS tab actually is and isn't:

- **Not a new store.** The "1bcoder runtime" section of this tab edits `config.yml` directly — model/ctx/timeout/run_timeout/other params — merge-preserving `vars`/`procs`/`hooks`/`active_project` it never touches, the same non-destructive-merge convention `~/.alkonost/config.json`'s own `POST /api/config` already follows (§9 "Implementation" of `ALKONOST.md`). No parallel settings system, no new schema.
- **Not the same store as Alkonost's own preferences.** Theme — and, once built, language — are Alkonost's own UI state, already living in `~/.alkonost/config.json` (`config.py`), entirely separate from 1bcoder's `config.yml`. One tab, two backing files; the two sections must stay visibly distinct so editing one never touches the other.
- **Project vs. global** needs the same choice 1bcoder's own `/config save [global]` already offers — a `.1bcoder/config.yml` override for this project vs. the machine-wide `~/.1bcoder/config.yml` default — not a single flattened settings screen.

### 3.2 How Buttons Actually Talk to 1bcoder: `--scriptapply`, Not a New Channel

Every "intervention" cell in §3's table (`redo`, `--review`, launching a flow) needs some actual mechanism to reach 1bcoder. Two were considered:

1. **A live IPC channel into an already-running 1bcoder REPL session** — a background HTTP listener thread inside `chat.py`, started by a new `/alkonost` command, with the UI's subprocess POSTing commands back into that specific session so a button click could reuse its live `chat.messages`/model state. Rejected for now: `chat.py`'s main loop blocks on `input()` (confirmed directly, ~line 4029) — making it receive commands concurrently means real surgery on that loop (a queue drained under a lock, or `select`/`msvcrt`-based interleaving), and `self.messages` is confirmed to be in-memory only, never persisted to disk, so this complexity would buy nothing beyond the one specific case of reusing a live conversation for `--ctx N`-style pre-seeding (`DEEPAGENT_AND_DEEPAGENT_MD.md` §5.1) — a narrower need than "every button."
2. **Subprocess-per-click, using 1bcoder's own existing headless mechanism** — adopted.

Confirmed directly in `chat.py`'s `main()`: `--scriptapply <script> --param k=v ...` already runs a `.1bcoder/scripts/*.txt` script headlessly, no REPL, no interactive prompt — and, critically, resolves `--host`/`--model` from `_load_startup_config()` (i.e. `.1bcoder/config.yml` / `~/.1bcoder/config.yml`) whenever they aren't passed explicitly on the command line. That's the exact file the SETTINGS tab (§3.1) edits. Leaving `--host`/`--model` off every subprocess call this UI makes turns the Settings tab into the single, live source of truth for which model and context every flow actually runs with — no separate bookkeeping in Alkonost, no risk of it drifting out of sync with what the user set.

This is not a new mechanism invented for this document — it's already in production use for exactly this kind of multi-flow chaining. Real scripts already in `_bcoder_data/scripts/`:

```
# deepagent_plan.txt
/param timeout 300
/flow deepagent_md --file {{task}} --maxdepth 2
/flow deepagent_spec {{plan}}
/flow deepagent_task {{plan}}
```

invoked as `1bcoder --scriptapply deepagent_plan.txt --param task=task.txt --param plan=plan2` — plus tree/tree_md variants (`deepagent_tree_spec_task.txt`, `deepagent_treemd_spec_task.txt`) and later-stage ones (`deepagent_implement.txt`, `deepagent_loop.txt`, `deepagent_next.txt`) already covering ARCHITECT/CODE-tab territory.

**Design**: ship a small library of thin `.1bcoder/scripts/alkonost_*.txt` wrapper scripts, one per button-action, each just a couple of `/flow ...` lines with `{{param}}` placeholders — an optional flag like `--review` is passed as a param whose *value* is either the flag text or an empty string (`{{review_flag}}` → `--review` or ``), no conditional script logic needed. A click becomes:

```
1bcoder --scriptapply alkonost_redo.txt --param plan_dir=<path> --param node_id=<id> --param review_flag=--review
```

Write these wrapper scripts into `.1bcoder/scripts/` on first run if missing, the same "create with generic defaults, then never touch it again" convention `deepagent_task` already uses for `colors.md`.

This is explicitly *not* the vyrii mistake (§0's forerunner discussion, and confirmed directly this session — `vyrii/flows/deepagent_md.py` is a full independent ~1400-line copy of the generation engine, not a wrapper). An `alkonost_*.txt` script is 2–4 lines invoking the one real engine already in `_bcoder_data/flows/`; nothing about generation logic is ever duplicated.

The cheaper alternative discussed for the narrow live-context-reuse case — a new `/export-ctx <file>` 1bcoder command dumping `self.messages` via the already-existing `_serialize_ctx`, with the PLAN tab's launch form optionally pointing at that file — remains un-built and is not required for the mechanism above to work; it's a separate, smaller addition if that specific need actually comes up.

---

## 4. What Stays Unchanged From ALKONOST.md v1

The TASKS tab is the current implementation, verbatim — board, backlog table, tag/`@module` colors, dashboard-flow filter, spec/item cross-navigation, autosave. Nothing about it changes; it simply moves from being *the whole product* to being *the pipeline's last screen*, gated behind a CREATE TASK button that only appears once the SPEC tab has real content.

---

## 5. Open Questions

1. **Axis-plan mechanics.** Plain `deepagent` (labels only) vs. a `deepagent_md`-style full-document pass for the axis plan itself — labels may be enough since its only job is naming groups, not writing prose. Needs a real test run to confirm, the same way `DEEPAGENT_SPEC.md` §2.3/§2.6 only settled its own generation questions after real runs falsified the first assumption.
2. **"Approve and expand" UI.** What does editing an axis plan actually look like — inline tree edit (add/remove/rename nodes) before any linking happens, or edit-after-link (risk: renaming a node orphans its links)?
3. **Link quality**: radogast similarity vs. RAG — is one sufficient alone, or do they need to run together with disagreement flagged for the user (a third use of the "surface uncertainty, don't auto-resolve it" principle from `DEEPAGENT_AND_DEEPAGENT_MD.md` §9)?
4. **Does every stage boundary need its own axis step**, or only research→spec? `deepagent_architect`'s fixed linear 5-level pipeline (§ ARCHITECT row above) may sidestep the mismatch entirely, since it never branches by content the way the C4/epic trees do — worth confirming before designing an axis step nobody needs there.
5. **Where does the axis-plan artifact live on disk** — a new `.1bcoder/axis/<plan>/` convention, sibling to `planMD`/`spec`/`tasks` (matching the existing top-level-artifact-folder convention), or folded into `spec/` itself?
6. **CODE/TEST tabs' `redo`/`--review` buttons** call the CLI mechanism `DEEPAGENT_AND_DEEPAGENT_MD.md` §9 already implements — this document only adds the UI surface, not a new engine. Confirm no additional engine-side work is needed before treating those two tabs as "just UI."

---

## 6. Related Documents

- [`ALKONOST.md`](ALKONOST.md) — v1 task board, retained here as the TASKS tab.
- [`DEEPAGENT_SPEC.md`](DEEPAGENT_SPEC.md) — §2.2/§2.5/§2.6, the orthogonal-axes problem this document resolves; the spec template the SPEC tab renders.
- [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md) — §9, the `redo`/`--review` CLI mechanism every intervention column above surfaces as buttons.
- [`DEEPAGENT_ARCHITECT.md`](DEEPAGENT_ARCHITECT.md) — `deepagent_architect`'s own fixed pipeline, relevant to the ARCHITECT tab's open questions.
