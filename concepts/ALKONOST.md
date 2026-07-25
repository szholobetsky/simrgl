# Alkonost — Task Board Viewer/Editor for deepagent_task Output

## Abstract

`deepagent_task` (see [`DEEPAGENT_SPEC.md`](DEEPAGENT_SPEC.md) §5) produces `.1bcoder/tasks/<plan_name>/tasks.md` — a flat, tag-based, nested-checkbox task list rolled up from a `deepagent_md` tree and its `deepagent_spec` units. Nothing in the 1bcoder/vyrii ecosystem can currently *view* or *edit* that file as a board — it's just a markdown file on disk. Alkonost is the new, standalone tool that closes that gap: a Trello-style board with drag-and-drop, a filterable/groupable backlog table, tag editing, and cross-navigation into the `spec_<id>.md`/`item_<id>.md` files each task row derives from.

Named after **Алконост** (Alkonost), the Slavic paradise-bird of joyful song — traditionally depicted paired with **Сирин** (Sirin, sorrow/night) in Old Russian/Slavic iconography, the two birds shown together. `syryn` already exists in this ecosystem (alongside `simargl`, `vyrii`, `yasna`, `radogast`, `svitovyd`); `alkonost` continues both the mythological naming convention and, specifically, that established pairing.

---

## 1. Why a New, Standalone Product — Not vyrii, Not 1bcoder

Two things were explicitly ruled out this session, for concrete, stated reasons, not by default:

**Not vyrii.** The obvious first instinct — build this as a new vyrii tab, reusing its Flask+FastAPI+Gradio scaffolding and the glossary-tab pattern already scoped (see the `purrfect-weaving-lightning` plan history) — was explicitly rejected by the user ("я не хочу зараз думати про вирій"). The decision: build standalone now; *if* it grows into something worth formalizing as its own ecosystem member, glossary's own not-yet-built viewer could move into Alkonost later too, rather than the reverse.

**Not a pure static page.** `/proc mdx` (`_bcoder_data/proc/mdx.py`) was checked directly as a candidate lightweight pattern: it generates a single self-contained HTML file (markdown via marked.js, LaTeX via KaTeX, diagrams via Mermaid, all CDN `<script src>`, zero build step) and opens it with `webbrowser.open("file://" + path)` — no server at all. This works for `mdx` because it's read-only. Alkonost cannot use it as-is: a `file://` page has no filesystem write access (browser sandbox), and Alkonost's core requirement — edit a row's tags, drag a card, hit **Save**, and have `tasks.md` actually change on disk — needs *something* listening for that write. Hence: a real, minimal local server, not a static file.

**Gradio was also considered and rejected** for the board UI specifically — a dense, custom, drag-and-drop interface "виглядає убого" (looks poor) in Gradio's component model, per direct user judgment. Flask, not Gradio, is the base.

---

## 2. Tech Stack

- **Flask** (not FastAPI, not Gradio, not raw `http.server`) — chosen once "no vyrii" was settled: Flask's routing/JSON convenience is worth the one small dependency, and this project already treats Flask as a known, acceptable quantity (it's vyrii's own primary backend).
- **Vanilla JS, no build step, no bundler** — same philosophy already proven in vyrii's `ui/app.js`: plain `<script>` tags, HTML5 Drag and Drop API for the board (native browser API, no external library required for the core interaction).
- **pip-installable**, matching the ecosystem convention already established by `1bcoder` (PyPI) and `vyrii` (pure-Python, no Rust): `pip install alkonost`.

---

## 3. Invocation Model

```
pip install alkonost
alkonost --port 8090
```

**Revised from the first draft of this document**: run Alkonost not from one specific project's own root, but from a *common ancestor* directory that may contain several separate projects side by side (e.g. `capestone/repository/`, with `AnimalAlert/` and others as siblings under it, each with its own project-local `.1bcoder/`). Alkonost recursively walks the directory it's started in, looking for every `.1bcoder/tasks/` folder it can find, groups the hits by the project root that owns each one (the folder directly containing that `.1bcoder/`), and presents a **two-level picker**: first which *project*, then which *plan* within it — rather than assuming a single fixed project root the way the first draft did. Still no separate config file telling Alkonost where anything lives; still resolves everything from where it was started, same convention `1bcoder`/`vyrii`/`deepagent_md`/`deepagent_task`/`deepagent_spec` already use, just one level more general — one Alkonost instance can now sit above an entire workspace of projects, not just one.

`GET /api/projects` lists every project root discovered this way; `GET /api/plans?project=<root>` lists that project's own `.1bcoder/tasks/<name>/` subdirectories (see §5) — a single running instance can browse *all* plans across *all* discovered projects, not just one, since switching which board you're looking at should never require restarting the server.

Recursive discovery has an obvious cost/safety question this document doesn't resolve yet — see open question 5.

---

## 4. What It Renders and Edits

Reuses the tag-based `tasks.md` format exactly as `deepagent_task` already produces it — no new format invented for Alkonost, it's purely a renderer/editor over an existing, deterministic artifact:

```
[ ] plan5-1.1.1. Create authorisation #backend @backend +active %user1 due:2026-07-29 _done:
```

- **Board** (top): cards in the `+active` lane, drag-and-drop between lanes. `+urgent` needs distinct visual treatment (red, a separate list, or pinned to the top) — see open question 1.
- **Dashboard flow filter**: a single editable row above the board (`.1bcoder/tasks/dashboard_flow.md`, one shared ordered list per project — same convention as `colors.md`) lets the user type e.g. `+selected, +active, +paused, +testing, +done, +releasing`. Only rows whose lane is in that list appear on the board, in that exact column order (columns render even when currently empty, like a normal Kanban board's fixed structure); lanes like `+backlog`/`+complete`/`+canceled`/`+closed` are hidden from the board without disappearing from the Backlog table below, which always shows every row regardless of this filter. Empty/missing file → no filter, falls back to showing every distinct lane found in the data (the original pre-filter behavior). The "move to lane" dropdown always offers the union of every real lane in the data plus every configured-but-currently-empty flow lane, so a hidden row can be moved onto the dashboard (or off it) from either the board or the Backlog table.
- **Backlog** (below): table view of everything else. A task doesn't disappear from the backlog table just because it's `+active` — it stays listed, visually marked (background color or icon) so its active state is visible from both views at once.
- **Tag discovery buttons**: every unique `#tag`, `@module`, `+flag`, `%user` found across the current plan's rows, surfaced as clickable filter buttons — same UX already used for glossary's term-list buttons (`_gl_search_terms`-style pattern in vyrii, reused as a design reference, not as shared code).
- **`@module` → color**: driven by `.1bcoder/tasks/colors.md` (`@module=color` pairs, one shared legend across all plans — a deliberate simplification made this session: originally considered per-plan `<plan>.colors.md`, rejected in favor of one shared file). `deepagent_task` creates this file with generic starter defaults the first time it's missing and never touches it again; Alkonost is the file's actual owner (add/edit/remove `@module=color` entries through its own UI).
- **Editing**: title and tag string per row, editable in place. **Autosave, debounced (400ms)** — revised from this doc's original "explicit Save button, not live-write" decision: every mutation (title/tag field commit on blur, drag-and-drop lane move, colors legend edit) now persists itself shortly after the user stops interacting, collapsing a burst of edits into one save call rather than firing per keystroke. The manual "Save Changes"/"Save Colors" buttons remain as an explicit "save now" fallback. A status indicator next to the button shows `unsaved changes… → saving… → saved`, or a persistent `⚠` message on failure (e.g. a 409 row-mismatch conflict) instead of a blocking `alert()` — autosave should never interrupt typing with a modal dialog.
- **Cross-navigation**: a task row links to its `spec_<leaf_id>.<i>.md` (via the `SOURCE:` line already written into every spec by `deepagent_spec`), which in turn links to the `item_<id>.md` node it was extracted from — the same provenance chain already built this session, just made clickable.
- **Adding tasks/subtasks by hand**: "+ Add Task" (Backlog header) and a per-row "+ Add SubTask" open the same modal — id field (auto-computed: next root number, or `parent.N+1` for a subtask, always editable), task text, `#`/`@`/`+` combobox fields (`<input list>` + `<datalist>`, so an existing value can be picked or a new one typed), an `+urgent` checkbox. These rows are written **directly into `tasks.md`** — no `item_<id>.md` is ever created for them, so they have no `planMD` node and no "source item" cross-navigation; they're purely manual entries deepagent_task itself would never generate. Insertion position: root tasks append at the very end of the file; a subtask lands right after its parent's *entire* existing subtree (deepest descendant included), matching the DFS-ordered layout `deepagent_task` itself always produces, and its own id only increments the parent's *direct* child count (deeper descendants' numbering is irrelevant). This required relaxing `tasks.py`'s `validate_row_ids` from an exact-match check to a subsequence check — existing ids must still all be present, in order, but new ones may now be interleaved anywhere, which is what actually made "Save" no longer 409 the moment a row was added.
- **"+ Add spec"**: any row without a matching `spec_<id>.<n>.md` (checked via a new `GET /api/spec_index` listing every spec id that exists for the plan) shows an "+ Add spec" button instead of the normal "spec" view link. Clicking it creates `spec_<task_id>.1.md` with a minimal starter template (title line + placeholder body + `SOURCE:` line, idempotent — re-clicking never overwrites real content) and opens it directly in the side panel's edit mode, so writing the actual content is the very next step rather than a separate round trip.
- **Editing the spec/item file itself**: a pencil button in the side panel header switches it from read-only rendered view to a raw-markdown editor (a plain `<textarea>` over the file's actual content) with a live preview below, using the same `md()` renderer — no WYSIWYG. A true WYSIWYG (bidirectional HTML↔markdown conversion that preserves code/table/LaTeX/Mermaid blocks intact) was considered and rejected as its own substantial project, not worth it for what's fundamentally "let a non-markdown-fluent user append a comment and see how it renders." Edits POST back to `PUT`-like `POST /api/spec` / `POST /api/item` routes, writing the raw text verbatim — no validation of the title/`SOURCE:` line format, same "trust the human" convention as `tasks.md`/`colors.md`. Navigating away (close button, breadcrumb source-item link, opening a different card) with unsaved edits prompts a confirm — the one place in Alkonost a blocking dialog is appropriate, since it's a deliberate user action, not an automatic background process like autosave.
- **Direct `planMD` browsing**: not only reachable by clicking through from a task row — the project/plan picker (§3) itself offers a straight jump into that plan's `.1bcoder/planMD/<plan_name>/` tree (the raw `deepagent_md` output, `item_<id>.md` files), for looking at the deepened source material on its own terms, independent of any specific task.

---

## 5. Routes (implemented — `C:\Project\alkonost\alkonost\flask_api.py`)

Raw project-root paths never appear in URLs (Windows paths contain `:`/`\`, awkward as path segments) — every route resolves an opaque `project=<id>` query param (`id = sha1(root)[:12]`) against a registry rebuilt fresh on every call from `discovery.discover_projects()` (cheap given the depth cap + ignore-list; sidesteps stale-cache invalidation since ids are a pure function of path):

```
GET  /api/projects                              -> [{id, root, name, plan_count}, ...]
GET  /api/plans?project=<id>                    -> [{name, item_count, has_spec, spec_count, has_tasks, task_count}, ...]
GET  /api/tasks?project=<id>&plan=<name>        -> {plan, exists, rows: [{id, title, depth, checked, tags_raw, tags}, ...]}
POST /api/tasks?project=<id>&plan=<name>        -> {ok, row_count} | 409 on row id/order mismatch
GET  /api/colors?project=<id>                   -> {colors: {"backend": "#4A90D9", ...}}
POST /api/colors?project=<id>                   -> {ok: true}
GET  /api/spec?project=<id>&plan=<name>&spec_id=<leaf_id>.<sec_index>  -> parsed spec fields + raw text
GET  /api/item?project=<id>&plan=<name>&item_id=<id>                   -> raw item_<id>.md + children + is_leaf
GET  /api/planmd/tree?project=<id>&plan=<name>  -> nested [{id, title, is_leaf, children:[...]}, ...]
                                                    (added beyond the original sketch — powers §4's
                                                    "direct planMD browsing" independent of any task row)
```

`POST /api/tasks` sends the entire row snapshot every time (not a diff) and the server does a whole-file rewrite either way — `tasks.py`'s `rewrite_tasks_md()` preserves `deepagent_task`'s own re-run contract (same id/order/depth, only title/tag-string content changes) and `validate_row_ids()` rejects a stale submission with a 409 rather than silently clobbering a row `deepagent_task` would still recognize as existing. Verified directly against the real `AnimalAlert/plan3` 297-row `tasks.md` and its sparse `plan1`/`plan2`/`plan4` counterparts (no `spec/`/`tasks/` yet) — both shapes round-trip correctly.

---

## 6. Relationship to the Rest of the Ecosystem

- Reads `.1bcoder/planMD/`, `.1bcoder/spec/`, `.1bcoder/tasks/` — all produced entirely by 1bcoder's `deepagent_md`/`deepagent_spec`/`deepagent_task` flows (see [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md), [`DEEPAGENT_SPEC.md`](DEEPAGENT_SPEC.md)). Alkonost itself makes **zero LLM calls** — pure viewer/editor, the same "deterministic compiler, not a generator" principle already established for `deepagent_task` and for `ENCYCLOPEDIA.md`'s Change History section.
- vyrii's `/vyrii/tools` external-services bar (Flask + FastAPI + Gradio, built this session) is the intended link point once Alkonost exists: register it as `{"name": "ALKONOST", "port": <N>}` and it's one click away from vyrii's own UI — vyrii never embeds or proxies it, just links out, the same way it already links to `SIMARGL`/`SVITOVYD`.

---

## 7. Open Questions (resolved for v1, implemented)

1. **`+urgent` visual treatment** — resolved: a red left-border (`--urgent` CSS var) plus a small "URGENT" badge, applied identically to board cards and backlog rows. `urgent` is a modifier inside `tags.plus`, never its own lane.
2. **Auth** — resolved for v1: none. Plain HTTP, local-network-trusted assumption, matching vyrii's own default-off auth posture. Revisit if Alkonost is ever exposed beyond localhost.
3. **`deepagent_task`/`deepagent_spec` triggering from within Alkonost** — resolved: no regenerate button in v1. User reruns 1bcoder's own flows and refreshes the Alkonost page afterward.
4. **Multi-user / concurrent editing** — still out of scope for v1, but no longer silent: `POST /api/tasks` validates the submitted row id set/order against what's on disk and returns 409 on mismatch (e.g. someone reran `deepagent_task` while the page was open), rather than clobbering silently.
5. **Recursive `.1bcoder/` discovery — resolved**: `discovery.py`'s `_walk()` uses a bounded manual recursion — default `max_depth=6` (CLI-overridable via `--max-depth`), an ignore-list (`.git`, `node_modules`, `venv`, `.venv`, `__pycache__`, `dist`, `build`, `.idea`, `.vscode`, plus dot-directories generally except `.1bcoder` itself), and no symlink following. Once a `.1bcoder` is found, its contents are read directly (`os.listdir`), never walked further — siblings/cousins elsewhere in the tree keep being searched for other projects. Verified against a real multi-project directory (`capestone/repository/`, 10 projects, one 5-plan project) with sub-second response times.

---

## 8. Related Documents

- [`DEEPAGENT_SPEC.md`](DEEPAGENT_SPEC.md) — `deepagent_task`'s own design and the exact `tasks.md` tag format Alkonost renders/edits (§5 there is the authoritative format spec, not repeated here).
- [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md) — the tree engine whose output (`item_<id>.md`) Alkonost cross-links to.
- [`GLOSSARY.md`](GLOSSARY.md) — source of the tag-discovery-as-clickable-buttons UX pattern Alkonost's filter bar borrows.

---

## 9. Implementation

Built at `C:\Project\alkonost\` (pip-installable, `alkonost = "alkonost.__main__:main"` entry point; `pip install -e .` + `alkonost --port 8090`). Modules: `discovery.py`, `planmd.py`, `spec.py`, `tasks.py`, `colors.py`, `dashboard.py`, `config.py`, `flask_api.py`, `__main__.py`, `ui/{index.html,app.js,style.css,themes/*.css}`. 58 passing Python tests (`tests/*.py`, `pytest`) against synthetic fixtures and regex/format contracts ported verbatim from `deepagent_task.py`, plus 48 passing client-side JS tests (`tests/js/*.test.js`, `node --test` — no build step/extra dependency, using `vm.createContext` to load `ui/app.js` in isolation) covering the markdown/LaTeX/mermaid renderer, autosave debounce, dashboard-flow filtering, colors-legend normalization, drag-and-drop rendering, the raw-markdown side-panel editor, last-selection persistence, and the Add Task/Subtask/Spec id/insertion logic; also smoke-tested end-to-end against the real `AnimalAlert` project (including a full Add Task + Add Spec round trip, reverted afterward since it was only a test). No PyYAML/`_deepagent_meta.yaml` dependency.

**"No config file" (§0) has been partially reversed**: `~/.alkonost/config.json` (same convention as vyrii's own `~/.vyrii/config.json`) now persists the *last-selected* project id, plan name, and theme — the one piece of state that belongs to the alkonost installation on this machine, not to any scanned project's own `.1bcoder/`. `GET`/`POST /api/config` is a flat merge-updatable blob (not project-scoped, no `project=` param). Saved immediately on each selection (not autosaved/debounced — a deliberate select action, not continuous typing); restored on next launch by re-selecting the project (if it still exists in the current scan) and its plan, then re-triggering the same load flow a manual selection would.

The spec/item side panel *does* render full markdown/tables/LaTeX (KaTeX)/Mermaid via a hand-rolled renderer ported from vyrii chat's own `md()` (see `ui/app.js`) — this reverses this doc's original "raw `<pre>` text" v1 simplification, added once the side panel became the primary way to read a whole spec. It's also now **editable**: a pencil button switches it to a raw-markdown `<textarea>` with a live preview (same `md()` renderer, no WYSIWYG — considered and rejected as its own substantial project given code/table/LaTeX/Mermaid round-tripping), writing back via `POST /api/spec` / `POST /api/item` (verbatim, no title/`SOURCE:` validation, same "trust the human" convention as `tasks.md`/`colors.md`).
