---
name: project-alkonost-build
description: Alkonost (renamed from alconost) is built — standalone Flask task-board viewer/editor for deepagent_task output
metadata: 
  node_type: memory
  type: project
  originSessionId: 59f4e9c6-779f-4f0a-9009-ff73ef6ea5a3
---

Alkonost — a standalone, pip-installable Flask tool that views/edits `.1bcoder/tasks/<plan>/tasks.md`, `.1bcoder/spec/`, and `.1bcoder/planMD/` artifacts produced by 1bcoder's `deepagent_md`/`deepagent_spec`/`deepagent_task` flows — was designed (concept doc `concepts/ALKONOST.md` in simargl) and fully implemented this session.

**Why renamed from "alconost" to "alkonost"**: user explicitly asked to rename the whole project (directory, package, entry point, all internal references) to match the more standard transliteration of Алконост. Done via directory rename + bulk find/replace across all `.py`/`.toml`/`.md`/`.html` files, followed by editable-reinstall and full retest — see [[feedback_notepad_indentation_gotcha]] session for the parallel debugging work done around the same time.

**Location**: `C:\Project\alkonost\` (matches the `C:\Project\<tool>\` live-tools convention — see AGENTS.md). Entry point `alkonost = "alkonost.__main__:main"`, `alkonost --port 8090 --root <dir>`.

**Architecture**: no import dependency on 1bcoder — independently re-implements parsing against the same on-disk formats (`item_<id>.md` trees, `spec_<leaf>.<sec>.md`, `tasks.md` row regexes ported verbatim from `deepagent_task.py`). Opaque project ids (`sha1(root)[:12]`) resolved via query params, not raw paths in URLs. Bounded recursive `.1bcoder/` discovery (`max_depth=6` default, ignore-list for `.git`/`node_modules`/etc.). No config file, no auth, no regenerate-from-UI button, no markdown rendering — all deliberate v1 simplifications.

**Testing discipline**: 35 passing pytest tests against synthetic fixtures (mimicking real sparse `.1bcoder/` shape) plus a live smoke test against the real `AnimalAlert` project (10 real projects discovered, real 297-row `plan3/tasks.md` parsed correctly, sparse plans with no spec/tasks handled correctly).

**Bug found+fixed during build**: `discovery.discover_projects()` initially applied `os.path.dirname()` twice to each found project root (the `_walk()` helper already returns the project root itself, not the `.1bcoder` path), collapsing all discovered projects onto their shared parent directory — caught by a fixture test, not shipped.

Related: [[project_vyrii_flask]] (packaging/serving pattern mirrored exactly — `Flask(static_folder=None)` + 3 static routes + `waitress.serve`, not `app.run()`).
