---
name: memory_1bcoder
description: 1bcoder tool — repository location, packaging, PyPI status, and key implemented commands
type: project
originSessionId: 457af3c4-1663-4f2f-a188-730d645cdea6
---
# 1bcoder — Implementation State (as of 2026-03-29)

## Repository & Distribution
- **GitHub**: https://github.com/szholobetsky/1bcoder (public)
- **PyPI**: `pip install 1bcoder` (published; current local version: v0.1.7, next publish → v0.1.8)
- **Main working directory**: `C:\Project\1bcoder\` (separate from simrgl)
- **simrgl integration**: `1bcoder/` in simrgl is a git submodule → szholobetsky/1bcoder
  (submodule setup may still be in progress — user was working on it)

## Packaging Architecture
- `_bcoder_data/` — wheel defaults (renamed from `.1bcoder/` for PyPI compatibility)
  - Contains agents/, proc/, scripts/, teams/, doc/, prompts/
  - Included in wheel via `package_data` in `pyproject.toml`
- `~/.1bcoder/` — user global dir; bootstrapped from `_bcoder_data/` on first run by `_bootstrap_global_dir()`
- `.1bcoder/` in project dir — project-local (highest priority, created by `/init`)
- `INSTALL_BCODER_DIR` → `_bcoder_data/` next to `chat.py`
- `HOME_BCODER_DIR` → `~/.1bcoder/`
- All `GLOBAL_*` constants point to `HOME_BCODER_DIR`

## Copyright
(c) 2026 Stanislav Zholobetskyi, Institute for Information Recording, National Academy of Sciences of Ukraine, Kyiv.
Dissertation: «Інтелектуальна технологія підтримки розробки та супроводу програмних продуктів»

## Recently Added Commands
- `/flow <name> [args]` — run deterministic Python pipeline from `_bcoder_data/flows/`; `/flow list`
- `/web search <term>` — DuckDuckGo search → context; `/web fetch <url>` — fetch+strip HTML → context
- `/websearch` alias → `/agent websearch`; `/webask` alias → `/flow webask`
- `/role <persona>` — set system persona; `/role show`; `/role clear`
- `/proc run <name> -f <file>` — run proc against external file instead of last reply
- `/ctx clear N`, `/ctx compose`, `/ctx compact N`
- `/parallel profile create/show/add/list`
- `/proc run md` / `mdx` — Markdown in terminal / browser
- `/translate` — four modes: online/mini/offline/lm; key:value config syntax

## Key Commands

### Navigation
- `/find <pattern> [-f/-c/-i/--ext/ctx]` — pure Python os.walk+re search
- `/tree [path] [-d N] [ctx]` — Unicode directory tree
- `/map index` — build identifier index (NEVER in agent mode)
- `/map find`, `/map trace`, `/map deps`, `/map diff`, `/map idiff`
- `/map keyword index` / `/map keyword extract <text> [-f/-a/-n/-c]`

### Editing
- `/insert <file> <line> [code]`
- `/save <file> [code] [overwrite|-ab|-aa|add-suffix]`
- `/patch <file>` — apply SEARCH/REPLACE block
- `/fix <file>` — AI proposes LINE N: fix

### Context / session
- `/ctx clear N` — remove last N messages (fixed)
- `/param num_ctx <N>`, `/format <description>`, `/bkup save`
- `/role <persona>` / `/role show` / `/role clear`
- `/config save` / `/config load` — persist session state to `.1bcoder/config.yml`

### Post-processors (`/proc`)
- `/proc run <name> [-f <file>]` — one-shot, optionally from external file
- `/proc on <name>` — persistent after every reply
- Built-in: extract-files, extract-code, extract-list, collect-files, add-save, grounding-check, md, mdx, regexp-extract

### Agent system
- `/agent [-t N] [-y] <task>`; `/ask`; `/advance`; `/plan`; `/fill`
- `-y` works at ANY position
- Named agents in `~/.1bcoder/agents/<name>.txt`

### Parallel
- `/parallel "prompt" profile <name>` or inline workers
- Profiles: `~/.1bcoder/profiles.txt` (global) + `.1bcoder/profiles.txt` (local)
- `profile create/list/show/add`

## Publication Plans
- JOSS (Journal of Open Source Software) — planned after adding tests + docs
- YouTube lecture series — planned
- Grant application (молодіжна наука) — open source status is a plus, not a problem
