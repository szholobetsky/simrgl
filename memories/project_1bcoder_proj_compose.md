---
name: 1bcoder /proj and /ctx compose — session 2026-04-09
description: New commands built in session: /proj project management, /ctx compose context builder, /ctx compact N
type: project
---

## /proj — project context management

Built into 1bcoder (chat.py), NOT a standalone package.
Storage: `<cwd>/.1bcoder/projects/<key>/` — local to working directory like .git/

**Commands:**
- `/proj set <key>` — activate project, create folder + project.txt
- `/proj status` — show active project + project.txt contents
- `/proj list` — all projects newest first, active marked *
- `/proj save <file>` — save current ctx to project folder
- `/proj load <file>` — load ctx from project folder (fuzzy filename match)
- `/proj show` — list ctx files in active project
- `/proj find <term> [-f|-c]` — search all projects in cwd; -f fast (project.txt + filenames), -c with content + line numbers
- `/proj keyword add <k1,k2>` — add keywords to project.txt
- `/proj file add <f1,f2>` — add files to project.txt
- `/proj index` — regex-extract file paths from /read /edit /patch etc. in ctx files

**project.txt format:**
```
Description: <text>
Keywords: k1, k2
Files:
path/to/file.py
```

**Auto-restore:** active project saved in config via `/config save`, restored on startup.
**Config bug fixed:** `_write_config_yml` was hardcoded and dropped `active_project` — added explicit write.

## /ctx compose — context composer

**Core idea:** pre-build thin "knowledge path" ctx files (tree → read chains), compose multiple paths into one merged ctx. Content-level dedup: identical message blocks appear once → shared tree root kept, unique branches appended.

**Queue workflow:**
- `/ctx compose add <file|N,M|all>` — add to queue (numbers from last /proj find)
- `/ctx compose list` — show queue with sizes + running total
- `/ctx compose clear` — clear queue
- `/ctx compose run [out.txt]` — merge with dedup → file or load into context
- `/ctx compose f1 f2 f3` — direct mode, no queue

**Path resolution:** bare name → .1bcoder/ctx/ → .1bcoder/projects/<key>/ → as-is

**`/proj find` integration:** results numbered [1],[2]... stored in `_last_find_results`. Hint shown: "use /ctx compose add <N,M|all>"

## /ctx compact N

New variant: compact last N messages in place (not whole context).
`/ctx compact 1` — compress one verbose LLM reply without touching rest of context.

## Other fixes in same session
- `DEFAULT_AGENT_TOOLS` now includes `tree` and `find`
- `AGENT_SYSTEM_BASIC` hints to start with `/tree` if structure unknown
- Startup tip line: `/agent ask <question>  /read <file>  /tree to explore`
- `/think` HELP_TEXT: added note about `/param think_exclude` for config persistence
- `/config save` now includes `active_project` key

**Why:** context limits on 1B models require surgical context management. Pre-built knowledge paths + composition = LLM wakes up knowing exactly the files it needs.
