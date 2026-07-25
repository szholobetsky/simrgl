# yasna — Implementation Plan

**Богиня долі, тче нитку пам'яті між сесіями розробника.**
Structured catalog of 1bcoder ctx files with search.

---

## Core concept

Problem: developer accumulates many ctx files, can't find the right one.
Solution: project folders + keyword/file index + search.

NO wake-up. NO spaCy. NO hooks. NO MCP in MVP.

---

## File extraction approach

NOT hooks in 1bcoder. NOT parsing ctx content for identifiers.
Instead: `yasna index` reads ctx files and extracts file paths
from 1bcoder COMMAND ARGUMENTS using regex:

```python
CMD_FILE_PATTERN = re.compile(
    r'^>?\s*(?:/read|/readln|/edit|/patch|/save|/insert)\s+([\w./\\-]+\.\w{1,6})',
    re.MULTILINE
)
```

This captures only files the user explicitly worked with — not tree output, not find results, not read file content. Clean and accurate.

---

## Package location

`C:\Project\codeXplorer\capestone\simrgl\yasna\`

```
yasna/
  pyproject.toml
  yasna/
    __init__.py
    __main__.py        ← python -m yasna
    cli.py             ← click commands
    db.py              ← SQLite operations
    indexer.py         ← regex extractor from ctx files
    search.py          ← yasna find logic
    config.py          ← ~/.yasna/config.json
  README.md
```

---

## Storage

```
~/.yasna/
  yasna.db             ← SQLite
  projects/
    ABC-123/
      index.txt        ← description + notes
      ctx_2026-04-01_ppcon.md
      ctx_2026-04-03_amort.md
    role-implementation/
      index.txt
      ctx_2026-04-07.md
```

---

## SQLite schema

```sql
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,      -- 'ABC-123' or any name
    description TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE keywords (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    keyword TEXT NOT NULL,
    source TEXT DEFAULT 'manual',  -- 'manual' only in MVP
    UNIQUE(project_id, keyword)
);

CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    filepath TEXT NOT NULL,
    source TEXT DEFAULT 'manual',  -- 'manual' or 'index'
    UNIQUE(project_id, filepath)
);
```

---

## CLI commands (MVP — 12 commands)

```bash
# Setup
yasna init                          # create ~/.yasna/ + yasna.db

# Project management
yasna switch <key>                  # set current project (create if not exists)
yasna status                        # current project + ctx count + keyword count
yasna list [--recent N]             # list all projects

# Saving content
yasna save [title] [--file path]    # copy ctx file into project folder with dated name
                                    # if no --file: reads from stdin (pipe from /ctx export)
yasna note "text"                   # append note to index.txt of current project
yasna file <filepath>               # manually add file to files table
yasna keyword add <word>            # manually add keyword to keywords table

# Indexing
yasna index [key|--all]             # parse ctx files in project folder,
                                    # extract file args from /read /readln /edit /patch /save /insert
                                    # add to files table with source='index'

# Search
yasna find <term>                   # search in: keywords, files, description, notes, ctx filenames
                                    # returns: list of projects with match location
yasna show [key]                    # list ctx files in project (current if no key)
                                    # output: number, date, title, size

# Keywords
yasna keyword list                  # show all keywords for current project
```

---

## yasna find — search logic

```python
def find(term: str) -> list[dict]:
    results = []
    # 1. keywords table — LIKE
    # 2. files table — LIKE
    # 3. projects.description — LIKE
    # 4. index.txt content — grep
    # 5. ctx filename — LIKE (dated name contains title)
    return results  # [{key, description, match_in, snippet}]
```

Output format:
```
ABC-123  [Виправити амортизацію]
  keywords: amortization, PPCon
  files:    finance/amort.py

role-implementation  [Реалізація /role команди]
  ctx file: ctx_2026-04-07_role_persona.md
```

---

## yasna show output

```
ABC-123  (3 ctx files)
  1. ctx_2026-03-15_ppcon_investigation.md      (12KB)
  2. ctx_2026-03-20_amort_rule18.md             (8KB)
  3. ctx_2026-04-01_oracle_stored_proc.md       (15KB)

Load: /ctx load ~/.yasna/projects/ABC-123/ctx_2026-03-20_amort_rule18.md
```

---

## 1bcoder integration

### Alias (add to ~/.1bcoder/config.yml or aliases.txt)
```
/project = /run yasna {{args}}
```

Usage:
```
/project switch ABC-123
/project save "PPCon investigation" --file .1bcoder/ctx/ppcon.md
/project keyword add PPCon
/project find PPCon
/project show
/project index
```

### No hooks needed
File extraction happens via `yasna index` from saved ctx files.
User runs it manually after saving a ctx, or periodically.

---

## Dependencies

```toml
[project]
name = "yasna-memory"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
]

[project.scripts]
yasna = "yasna.cli:main"
```

Zero ML dependencies. sqlite3 and re from stdlib.

---

## index.txt format

```
Description: Виправити розрахунок амортизації в legacy Oracle модулі
Notes:
  PPCon відповідає за завантаження конфіга — не чіпати без Василя
  Legacy stored procedure — finance_pkg.get_amort(account_id, period)
```

Keywords and files are stored in SQLite only (not duplicated in index.txt).

---

## MVP Roadmap

### v0.1 — done when:
- [ ] `yasna init`, `switch`, `status`, `list`
- [ ] `yasna save`, `show`
- [ ] `yasna note`, `file`, `keyword add/list`
- [ ] `yasna find` — search across keywords + files + description + notes + filenames
- [ ] `yasna index` — regex extraction from ctx files
- [ ] 1bcoder alias `/project`
- [ ] README

### v0.2 — later:
- [ ] MCP server (FastMCP, same pattern as simargl)
- [ ] `yasna wake-up` (if proves useful)
- [ ] FTS5 for ctx content search
- [ ] `yasna ont` ontology commands

---

## Key design decisions

1. **No hooks in 1bcoder** — yasna is standalone, zero 1bcoder modification
2. **No auto-extraction at index time from random content** — only from command arguments
3. **Regex pattern** extracts file from command args: `/read path/to/file.py` → `path/to/file.py`
4. **Project key** — any string, not just Jira keys (e.g. 'role-implementation', 'ppcon-debug')
5. **ctx files** are copied INTO ~/.yasna/projects/<key>/ on `yasna save` with dated names
6. **Search** returns project list, not ctx content — user then picks ctx to load

---

*Plan written: 2026-04-08*
*Context: session with implementation ready to start next session*
