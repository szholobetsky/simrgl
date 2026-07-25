---
name: /parallel redesign specification
description: Full spec for redesigned /parallel command — new syntax, collect:, ctx compact, aliases
type: project
---

## /parallel new syntax

```
/parallel [list: p1, p2, p3]  profile: name  [ctx: full|last|none]
          [file: path [-n N]]  [collect: compact [profile: name]]  [-> var]
```

### Key changes from old syntax
- No quotes required for prompts
- `profile:` instead of bare `profile` keyword
- `ctx:` instead of `--ctx`/`--last`/`--no-ctx` flags  
- `list:` prefix optional — comma-separated prompts work with or without it
- `$` and `~` expanded inline (no shell quoting issues)
- `{{vars}}` expansion works in prompts
- `file: path -n N` — split file into N chunks, one per worker; -n auto = chunks = worker count
- `===` separator in file also supported (reuse existing file: splitter logic)
- `collect: compact [profile: name]` — after all workers finish, run /ctx savepoint compact
- `-> var` — capture compact result into session variable

### collect: behavior
- Before running workers: automatically sets a ctx savepoint
- After all workers write to ctx: runs `/ctx savepoint compact [profile: name]`
- If `-> var` present: captures compact summary into named variable
- No file management needed — everything stays in ctx pipeline

### Examples
```
/parallel list: q1, q2, q3  profile: three-machines  ctx: last  collect: compact profile: small -> brainstorm_results
/parallel file: bigfile.txt -n auto  profile: cluster  collect: compact
/parallel $  profile: short  ctx: last
/parallel {{q1}}, {{q2}}, {{q3}}  profile: three-machines
```

## /ctx compact extension

```
/ctx compact [savepoint] [profile: name]
```

- `compact` — current behavior (full ctx, main model)
- `compact savepoint` — only messages since savepoint (alias for savepoint compact)
- `compact profile: small` — use external parallel worker for summarization
- `compact savepoint profile: small` — both

## aliases.txt updates

```
/short    = /parallel {{args}}  profile: short    ctx: last
/explain  = /parallel {{args}}  profile: explain  ctx: last
/small    = /parallel {{args}}  profile: small    ctx: last
/thinking = /parallel {{args}}  profile: thinking ctx: last
```

(quotes removed, profile: syntax, ctx: last for all)

## assist.py update
Output: `ACTION: /parallel $ profile: short ctx: last`

**Why:** ctx: last — the hint model sees only the last agent turn, not the full history
