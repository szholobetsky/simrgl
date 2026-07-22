---
name: 1bcoder default files location
description: Where to edit 1bcoder default configs — always _bcoder_data/, never ~/.1bcoder/
type: feedback
originSessionId: 457af3c4-1663-4f2f-a188-730d645cdea6
---
Always edit defaults in `C:\Project\1bcoder\_bcoder_data\` — NOT in `~/.1bcoder/`.

**Why:** `_bcoder_data/` is the package source that gets shipped in the wheel and bootstrapped to `~/.1bcoder/` on first run. Editing `~/.1bcoder/` directly only affects the current machine's install, not the distributed package.

**How to apply:** When adding aliases, agents, scripts, procs, or any default config for 1bcoder — always write to `C:\Project\1bcoder\_bcoder_data\<subdir>\<file>`.
