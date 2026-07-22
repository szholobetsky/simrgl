---
name: Legacy code caution — don't remove lines without strong justification
description: Every line in 1bcoder was added to fix a real bug. Do not delete code to fix a minor issue without explicit permission.
type: feedback
originSessionId: 457af3c4-1663-4f2f-a188-730d645cdea6
---
Don't remove existing lines of code (especially `print()` calls, guards, or fixes) to address a minor visual issue. Each line in 1bcoder was added to fix a real bug — removing it regresses that fix.

**Why:** User explicitly called this out: the blank line after `_print_status()` was added to fix the VS Code status line disappearing bug. Proposing to remove it as a "fix" for a different minor issue is wrong.

**How to apply:** Before deleting any line, ask: was this added intentionally? If yes — don't touch it. If the symptom is standard terminal behavior, say so and stop.
