---
name: Show findings before editing
description: Report what existing code already does before proposing or applying changes
type: feedback
---

Before making any code change, report what the relevant existing code already does — especially if it partially covers the need. Don't silently skip over existing logic and just apply the fix.

**Why:** User caught that SKIP_DIRS already had explicit hidden dir entries (.git, .1bcoder, etc.) — this context should have been surfaced before editing, not discovered after.

**How to apply:** When reading code prior to an edit, if you find existing logic related to the change (skip lists, filters, guards), mention it explicitly: "X is already handled by Y, we just need to add Z."
