---
name: Windows Terminal drops lines at scroll boundary
description: Windows Terminal loses lines (banner rows, user queries, LLM response first/last lines) when content scrolls past the top edge
type: feedback
originSessionId: 457af3c4-1663-4f2f-a188-730d645cdea6
---
Windows Terminal has a known behavior of dropping lines when content is pushed past the upper edge of the viewport during scrolling. Affected content: ASCII art banner rows (specifically rows with box-drawing chars like ╚, ═), user input lines, and first/last lines of LLM responses.

**Why:** Likely related to how Windows Terminal handles "ambiguous width" Unicode characters or soft-wrapped lines in its scrollback buffer. Other terminals (Eclipse, PowerShell console) do not exhibit this.

**How to apply:** Do not investigate or try to fix this unless the user explicitly asks. It is a known Windows Terminal quirk, not a 1bcoder bug. Do not remove the blank line after `_print_status()` or other existing fixes as a workaround.
