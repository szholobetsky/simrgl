---
name: 1bcoder keybinding restrictions on Windows
description: Ctrl+S and Ctrl+Q intercepted by pyreadline3 on Windows before prompt_toolkit
type: feedback
originSessionId: 457af3c4-1663-4f2f-a188-730d645cdea6
---
Ctrl+S and Ctrl+Q cannot be used as keybindings inside prompt_toolkit on Windows with pyreadline3 — they are intercepted at the Windows Console API level before prompt_toolkit receives them. Ctrl+S triggers `forward-i-search` mode, Ctrl+Q triggers XON resume.

**Why:** pyreadline3 hooks at a lower level than POSIX tty, so it affects even prompt_toolkit raw mode sessions.

**How to apply:** When choosing keybindings for prompt_toolkit in 1bcoder, avoid Ctrl+S and Ctrl+Q. Also avoid Ctrl+C (SIGINT), Ctrl+Z (SIGTSTP), Ctrl+D (EOF — already used for submit). Safe options for custom bindings: Ctrl+G, Ctrl+T (but these are readline-reserved too). Text commands like `/end` and `/save` are the most reliable cross-platform approach.
