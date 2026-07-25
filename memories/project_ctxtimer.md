---
name: project-ctxtimer
description: The /flow ctxtimer tool built for 1bcoder — empirically measures max safe context length per model/hardware before timeout
metadata: 
  node_type: memory
  type: project
  originSessionId: 59f4e9c6-779f-4f0a-9009-ff73ef6ea5a3
---

# ctxtimer — empirical context-limit measurement for 1bcoder

Built 2026-07-05. Concept doc: `C:\Project\codeXplorer\capestone\simrgl\concepts\CTXTIMER.md`. Implementation: `C:\Project\1bcoder\_bcoder_data\flows\ctxtimer.py`, base test text: `C:\Project\1bcoder\_bcoder_data\ctxtimer\base_prompt.txt` (~17K tokens, English, transformer/LLM technical content).

**Why:** Local models on CPU-only hardware ([[project_pantheon_hardware|Пантеон]]) hit a hard wall where the model stops responding within timeout once context crosses some threshold (often 3-10K tokens depending on model architecture — dense vs sparse MoE saturate differently). No universal formula predicts this threshold; it must be measured empirically per model+hardware+timeout combo.

**What it does:** Slices the base prompt to N tokens (1 token ≈ 4 chars, matching 1bcoder's own estimate), sends it through the currently-configured `chat._stream_chat()`, and checks whether the model returns a first token before timeout. Supports `--seq` (test start, start+step, ... stop at first failure) and `--bin` (binary search between --start/--end) modes, both parameterized by `--step` (default 1000 tokens). Results append to `.1bcoder/ctxtimer/report.csv`; `/flow ctxtimer report` displays them as a table, `/flow ctxtimer report clear` deletes the file (with y/N confirmation).

**How to apply:** Flow is provider/model/OS-agnostic — it inherits whatever model+timeout is active in 1bcoder via the `chat` object, no separate config needed.

## Key bug found and fixed (2026-07-05)
`chat._stream_chat()` (chat.py:3340-3513) never lets timeouts or Ctrl-C escape as Python exceptions — by design, for chat UX:
- Read timeout / connection error → caught internally, prints `"error: ..."` itself, returns `""`
- Ctrl-C → caught internally, prints `"[interrupted]"` itself, returns `None`

First two fix attempts (by a Haiku-run session) wrapped `try/except` around the `_stream_chat()` call and manipulated stdout — this could never work because no exception ever escapes that function. The real fix: inspect the **return value** (`None` = was Ctrl-C, re-raise `KeyboardInterrupt`; `""` = check captured stdout text for `"error:"` to distinguish real failure from a genuinely empty reply) instead of relying on exception handling. This is the actual root-cause fix, not a symptom patch — worth remembering as a case study on this codebase's error-handling convention (chat.py swallows exceptions internally, prints itself, returns sentinel values `None`/`""` — any code calling `_stream_chat()` must handle it this way, not by wrapping try/except).
