---
name: project-limited-context
description: "/ctx window sliding-window context concept — RS/BM25/DP/TextRank algorithms to retain gist beyond the ctxtimer-measured context ceiling, for 1bcoder and vyrii"
metadata: 
  node_type: memory
  type: project
  originSessionId: 59f4e9c6-779f-4f0a-9009-ff73ef6ea5a3
---

# LIMITED_CONTEXT — sliding-window context management

Concept doc: `C:\Project\codeXplorer\capestone\simrgl\concepts\LIMITED_CONTEXT.md` (written 2026-07-06, not yet implemented).

**Why:** [[project_ctxtimer|ctxtimer]] showed KV-cache stops helping past a hard per-model ceiling (3-8K tokens on weak hardware like [[project_pantheon_hardware|Пантеон]]). If KV-cache reuse is dead past that point anyway, there's no reason to keep feeding the model the full accumulated conversation — better to explicitly window it, accepting mid-conversation detail loss in exchange for effectively unbounded conversation length.

**Command syntax (increasing complexity):**
```
/ctx window last:N                              # just last N tokens
/ctx window off
/ctx window first:M last:N                      # fixed head (where the conversation started) + sliding tail
/ctx window first:M last:N mid:rs   limit:K      # + reservoir sampling from the middle
/ctx window first:M last:N mid:bm25 limit:K      # + BM25 retrieval, query = current message
/ctx window first:M last:N mid:dp   limit:K      # + budgeted knapsack (DP) packing
/ctx window first:M last:N mid:tr   limit:K      # + TextRank/LexRank graph ranking
```

**Key reuse finding:** `chat.py:167` in 1bcoder already has `_fts_rank()` — BM25 via in-memory SQLite FTS5, currently used for file-content RAG ranking. Directly reusable for `mid:bm25` (index middle messages as documents, query = current user message).

**Architecture decision:** implement all 4 command variants + all 4 mid-algorithms in 1bcoder FIRST (sandbox for comparing methods on real long conversations), THEN port to vyrii. Unlike ctxtimer (which needed a fully separate vyrii module due to error-handling convention differences), this one is a message-list preprocessing step before `stream_chat()`/`complete()` — same for all 3 vyrii UIs — so it belongs centralized in `vyrii/engine.py` (or a sibling `vyrii/ctxwindow.py`), called once from each backend, not duplicated per-UI.

**Existing 1bcoder mechanism to reuse for LLM-based summarization:** `/ctx compact profile: <name>` already summarizes via an external model (not the main chat model) — solves the "don't disturb the main model's KV-cache/VRAM" requirement the user cares about, already exists, just needs to be used alongside `/ctx window` rather than instead of it.

**Research angle:** success here is about answer-quality retention, not speed (unlike ctxtimer). Proposed eval protocol mirrors exp2/exp3's MAP/MRR methodology: long conversation with facts buried mid-conversation, ask questions requiring recall, compare full-context vs. windowed variants — a secondary, citable applied contribution about practical context management in the SIMARGL tool ecosystem.

**Status:** concept only, nothing implemented. Implementation order specified in the doc: (1) `last:N`/`off` MVP in 1bcoder → (2) `first:M last:N` → (3) `mid:rs` → (4) `mid:bm25` (reuse `_fts_rank`) → (5) `mid:dp`/`mid:tr` → (6) port to vyrii `engine.py` → (7) quality-eval protocol.
