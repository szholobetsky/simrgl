# DeepAgent_Search: Iterative Search Over an Unbounded Space

**Component**: 1bcoder `/flow deepagent_search`
**Date**: 2026-09-30
**Status**: Draft / Concept — not yet built, crystallized from a chat brainstorm, no real runs yet

---

## Abstract

Every `deepagent_*` flow so far walks a space whose size is known before generation starts — a tree it's building itself, or a list handed to it from outside (`git diff`, a commit log). `deepagent_search` is for a different shape of problem: find N things that satisfy a set of criteria, in a space that is effectively unbounded (the open web) and cannot be enumerated or materialized up front. The concrete motivating requests, in the author's own words: *"find textbooks on big data covering neural networks/kafka/hadoop"*, *"find 10 hotels in Marmaris on the second line with a private beach"*, *"find 10 socket-775 CPUs with DDR3 support and integrated graphics"*. The shape is always the same — a query, some orthogonal sub-topics to also search, a checklist every result must satisfy, and a result list built up one small, fresh-context step at a time.

The core design constraint, stated directly by the author and taken as a hard requirement throughout: *a small local model (1B–4B, CPU) must be able to run this end to end.* Every existing web-facing flow in 1bcoder either already respects that constraint in a narrow way (`webanalys`'s one-page-at-a-time scoring) or explicitly doesn't (`/agent websearch`'s own header: *"for capable models (4B+)"*). `deepagent_search` composes the pieces that already work into an orchestration layer none of them have — multi-query search, criteria-driven verification, and a search that can run unattended for hours or days with honest, resumable progress.

---

## 1. What already exists, and what it actually does

Verified directly against the code in `C:\Projects\1bcoder\_bcoder_data\flows\` and `_bcoder_data\agents\` before designing anything new, because two assumptions made earlier in this design's own brainstorm turned out to be wrong on inspection:

| Flow/agent | What it really does | Relevant to this design because |
|---|---|---|
| `webask.py` | One DDG search, fetches up to `depth` pages, concatenates ALL of them into **one** prompt, one answer. | This is the single-prompt-ceiling failure transplanted to web search — the exact failure mode `deepagent_search` exists to avoid. |
| `/agent websearch` (`websearch.txt`) | Multi-turn ReAct loop: search → review → fetch → decide when "enough." Own header comment: **"for capable models (4B+)."** | Confirms directly, in the author's own prior work, that a self-directed multi-turn loop is not the small-model-safe shape — external, bounded steps are. |
| `webanalys.py` | For each DDG result, fetches the page and asks the LLM to rate it 0–5 with a one-line explanation. One bounded call per page. | This is already the right *shape* of per-candidate judgment — small, fresh context, one page at a time. `deepagent_search`'s own per-candidate verdict call (§4) is a direct generalization of this, with a criteria checklist instead of a bare 0–5 scale. |
| `webcrawl.py --mode llm` | "2 calls per page (filter relevance, then extract)," already writes CSV. | Proof the filter-then-extract split already works in this codebase — though §4 below merges it back into one call once the checklist is externally structured (`criteria:`), since a structured template doesn't need two round trips (same reasoning `deepagent_spec`'s own WHO/WHAT/SMART template uses: one call, several structured fields). |
| `webindex.py` | **Corrected after re-reading the full file, not just the first 100 lines**: BFS-crawls one seed URL, saves pages as `.txt`, runs `simargl index`. **No LLM call anywhere in this file.** Ranking, if any, happens later and separately, via `simargl retrieve`'s embedding cosine similarity — not here. | An earlier draft of this same design session mis-attributed LLM page-rating to this flow; the real rating flow is `webanalys.py`. Recorded here so the mistake isn't repeated. `webindex` is a different tool for a different job (crawl and RAG-index one known site), not a relative of `deepagent_search`. |
| `_web_ddg_search` (in `chat.py`) | The actual search backend already wired in — DuckDuckGo, not Google. Supports `"exact phrase"`, `-exclude`, `site:`, `filetype:`, `intitle:`, `inurl:`, `region:` operators (documented in `websearch.txt`). | `filetype:`/`site:` are directly useful for later, project-specific instances of this flow (a literature search would want `filetype:pdf`) — not designed into the core flow itself, just noted as available. |

Nothing here is discarded — `deepagent_search` is the orchestration layer that was missing, composed mostly of things already proven, not a new judgment mechanism from scratch.

---

## 2. Six parameters, one job each

Worked out over several rounds of a real example, converging from an initial (wrong) framing as "coverage" (`list:`/`cover:`, borrowed from `deepagent_foreach`'s facet-completeness idea) to the actual, simpler shape below, once it became clear the author wanted **query generation**, not coverage tracking.

```
/deepagent_search <root query, free text>
  aspect:   <comma-separated orthogonal sub-topics/branches>
  criteria: <comma-separated must-check items — applied uniformly to every candidate>
  add:      <N — LLM-generated expansion queries per branch, default small (2–3)>
  limit:    <N — results fetched per individual query>
  collect:  <csv | md | txt>
```

| Parameter | Role | What it is *not* |
|---|---|---|
| root query | The base intent, free text. May already contain some of what belongs in `criteria:` — that overlap is fine, see §3. | Not itself split into sub-parts; this is the one thing the user writes in natural language. |
| `aspect:` | Independent search branches — different regions/sub-topics that each get their **own** full query→expand→fetch chain. Multiplies the whole pipeline, once per aspect. | Not a coverage requirement ("must find something for each aspect") — there is no completeness check, no radogast-style tracking. An aspect branch that finds nothing just finds nothing. |
| `criteria:` | A flat checklist, applied the same way to **every** candidate regardless of which query/aspect found it — the per-candidate verification rubric (§4). Optionally scored via a structured decision-model call instead of free text — see `--jev`, §4.1. | Not automatically mixed into the query text (§3) — that's the user's own call, left out of the engine entirely. |
| `add:N` | Per aspect-branch, after the branch's own first query returns, one LLM call reads the result snippets and proposes `N` refined/expanded queries grounded in real signal from that search — **pseudo-relevance feedback** (a real, established IR technique — Rocchio-style query expansion from top-result terms, not an invented heuristic). Nested inside each aspect. | Not a synonym list supplied up front — the queries are generated *from* live search results, not guessed before searching. |
| `limit:N` | How many results get fetched for one individual query (the root query, or one `add:`-generated expansion query). The one truly mechanical, mundane bound. | Not a total-run budget — see §6 for why no such cap is enforced. |
| `collect:` | Output shape, and — for `md` specifically — also changes what gets generated per candidate (an added reasoning paragraph), not just how the result is serialized. Precedent for format choice affecting generation depth: `deepagent_tree` (one sentence per section) vs. `deepagent_tree_md` (full paragraph) is exactly this same move, already established in `DEEPAGENT_TREE.md` §2. | Not itself a source of LLM calls beyond the one per-candidate verdict call (§4) — the final render is a deterministic compile pass (§5), same as `deepagent_task` over `deepagent_spec`. |

Total query count for one run: `len(aspects or [None]) × (1 + add)`. Total planned fetch slots: that, times `limit`. Both computable before the run starts — this matters for §7 (progress reporting) and is the reason an earlier, dynamic "keep searching until covered" design (§9, superseded) was dropped in favor of this fixed, countable shape.

---

## 3. `criteria:` is not folded into the query automatically

Considered and rejected: auto-appending every `criteria:` item to the generated search-engine query text, to bias results toward pages likely to mention them. Rejected because (a) jamming many criteria terms into one search-engine query risks an over-narrow query returning nothing, and (b) the user may already have written some or all of the criteria straight into the free-text root query, and a mechanical auto-append would double them clumsily or fight the user's own phrasing.

**Decision: `criteria:` is used only for per-candidate verification (§4), never auto-mixed into query text.** Whether and how to fold a criterion into the query itself is left entirely to the user, in the root query's own free text. Documentation should say this explicitly, and should also say plainly that **a criterion may deliberately repeat something already in the root query** — that's not redundancy to avoid, it's the only way to get an honest, isolated check of exactly that one point against exactly this one candidate, rather than trusting the search engine's own relevance scoring for it.

---

## 4. The per-candidate pipeline: one LLM call, not several

An earlier round of this design proposed two or three separate calls per candidate (a relevance filter, then a structured extract, mirroring `webcrawl --mode llm`'s two-call shape). Once `criteria:` exists as an explicit, structured checklist, that split is unnecessary — the same reasoning `deepagent_spec`'s own template uses (one call fills several structured fields, WHO/WHAT/SMART/INVEST all at once, §3 of `DEEPAGENT_SPEC.md`) applies here:

1. **Mechanical, no LLM**: fetch the URL, extract plain text (HTML only for v1 — see §9, open question 1).
2. **One LLM call, fresh context**: page text + the full `criteria:` list + the root query/task → produce (a) one short reasoning paragraph, (b) a match score per criterion (0–100%, not just yes/no — matches the worked example's `s775 - 100% ddr3 - 0%`).
3. **Mechanical, no LLM**: append one line to the raw result log (§5) — the candidate's URL, title, which query/aspect found it, the reasoning text, and the per-criterion scores, exactly as they came back from step 2. No formatting, no numbering, no ranking happens here.

That's the whole per-candidate cost: **one bounded LLM call, fresh/empty context, per candidate** — every other number in this pipeline (how many candidates total) is a mechanical consequence of `aspect`/`add`/`limit`, not additional LLM cost.

### 4.1 `--jev`: structured rubric scoring instead of free-text percentages (concept, not built)

Noted here the same day it became available, so it doesn't get lost — **unverified, no real run, treat as a sketch, not a design.** Ollama 0.35 (released 2026-09-30) added a decision-model API, `/v1/systemone`, based on TypeSafe's Jev API — a genuinely different shape from chat completion: you send `state` (the text to judge) plus 1–64 named `questions`, each typed `choice` (pick one, with a probability per option), `noul` (probability a condition is true), or `score` (a level on an ordered rubric, returned as a probability-weighted number plus a human-readable legend). The response is structured numbers, not prose that then has to be parsed. Three models exist at launch: Bespoke Labs' `nimble` (9B) and Together AI's `tev1` in 4B and 0.8B — the 0.8B `tev1` (812MB) is, as of this writing, the closest thing to a genuinely 1B-class Jev model.

**Why this matters for §4 specifically**: the current verdict call asks the model to produce per-criterion percentages *inside free text* (`s775 - 100% ddr3 - 0%`) — exactly the kind of formatting a small model can get subtly wrong (misaligned numbers, a skipped criterion, inconsistent scale). A `score` question per `criteria:` item sidesteps that class of failure entirely: the number comes back structured, from a model purpose-built to answer exactly this shape of question, not coaxed out of a chat model's prose.

**Proposed shape, `--jev` flag, engine-internal only — no change to the `criteria:` syntax itself**: when set, step 2 of §4's pipeline compiles each `criteria:` item into one `score` question (template: *"How well does this text confirm: {criterion}?"*), sends them all as one `/v1/systemone` request (`state` = the fetched page text), and writes the returned scores straight into `results.jsonl`'s `criteria` field — no free-text parsing at all for that part.

**What's still open, not resolved by this sketch:**
- **Two calls, not one, probably.** `/v1/systemone` answers typed questions — it doesn't write prose. The reasoning paragraph `collect:md` wants (the worked example's *"Цей процесор поєднує..."*) has no obvious home in a `score`/`choice`/`noul` answer. Either `--jev` mode drops the prose paragraph and renders each `score` question's own returned *legend* instead (keeps the one-call property, but changes the artifact's texture — a Q&A table instead of a paragraph), or it keeps a second, small chat call just for the paragraph, trading the "one call per candidate" property (§4) for better-structured scores. Not decided.
- **A second, separate model setting is implied.** A Jev-capable model (`tev1`/`nimble`) is not the same model `--model`/Settings already points at for everything else in this flow (and the rest of the `deepagent_*` family) — `--jev` needs its own model selector, the first place in this whole family where a flow would genuinely need two different models configured for two different calls in the same run. No precedent for this yet anywhere in 1bcoder's existing config surface.
- **Version-gated.** Requires Ollama 0.35+ specifically — every other flow in this family works against any reasonably recent Ollama. `--jev` would need to detect/report a clear error on an older Ollama rather than fail confusingly against a missing endpoint.

---

## 5. The collector: raw log now, formatted artifact later

Directly answers the question that triggered this section of the design: *how many calls, and how do we even assemble something like the worked `#3. Intel Celeron N630...` block without asking the model to get formatting/numbering/ranking right inside a single, fresh-context call that has no idea what candidate #1 or #2 looked like?*

**Answer: don't ask it to.** Split generation from compilation, the same split `deepagent_task` already makes over `deepagent_spec` (§5 of `DEEPAGENT_SPEC.md`: *"the deep artifact is generative, the flat rollup is a compiler, not a second LLM pass"*):

- **Raw log** (`results.jsonl`, append-only, written immediately after each candidate's single verdict call, step 3 above): one JSON object per line — `aspect`, `query`, `url`, `title`, `reasoning`, `criteria: {name: pct, ...}`. JSONL specifically, not CSV, at this stage — a reasoning paragraph can contain commas and quotes freely; nothing here needs escaping discipline because nothing here is the final rendered artifact yet.
- **Compile pass** (`collect:`, zero LLM calls, deterministic, re-runnable any number of times over the same log without re-searching anything):
  1. Read every line of `results.jsonl`.
  2. Compute `rank` per candidate — the mean of its criteria percentages. **Arithmetic, not a model judgment** (the worked example's `rank: 0.5` is exactly `(100 + 0) / 2` — confirmed this is simple averaging, not something that needs its own LLM call).
  3. Sort descending by rank, number the results `#1, #2, #3, ...`.
  4. Render per `collect:` type:
     - `md` — the numbered-block format from the worked example (title line, reasoning paragraph, criteria-score line).
     - `csv` — via Python's real `csv` module (handles embedded commas/quotes correctly per RFC 4180 automatically — this was flagged as a worry during the brainstorm and isn't actually a design problem, just a reminder not to hand-join strings with commas).
     - `txt` — a plain flat list, no reasoning paragraph (matches the family's existing convention that a leaner output format also means less gets generated for it, not just less gets rendered).

---

## 6. No enforced budget cap — deliberately

An earlier round of this design proposed a mandatory total-fetch budget (mirroring `deepagent_spec`'s external-depth discipline). Explicitly rejected by the author, for a reason worth recording precisely because it's a real difference from every other `deepagent_*` flow's cost profile: **this runs against a local model.** There is no per-token bill accumulating — the only real cost is wall-clock time, and the author's own stated use case is letting a run go overnight, or across a weekend, unattended. Capping `add:`/`limit:` mechanically would protect against a cost that, for this specific runtime, doesn't exist.

What actually matters instead, and is a real requirement: **visible, steady progress, and safe interruption** (§7). The guidance that replaces a hard cap is documentation, not code: examples in the flow's own `--help`/docstring should use small numbers (`add:2` or `add:3`), so a first-time user doesn't accidentally ask for `add:10` across five aspects with `limit:20` and only discover the arithmetic (`5 × 11 × 20 = 1100` fetches) after the fact. If they do ask for that deliberately — that's the point of not capping it.

---

## 7. Progress and resume

Because total planned fetch slots (`aspect × (1+add) × limit`) is known before the run starts (§2), a progress counter (`45/330, 32%`) is free — no estimation needed, just a count against a known denominator. What the denominator actually counts needs to be stated precisely, because deduplication (the same URL found by more than one query) means "fetches attempted" and "unique candidates" are different numbers: report both — `45/330 fetched, 12 accepted` — rather than one figure that looks like it's stalled once duplicates start getting skipped.

**Resume, not continue** — the family already has this exact naming split, documented at length in `runner.py` (Alkonost's launch layer) for `deepagent_md`/`deepagent_tree`: `resume` finishes an interrupted run exactly as originally planned; `continue` deliberately grows an already-finished run with something new. What's being asked for here is `resume`. A hypothetical `deepagent_search continue <run_dir> add:+5` (ask for more after a clean finish) is a different, later feature, not designed here.

The one genuinely new wrinkle `resume` has here, with no analogue in the tree-shaped flows: **the `add:`-generated expansion queries must be frozen the moment they're generated, never regenerated on resume.** A tree node's content is deterministic-enough to re-derive from its parent section on `redo`; an LLM call that proposes "10 better queries based on this search's results" is not guaranteed to produce the same 10 queries twice, and regenerating them on resume would silently abandon or duplicate whatever partial fetching already happened against the *original* 10. So the expansion-query *text* itself is part of the persisted run state, written once, right after the pseudo-relevance-feedback call returns — before any of those queries are executed.

`results.jsonl` (§5) already carries enough to resume cheaply — its `url` field is the dedup key, and "already in the log" is "already processed," the same "resume = which files already exist on disk" principle the whole family already relies on. What it does *not* carry is the frozen expansion-query list itself (that's metadata about the plan, not a result), so a small sidecar — `_deepagent_search_state.yaml`: root query, aspects, criteria, limit, add, and per-aspect the frozen list of query texts with a `done`/`pending` + `fetched` count each — is needed alongside the log, not instead of it.

---

## 8. What this is not

- **Not PDF-capable.** Explicitly out of scope by the author's own instruction this session — v1 only handles HTML pages. A literature-search instance of this flow (the original motivating use case) would need PDF text extraction as separate, later work; nothing in this design assumes it exists.
- **Not validated against a real run.** Every number and shape here (the single-verdict-call-per-candidate claim, the rank-as-mean formula, the resume sidecar shape) is design, not measurement — unlike `DEEPAGENT_TREE.md`, which is backed by five real generation runs. Treat defaults as a starting guess.
- **Not a coverage/completeness tool.** An earlier version of this design (`list:`/`cover:`, borrowed from `deepagent_foreach`) tried to guarantee every aspect gets represented in the output, the way `radogast`'s `key_terms` coverage check works elsewhere in this project. That idea was deliberately dropped — `aspect:` here only generates more queries, it makes no promise about what they'll find.

---

## 9. Open questions

1. **HTML-only fetch/extraction for v1** (§8) — confirmed deliberate, not an oversight, but means any instance of this flow that needs PDF/DOCX sources needs a separate extraction step bolted on later, not assumed solved here.
2. **How many `criteria:` items is too many for one verdict call?** §4 merges all criteria into one structured call regardless of count. At what point (5? 10?) does a single call's output quality degrade on a 1B model, and is a split-by-criterion-group fallback needed? Untested.
3. **`add:` grounding signal** — §2 specifies snippet-only pseudo-relevance feedback (cheap, no page fetch needed before generating expansion queries). Whether snippets alone give the LLM enough signal to propose genuinely better queries, versus needing at least one fully-fetched page from the root query first, is unverified.
4. **Per-aspect vs. global expansion-query budget** — `add:N` is specified per aspect (§2, "nested"). Whether a user might reasonably want a single global `add:` pool spread unevenly across aspects (more expansion queries for an aspect that's finding less) is unresolved; the simpler per-aspect-fixed model is the default for v1 specifically because it keeps the total-fetch-count arithmetic in §2/§7 exact and trivial.
5. **Criteria-in-query-text guidance** — §3 leaves this entirely to the user. Whether the flow's own `--help` should show a worked example of *good* practice here (e.g., fold only the one or two most search-engine-friendly criteria into the root query, verify the rest only in §4's per-candidate pass) is a documentation task, not an engine decision, but isn't written yet.
6. **`collect:` formats beyond csv/md/txt** — not discussed this session; `deepagent_foreach`'s own `collect` vocabulary (`DEEPAGENT_FOREACH_REVIEW.md` §2.2) may be the right place to eventually unify this rather than growing a second, separate set of output-format conventions.
7. **`--jev` (§4.1) is a same-day sketch, not a design** — one-call-vs-two tradeoff unresolved, needs its own model-selection setting (first case in the family of a flow needing two configured models at once), and is gated on Ollama 0.35+. Needs a real run against `tev1:0.8b` before any of §4.1's claims can be trusted.

---

## 10. Implementation location (not yet built)

- `C:\Projects\1bcoder\_bcoder_data\flows\deepagent_search.py` — independent clone, per `AGENTS.md`'s submodule-migration table; never the pinned `C:\Projects\simrgl\1bcoder\` snapshot.
- Reuses `chat._web_ddg_search` (search backend) and `chat._web_strip_html` (HTML extraction) unchanged — both already used by `webask.py`/`webindex.py`/`webanalys.py`, no new dependency for v1's HTML-only scope.
- Output: `.1bcoder/search/<run>/results.jsonl` (raw log, §5) + `.1bcoder/search/<run>/_deepagent_search_state.yaml` (frozen expansion queries + per-query progress, §7) + `.1bcoder/search/<run>/result.{csv,md,txt}` (compiled artifact, regenerable from the log alone).

---

## 11. Related documents

- [`DEEPAGENT_FOREACH_REVIEW.md`](DEEPAGENT_FOREACH_REVIEW.md) — origin of the `collect:` terminal-operation vocabulary reused in §2/§5, and of the `list:`/`cover:` coverage idea this design considered and dropped (§8).
- [`DEEPAGENT_SPEC.md`](DEEPAGENT_SPEC.md) — §3's structured-template-in-one-call precedent, reused in §4; §5's "generative artifact vs. deterministic compiler" split, reused wholesale in §5 of this document.
- [`DEEPAGENT_TREE.md`](DEEPAGENT_TREE.md) — §2's precedent for output format changing generation depth, reused for `collect:md` vs. `collect:txt` in §2.
- [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md) — §9's `resume`/`continue` naming split, reused precisely in §7; the CLI-level `redo`/`--review` precedent this document does not yet adapt (no per-candidate `redo` designed here — open for a future pass).
