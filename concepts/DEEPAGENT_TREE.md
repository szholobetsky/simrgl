# DeepAgent_Tree: Structured Decomposition Is Not Prose

## Abstract

`/flow deepagent_md` (see [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md)) recursively deepens a topic into a tree of markdown documents, one LLM call per node, state on disk instead of in context. Its own prompt asks the model to "write a detailed markdown analysis" — a framing calibrated for **prose**: an essay is free to revisit a theme from a new angle at any depth, a parent section doesn't have to be the literal sum of its subsections, and nothing requires siblings to be the same size or kind of thing.

That freedom is exactly wrong when the tree's purpose isn't a document to *read* but a structure to *build from* — a backlog, a module breakdown, a work-breakdown structure that `deepagent_spec`/`deepagent_task`/`deepagent_architect` will walk mechanically, assuming every leaf sits at a comparable grain and every branch owns a disjoint slice of the whole. `deepagent_tree` and `deepagent_tree_md` are forks of `deepagent_md`'s engine — same recursion, same resume/compose/web-research machinery — with the generation prompt replaced by an explicit, mechanically-testable contract for what "structured" means, instead of leaving it to how the task text happens to be phrased.

This document is the concept doc that was missing: it names the distinction, gives the formal contract, and — unusually for a concept doc written before a single line of code existed — is backed by five real generation runs (project `RRTrack`, four different local models) with measured failure rates, not projected ones.

---

## 1. The distinction: two genres, two contradictory quality criteria

Take two decompositions of superficially similar shape:

> **A.** "The Dramatism of Shakespeare's Plays" → discusses Sonnet 2 as a warm-up, compares *Othello* and *Romeo and Juliet*, circles back to Desdemona at the end.
>
> **B.** "Design a system that tracks how often a rarely-restocked household item is used, and notifies the user when the number of uses since the last restock suggests supply is running low."

Both look like "give the model a topic and let it write." But A is asking for **prose** — an argument, free to loop, revisit, and juxtapose across its own sections, because that is what makes an essay good. B is asking for **structure** — a thing that will be cut into tickets, specced, and built, where a good decomposition is judged by whether it can be handed to someone else, one atomic piece at a time, with each piece independently intelligible.

The two genres are not just different in tone; the definition of a *defect* inverts between them:

| | Prose (essay) | Structure (backlog/tree) |
|---|---|---|
| A theme revisited at multiple points | Often a *strength* (cumulative argument) | Always a *defect* (redundancy, see §3.2) |
| A section broader or narrower than its siblings | Fine — depth follows the argument | A *level violation* (see §2, HOMOGENEITY) |
| Parent doesn't equal the sum of its children | Normal — a section can gesture beyond its own subsections | A *defect* — the parent is no longer a real container (see §2, PART-OF) |
| Section N mentions something section N+2 will explain in full | Normal foreshadowing | A *floor violation* if it invents facts that belong to a level not yet reached |

`deepagent_md`'s prompt is honest and correct for the left column. It was never wrong for essays, hotel-room feature lists, or heat-equation walkthroughs — the actual failures that motivated this fork all came from asking that same prompt to produce the *right* column's genre instead, and getting graded — by the human reader, and later by `deepagent_spec`'s own template — against right-column criteria it was never asked to satisfy.

---

## 2. The formal contract (`_DECOMPOSITION_RULES`)

`deepagent_tree`/`deepagent_tree_md` replace `deepagent_md`'s "write a detailed analysis" framing with an explicit, numbered contract, reproduced here because the contract itself *is* the concept:

1. **PART-OF, NOT ABOUT** — a sub-part must be a genuine constituent of the parent, not a topic merely related to it. Test: *if every sub-part were removed, would the parent cease to exist?* If yes, the split is real.
2. **HOMOGENEITY** — every sub-part must be the same kind and size of thing as its siblings. A whole subsystem next to a single field is a level violation.
3. **CEILING** — do not restate, re-justify, or re-derive anything that belongs to an ancestor.
4. **FLOOR** — do not reach into anything that belongs to a level not yet decided (a sub-part's own internal breakdown, specific file names, method signatures).
5. **NO SMEARING** — each sub-part owns a distinct slice with zero overlap; a concept two siblings both need becomes its own sibling instead of being repeated.
6. **NO BARE HEADINGS** *(`deepagent_tree` only, see §3.1)* — every heading needs at least one real sentence under it; never a title with nothing below.

`deepagent_tree_md` keeps 1–5 and replaces 6 with **FULL ARTICLE**: a full paragraph of explanation is required per sub-part, not one sentence — same discipline, different verbosity contract. Neither variant touches `deepagent_md` itself, which remains the right tool whenever the actual desired output is prose.

Two structural additions, not part of the original five/six rules, were added after real runs exposed a gap the rules alone didn't close (§4):

- **Position breadcrumb** (`--noposition` to disable, on by default): every generation prompt is told its place — ancestor titles and their own one-line gist, plus numeric depth ("You are now at: class (step 4 of 4)"). Without it, a node has no way to judge whether it's already at maximum useful granularity.
- **Finalize instruction** (`--finalize ["<text>"]`, opt-in): at the deepest level, an explicit "this is the last pass before atomization" note, composed as universal-text-always-present *plus* caller-supplied text appended (never replacing the universal note — the layering is deliberate, see `deepagent_tree.py`'s own flag-parsing comments).

---

## 3. Empirical record: five runs, one project, four models

All runs share the same root task family (a household-inventory usage tracker) against project `RRTrack`, generated across a deliberate capability ladder: gemma4:26b → lfm2.5:8b-a1b (MoE) → gemma3:4b → gemma3:1b. Numbers below are measured, not estimated — grep counts and file-size distributions over the actual generated trees.

### 3.1 The bare-heading failure and its fix

First real run (`plan1`, `deepagent_tree`, gemma4:26b, before rule 6 existed): several nodes wrote `## N. Title` headers with *nothing* underneath. `deepagent_spec`'s own system prompt ("never invent unstated details") then correctly, honestly refused to spec them: *"Please provide the Source Text... I cannot invent unstated details."* Measured: **6/53 specs (11%) were refusals**, all traceable to empty source sections.

Root cause, read directly out of the generated tree, not inferred: `deepagent_tree`'s FLOOR rule ("don't reach into what's below this level") was apparently read by the model as license to be terse *in general*, not just about descendants — conflating depth-of-explanation (wanted, at the node's own level) with depth-of-decomposition (forbidden, below it). Rule 6 (NO BARE HEADINGS) exists specifically to separate those two axes and mechanically enforce the first even when a weak model blurs the distinction; `_bare_heading_titles()` re-checks generated content with the same header-detection regex as `_parse_sections`, and retries once with a targeted correction naming exactly which sub-parts are bare.

Second run (`plan2`, `deepagent_tree_md`, same task, same model, only the prompt's rule 6 replaced with FULL ARTICLE): **0/50 refusals.** Confirms the fix works via richer per-node content, not luck — same task, same model, only the prompt discipline changed.

### 3.2 The redundancy failure — worse than emptiness, harder to see

`plan4` (gemma3:4b, a richer task explicitly grounding the system in an MCP-server integration) produced a structurally *excellent*-looking tree — six homogeneous top-level branches, zero bare headings, zero spec refusals across 907 specs. And yet, read at the compact `tasks.md` rollup level (not visible from sampling individual specs — the redundancy only becomes visible when many sibling *titles* are seen side by side), branch 2 ("Usage Tracking Logic") turned out to independently re-derive the same five or six underlying concepts (fetch → count → restock-date → period → threshold → notify) under six *different* sibling headings, each restating the whole pipeline instead of owning one disjoint slice of it.

This is a **PART-OF violation dressed as valid structure**: no bare heading, no homogeneity mismatch between any *pair* of siblings taken in isolation — the defect only shows up as an aggregate pattern across all six. None of rules 1–6 catch it mechanically as written; catching it required a human to read the flattened title list, exactly the kind of check §5 (radogast integration, planned) exists to automate.

Contrast: branch 6 ("Item Selection & Configuration") of the same tree, same run, same model — five genuinely orthogonal children (UI search field, threshold config UI, notification preferences, MCP integration parameters, historical-data access) with almost no restatement. Same prompt, same model, wildly different redundancy outcome — the contract narrows the failure mode, it does not eliminate it.

### 3.3 The wrong-abstraction-level failure

Branch 1 of the same `plan4` tree ("Data Source Integration" → "MCP Server Connection Establishment") independently reinvented a raw TCP/DNS/TLS client from first principles — SYN/SYN-ACK/ACK packet exchange, DNS `A`/`AAAA` resolution, TLS cipher-suite negotiation — for a protocol (MCP) that has existing client libraries where "connect" is normally one function call. Structurally flawless (homogeneous siblings, every heading has real content, `deepagent_spec` produced clean, well-formed specs for all of it, including hallucinated-but-plausible details like a `400 Bad Request` HTTP status code for what is nominally a raw DNS lookup). The tree obeyed every rule in §2 and was still, in substance, wrong — because none of those rules encode *domain* knowledge, only *shape*.

Confirmation this was genuinely a knowledge gap and not scope confusion: the same tree's branch 6.4 ("MCC Server Integration Parameter" — sic, the model even misspelled the acronym) independently arrived at the *correct* abstraction level for the identical underlying concern — endpoint URL, auth credentials, schema mapping, rate limits, error handling — without any raw protocol detail. Two branches of the same run, same model, same "topic" (talk to the MCP server), one right, one wrong. This rules out "the model doesn't understand MCP at all" and points instead at something closer to per-branch framing luck.

### 3.4 The disguised-refusal failure (weaker model, worse-looking failure)

`plan3` (lfm2.5:8b-a1b, running against `deepagent_tree` *before* rule 6 existed in that codepath) produced specs that pass every structural check — WHO/WHAT/DESCRIPTION/ACCEPTANCE CRITERIA/BOUNDARIES, correctly ordered, correctly formatted — whose actual content is a polite refusal wearing the template: *"No actionable specification available due to missing source text."* This is strictly worse than `plan1`'s honest meta-commentary refusal, because it doesn't trip any of the refusal-phrase greps that catch the `plan1`-style failure; it has to be read to be caught. Traced to source: the underlying `item_*.md` node had three consecutive bare `## N.` headings with zero content — the same root cause as §3.1, on a model that responds to emptiness by dressing it up rather than declining outright.

### 3.5 The capability floor (gemma3:1b)

`plan5` (gemma3:1b, root task phrased with an explicit `"Provide implementation details"` instruction — itself a violation of the atomic-seed principle, see §4) exhibits two effects that need to be told apart:

- **Task-framing effect** (not a model-weakness signal): the explicit "provide implementation details" instruction dominates from `index.md` onward — full JSON schemas and a MongoDB choice appear at the *root* of the tree, not just at leaves. `root_task` is injected verbatim into every single node's prompt; anything compound or implementation-flavored in it propagates everywhere, immediately, regardless of model strength.
- **Model-weakness effect** (a genuine capability-boundary signal, previously undetected): **218/240 tree files (91%) open with chatty meta-commentary** ("Okay, here's a breakdown structured precisely as outlined...") that the prompt explicitly forbids, and which none of rules 1–6 check for (they check what's *under* a heading, not what precedes the first heading). **247/1255 sections (19.7%) are bare** even with the position breadcrumb active — well above `plan1`'s pre-fix rate, and gemma3:1b's own refusal rate at the spec stage (3.9%) is *lower* than gemma4:26b's early unpatched run (11%), which is not a sign of better quality: it means the small model is *worse* at recognizing when it lacks grounding and more prone to confidently fabricating content instead (one sampled spec invented a `Demand_Level__c` Salesforce-style field and a Tableau dashboard integration neither the source nor the task ever mentioned).

The preamble-violation finding is new: it is a distinct failure axis from bare-heading, uncaught by any existing rule, and — going by the 91% rate — the dominant failure mode at this capability floor, ahead of the emptiness problem the earlier rules were built to catch.

---

## 4. A related, separate principle: the atomic seed

Not a `deepagent_tree`-internal rule but a task-authoring discipline this project arrived at independently, and worth recording alongside the contract because the same underlying mechanism (`root_task` injected verbatim into every node, forever) explains both:

> "Драматизм творів Шекспіра" (clean, atomic) vs. "Драматизм творів Шекспіра на прикладі Ромео і Джульєтти, Отелло і сонету №2, причому спочатку розкажи про сонет, а потім порівняй..." (compound, pre-sequenced) — the second is "a crime against Ralph": it hands the recursive loop an already-decomposed, pre-ordered set of decisions in the one slot that's supposed to hold a single seed idea, and that compound instruction then re-fights the "decompose this one node" instruction at every depth of the tree, not just at the root.

Corollary: a genuine external constraint (a fixed tech stack, say) does *not* belong smeared into `root_task` either — it belongs in a separate, lower-priority channel (`list:` aspects, or `--focus` at the `deepagent_spec` stage) that doesn't compete grammatically with the primary framing sentence the way `root_task` does. `deepagent_spec`'s own code carries a scar from getting this wrong once already: `root_task` is deliberately *excluded* from its prompt entirely, after a real run showed the model treating stale root-task instructions ("never mention backend or database") as still-binding at a pipeline stage where they no longer applied.

---

## 5. Known gaps, not yet closed

- **Cross-sibling redundancy (§3.2) has no mechanical check.** The plan is an embedding-based one, reusing `radogast`'s existing `metrics/drift.py` (`sentence_transformers` + cosine angle) against sibling pairs instead of its native goal-vs-conversation framing, plus `Target.out_of_scope`'s existing trend-tracking for a mechanically-derived (not LLM-generated — see below) block-list per branch built from sibling titles. Undesigned in detail as of this writing: retry semantics for a caught duplicate are straightforward for parent-fit/out-of-scope (self-contained per node, same shape as rule 6's retry) but awkward for sibling-redundancy under `deepagent_tree`'s current parallel (BFS, `ThreadPoolExecutor`) per-level generation — a clean auto-retry wants sequential within-level generation instead, an open tradeoff (speed vs. clean auto-fix) not yet resolved.
- **Out-of-scope term lists should not be self-generated by the same model that will be checked against them**, for the exact reason §3.3 demonstrates: the model's blind spot (not knowing MCP has client SDKs) is precisely the kind of gap it cannot be expected to name as forbidden. Terms derived *mechanically* from sibling titles (no LLM involved, no circularity) are trusted for the redundancy axis; terms for the wrong-abstraction-level axis need either a small hand-curated, project-independent list or a genuinely different model as a second opinion — not the generating model grading its own output.
- **No verification against external reality exists in either variant.** Every run this document is based on used `use_web: false`. Nothing here has been checked against an actual API schema, a real MCP client library's capabilities, or any other ground truth; §3.3's DNS/MCP finding was caught by a human reading the tree, not by the system. `DEEPAGENT_SPEC.md`'s own open question ("RAG-grounding reliability in standalone mode is unmeasured") still stands, un-superseded by anything in this document.
- **`deepagent_tree` vs. `deepagent_tree_md` — which framing wins is still an open, actively-running A/B comparison**, not a settled recommendation. Terse (one sentence per sub-part) vs. full-article (a paragraph) are both structurally disciplined; which produces a more *useful* downstream backlog is exactly the RRTrack side-by-side this document's evidence comes from, and it isn't concluded.

---

## 6. Related Documents

- [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md) — the tree engine `deepagent_tree`/`deepagent_tree_md` fork from; read first.
- [`DEEPAGENT_SPEC.md`](DEEPAGENT_SPEC.md) — the transform pass downstream of this tree; §2.5's horizontal-vs-vertical decomposition distinction and the root-cause finding behind §4's atomic-seed principle both originate there.
- [`DEEPAGENT_ARCHITECT.md`](DEEPAGENT_ARCHITECT.md) — the next stage downstream of `deepagent_spec`; also received the position-breadcrumb/finalize treatment (§2) adapted to its fixed linear interface→abstract→approach→class pipeline instead of a variable-depth tree.
