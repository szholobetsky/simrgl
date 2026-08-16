# Conversations: Multi-Agent Deliberation and Ensembling for Small-Model Robustness

**Status**: Concept — no code written. §0 is the only empirically-grounded claim in this document; everything downstream is design reasoning from a single working session, not measured.
**Date**: 2026-08-16
**Origin**: grew out of debugging `simargl`'s RAG chunk-retrieval (see [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md) for the retrieval side); this document is the generation side of the same session.

---

## Abstract

A single small local model, even given exactly the right retrieved context, does not reliably produce a grounded answer — §0 below is a same-session, reproducible demonstration of this, not a projected risk. The question this document works through is not "how do we make one small model reliable" (External Supervision, see [`AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md`](AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md), already answers that for *thread management*) but a narrower, adjacent question: **can text be enriched — made more grounded, more robust, more multi-perspectived — by exposing it to more than one model or more than one sample before it is finalized?**

Seven variants were worked through in one sitting, each correcting a flaw found in the one before it. That sequence is preserved here deliberately (rather than flattened into a clean taxonomy after the fact) because the corrections are the actual content: a reader who only sees the final "best" design would not learn why the earlier, plausible-looking designs fail, and would be liable to reinvent them.

The throughline: **every variant reduces to the same generative/compiler split already established by `deepagent_spec` → `deepagent_task`** — the deliberation or ensemble step is generative (LLM, expensive, non-deterministic), the step that turns its output into a usable artifact is a deterministic rollup over structured records, never a second LLM pass over free text. Every version of this document's ideas that violated that split turned out to be the one with a real flaw.

---

## 0. Empirical starting point

During the same session, `gemma3:1b` was given the *actual correct retrieved chunk* for a factual question (post-fix RAG context, verified relevant by inspection) and still produced a fluent, on-topic, confidently wrong answer — inventing a garbled term and a distinction not present in the source. The same context given to a ~4B model (MamayLM-Gemma-3-4B) produced an answer that visibly quoted real terms from the source (`CEILING`, `FLOOR`, `NO SMEARING`, `NO BARE HEADINGS`) even though its synthesis was imperfect elsewhere.

This is the concrete evidence behind two assumptions used throughout this document: (a) correct retrieval is necessary but not sufficient — grounding can fail at the generation step even with perfect context; (b) the failure is capability-dependent, not a fixable prompt-wording issue, which is why every mechanism below spends effort on *cross-checking or cross-sampling* rather than on writing a better single prompt.

---

## 1. Consilium (`/team`): debate with voting

**What it is**: N models (or N role-framings of one model) debate a single contested claim — propose, critique, amend, vote — modeled loosely on adversarial legislative revision.

**Implementation shape**:
- Atomic unit is one **claim**, one file — same grain as `term.md` in [`GLOSSARY.md`](GLOSSARY.md), not "the whole document argued over at once." Each claim's debate is independent and parallelizable.
- **PROPOSE** → a model writes the claim under a hard length ceiling (forced concision, the inverse of `deepagent_tree`'s `FULL ARTICLE` contract).
- **CRITIQUE** → a different model sees *only* the current claim text (never debate history), and must tag its objection as **descriptive** ("this is factually wrong because X") or **evaluative** ("this framing is weak") — reusing the descriptive/evaluative distinction already named as aspect 4 of the External Supervision concept (Pospelov). Tagging exists so a downstream deterministic compiler can weight the two kinds of objection differently instead of pooling them.
- **AMEND** → the original author sees the claim + *one* critique (not accumulated history) and rewrites — same parent-scoping discipline as `deepagent_md`'s `_extract_parent_section`, applied to a debate turn instead of a tree node.
- **VOTE** → needs only the current claim version vs. the immediately preceding one (diff-sized context), never the full history.
- **Termination is a round cap, not model-judged convergence** — 2–3 amend cycles, then a forced final vote. This is the same "automaton counts, model doesn't decide when it's done" principle as the ladder gate, applied to debate length instead of agent-thread length.
- **Role assignment is inverted by context-cost**: spend the smartest-but-context-poor model on the *short* high-judgment steps (critique, vote), spend the weaker-but-context-tolerant model on steps needing to hold more text (initial long-form propose).
- **Rollup is a deterministic compiler**, never a second LLM synthesis pass: `/team` emits a claims-ledger (`claim_id, final_text, votes, critique_tags, amendment_count`); the calling flow decides what document shape to render it into (flat glossary, `spec_<id>.md` node, architecture decision record) — `/team` itself is document-shape-agnostic.

**Benefit over a single model**: removes the single-sample confabulation risk from §0 for genuinely *contested* claims, where the value is specifically in surfacing disagreement, not consensus.

**Known flaw carried forward**: debate format has order/anchoring risk — a critiquing model can be unduly swayed by what it's shown first (see §4 for the design that removes this entirely).

---

## 2. Two-tier draft-then-senior-review (not `/team`)

**What it is**: a *different*, simpler mechanism, easy to mistake for §1 but structurally distinct — a strict hierarchy (one strong reviewer, several weak drafters) rather than a multi-party debate among peers. Arose from a concrete example: a 30B model writes a plan; the plan is cut into points; a small model expands each point (having read the whole plan as anchor, same mechanic as `root_task` + `--max_parent_ctx`); the 30B model then reviews and corrects each expanded point in isolation.

**Two corrections made to the original proposal, both worth stating explicitly because the mistakes are natural ones**:

1. **The reviewer cannot have zero context.** Given only the isolated drafted paragraph, a reviewer can judge prose quality but not correctness-to-intent. Minimum required context is the *specific plan point's own definition* (one or two sentences) — the same `_extract_parent_section` scoping as always, applied to (plan-point, draft) instead of (parent-node, child-node). Still tiny; just not literally nothing.
2. **The correction must not round-trip back to the weak drafter for re-writing.** "Here is a critique, please incorporate it" is itself a multi-step instruction — exactly the class of instruction small models execute unreliably (the whole premise of External Supervision). The reviewer's correction should directly *become* the final text for that point. One hierarchical patch pass, not a critique-then-amend cycle.

**Why "senior just fixes junior's bugs" is not a wasteful use of the strong model**: it is the correct allocation given inverted context-cost — the expensive, context-poor resource does a narrow-scope review pass instead of re-deriving the whole artifact, exactly as a human senior engineer reviews a PR without re-reading the whole codebase. This is not the compiler pattern (a human/LLM judgment is still involved) — it is closer to `deepagent_spec`'s "Layered" mode: parent-chain-grounded refinement, not deterministic rollup.

**Benefit over §1**: no voting machinery needed (strict hierarchy has an obvious tie-breaker: the senior), no critique-tagging needed (single reviewer, not reconciling multiple opinions), fewer LLM calls than a full debate round.

---

## 3. `deepagent_rewrite`: a mode, not a new engine

**What it is**: the realization that §2's pattern generalizes past "plan expansion" — any existing artifact drafted by a weak model can be handed a second pass by a strong model — but should **not** be built as a third traversal engine alongside `deepagent_spec`/`deepagent_translate`/`deepagent_code`. It should be a `--refine` mode flag on the *existing* walkers.

**Implementation shape**:
- The correct scoping for a rewrite pass is **inherited from the shape of the source being rewritten**, not chosen freely:
  - Real hierarchy (nested `item_<id>.md` tree from `deepagent_md`) → `deepagent_spec`-style parent-chain scoping, because correctness genuinely depends on tree position.
  - Flat/chunked source (a plain document cut into windows purely for context-length reasons, e.g. `simargl`'s `chunk_text`) → `deepagent_translate`-style local/sequential scoping, because there is no real hierarchy to ground against.
- `--refine` changes three things on the existing walker, reusing its scoping code unchanged: (1) swap `profile:` to the stronger model, (2) change the prompt from "write this" to "here is an existing weaker draft — correct it", (3) thread the draft-so-far into the call as additional input.
- Generalizes to code for free: a second `deepagent_code` pass with a stronger model over a weaker model's leaf implementations, using the same yaml-state-file audit trail already described in §7–8 of [`DEEPAGENT_CODE.md`](DEEPAGENT_CODE.md).

**Benefit**: zero new traversal/scoping code. `rewrite` is an orthogonal axis (weak-draft → strong-refine) applicable to any walker that already exists, not a sibling command competing with them.

---

## 4. Embedding-centroid ensemble voting (medoid selection)

**What it is**: N structurally different small models (different training lineages — Qwen, Gemma, LFM, DeepSeek, VibeThinker, not the same model resampled) each generate independently and in parallel, with **no visibility into each other's output**. A coordinator embeds all N outputs, computes a (possibly weighted) centroid, and picks the actual candidate closest to it by cosine similarity — the medoid, not a synthesized average text.

**Why this removes a flaw in §1**: independent parallel generation has no order/anchoring bias at all — no model ever sees another's output, so there is nothing to be swayed by. The selection step itself is fully deterministic (cosine distance), not an LLM vote — closer to "automaton tracks the field" than §1's LLM-adjudicated voting is.

**Implementation shape**:
- Weighting reuses code that already exists in this codebase: `simargl.searcher._search_aggr`'s weighted-centroid math (`weights = scores / scores.sum(); centroid = (vecs * weights[:,None]).sum(axis=0)`) — substitute per-model trust weight for the existing per-hit similarity score.
- **Trust weight should not default to raw parameter count.** Parameter count is a weak proxy. The correct source is empirical: calibrate weights against the same ground-truth validation set already planned for the External Supervision MVP (sonar.db recall measurement) — the same infrastructure serves double duty.

**Real flaw, stated precisely (do not oversell "no bias")**: centroid selection removes *order* bias but not *systematic* bias. It protects against one model's idiosyncratic, random noise (a rare slip), not against **correlated hallucination** — several models sharing similar training-data priors independently converging on the same plausible-but-wrong answer for an underspecified prompt. In that failure mode the wrong-but-similar cluster dominates the centroid and the one model that got it right is discarded as the outlier. §0's confabulation risk is not fully solved by this mechanism when the confabulation is a *shared* blind spot, only when it's model-specific noise. Sentence-embedding cosine similarity measures topical/stylistic agreement, not factual faithfulness to a source — a grounded-but-differently-phrased correct answer can sit farther from centroid than two fluent, similar, wrong ones.

**Mitigation**: measure agreement tightness, not just centroid proximity. A tight single cluster → trust the medoid cheaply. A wide spread or two similarly-sized clusters → treat as "small models did not converge," and escalate to §3 (`deepagent_rewrite` by a stronger model) rather than silently returning a low-confidence medoid.

---

## 5. Distributed weak-device swarm

**What it is**: §4 generalized to N *physically separate* weak devices (e.g. a rig-miner farm, each node capable of `gemma3:1b` but nothing larger), constrained to text-only inter-node communication.

**Key clarification (the constraint sounds like it blocks §4, it does not)**: text-only transport only means worker devices never embed anything themselves. Embedding and centroid computation happen centrally, once, at a coordinator that receives N text responses over the network. No device needs an embedding model loaded.

**Two distinct noise sources, worth not conflating**:
- **Self-consistency** (the same model, resampled 2–3× at different temperature/top-k on one device, own internal medoid taken first) removes *random* per-sample noise — but does not remove the model's own systematic blind spot, since it's still the same weights answering.
- **Cross-model ensembling** (§4, across architectures) removes *systematic*, architecture-specific bias — but is a weaker source of protection against a single unlucky sample from any one model.
- These attack different problems and combine cleanly: each device self-filters its own noise first (cheap, local), and only its one cleaned candidate enters the cross-device pool (§4's weighted centroid) — avoiding N raw, un-filtered samples going straight into the vote.

---

## 6. Cost economics, and why "cheap scouts" is not a free fix

**The critique that forced this section**: running N *full* generations just to pick one, when the useful signal ("does this text go in a sane direction") is far lower-dimensional than a full artifact, is genuinely wasteful — like shipping five full truckloads of fragile cargo to find out which route breaks least on average, instead of sending five cheap scouts to agree on a route first.

**The obvious fix (send a cheap short "direction" statement first, only the winner writes the full text) has a real flaw, not just a cost/quality tradeoff**: autoregressive generation's divergence between samples accumulates over token-steps — early tokens sit close to the shared prompt-conditioned attractor, and meaningfully differentiated, idiosyncratic content (the kind that makes a model's *full* output interesting) tends to emerge only after some "warm-up" length. A short direction statement from N models, therefore, is not a cheap proxy for the real signal — for creative/exploratory generation it can measure "nobody has diverged yet" and report false high agreement, not real consensus.

**This makes applicability task-dependent, not universal**:
- **Short factual claims** (does the source say X or Y) diverge immediately — a short statement genuinely carries signal here. §4/§5 truncation is safe for grounding/fact-checking use cases.
- **Creative/exploratory prose** (the actual reason `deepagent_md` exists — see [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md)) needs the warm-up length before anything worth ensembling over even exists.

**A sharper, more uncomfortable implication**: for creative tasks, centroid/consensus selection (§4) and the value of unconstrained long generation may be in direct tension, not just cost tension. If a model's value in going long is precisely that it reaches an idiosyncratic, non-average trajectory, then averaging across several long generations systematically discards exactly that trajectory in favor of the blandest, most-repeated one. **Recommendation**: scope ensemble/voting mechanisms (§4, §5, §7) to fact-grounded, verifiable-against-source tasks; do not apply them to `deepagent_md`'s creative elaboration, where per-model idiosyncrasy after warm-up is the feature, not the noise.

**Cost fixes that don't truncate generation length** (used instead, where ensembling of creative content is still wanted): ensemble less *often* — once per section, not once per leaf node — trading granularity for cost instead of trading signal for cost; or empirically measure the divergence-vs-length curve (pairwise cosine similarity across samples at increasing checkpoints) instead of guessing a cutoff, consistent with the "measure it, don't assume it" discipline already used for the locked findings in `exp1`–`exp3`.

---

## 7. `deepagent_rada`: legislative-style deliberation

**What it is**: §1 generalized with a second diversity axis. Real legislatures diversify not just by *who* votes but by *what interest* each voter represents (regional, industrial, ideological). Applying that: diversify by **aspect** (role/interest framing) in addition to, or instead of, model architecture.

**Why aspect-diversity is cheaper than architecture-diversity, not just different**: it does not require N different model weights loaded (and the sequential model-swap cost on constrained VRAM that implies, per §5) — it reuses the `list: a1, a2, a3` "analytical lenses" parameter that `/deepagent` already has, applied to a single loaded model instead of a tree of nodes. One model, N role-prompts, gets most of the diversity benefit of N architectures at a fraction of the hardware cost.

**Process**:
1. Draft an initial plan (`deepagent_tree_md`, dot-notation node IDs).
2. Per plan point, solicit amendments from aspect-framed models.
3. Round-robin each amendment to all aspect-deputies for a vote.
4. **Vote validity gate, not a raw yes/no trust**: ask each deputy to paraphrase the amendment first, then embed the paraphrase and compare it to the **amendment's own source text** (not to other deputies' paraphrases). Low fidelity → that deputy misunderstood the amendment; discount or discard its vote regardless of which way it voted. This reuses the exact source-faithfulness check already built for RAG chunk grounding (see [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md)'s companion retrieval work), applied to vote validity instead of answer grounding.
5. **Adversarial loophole check**: the strongest available model generates a misuse scenario per amendment ("how could a bad-faith actor exploit this"). Must be calibrated against a known-safe control amendment first — an eager model asked to find an exploit will usually find *something* regardless of whether the amendment is actually exploitable; an uncalibrated raw verdict risks flagging everything.
6. **Deliberation length is capped procedurally, not left open-ended.** "Let it run for two months" reintroduces exactly the unbounded-thread problem External Supervision exists to prevent — just at the level of the whole deliberation instead of a single agent's turn. Real legislatures converge via fixed readings and a closing vote, not indefinite debate; `deepagent_rada` needs the same hard round cap and forced final vote per amendment.
7. **Every exchange must emit a structured record as it happens** — `{amendment_id, text, proposed_by_aspect, verdicts: [...], loophole_scenarios: [...], status}` — never accumulate as free-text transcript. This is the load-bearing fix: without it, "how do we later analyze gigabytes of negotiation" has no good answer, because analyzing free text at that scale reintroduces §0's synthesis-over-long-context failure at a much larger scale. With structured live records, the final rollup is a deterministic filter/aggregation, not an LLM pass.

**Naming split, mirroring `deepagent_spec` → `deepagent_task`**: `deepagent_rada` is the generative deliberation engine (produces the structured amendment ledger); a separate compiler (tentatively `deepagent_codex`) performs the deterministic rollup from accepted amendments into final document text. Splitting the name enforces the generative/compiler split at the architecture level, not just as a convention to remember.

---

## 8. Applied to commercial/product decisions, not just legislative ones

**Motivating observation**: most real usage is product/UX/commercial decisions ("where should a reorder reminder appear in a checkout flow"), not high-stakes legislative-style questions. Several things change for the better in this domain, not just "the same mechanism, lower stakes":

- **The aspect list is more tractable.** Political factions are often genuinely irreconcilable; product-review disciplines (UX/usability, growth/conversion, brand/tone, accessibility, legal/dark-pattern compliance, engineering feasibility) are an established, well-understood role set already used in real product reviews — easier to define and justify than political interest groups.
- **§7's loophole check maps directly onto a known, regulated concern**: a "how could this be abused" scan over a UI amendment is structurally identical to dark-pattern detection (forced continuity, nagging, confirm-shaming — categories with real regulatory weight, e.g. EU DSA / FTC dark-pattern rules). The "bad-faith actor" here is often not an external attacker but the client's own growth incentives overriding a usability concern — the mechanism doesn't need external adversaries to be useful.
- **Ground truth exists and is cheap to obtain (A/B testing), which changes the required trust level.** A legislative verdict has no empirical check; a UI verdict does. This means `deepagent_rada`'s output for commercial decisions should be treated as a **prioritized set of hypotheses for empirical validation**, not a binding decision — a much lower, more defensible trust bar than "the deliberation decided correctly."
- **Verdict states should be richer than binary accept/reject**, given empirical validation exists downstream: `accepted`, `rejected`, `accepted — pending A/B`, `deferred — needs legal review`. Forcing a binary outcome discards the legitimate "let's just test both" resolution that real product decisions often want.
- **The actual value for small decisions is not "getting the objectively right answer" — a competent human resolves those in under a minute, and an A/B test settles it empirically regardless.** The value is the *byproduct*: a cheap, structured, retrievable rationale trail (why was X placed here and not there; what alternative was considered and rejected; was the dark-pattern risk explicitly checked) — useful for onboarding, for defensibility if a decision is later questioned, and directly retrievable by this project's own task-to-code retrieval research (a documented design decision is exactly the kind of artifact `simargl` is built to index and retrieve).

---

## 9. Resulting artifact structure and tooling

**Document shape**: a `deepagent_tree_md` plan (dot-notation node IDs) with a **sibling structured-data file per node** holding that node's amendment ledger (`amendments_<id>.json`, an array of the §7 record shape) — not free prose attached to the tree.

**Why this is very likely an [`ALKONOST.md`](ALKONOST.md) extension, not a new product**: Alkonost already discovers and renders `.1bcoder/planMD/`, `.1bcoder/spec/`, `.1bcoder/tasks/` by the same dot-notation ID convention, independently re-implementing the file format with zero LLM calls, zero import dependency. A structured `amendments_<id>.json` sibling file is one more file type at the same convention — it needs one new renderer (an expandable panel per node), not a new discovery or navigation paradigm. This holds as long as the ledger stays structured data (§7's fix) rather than degrading into free transcript.

**Where a genuinely separate tool becomes justified**: if ledger volume grows large enough to need cross-node analytics (vote-tally dashboards, "which aspect votes against most often," filtering across the whole corpus) rather than per-node display — that is a different interaction pattern than Alkonost's two-level project→plan picker was designed for. This fork point should be decided empirically once real volume exists, not designed for preemptively.

**Search over the ledger at scale is not a UI concern at all — it is a `simargl` retrieval task.** "Find every amendment where the accessibility deputy voted against" is the same semantic search problem this whole document's originating debugging session was about, just indexing structured amendment records instead of markdown concept docs. This closes the loop: the tool whose retrieval bug started this session becomes the natural search layer for the artifact this session's design work produces.

---

## 10. Known gaps, not yet closed

- **The central empirical claim of §6 — that divergence-vs-generation-length actually behaves the way described — is not measured, only argued from first principles about autoregressive sampling.** Before relying on it, run the same measurement suggested there: pairwise cosine similarity across N samples at increasing length checkpoints, on this project's actual models, not assumed.
- **Trust-weight calibration (§4) against sonar.db ground truth has not been done.** Currently would default to parameter count, which this document itself argues is a weak proxy.
- **§7's loophole-detector calibration (control amendment → measured false-positive rate) is described but not designed in detail** — no concrete control-amendment set or acceptance threshold proposed yet.
- **§7's procedural round cap has no proposed number.** "Capped, not open-ended" is a principle, not a value — how many amendment cycles, how ties in the final vote resolve, is undecided.
- **Whether aspect-diversity (§7) actually produces measurably more different outputs than architecture-diversity (§4) is an assumption, not a tested result.** It is plausible (interest-conditioning targets the axis of disagreement more directly than training-data idiosyncrasy does) but no comparison has been run.
- **The Alkonost-extension-vs-new-product fork point (§9) is explicitly deferred to future volume data** — this is a known gap by design, not an oversight, but it means §9 has no concrete trigger threshold defined yet.
- **No mechanism in this document has been run once, on any real task, end to end.** Every benefit claimed is either reasoned from principles established elsewhere in this codebase (External Supervision, the RAG-grounding fix, `_search_aggr`'s existing weighted-centroid code) or extrapolated from the single §0 observation. This document is a design map, not a validated result.

---

## 11. Related Documents

- [`AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md`](AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md) — the External Supervision thesis this whole document sits downstream of; the descriptive/evaluative distinction (§1) and the "automaton counts, model doesn't judge convergence" principle (§1, §7) both originate there.
- [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md) — `root_task`/parent-section-injection, reused throughout (§1, §2, §3) as the scoping discipline; also the retrieval-side debugging session §0's evidence comes from.
- [`DEEPAGENT_TREE.md`](DEEPAGENT_TREE.md) — the structured-vs-prose distinction behind §6's recommendation to keep ensembling off creative `deepagent_md` content; capped-round discipline precedent.
- [`DEEPAGENT_SPEC.md`](DEEPAGENT_SPEC.md) / [`DEEPAGENT_ARCHITECT.md`](DEEPAGENT_ARCHITECT.md) — the generative/compiler split (spec tree → deterministic `deepagent_task` rollup) that every section of this document reduces to.
- [`DEEPAGENT_CODE.md`](DEEPAGENT_CODE.md) — verification-as-ground-truth (test pass/fail as milestone, not model judgment), the same principle §7's vote-validity gate and loophole check both apply to text instead of code.
- [`ALKONOST.md`](ALKONOST.md) — the viewer this document's artifact (§9) is proposed to extend rather than replace.
- [`GLOSSARY.md`](GLOSSARY.md) — the `term.md`/`glossary.md` generative/compiler pair §1's atomic-claim-file grain is modeled on.
