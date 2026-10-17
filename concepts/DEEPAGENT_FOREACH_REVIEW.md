# DeepAgent_Foreach and DeepAgent_Review: A Generic External-List Loop, First Applied to Code Review

**Component**: 1bcoder `/flow deepagent_foreach` (engine) + `/flow deepagent_review` (first instance)
**Date**: 2026-09-18
**Status**: Draft / Concept — not yet built, written for a first pass of revision before any design is locked

---

## Abstract

Every `deepagent_*` flow so far builds its own list to walk: `deepagent_md`/`deepagent_tree` generate a tree and then walk it; `deepagent_spec` walks the sections a generation leaf happens to contain. This document describes a different shape, needed for a different class of task: the list already exists **outside** the model — a `git diff`, a commit range, a linter's JSON output — and the job is to run one judgment or transform pass over each item, independently, with an externally supplied rubric deciding what "judgment" means. `deepagent_foreach` is the proposed generic engine for that shape; `deepagent_review` (rubric-driven code review over a diff) is worked out here as its first concrete instance, with `deepagent_relnote` (commit log → release notes) sketched as the second to keep the engine/instance boundary honest rather than accidentally specific to review.

This reuses, deliberately, mechanisms already proven elsewhere in the family rather than inventing new ones: `deepagent_spec`'s externally-controlled-depth discipline (§2.3 of `DEEPAGENT_SPEC.md`), `deepagent_tree`'s position breadcrumb, `deepagent_architect`'s research-grounding-over-imagination step, and `DEEPAGENT_AND_DEEPAGENT_MD.md` §9's `--review` human gate. What's actually new is narrow: a pluggable list-source adapter and a chunking layer for when a single item is itself too large for the effective context window.

---

## 1. Motivation: the shape recurs, the list source doesn't

Three concrete needs, on inspection, are the same loop with a different front door:

- **Code review**: given a diff (a branch, a PR, a commit range), flag naming-convention violations, antipatterns, and security smells per changed unit, optionally proposing a fix.
- **Release notes**: given a commit range, translate terse commit-message language into user-facing release-note language, however large the release — by never looking at more than one commit's worth of scope at a time.
- **Lint-driven cleanup**: given a linter's findings list, either comment on each (explain why it matters, in this project's context) or fix it, per finding.

All three share three properties that none of the existing `deepagent_*` engines assume:

1. The list is **discovered from outside**, not decomposed by the model — a diff already has hunks, a commit log already has commits, a linter already has findings. There is no BFS tree to build; the "depth" question `deepagent_spec` had to solve (§2.3) doesn't arise because there is nothing to decompose.
2. Items are **independent enough to process in any order**, closer to `deepagent_md`'s single-level BFS siblings than to a genuinely hierarchical tree.
3. A single item can **exceed the effective context window on its own** (one commit touching forty files, one hunk that's actually a generated lockfile diff) — a problem none of the tree-shaped flows have, because their nodes are sized by construction (one `## N.` section, one leaf file).

Rather than writing `deepagent_review`, `deepagent_relnote`, and eventually `deepagent_lintfix` as three fully independent copies from day one — the pattern the project already accepted for `deepagent_tree`/`deepagent_tree_md`/`deepagent_spec`/`deepagent_test` per `DEEPAGENT_AND_DEEPAGENT_MD.md` §9.5 ("each a fully independent copy of the machinery... the same patch was applied three times rather than factored into a shared module") — this document proposes factoring the genuinely common 80% (list intake, chunking, resume, the `--review` gate) into one engine first, on the theory that the common part here is larger and more load-bearing than it was for the tree forks. Whether that theory survives contact with a real build is an open question (§9), not a settled claim, precisely because the project's own track record leans the other way.

---

## 2. Core idea: engine vs. instance

```
deepagent_foreach                          ← the engine
  ├─ list source adapter                   ← pluggable, §3
  ├─ chunker                               ← §4, external budget, not model judgment
  ├─ per-item prompt = external rubric     ← §5, pluggable
  ├─ resume / redo / --review              ← reused unchanged from deepagent_md family
  └─ rollup / compose                      ← instance-specific grammar, §5–§6

deepagent_review    = deepagent_foreach + git_diff adapter   + code-review rubric
deepagent_relnote   = deepagent_foreach + git_log adapter    + release-note rubric
deepagent_lintfix   = deepagent_foreach + lint_json adapter  + comment-or-fix rubric   (sketched only, §9)
```

An instance is, concretely: which adapter (§3) supplies the list, which rubric (§5) frames the per-item prompt, and which rollup grammar (§6/§9) the flat output is compiled into. Everything else — chunk-or-not, resume, the human review gate — is engine code, written once.

### 2.1 What actually flows: the Item stream

The split above still leaves the interior of "engine" undefined: what vocabulary of operations is allowed on the list while it's being processed? The right model isn't "walk a list, call one prompt per item" — it's closer to a Unix pipe (`cmd | grep | awk | ...`) crossed with a Java-style `Stream<T>` (typed elements, a small fixed vocabulary of `filter`/`map`/`flatMap`, and exactly one terminal operation — `forEach`, `collect`, or `reduce` — deciding the final shape). Both analogies exist for the same reason: "transform each element of a sequence, then separately decide how to combine the results" is a genuinely reusable shape, not something specific to code review.

The stream element is the `Item` type from §3 (`id`, `title`, `body`, `metadata`) — structured, not a raw text line the way a Unix pipe would use, because a stage needs to address specific fields (`body` for diff text, `metadata.severity` for a filter predicate) rather than re-parsing text at every stage the way `awk`/`grep` do.

### 2.2 Stage vocabulary

| Stage | Arity | LLM? | Example |
|---|---|---|---|
| `source` | 0 → N | no | `git-diff main..HEAD --hunk` (§3 adapters) |
| `filter` | 1 → 0 or 1 | usually no | `severity >= warning`, `ext in {.py,.ts}` — the `grep` equivalent |
| `map` | 1 → 1 | optional | `llm-judge rubric=code_review.yaml` (LLM, adds `metadata.verdict`) or `normalize-path` (pure) — the `awk` equivalent when pure, the actual judgment step when LLM-backed |
| `flatmap` | 1 → N (or N → 1) | optional | `chunk-if-large budget=6000tok` (one oversized item → many breadcrumbed sub-items, §4) or `group-by file` (many lint findings → one item per file) |
| `foreach` *(terminal)* | N → side effect | optional | write one `verdict_<id>.md` per item; post one PR comment per item |
| `collect` *(terminal)* | N → 1 artifact, no folding | no | assemble a `tasks.md`-style rollup, a flat table, a JSON array |
| `reduce` *(terminal)* | N → 1 value, pairwise fold | optional | fold per-commit release notes into one running draft, smoothing transitions on the way (echoes `deepagent_md`'s `compose flat` mode) |

Exactly one terminal stage ends a pipeline — mirroring Java Streams' own rule that a stream is consumed once, by exactly one terminal operation. Non-terminal stages (`filter`/`map`/`flatmap`) compose freely and repeat.

### 2.3 Instruction format: a declarative pipeline, not hardcoded logic per instance

An instance becomes a **pipeline description** the engine interprets, rather than a new Python function per instance — the same move `deepagent_md` already made by treating `plan:`/`list:` as external, textual, per-run configuration instead of hardcoded prompt branches:

```
/flow deepagent_foreach \
  source:  git-diff main..HEAD --hunk \
  filter:  ext in {.py,.ts} \
  flatmap: chunk-if-large budget=6000tok \
  map:     llm-judge rubric=rubric/code_review.yaml \
  filter:  verdict != PASS \
  sink:    collect tasks.md
```

```
/flow deepagent_foreach \
  source: git-log main..HEAD \
  map:    llm-translate style=release-note.yaml \
  sink:   reduce release-notes.md
```

Two consequences beyond syntactic sugar:

- **A new instance costs a pipeline description, not new Python.** The sketched `deepagent_lintfix` (§9) is `source: lint-json | filter: severity>=warn | map: llm-fix-or-comment | sink: foreach write-comment` — no new flow file, provided the stage vocabulary already covers `llm-fix-or-comment`.
- **Escalation falls out of the vocabulary for free.** §6's "cross-model check only on items the cheap judge flagged" is just `filter: verdict != PASS` between two `map` stages — no bespoke escalation branch needed.

This is explicitly analogy-driven design, not yet validated against a real run — see §9 for what's still open, chiefly whether the pipeline is lazy (true streaming, low memory, harder to resume mid-pipeline) or materialized (a plain in-memory list between stages, simpler resume, closer to what `deepagent_spec` already does).

---

## 3. List sources: a uniform `Item`, adapter-per-source

Regardless of source, every adapter reduces its input to the same shape:

```python
Item(id: str, title: str, body: str, metadata: dict)
```

| Source | `id` | `body` | `metadata` |
|---|---|---|---|
| `git_diff` (file granularity) | file path | unified diff for that file | insertions/deletions, is_binary |
| `git_diff` (hunk granularity, `--hunk`) | `path#hunk_index` | one `@@ ... @@` hunk | same, per-hunk |
| `git_log` | short commit sha | commit message + optional `--stat` | author, date |
| lint JSON | `file:line:rule_id` | message + a few lines of surrounding source | severity, rule id |

This is the same "one small adapter file, registered in a dict, nothing else needs to change" shape `yasna` already uses for its per-agent session adapters (`yasna/adapters/__init__.py`'s `ALL` dict) — proposed here as `adapters/{git_diff,git_log,lint}.py`, each exposing a source name and a `parse(...) -> list[Item]` function, so a fourth source (a Jira ticket range, a test-failure list) is a new file, not a change to the engine.

---

## 4. Chunking: external budget, never a model judgment call

Same discipline `deepagent_spec` already established for depth (§2.3 of `DEEPAGENT_SPEC.md`): the chunk boundary is a deterministic token-count check against a configured budget, never "does the model think this hunk is too big." Two levels of chunking, both mechanical:

- **Across items**: if the whole item list doesn't fit in one context, process it in ordered batches — no different from BFS processing one level at a time.
- **Within one item**: if a single item's own `body` (a huge commit's diff) alone exceeds budget, split it further — by file within the commit, by hunk within the file — recursively, until each piece fits. This is a size-driven split, not a semantic one; it has nothing to do with `deepagent_tree`'s HOMOGENEITY/PART-OF contract, which governs *meaning*, not *bytes*.

When an item is split into chunks, each chunk's prompt carries the same **position breadcrumb** `deepagent_tree` already uses (`DEEPAGENT_TREE.md` §2): "chunk 3 of 7 of item `<id>`", plus a running one-line summary of the verdicts already reached for earlier chunks of the same item — the direct analogue of parent-section injection, just along a size axis instead of a depth axis. Final per-item verdict aggregation across its own chunks should default to a **deterministic** rule (worst-severity-wins, or majority) rather than another free-text LLM synthesis call — an LLM aggregation pass is reserved for the rarer case where chunk verdicts genuinely conflict (one chunk says PASS, another says FAIL on the same named concern) and a human wants a one-line explanation of the disagreement, not a re-vote.

---

## 5. Per-item task: the rubric is external, the output is structured

The rubric is a plain checklist supplied from outside the flow, never hardcoded into the prompt template — the same principle as `--maxdepth`/`plan:` labels being external configuration for `deepagent_md`, applied here to "what counts as a defect" instead of "how deep to go":

```yaml
rubric: code_review
checks:
  - naming:      "Identifiers follow this file's existing snake_case/camelCase convention"
  - antipattern: "No god-function over ~50 lines, no bare except, no mutable default argument"
  - security:    "No hardcoded secret, no string-built SQL, no eval/exec on external input"
```

The per-item prompt is built from: the item's `body`, the rubric's checks, and — reusing `deepagent_architect`'s research-grounding precedent (`DEEPAGENT_ARCHITECT.md` §4, "grounding each step in the state of the real world, not the model's imagination") rather than reviewing in a vacuum — optionally a bounded read-only lookup of the surrounding file or the project's own naming conventions, before the verdict is written.

Output is structured, not free prose, so the rollup (§6) can be mechanical rather than another LLM pass:

```markdown
### <item_id>
verdict: PASS | FLAG | FAIL
findings:
  - [naming]     `getUsrData` should be `get_user_data` (line 42)
  - [security]   raw f-string SQL query (line 88) — potential injection
suggested_fix: <patch, or "none">
```

---

## 6. Validation methods for the rubric pass itself

This is the direct continuation of the chat that produced this draft — a taxonomy of ways to validate LLM-generated (or LLM-reviewed) code, mapped onto what this project already has working, so `deepagent_review` doesn't reinvent verification machinery that exists elsewhere in the family.

| Class | Method | Reuse in this project | Notes |
|---|---|---|---|
| Mechanical (cheap, first filter) | Syntax/parse check | `deepagent_code`'s existing syntax check | Already proven; run before any LLM call, not after |
| Mechanical | Linters/type checkers (ruff, mypy) | new dependency, no existing wrapper | Catches most naming/style before the rubric pass ever runs |
| Mechanical | Security scanners (bandit, semgrep, gitleaks) | new, but same shape as `tool-allowlist-gate`'s "PASS/FAIL gate proc" (`DEEPAGENT_ARCHITECT.md` §4) | Semgrep custom rules are the natural home for project-specific naming/antipattern checks — deterministic, no LLM needed for what a rule can express |
| Mechanical | Complexity/duplication (radon, jscpd) | new | Flags god-functions/copy-paste without spending an LLM call |
| Dynamic | Run existing tests against the change | none yet in this flow family | Straightforward for `deepagent_review` when the diff touches tested code |
| Dynamic | Differential testing (old vs. new output on same input) | none yet | Best fit for refactor-shaped diffs, not new features |
| Model-based | LLM-as-judge against the rubric | this document's baseline mechanism | The rubric pass itself |
| Model-based | **Cross-model verification** (different/stronger model reviews the diff or the first judge's verdict) | new — proposed as `--judge-model`, separate from `--model`, same shape as `--profile` already separating generation workers | Verification is often cheaper and more reliable than generation; reserve for items already flagged by the cheap mechanical pass, not every item, to control cost |
| Model-based | Self-consistency / majority vote (N independent judge calls) | none yet | Useful when a single judge call is noisy on borderline items |
| Model-based | Adversarial self-critique (a call whose only instruction is "find flaws," never "confirm this is fine") | none yet, but same spirit as `deepagent_architect`'s separation of research from decision | Avoids the failure mode where a judge agrees with output from a similar model family |
| Model-based | Retrieval-grounded review (ground the rubric prompt in actual project conventions, not the model's assumption of them) | direct reuse of `deepagent_architect`'s research step (§4) | Already has an enforced read-only allow-list (`tool-allowlist-gate`) to reuse as-is |
| Model-based | Drift check on the review output itself | `radogast`'s existing `drift_status` mechanism, applied to the judge's own output instead of a chat session | Catches a judge comment that has wandered off the actual diff/rubric |
| Human-in-the-loop | Per-item approval gate | `DEEPAGENT_AND_DEEPAGENT_MD.md` §9's `--review` (`[y]es [n]o [a]ll`, no subtree here so `[b]ranch` doesn't apply — same simplification already made for `deepagent_spec`/`deepagent_test`, §9.4) | Free-text rejection feedback folds back into a re-run of that one item, unchanged mechanism |
| Human-in-the-loop | Escalation threshold | new | e.g. any `FAIL` or any `security` finding forces the gate open even under `--all` auto-trust — same "safety override" principle already specified for `--review`'s `b`/`a` modes |

The recommended default pipeline, cheapest-first: **mechanical scanners → LLM rubric judge → cross-model or adversarial re-check only on items the first judge flagged → human gate on anything still flagged or security-tagged.** Nothing here mandates spending the most expensive check on every item.

---

## 7. Relation to the family / the Ralph loop

| Ralph ingredient | `deepagent_foreach` equivalent |
|---|---|
| External state (`fix_plan.md`) | One verdict file per item, on disk, not in context |
| Fresh context per iteration | Each item's (or chunk's) judge call is independent |
| "Only one thing" per loop | One item, one verdict, per pass |
| Failures piped back as context | Rejection feedback from `--review` folds into that item's re-run |
| No LLM-judged stopping | List membership and chunk boundaries are both external/mechanical, never model-decided |
| Real, enforced boundaries | Read-only research grounding reuses `tool-allowlist-gate`, unchanged |

Unlike `deepagent_md`/`deepagent_tree`, there is no tree here — the list is flat, so "BFS" degenerates to a single level, and "resume" is simply "which items already have a verdict file on disk."

---

## 8. What this is not

- **Not a git-operations tool.** It reads a diff/log/lint output and writes verdict files; it does not commit, push, or post PR comments itself in v1 — those are separate, higher-blast-radius integrations (§9) deliberately out of scope for a first draft, consistent with this project's general caution around actions that touch shared/external state.
- **Not a replacement for CI.** Mechanical checks (§6, class "Mechanical") duplicate what a linter/CI job already does faster and without an LLM call; the point of this flow is the layer CI doesn't cover — rubric judgment, cross-model verification, human-in-the-loop escalation — not re-implementing `ruff` badly.
- **Not verified against real runs yet.** Everything in this document is design, not measurement — unlike `DEEPAGENT_TREE.md`, which is backed by five real generation runs, this draft has zero. Treat every default suggested here (chunk-by-file-then-hunk, cheapest-first pipeline order) as a starting guess to be revised once a real diff is actually run through it.

---

## 9. Open questions

1. **Engine-first vs. fork-first.** §1 proposes factoring a shared `deepagent_foreach` engine before writing `deepagent_review`, against the project's own track record of forking three times instead (`DEEPAGENT_AND_DEEPAGENT_MD.md` §9.5). Not resolved — worth revisiting after `deepagent_review` alone is built, to see how much of it actually generalizes cleanly to `deepagent_relnote`.
2. **Default granularity for `deepagent_review`** — file-level or hunk-level. Leaning file-level default with a `--hunk` opt-in, but untested against a real diff.
3. **Auto-fix vs. comment-only default.** Should default to comment-only (`suggested_fix` written but never applied automatically); an explicit flag would be required to let the flow write a patch, consistent with this project's general default of not silently mutating files.
4. **Where verdicts get posted.** v1 target is a local markdown file only. Posting to an actual GitHub PR as inline comments is a real, separate integration with its own risk profile (visible to others, effectively unreversible once posted) — not designed here, and not to be conflated with the local-file version when this is eventually built.
5. **Cross-model judge cost policy.** §6 suggests reserving the stronger/second model for items already flagged by a cheaper pass. Not yet decided whether that threshold is itself configurable per rubric or fixed by the engine.
6. **`deepagent_relnote`'s rollup grammar is genuinely different from a checkbox list** — a release note doesn't have a "done" state the way a `deepagent_task` line does. Likely a categorized markdown (Features / Fixes / Breaking Changes) instead, meaning the "rollup" step is not fully shared between instances even if the judge/chunk machinery is. Not designed in detail here.
7. **`deepagent_lintfix` is sketched, not designed** — included in §2 only to keep the engine/instance boundary honest (two instances risk looking like the engine was designed around one specific case). Needs its own pass once `deepagent_review` exists to validate against.
8. **Lazy stream vs. materialized list.** A true stream (§2.1–2.3) processes items one at a time without ever holding the full list in memory — attractive for a huge commit range, but it complicates resume/redo: `--review`'s per-item gate and `redo <item_id>` (§9.5 of `DEEPAGENT_AND_DEEPAGENT_MD.md`) both assume you can address a specific item after the fact, which is trivial against a materialized list (or one file per item on disk, as this family already does everywhere) and genuinely awkward against a one-pass stream, especially once a `flatmap` (chunking) stage has changed the item count and given sub-items derived ids (`item#chunk3`) that must stay stable across reruns. Leaning toward "materialized list of `Item`s, streaming vocabulary for the *code*, not the *runtime*" — i.e. borrow the `filter`/`map`/`flatmap`/terminal naming and composability, without actually committing to single-pass, unbuffered execution. Not decided.
9. **Pipeline description format.** §2.3 sketches a shell-flag style (`source: ... | filter: ... | sink: ...`) matching existing `/flow --flag` conventions. A YAML pipeline spec was also considered — nests more cleanly once a stage's config is itself multi-line (a rubric is already its own file, §5) — but breaks the "it reads like a `/flow` command" property every other tool in this family has. Not decided; likely resolved by writing `deepagent_review` once and seeing which one is actually less annoying to type.
10. **(Parking lot, deliberately deferred) An event-driven/queue-backed runtime would collapse several of the above into one mechanism** — a topic *is* the transport interface (§2.3), consumer semantics *is* the single gate/backpressure point (§4's chunk budget, §6's escalation threshold), and a durable log *is* the resume/redo state (§9.8's stable-item-id problem). Two versions of this are not the same commitment:
    - **(a) File-log-as-topic, no broker at all** — an append-only JSONL file per pipeline run as the "topic," a sidecar offset file (or the existing convention: a written `verdict_<id>.md` *is* the "processed" marker) for consumer position, and a maildir-style pending-directory + atomic rename as the claim operation for `--profile` workers pulling the next item themselves instead of today's static round-robin assignment (`DEEPAGENT_AND_DEEPAGENT_MD.md` §4.1) — this fixes a real, already-known weakness (round-robin ignores uneven chunk sizes) using only files, zero new dependencies, fully consistent with every other `deepagent_*` flow's "state on disk, no external service" convention. Plausible for v1, not just a parking-lot idea.
    - **(b) An actual broker (Kafka, or a live transport like ZeroMQ for push-based fan-out to already-busy workers)** — genuinely changes the deployment model for the whole family (an external service or a socket-level dependency beyond git), and buys little over (a) unless multiple independent consumers need to read the same event log concurrently, which no current instance (`deepagent_review`/`deepagent_relnote`) actually needs. Stays deferred until a real multi-consumer case shows up.

---

## 10. Implementation location (not yet built)

- `C:\Projects\1bcoder\_bcoder_data\flows\deepagent_foreach.py` (engine, or first fork — per open question 1) — independent clone, per `AGENTS.md`'s submodule-migration table; never the pinned `C:\Projects\simrgl\1bcoder\` snapshot.
- `C:\Projects\1bcoder\_bcoder_data\flows\adapters\{git_diff,git_log,lint}.py` — list-source adapters, same registration-dict shape as `yasna/adapters/__init__.py`.
- Output: `.1bcoder/review/<run>/verdict_<item_id>.md`, mirroring the `planMD`/`arch` convention already used elsewhere in the family.

---

## 11. Related documents

- [`DEEPAGENT_AND_DEEPAGENT_MD.md`](DEEPAGENT_AND_DEEPAGENT_MD.md) — origin of the `--review` human gate (§9) reused unchanged here.
- [`DEEPAGENT_SPEC.md`](DEEPAGENT_SPEC.md) — origin of the "depth/chunking is external, never model-judged" discipline (§2.3), directly reused for §4's chunk boundaries.
- [`DEEPAGENT_TREE.md`](DEEPAGENT_TREE.md) — origin of the position-breadcrumb mechanism, reused in §4 along a size axis instead of a depth axis.
- [`DEEPAGENT_ARCHITECT.md`](DEEPAGENT_ARCHITECT.md) — origin of the research-grounding-before-decision step and the `tool-allowlist-gate` mechanism, both reused in §5–§6.
