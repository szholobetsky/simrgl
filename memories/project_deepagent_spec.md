---
name: project-deepagent-spec
description: "deepagent_spec/deepagent_task design work and deepagent_md's new continue subcommand (2026-07 session, AnimalAlert validation)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 59f4e9c6-779f-4f0a-9009-ff73ef6ea5a3
---

Designing `deepagent_spec`/`deepagent_task` (concept doc: `C:\Project\codeXplorer\capestone\simrgl\concepts\DEEPAGENT_SPEC.md`) — turning a `deepagent_md` prose tree into executable specs, validated empirically against the author's own training project **AnimalAlert** (`C:\Project\codeXplorer\capestone\repository\AnimalAlert\`).

**Why:** existing LLM-generated AnimalAlert planning docs (plan.md, implementation_plan.md, etc.) read as ideas, not plans — Roadmap bullets bundle multiple unrelated actions into one line ("взяв і зробив" granularity was the author's own diagnostic test). Building a general tool (not AnimalAlert-specific) to fix this.

**Key empirical findings (real runs, not just code-reading):**
- Organic leaf-detection in `deepagent_md` (originally assumed: model stops subdividing once content is "concrete enough") is **falsified**, not just unverified — 3 real runs (`plan1`/`plan10` at maxdepth1, `plan2` at maxdepth2, `plan3` at maxdepth4, all under AnimalAlert's `.1bcoder/planMD/`) all show the model *never* stops on its own; every leaf still contains further `## N.` sections that would expand if allowed.
- Branching factor measured directly: ~3–5 children per node at every depth tested. Cost is exponential in depth, not linear (`plan3`'s `item_1` subtree alone: 57 files, ~97 minutes, still not done when interrupted).
- Depth must be **externally capped** (same convention as `deepagent_code`'s `is_leaf = current_depth >= max_depth`), never a generation-time model decision.

**Design resolution:** two-pass architecture — (1) generate via unmodified `deepagent_md` to whatever `--maxdepth`, (2) a separate transform pass splits each real leaf's own `## N.` sections into atomic spec units (WHO/WHAT/WHEN, SMART, INVEST, acceptance criteria, boundaries). C4-model decomposition (Container/Component/Code) used as the architecture-lens vocabulary for `--plan:` labels, not a mandatory Agile epic/story/task/subtask taxonomy.

**Open, unresolved (flagged explicitly in the doc, not glossed over):**
- Non-leaf C4 nodes (`item_1.2`) are NOT equivalent to Agile epic/story — C4 is a horizontal (architecture-location) slice, epic/story is vertical (user-capability, cross-branch). A genuine epic/story tree can likely be produced by running `deepagent_md` a **second, independent time** over the same root task with business-capability-phrased `--plan:` labels (e.g. `epic, user story`) instead of clustering the C4 tree after the fact — promising, not yet validated empirically.
- Non-leaf/container nodes may need their own lighter "Integration spec" template (contract between children, integration order, integration-level acceptance criteria) distinct from leaf-level unit specs — proposed, not yet written into the doc.

**Implemented this session** (not just designed): `deepagent_md.py` gained a `continue <plan_dir> plan: <new label>` subcommand — persists run params (task/plan_labels/aspects/cfg) to `<plan_dir>/_deepagent_meta.yaml` on every fresh run, lets you expand an existing tree's current leaves by exactly one more labeled level without re-guessing `--maxdepth` upfront or paying for a uniform extra level everywhere. Reuses `_expand()` unchanged (pointing it at an existing leaf file with a raised `max_depth` was sufficient — file-exists-skip-then-recurse was already the right behavior). See [[feedback-yaml-over-json]] — meta file uses YAML per explicit user preference, not JSON.

Related: [[feedback-explicit-consent-destructive]] pattern of correction-and-fix also applied here (the original "organic leaf detection" claim in the doc was corrected in-place after empirical falsification, not just appended as a caveat).
