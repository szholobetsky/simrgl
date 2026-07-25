# Prompt and Context Management: Radogast & Prove

*Stanislav Zholobetsky, 2026*

---

## Core Observation

Context state matters more than model size for task solving. A large model with a drifted
context will fail. A smaller model with a clean, targeted context will succeed. Context is
the working memory of an LLM — if it fills with noise, the model cannot focus regardless
of parameter count.

This document describes a universal framework for monitoring, measuring, and correcting
context state across any AI agent system.

---

## The Two Tools

**Radogast** — static analysis. Given context C and target T, measures alignment, coverage,
noise, and current process state. Named after the Slavic god of hospitality: prepares the
space for the task.

**Prove** — dynamic validation. Verifies through a metacognitive probe whether the LLM
actually internalized the context. Named after the Slavic god of law: confirms that the
context holds what it claims to hold.

Both tools are external to any specific agent and connect via MCP server or stdin pipe.

---

## The Target Document

A target is not a single sentence goal. It is a structured specification with temporal
stages, domain markers, and falsification criteria.

### Minimal Target (3 fields, always required)

```
goal:           one sentence — what we want to achieve
key_terms:      domain vocabulary that must appear in context
critical_tests: conditions that would prove the goal was NOT met
```

### Full Target Schema

```yaml
target:
  goal: "develop REST API for tracking visitor dwell time at a café"

  milestones:
    - name: "domain_understood"
      markers: [visitor, establishment, entry, exit, timestamp]
      evidence: "definition of 'visitor' as an entry+exit event pair"

    - name: "data_model_defined"
      markers: [schema, table, field, device_id, coordinates]
      evidence: "structure of input data is specified"

    - name: "api_designed"
      markers: [endpoint, POST, GET, /visitors, /sessions, response]
      evidence: "at least one endpoint with request/response schema"

    - name: "calculation_specified"
      markers: [average time, sum, count, formula, minutes]
      evidence: "formula for average dwell time is present"

  key_terms: [REST, API, tracking, session, timestamp, geolocation]

  marker_words:
    domain:  [café, store, visitor, client]
    data:    [schema, model, field, type]
    api:     [endpoint, route, method, status code]
    result:  [average time, dashboard, report, analytics]

  falsification:
    critical_tests:
      - "if 'visitor' is never defined — the problem is not formulated"
      - "if there is no formula — the API exists but the task is unsolved"
      - "if only REST theory, no domain terms — context has drifted"
    minimum_evidence:
      - at least one endpoint specification
      - a formula or pseudocode for dwell time calculation
      - definition of input data structure

  out_of_scope: [authorization, billing, UI, scaling]
```

---

## The Smetana Principle

Context evolves through time according to process stages. Marker words signal which stage
is currently active in the context window.

Dumpling example:
- **flour** appears → dough stage is active
- **boiling water + skimmer** appear → cooking stage is active
- **smetana** appears → serving stage: task is complete

Smetana (the finishing marker) is meaningful only when it appears after the earlier stage
markers. If smetana appears before flour — something is wrong.

This is not always a linear sequence. For iterative tasks (ML training loops, refactoring
cycles), each iteration reaches its own "smetana" and restarts. In these cases, milestone
ordering should not be strictly enforced — the system should detect cycles rather than
linear progress.

### Process Type Detection

| Pattern | Type | Milestone enforcement |
|---|---|---|
| Sequential stages, each appears once | Linear process | Strict ordering |
| Same markers repeat N times | Iterative/cyclic | Detect cycles, no strict order |
| Stages overlap or skip | Exploratory | Soft ordering only |

---

## What Radogast Measures

### 1. State Detection

```
current_window = last [1, 3, 5] messages  (configurable, default: all three)
for each milestone → count(markers ∩ current_window)
active_state = milestone with highest score
```

Multiple window sizes run simultaneously. They vote: if the 1-message window shows
"result" markers but the 5-message window still shows "domain" markers — the context
is transitioning, not settled.

### 2. Term Coverage

For each key term from the target:

| Status | Condition |
|---|---|
| `defined` | definition pattern found ("X is...", "X — це...", section header) |
| `mentioned` | term appears but without surrounding definition |
| `absent` | term does not appear in context |

Coverage ratio = defined_terms / total_key_terms

### 3. Drift Score

```
origin_vector = embed(goal + key_terms)
current_vector = embed(current_window)
drift_angle = arccos(cos(origin_vector, current_vector))
```

| Angle | Status |
|---|---|
| 0°–40° | On track |
| 40°–65° | Warning: divergence detected |
| 65°+ | Critical: context has drifted |

Threshold: 40° (configurable). Above this → alert.

### 4. Balance Check

```
coverage_scores = [freq(term) for term in key_terms]
bias = max(coverage_scores) / mean(coverage_scores)
```

If bias > 3.0: one term dominates. The context explains REST theory at length
but barely mentions the domain it serves.

### 5. Transition Validation (linear process only)

If active milestone N is detected but milestone N-1 has no evidence → warning:
"Building the API without having defined the domain model."

### 6. Falsification Check

For each `critical_test` in the target: verify the evidence condition.
If `minimum_evidence` is not met → `fail`.

---

## What Prove Validates

Static analysis tells you what is in the context. Prove asks the LLM what it
understood.

**Metacognitive probe**: after building context, Prove sends:
> "Based on our conversation so far, define the following terms: {key_terms}"

Compare the LLM's definitions against the glossary extracted by Radogast:
- term defined correctly → pass
- term undefined or wrong → context coverage insufficient for that concept

This catches cases where context contains the words but not the meaning — the
difference between mentioning "timestamp" and actually explaining what event
generates it and what format it uses.

---

## Glossary Extraction

Radogast can extract a glossary from context automatically.

Definition detection patterns:
```
"{term} is ..."
"{term} — це ..."
"{term} refers to ..."
"{term}: ..." (inline definition style)
"## {term}" (section header followed by body)
first occurrence of term + surrounding 2 sentences
```

Output:
```
GLOSSARY (4/6 terms extractable):
  REST API: "architectural style for web services using HTTP..."
  session:  "pair of entry and exit events for one visitor..."
  dwell time: "duration = exit.timestamp - entry.timestamp"
  visitor: [not defined — only mentioned]  ← gap
```

---

## Hybrid Measurement

Both embedding similarity and marker word counting run in parallel. They measure
different things:

| Method | Detects | Blind to |
|---|---|---|
| Marker words | exact domain vocabulary | synonyms, paraphrase |
| Embeddings | semantic direction | precise term usage |

A drift alert fires when **either** method exceeds threshold. Agreement between both
methods gives higher confidence. Disagreement (high embedding similarity but low
marker coverage, or vice versa) signals an interesting edge case worth inspecting.

---

## Configuration

```yaml
radogast:
  windows: [1, 3, 5]          # message window sizes to evaluate simultaneously
  drift_threshold_deg: 40     # alert above this angle
  bias_threshold: 3.0         # alert when max/mean coverage ratio exceeds this
  embedding_model: "bge-large" # or any sentence-transformers compatible model
  hybrid: true                 # use both marker words and embeddings
```

---

## Architecture: Universal Connection

Radogast and Prove connect to any AI agent system.

```
┌──────────────────────────────────────────────────────────┐
│                    AI AGENT ECOSYSTEM                     │
│  1bcoder · OpenCode · Codex · aider · Continue.dev (cn)  │
│  Claude · Gemini · nanocoder · pi                        │
└───────────────┬──────────────────────────────────────────┘
                │  MCP protocol  /  stdin pipe  /  file watch
                ▼
┌──────────────────────────────────────────────────────────┐
│                      YASNA (extended)                    │
│  context reader + session indexer                        │
│  keyword search     ← existing                           │
│  vector search      ← new: semantic session retrieval    │
│  drift history      ← new: drift_angle timeline per session│
└───────────┬──────────────────────────────────────────────┘
            │
    ┌───────┴────────┐
    ▼                ▼
┌──────────┐    ┌──────────┐
│ RADOGAST │    │  PROVE   │
│ static   │    │ dynamic  │
│ coverage │    │ probe    │
│ drift    │    │ glossary │
│ glossary │    │ verify   │
│ balance  │    │          │
└──────────┘    └──────────┘
```

### Connection modes

**MCP server** (broadest compatibility — Claude, Continue.dev, Codex CLI):
```
radogast --mcp-port 3700
```
Exposes tools: `analyze_context`, `get_drift_score`, `suggest_refocus`

**stdin pipe** (any tool that can export history):
```bash
cat ~/.aider/history.json | radogast --target task.target.yaml
opencode export | radogast --watch
```

**File watch** (tools that write session files):
```bash
radogast --watch ~/.continue/history/ --target task.target.yaml
```

---

## Yasna Extension: Vector Search

Current yasna: keyword (BM25) search over context history.

Extended yasna:
1. **Embed on write**: each message/session receives a vector at save time
2. **Vector search**: "find sessions where I was solving REST API tracking problems"
   → embed query → cosine search over stored session vectors
3. **Drift index**: for each session, store `drift_timeline` — array of drift angles
   over time, queryable: "find sessions where I stayed on track throughout"
4. **Good context retrieval**: sessions where drift stayed below 40° AND task was
   completed → use as positive examples for context construction

---

## Target Derivation Without an Explicit Target

Any prompt contains an implicit target. Derivation algorithm (no LLM required):

1. Extract intent verb: write, develop, explain, compare, debug, refactor
2. Extract subject: article, API, explanation, ...
3. Extract domain: Mozart, REST, Python, ...
4. Generate critical tests from templates:
   - "if result contains no {domain} terms → goal not met"
   - "if {subject} is absent from output → goal not met"
5. Extract key_terms: nouns from prompt, filtered by domain specificity

This gives a minimal working target for any prompt, even one written with no
engineering intention.

---

## /target Commands (1bcoder integration)

```
/target set <file.yaml>      load target for current session
/target show                 display current target
/target status               current milestone, drift score, falsification status
/target milestone next       manually advance to next stage
/target milestone list       show all milestones and their evidence status
```

Radogast runs automatically after each message when a target is active.
Alert threshold crossing → inline warning before next AI response.

---

## Open Questions for Empirical Validation

The following require testing on real sessions before finalizing weights:

1. Which window size (1, 3, or 5 messages) correlates best with human judgment
   of "context relevance"?
2. Does the 40° drift threshold correctly separate successful from failed sessions?
3. Do marker words outperform embeddings for state detection on small models (1b–8b)?
4. For cyclic tasks: what is the right metric when smetana appears on every iteration?
5. Does the metacognitive Prove probe correlate with actual task success rate?

Methodology: run Radogast on sessions where LLM succeeded and where it failed.
Compare metric distributions. Adjust thresholds and weights accordingly.

---

## Relation to Existing Concepts

- `PROMPT_AND_CONTEXT_EVALUATION.md` — metrics for prompt quality and reply scoring
  without LLM calls. Complementary: that document scores prompts; this one manages
  the context lifecycle.
- `yasna_system_spec.md` — context memory architecture. Radogast/Prove depend on
  yasna for context access and historical search.
- `FILTERING_CONTEXT_CONCEPT.md` — context filtering strategies. Radogast provides
  the signal (what is noise); filtering applies the correction.
