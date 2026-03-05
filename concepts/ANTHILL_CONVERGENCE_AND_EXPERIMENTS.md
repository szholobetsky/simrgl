# Anthill Convergence: Conceptual Lineage and Experimental Validation Framework

## Abstract

This document serves two purposes. First, it demonstrates that the Anthill Distributed Cognitive OS is not an architectural rupture within the SIMARGL project but a **convergence point** — the natural destination toward which every prior concept was independently moving. We trace seven direct conceptual lineages from earlier SIMARGL documents to specific Anthill components, showing that the ideas did not need to be invented anew but only recognized as a system. Second, we propose a structured program of **twelve experiments** across four experimental families, designed to provide empirical evidence for (or against) the Anthill architecture's core hypotheses. We define eight new metrics specific to the Anthill paradigm, and show how existing SIMARGL metrics (Novelty@K, Structurality@K, SES, Grounding Accuracy) apply — often unchanged — to the new context.

---

## Part I: Conceptual Lineage — The Ideas That Were Already There

> "We do not invent — we recognize. The concepts were always related; we simply lacked the architectural frame in which to see them as a single system."

The table below maps each major SIMARGL concept to its Anthill analog. The mapping is not loose metaphor — it is structural correspondence.

| SIMARGL Concept | Source Document | Anthill Realization | Role in Anthill |
|-----------------|-----------------|---------------------|-----------------|
| Entity Map (business terms ↔ code files) | `KEYWORD_ENTITY_MAPPING.md` | Ontological Knowledge Graph (OKG) | The OKG is the entity map made persistent, versioned, and machine-queryable at runtime |
| Object Passport (first described implicitly) | `FINAL_PRODUCT.md` § codeXport | Object Passport (YAML) | Identical concept, formalized as the agent's minimum context unit |
| Keyword as Semantic Coordinates (positive/negative space) | `KEYWORD_INDEXING.md` | OKG Context Tags + Passport Architectural Tags | Tags are the Anthill's encoding of the semantic coordinate idea |
| Two-Phase Reflective Agent (Reason → Reflect) | `TWO_PHASE_REFLECTIVE_AGENT.md` | Full Anthill pipeline (Architect → Coder → Auditor) | Phase 1 = Architect+Coder; Phase 2 = Adversarial Critic + Auditor |
| Three Specialized Agents (proposed in FURTHER_RESEARCH) | `FURTHER_RESEARCH_RECOMMENDATION.md` §2.1 | 10-role agent roster on GPU farm | Direct generalization: 3 → 10, conceptually identical division of cognitive labor |
| Dual-Server RAG (RECENT vs ALL) | `DUAL_SERVER_RAG_EVALUATION.md` | Memory hierarchy tiers L3–L4 | RECENT = active OKG; ALL = historical vector RAG (Qdrant/Chroma) |
| Cross-Layer Transformation Embeddings | `CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md` | OKG Pattern Recognition (Saga, CQRS) | A Saga IS a specific cross-layer transformation topology; the OKG encodes it as a subgraph pattern |
| Symbol Grounding Problem (LLMs lack embodiment) | `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` §3 | Qualia Horizon + Map of the Unknowable | Instead of solving ungroundability, Anthill maps it and routes around it |
| Hermeneutic Circle (text ↔ system, iterative understanding) | `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` §6 | Kantian Observer Loop (code ↔ OKG synchronization) | The Observer Loop IS the hermeneutic circle operationalized as a validation trigger |
| SIMARGL Novelty/Structurality metrics | `SIMARGL_concept.md` | OKG-guided change validation | Changes violating structurality (cross-module) trigger Auditor escalation |
| Adaptive Fusion (RECENT + ALL query weighting) | `DUAL_SERVER_RAG_EVALUATION.md` | Memory tier selection in the Librarian agent | Librarian queries L3 (OKG) first, escalates to L4 (RAG) on miss |
| Affordances and Constraints (digital embodiment proxies) | `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` §5 | SKILL.md tool manifest + OKG schema constraints | SKILL.md encodes affordances; OKG schema enforces constraints |

### 1.1 The codeXport → OKG Lineage

The most direct lineage runs from **codeXport** to the **Ontological Knowledge Graph**. `FINAL_PRODUCT.md` describes codeXport as building:

1. A Word2Vec model trained on code identifiers
2. A `keyword_index` table linking identifiers to semantic vectors
3. An `entity_map` linking business terms to files and related entities
4. Positive/negative space filtering on these entities

The Anthill OKG is this identical structure, extended in three dimensions:

- **Runtime persistence**: codeXport builds the map once before deployment; the OKG is maintained continuously by the Notary agent
- **Graph structure**: codeXport uses an entity map (JSON/SQLite); OKG uses RDF triplets in a graph database (Neo4j/FalkorDB), enabling multi-hop traversal
- **Validation feedback**: codeXport is read-only at inference time; OKG is updated after every code change via the Observer Loop

In other words: **codeXport is the OKG without the runtime**. Add the Notary agent and the Observer Loop to codeXport, and you have the core of the Anthill OS knowledge layer.

### 1.2 The Two-Phase Agent → Anthill Pipeline Lineage

`TWO_PHASE_REFLECTIVE_AGENT.md` proposes:
- **Phase 1**: Reasoning — extract keywords, search, build initial hypothesis
- **Phase 2**: Reflection — critique Phase 1 output, verify consistency, refine

`FURTHER_RESEARCH_RECOMMENDATION.md` §2.1 explicitly proposes three specialized agents: a Reasoning Agent, a Search/Context Agent, and a Refinement/Synthesis Agent, with a "meta-agent orchestrator."

The Anthill pipeline is this proposal, scaled and made concrete:

```
Two-Phase Agent           Three Specialized Agents      Anthill Pipeline
─────────────────         ──────────────────────────    ─────────────────────────
Phase 1: Reason     →     Reasoning Agent          →    Router + Architect
Phase 1: Search     →     Search/Context Agent     →    Librarian (MCP queries)
Phase 2: Reflect    →     Refinement Agent         →    Auditor (DeepSeek-R1)
[implicit]          →     Orchestrator             →    Router (LangGraph)
[implicit]          →     [no equivalent]          →    Coder + Tester + Notary
```

The Crooked Wall Principle (adversarial integrity) is the operationalization of what `FURTHER_RESEARCH_RECOMMENDATION.md` §3.3 calls "Continuous Verification & Feedback" — the same idea, but instead of post-hoc learning from developer corrections, Anthill performs synchronous OKG consistency checking at the moment of change.

### 1.3 The SIMARGL Metrics → OKG Validation Lineage

The SIMARGL Novelty/Structurality framework was designed to classify whether a recommended code change is EVOLUTION (ideal), DISRUPTION (risky), MAINTENANCE (safe), or STAGNATION (useless).

In the Anthill OS, these same metrics can be computed *automatically by the Observer Loop* after any code change:

- **Novelty**: does this change add a new edge to the OKG (new relationship) or modify an existing one?
- **Structurality**: do the source and target nodes of the new edge belong to the same module in the OKG?

This means the SIMARGL 2×2 matrix becomes a **real-time architectural health monitor**. An Anthill Auditor that observes a code change scoring high Novelty + low Structurality (DISRUPTION) can:
1. Flag it for mandatory human review
2. Require architectural justification before proceeding
3. Route to the Oracle for large-model assessment

The SIMARGL metrics were originally designed as offline evaluation tools. The Anthill architecture converts them into online, per-change decision triggers.

### 1.4 The Phenomenological Grounding → Observer Loop Lineage

`PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` Section 6 develops the hermeneutic circle: to understand a code element, one must understand the system; to understand the system, one must understand its elements. This circularity is the fundamental structure of interpretation.

The Kantian Observer Loop in Anthill is this hermeneutic circle made mechanical:

```
Hermeneutic Circle                Observer Loop
──────────────────────            ──────────────────────────────
Read element in context    →      Scanner reads changed file
Interpret against system   →      Compare against OKG (system model)
Update system understanding →     Update OKG with new triplets
Re-read element            →      Re-validate changed file
```

The loop terminates when hermeneutic convergence is achieved: the element (file) and the system (OKG) are mutually consistent. This is not philosophical decoration — it is the precise termination condition for the Observer Loop.

---

## Part II: Experimental Validation Framework

### Preamble: What Are We Trying to Prove?

The Anthill architecture rests on several falsifiable claims — hypotheses in Popper's strict sense. Each experiment below is designed to either confirm or refute one or more of these claims. We do not assume the architecture is correct; we assume it is a well-formed hypothesis deserving rigorous testing.

**Core Hypotheses:**

| ID | Hypothesis | Prediction |
|----|-----------|------------|
| H1 | OKG Memoization | Small models (3B) with OKG context match large models (14B) cold on code modification tasks |
| H2 | Adversarial Integrity | Pipelines with a Critic agent produce fewer OKG-inconsistent changes than single-model pipelines |
| H3 | Specialization | A role-tuned 1B model outperforms a general 7B model on its specific sub-task |
| H4 | PCIe Sufficiency | Mining rig (x1) agent parallelism achieves comparable throughput to x16 single-GPU sequential |
| H5 | Qualia Horizon | Task types with emergence threshold > 14B are empirically distinguishable by surface features |
| H6 | Precomputed Memoization | Precomputed OKG reduces LLM call count by ≥50% compared to cold-start for identical tasks |
| H7 | SIMARGL Coherence | OKG-guided changes score higher Evolution@K than unguided changes |
| H8 | Oracle Privacy | Reduced-context (passport-only) Oracle queries achieve ≥80% of full-context query quality |

---

### New Metrics for Anthill Validation

Before the experiments, we define eight metrics not present in existing SIMARGL documentation. All existing SIMARGL metrics (MAP, MRR, Novelty@K, Structurality@K, SES, HES, Grounding Accuracy, Horizon Completeness) carry over unchanged.

#### M1: Passport Retrieval Accuracy (PRA@K)

**Definition**: Given a task description, PRA@K measures the fraction of Object Passports returned by the Librarian agent that correspond to files actually modified in the ground-truth commit.

```
PRA@K = |{passports in top-K} ∩ {ground truth files}| / |{ground truth files}|
```

**Relation to existing metrics**: PRA@K is the retrieval accuracy of the OKG layer, analogous to MAP but measured on passports rather than raw file embeddings. Comparison between PRA@K and standard MAP@K (raw embedding retrieval) directly tests H1: if PRA@K > MAP@K for small models, the OKG is adding information beyond embedding similarity.

**Target**: PRA@K ≥ 0.70 at K=10 (vs MAP≈0.37 baseline from exp2)

#### M2: OKG Consistency Score (OCS)

**Definition**: After a code change, the fraction of OKG assertions about the modified file(s) that remain logically consistent (no dangling references, no violated type constraints, no contradictory relationship claims).

```
OCS = 1 - (|inconsistent triplets| / |total triplets involving changed nodes|)
```

**Measurement**: Requires a consistency checker (a deterministic rule set, not an LLM) that validates OKG graph properties: no method listed in a passport exists only in a deleted file, no dependency listed in a passport points to a non-existent node, etc.

**Target**: OCS ≥ 0.95 after Observer Loop completes (i.e., the loop may need multiple iterations, but must converge)

#### M3: Memoization Hit Rate (MHR)

**Definition**: Fraction of agent sub-queries (calls to `get_context()`, `get_passport()`, `query_pattern()` via MCP) that are answered from the OKG cache without requiring a fresh LLM derivation or file read.

```
MHR = |sub-queries answered from OKG| / |total sub-queries|
```

**Interpretation**: MHR measures the practical effectiveness of memoization. If MHR is low, the OKG is not dense enough to support agent reasoning — agents must read raw files, defeating the compression purpose. If MHR is high (> 0.80), the OKG is providing the functional equivalent of the large model's internalized knowledge.

**Target**: MHR ≥ 0.80 for mature OKG (after ≥ 500 tasks processed)

#### M4: Agent Hallucination Rate (AHR)

**Definition**: Fraction of generated code changes containing at least one factual error about the codebase: wrong method signature, non-existent class reference, incorrect import path, wrong argument type.

```
AHR = |changes with ≥1 factual error| / |total changes generated|
```

**Measurement**: Automated static analysis (AST parsing) plus compilation attempt. A change that fails to compile or that references non-existent symbols is flagged as a hallucination.

**Comparison baseline**: AHR for same model (3B, 7B, 14B) working without OKG context (raw file read only).

**Target**: AHR ≤ 0.10 with OKG context (vs expected AHR ≈ 0.35–0.50 for cold 3B model)

#### M5: Cascade Correction Depth (CCD)

**Definition**: Average number of Critic-Coder iteration cycles before the Auditor approves a change.

```
CCD = (1/N) Σ iterations_until_approval_i
```

**Interpretation**: CCD = 1 means the Coder produced correct output on the first attempt. CCD = 3 means the Critic caught errors twice before the Auditor approved. CCD > 5 for most tasks suggests the architecture is not converging and the task may exceed local capability.

**Target**: CCD ≤ 2.5 for "standard" tasks (single-module modifications); CCD > 5 triggers Qualia Horizon escalation

#### M6: Qualia Boundary Detection Precision (QBDP)

**Definition**: Precision of the Router's classification of a task as "requires large model / human." Measured against ground truth: tasks where the local pipeline consistently fails (CCD > 5 for ≥ 3 independent runs) are labeled as true positives.

```
QBDP = |correctly flagged Qualia tasks| / |total flagged Qualia tasks|
```

**Complement metric**: Qualia Boundary Detection Recall (QBDR) = fraction of true Qualia tasks that were correctly flagged (measures learned helplessness risk: low QBDR means the system is routing too many tasks to Oracle, wasting resources).

**Target**: QBDP ≥ 0.85, QBDR ≥ 0.70

#### M7: Human Attention Time Fraction (HATF)

**Definition**: Fraction of total task completion time (wall clock) during which a human must be actively present. The remainder is autonomous pipeline time.

```
HATF = human_active_time / total_completion_time
```

**Interpretation**: HATF = 1.0 means the human must be present the entire time (no automation benefit). HATF = 0.05 means 95% of task time is autonomous. This is the primary metric for the "freeing human time" hypothesis of the Anthill OS.

**Target**: HATF ≤ 0.10 for standard development tasks (feature addition, bug fix)

#### M8: Structural Drift Rate (SDR)

**Definition**: Over N consecutive code changes, the rate at which the OKG's structural metrics (module cohesion, dependency depth, pattern density) degrade compared to the initial state.

```
SDR = (structural_quality_t0 - structural_quality_tN) / N
```

Where structural quality is a composite: mean module cohesion (fraction of intra-module edges), pattern coverage (fraction of modules with recognized architectural pattern label), and OCS.

**Interpretation**: SDR measures whether the Anthill OS, over time, helps the codebase evolve cleanly (SDR ≈ 0 or negative, meaning improvement) or allows it to drift toward architectural incoherence (positive SDR).

**Target**: SDR ≤ 0.0 over 100 consecutive changes (system should maintain or improve structural quality)

---

### Family A: OKG Effectiveness Experiments

These experiments test whether the Ontological Knowledge Graph actually provides the context efficiency gains claimed in H1 and H6.

#### Experiment A1: Passport vs Raw — Context Compression Battle

**Hypothesis tested**: H1 (OKG Memoization)

**Design**: Take 100 code modification tasks from the SIMARGL ground-truth dataset (sonar.db, kafka.db). For each task:

- **Condition 1 (Baseline)**: Give a 3B model the raw source files of the relevant modules (average: ~3000 tokens per task) and ask it to generate the required code change
- **Condition 2 (OKG)**: Give the same 3B model only the Object Passports of the relevant modules (average target: ~300 tokens per task) and ask it to generate the required code change
- **Condition 3 (Oracle)**: Give a 14B model the raw source files (this is the performance ceiling)

**Metrics**: AHR (M4), correctness (does the change compile and pass unit tests), PRA@5 (M1)

**Expected result**: Condition 2 (3B + OKG) significantly outperforms Condition 1 (3B + raw) on AHR, and approaches Condition 3 (14B + raw) on correctness.

**Falsification condition**: If Condition 2 does not improve over Condition 1, the OKG compression is not adding value — raw file reading is equivalent or better for 3B models.

**Data source**: Existing sonar.db, kafka.db from exp3 (9,799 tasks with ground truth file changes). Extract 100 tasks with single-module changes for cleaner measurement.

---

#### Experiment A2: OKG Density and Memoization Maturity

**Hypothesis tested**: H6 (Precomputed Memoization)

**Design**: Run the Anthill pipeline on the same 50 tasks three times:

- **Phase α (cold OKG)**: OKG contains only automatically extracted AST triplets, no task history
- **Phase β (warm OKG)**: OKG updated after processing the first 25 tasks (passports updated, patterns recognized)
- **Phase γ (mature OKG)**: OKG updated after processing all 50 tasks, plus overnight enrichment pass

**Metrics**: MHR (M3) measured at each phase, CCD (M5), wall-clock time per task

**Expected result**: MHR increases monotonically from α to γ (memoization becomes more effective as the OKG fills in). CCD decreases from α to γ as agents find more answers in the graph without needing to re-derive them.

**Falsification condition**: If MHR plateaus at < 0.5 even in Phase γ, the OKG is not dense enough to support agent reasoning for this codebase scale.

---

#### Experiment A3: OKG Pattern Recognition — Saga Detection

**Hypothesis tested**: Prerequisite for H5 — can the OKG encode and recognize architectural patterns?

**Design**: Manually label 10 subgraphs in kafka.db as "Saga pattern" (Kafka is an event-driven system with compensation logic — ideal ground truth). Train the Pattern Recognizer (GNN or prompted 4B model with OKG context) to classify subgraphs. Evaluate on 10 held-out subgraphs.

**Metrics**: Pattern recognition accuracy (standard classification), PRA@3 for pattern-aware retrieval (does knowing "this is a Saga" improve file retrieval?)

**Expected result**: ≥ 80% pattern recognition accuracy. PRA@3 improves by ≥ 0.15 when pattern context is provided to the retrieval agent.

**Significance**: This experiment validates the core claim that the OKG enables architectural knowledge to be encoded as machine-queryable structure — the prerequisite for all higher Anthill intelligence.

---

### Family B: Multi-Agent Integrity Experiments

These experiments test the Adversarial Integrity (H2) and Specialization (H3) hypotheses.

#### Experiment B1: The Crooked Wall — Does Adversarial Architecture Reduce Hallucinations?

**Hypothesis tested**: H2 (Adversarial Integrity)

**Design**: 60 code modification tasks (medium complexity — 2–3 file changes each). Three conditions:

- **Condition Single**: One 7B model generates the complete change, no review
- **Condition Compliant Pair**: 3B Coder + 3B Reviewer, but the Reviewer is instructed to be "helpful and supportive" (simulate RLHF compliance training)
- **Condition Adversarial Pair**: 3B Coder + 3B Critic, where the Critic is instructed: "Find everything wrong. Your job depends on finding errors. Do not agree with the Coder."

**Metrics**: AHR (M4), OCS (M2) post-change, CCD (M5)

**Expected result**: Adversarial Pair has lower AHR and higher OCS than both Single and Compliant Pair. CCD for Adversarial Pair is higher than Single (more iterations) but total quality is higher.

**Critical finding expected**: Compliant Pair may perform *worse* than Single — two models agreeing on a hallucination is worse than one model hallucinating alone. This would empirically validate the RLHF compliance trap hypothesis.

**Falsification condition**: If Compliant Pair ≈ Adversarial Pair on AHR, adversarialism is not necessary — standard multi-model review is sufficient.

---

#### Experiment B2: Specialization vs Generality — The 1B vs 7B Showdown

**Hypothesis tested**: H3 (Specialization)

**Design**: Take three sub-tasks that appear in the Anthill pipeline:
1. **Test generation**: Given a function signature + docstring, write pytest tests
2. **Dependency extraction**: Given a Python file, list all external imports and their roles
3. **OKG passport generation**: Given an AST of a Java file, produce a valid Object Passport YAML

**Models compared**:
- 1B model (Llama-3.2-1B) with role-specific few-shot examples (5 examples per sub-task)
- 7B model (Llama-3-7B) with zero-shot prompt
- 7B model (Llama-3-7B) with role-specific few-shot examples

**Metrics**: Task-specific quality score (test coverage %, dependency extraction F1, passport schema validation %)

**Expected result**: 1B + few-shot beats 7B zero-shot on all three narrow sub-tasks. 7B + few-shot sets the ceiling. The gap between 1B + few-shot and 7B + few-shot is ≤ 15% — suggesting specialization closes most of the parameter-count gap.

**Implication**: If this holds, the Anthill architecture of many specialized small models is empirically justified over a single large generalist model for pipeline sub-tasks.

---

#### Experiment B3: Majority Voting — Does Multi-Instance Debate Help?

**Hypothesis tested**: Extension of H2

**Design**: For 30 tasks with objectively verifiable correct answers (e.g., "add null check to this method"), compare:
- **Single 3B**: One attempt, take the result
- **3×3B with voting**: Three independent 3B instances, Critic selects best
- **3×3B with debate**: Three instances, each sees the others' output and can revise once

**Metrics**: AHR (M4), percentage of tasks producing a compilable, correct change

**Expected result**: 3×3B with debate > 3×3B with voting > Single 3B. If debate adds less than 10% over voting, simpler majority voting is preferred (lower compute).

---

### Family C: Hardware Architecture Experiments

These experiments test H4 (PCIe Sufficiency) — the mining rig feasibility hypothesis.

#### Experiment C1: PCIe x1 vs x16 Throughput — Text-as-API Measurement

**Hypothesis tested**: H4 (PCIe Sufficiency)

**Design**: On a single machine with multiple GPU slots, compare:
- **Setup A**: Single 7B model on GPU in x16 slot, processes 10 tasks sequentially
- **Setup B**: Two 3B models, one on x16 slot, one on x1 riser slot, process 10 tasks with the split: Model 1 does planning, Model 2 does code generation
- **Setup C**: Two 3B models, both on x1 riser slots, same split as Setup B

**Metrics**: Total wall-clock time for 10 tasks, tokens-per-second per GPU, inter-model communication latency (time for message to travel from one Ollama instance to another)

**Expected result**: Communication latency between models via HTTP (text) is < 5ms regardless of PCIe slot configuration. Throughput of Setup B ≈ Setup C, both within 15% of Setup A. The bottleneck is inference time, not the bus.

**Measurement method**: Instrument the LangGraph orchestrator to log precise timestamps for: task received, first model call issued, first model response received, second model call issued, second model response received, result returned.

---

#### Experiment C2: Mining Rig Scaling — Linear Throughput Verification

**Hypothesis tested**: Extension of H4 — does adding more GPUs produce proportional throughput gains?

**Design**: On the full 10-GPU mining rig, run batches of 10, 20, 40, and 80 independent tasks (tasks that do not depend on each other — e.g., 80 different single-function modifications). Measure:
- Total wall-clock time
- Per-task average time
- GPU utilization per card

**Expected result**: Throughput scales approximately linearly with GPU count for independent tasks. Per-task time when running 80 tasks in parallel across 10 GPUs approaches per-task time for 10 tasks on 1 GPU (i.e., no queueing bottleneck).

**Falsification condition**: If per-task time doubles when parallelism increases from 5 to 10 GPUs, the orchestration overhead (Router, message bus) is bottlenecking — the architecture needs optimization.

---

### Family D: Qualia Horizon and Meta-Cognitive Experiments

These experiments address the most philosophically interesting hypotheses: H5, H7, and H8.

#### Experiment D1: Mapping the Qualia Horizon — Empirical Threshold Discovery

**Hypothesis tested**: H5 (Qualia Horizon)

**Design**: Create a benchmark of 100 diverse software tasks, manually labeled on two axes:
1. **Type**: Structural (logic, data manipulation, API changes) vs Aesthetic (UI design, naming, style) vs Architectural (pattern selection, refactoring strategy)
2. **Scale requirement**: what is the minimum model size that consistently (≥80% success) produces a correct result? (Determined empirically by testing on 1B, 3B, 7B, 14B, and 30B+ models)

Run the Anthill Router on these 100 tasks (blind — Router does not see the labels). Measure QBDP (M6) and QBDR against the manual ground truth.

**Expected result**: Router achieves QBDP ≥ 0.80 for Aesthetic/Architectural tasks. Surface features that predict Qualia boundary: presence of terms like "beautiful," "elegant," "natural," "refactor all," "redesign" in the task description; number of files in ground truth > 15; cross-module dependency count > 3.

**Scientific value**: This experiment establishes an empirical vocabulary of Qualia-boundary indicators that can be formalized into the OKG Meta-Knowledge schema.

---

#### Experiment D2: SIMARGL Metrics as Real-Time Architectural Health Monitor

**Hypothesis tested**: H7 (SIMARGL Coherence)

**Design**: Process 50 tasks with the Anthill pipeline (OKG-guided changes). Process the same 50 tasks with a baseline (single 7B model, no OKG, changes applied directly). After all 100 runs (50 per condition), compute SIMARGL metrics on the resulting set of code changes:
- Novelty@10, Structurality@10, SES, HES
- Evolution@10 (fraction of changes that are new + within-module)
- SDR (M8) — structural drift over the sequence of 50 changes

**Expected result**: Anthill pipeline produces significantly higher Evolution@10 and SES than baseline (changes are more often new relationships within existing module boundaries, rather than disruptive cross-module connections). SDR is lower (or negative) for the Anthill pipeline.

**Significance**: This experiment directly validates the claim that the Anthill OS acts as an architectural guardian — not merely a code generator, but a system that preserves and improves structural health as it works.

---

#### Experiment D3: Oracle Privacy Preservation — Passport-Only Query Quality

**Hypothesis tested**: H8 (Oracle Privacy)

**Design**: Take 20 Qualia-boundary tasks (identified in Experiment D1). For each task:
- **Full context query**: Send the complete source files to the Oracle (Claude API) — this is the theoretical maximum quality
- **Passport-only query**: Send only Object Passports (no source code) to the Oracle — this is the privacy-preserving version

**Metrics**: Task completion quality (human-rated 1–5 scale), information leakage assessment (does the Oracle response reveal any proprietary business logic or internal identifiers from the source code?)

**Expected result**: Passport-only query achieves ≥ 80% of full-context quality on architectural guidance tasks (Oracle can give correct pattern-level advice without seeing implementation details). Information leakage: zero proprietary identifiers appear in Oracle responses when using passport-only mode.

**Falsification condition**: If quality drops below 60% with passport-only queries, the privacy-preservation protocol is too aggressive — more context must be included.

---

#### Experiment D4: Night Experiment Protocol — Autonomous Capability Expansion

**Hypothesis tested**: Can the Map of the Unknowable shrink through automated experimentation?

**Design**: Identify 5 tasks marked as Qualia-boundary (CCD > 5 historically). Run automated overnight experiments:
- 10 different decomposition strategies per task (different prompt structures, different agent assignments, different OKG query strategies)
- All run while the system is idle
- Each attempt logged with its strategy and outcome in the OKG

After 30 days of overnight experiments (5 tasks × 10 strategies = 50 new experiments), evaluate:
- Have any previously Qualia-boundary tasks become reliably solvable by local agents?
- Which strategy types correlate with success on which task types?

**Metrics**: Change in QBDR over time (recall should improve as the system learns which tasks it can now solve). Strategy success rate by task type.

**Expected result**: At least 2/5 previously Qualia-boundary tasks become locally solvable after discovering effective decomposition strategies. The system's capability boundary measurably expands — the Map of the Unknowable shrinks.

**Significance**: This is the most philosophically significant experiment. If the Map of the Unknowable can shrink through autonomous experimentation, the Anthill OS is capable of genuine learning — not weight update learning, but epistemological learning: learning what it can know.

---

## Part III: The Experimental Roadmap

The following ordering is recommended based on dependency relationships and infrastructure requirements:

```
Phase 0: Infrastructure (prerequisite for all)
  ├── Build Ontology Extractor (AST → RDF triplets)
  ├── Deploy OKG (Neo4j or FalkorDB)
  ├── Implement Notary agent (file-watcher + OKG update)
  └── Implement MCP server (get_passport, get_neighbors, query_pattern)

Phase 1: Core Validation (no GPU farm required — runs on single machine)
  ├── A1: Passport vs Raw (validates OKG premise)
  ├── B1: Crooked Wall (validates adversarial integrity)
  └── B2: Specialization (validates role-specific small models)

Phase 2: System Integration (requires LangGraph orchestrator)
  ├── A2: OKG Density and Memoization Maturity
  ├── B3: Majority Voting
  └── D2: SIMARGL as Health Monitor

Phase 3: Hardware Scaling (requires multi-GPU rig)
  ├── C1: PCIe x1 vs x16 Throughput
  └── C2: Mining Rig Scaling

Phase 4: Meta-Cognitive (requires Phase 1-3 infrastructure)
  ├── A3: Pattern Recognition (Saga Detection)
  ├── D1: Mapping the Qualia Horizon
  ├── D3: Oracle Privacy Preservation
  └── D4: Night Experiment Protocol
```

**Critical path**: A1 → A2 → D2. These three experiments form the minimum evidence set for the Anthill OS claim. If A1 shows no OKG benefit, A2 and D2 become irrelevant. If A1 succeeds, A2 confirms the memoization principle, and D2 shows that SIMARGL architectural quality is preserved.

---

## Part IV: Connection to SIMARGL Research Papers

The experimental framework above suggests three new academic papers extending the SIMARGL research:

### Paper 1: "Ontological Memoization for Small Language Model Code Assistance"

**Core contribution**: Introduces the OKG as a DP memoization table for LLM reasoning. Experiments A1, A2 provide evidence. Metrics: PRA@K (M1), MHR (M3). Shows that context compression via Object Passports enables 3B models to approach 14B quality on targeted code modification tasks.

**Connection to existing SIMARGL papers**: Extends the task-to-code retrieval work (MAP results from exp2/exp3) by showing that retrieval quality measured by PRA@K (OKG-aware) exceeds MAP (embedding-only). The OKG provides a qualitative layer that embeddings alone cannot.

### Paper 2: "Adversarial Multi-Agent Architecture for Codebase Structural Integrity"

**Core contribution**: Introduces the Crooked Wall Principle as a formal architectural pattern. Experiments B1, B3, D2 provide evidence. Metrics: AHR (M4), OCS (M2), Evolution@K (SIMARGL). Shows that adversarial agent design reduces hallucinations and preserves SIMARGL structural metrics better than single-model or compliant multi-model approaches.

**Connection to existing SIMARGL papers**: Directly uses the SIMARGL Novelty/Structurality 2×2 framework in a new context — as a real-time change validation tool rather than an offline evaluation metric.

### Paper 3: "The Qualia Horizon: Empirical Mapping of Emergence Thresholds in LLM-Assisted Software Engineering"

**Core contribution**: First empirical study of the relationship between parameter scale and task-type solvability in multi-agent code generation systems. Experiments D1, D4 provide evidence. Metrics: QBDP/QBDR (M6), Night Experiment convergence rates. Shows that emergence thresholds for different task types can be empirically characterized and used for principled routing decisions.

**Connection to SIMARGL phenomenological work**: Grounds the philosophical concept of Qualia (from `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md`) in measurable empirical phenomena — the emergence of irreducible capabilities at specific parameter scales.

---

## Conclusion: The System That Was Already There

The deepest finding of this analysis is structural rather than technical: the Anthill Distributed Cognitive OS was implicit in the SIMARGL research from the beginning. The phenomenological analysis of code (why do LLMs fail to understand it?) pointed toward the need for externalized knowledge (codeXport). The codeXport component pointed toward an ontological graph. The two-phase agent pointed toward a multi-agent pipeline. The SIMARGL metrics pointed toward a real-time architectural health monitor. Each concept, followed to its logical conclusion, arrives at Anthill.

This is not coincidence — it reflects the genuine coherence of the underlying problem. The gap between natural language intention and executable code is not merely a retrieval problem (solved by embeddings) or a reasoning problem (solved by large models). It is an epistemological problem: the agent cannot reason about what it cannot know. The Anthill architecture, by making knowledge explicit, persistent, and queryable, transforms the epistemological problem into an engineering problem. And engineering problems, unlike epistemological ones, have solutions.

The experiments proposed here are not tests of whether the Anthill idea is interesting — it is. They are tests of whether it is true.

---

**Document Version**: 1.0
**Created**: 2026-03-03
**Status**: Experimental Framework — Ready for Phase 1 Implementation
**Dependencies**: `ANTHILL_DISTRIBUTED_COGNITIVE_OS.md`, `SIMARGL_concept.md`, `FINAL_PRODUCT.md`, `FURTHER_RESEARCH_RECOMMENDATION.md`, `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md`

---

## References

### SIMARGL Internal Documents
1. `ANTHILL_DISTRIBUTED_COGNITIVE_OS.md` — Primary theoretical foundation
2. `SIMARGL_concept.md` — Novelty, Structurality, SES, HES metrics
3. `TWO_PHASE_REFLECTIVE_AGENT.md` — Agent pipeline architecture
4. `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` — Hermeneutic circle, symbol grounding
5. `FURTHER_RESEARCH_RECOMMENDATION.md` — Multi-agent specialization proposals
6. `KEYWORD_INDEXING.md` — Semantic coordinates, positive/negative space
7. `CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md` — Cross-layer pattern topology

### External Scientific References
8. Popper, K. (1959). *The Logic of Scientific Discovery*. — Falsificationism as experimental design principle
9. Amdahl, G. (1967). "Validity of the single processor approach." *AFIPS*. — Scaling limits of parallelism
10. Wei, J. et al. (2022). ["Emergent Abilities of Large Language Models."](https://arxiv.org/abs/2206.07682) *TMLR*. — Empirical basis for Qualia Horizon
11. Wang, X. et al. (2023). ["Self-Consistency Improves Chain of Thought Reasoning."](https://arxiv.org/abs/2203.11171) *ICLR*. — Majority voting baseline for Experiment B3
12. Du, Y. et al. (2023). ["Improving Factuality and Reasoning in LLMs through Multi-Agent Debate."](https://arxiv.org/abs/2305.14325) — Multi-agent debate foundation for Experiment B3
13. Liang, T. et al. (2023). ["Encouraging Divergent Thinking in LLMs through Multi-Agent Debate."](https://arxiv.org/abs/2305.19118) — Adversarial multi-agent design
14. Hu, E. et al. (2021). ["LoRA: Low-Rank Adaptation of Large Language Models."](https://arxiv.org/abs/2106.09685) — Specialization mechanism for Experiment B2
