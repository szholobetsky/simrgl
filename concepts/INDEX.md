# SIMARGL Concepts Index

This document provides a structured overview of all metrics, concepts, and approaches developed in the SIMARGL project.

---

## 1. Core Metrics (SIMARGL Framework)

| Metric | Description | Source File | Opposing/Related Metric |
|--------|-------------|-------------|------------------------|
| **Novelty@K** | Fraction of recommended relationships that are NEW (not in existing codebase) | [SIMARGL_concept.md](SIMARGL_concept.md) | Structurality@K |
| **Structurality@K** | Fraction of recommended relationships where BOTH source and target belong to the SAME module | [SIMARGL_concept.md](SIMARGL_concept.md) | Novelty@K |
| **Evolution@K** | New + intra-module relationships (ideal zone) | [SIMARGL_concept.md](SIMARGL_concept.md) | Stagnation@K |
| **Disruption@K** | New + inter-module relationships (risky zone) | [SIMARGL_concept.md](SIMARGL_concept.md) | Maintenance@K |
| **Maintenance@K** | Old + intra-module relationships (safe zone) | [SIMARGL_concept.md](SIMARGL_concept.md) | Disruption@K |
| **Stagnation@K** | Old + inter-module relationships (useless zone) | [SIMARGL_concept.md](SIMARGL_concept.md) | Evolution@K |
| **SES** | Structural Evolution Score = sqrt(Novelty@K × Structurality@K) | [SIMARGL_concept.md](SIMARGL_concept.md) | HES |
| **HES** | Harmonic Evolution Score = 2×N×S/(N+S), more sensitive to imbalance | [SIMARGL_concept.md](SIMARGL_concept.md) | SES |

---

## 2. Compositional Embedding Metrics

| Metric/Concept | Description | Source File | Related Concept |
|----------------|-------------|-------------|-----------------|
| **Additive Composition** | v_d = v_a + v_b for parallel/merging calls | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Multiplicative Composition |
| **Multiplicative Composition** | v_d = v_a ⊙ v_b (Hadamard product) for function application | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Additive Composition |
| **Weighted Composition** | α·v_a + β·v_b based on call frequency | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Composition MSE |
| **Composition MSE** | Mean squared error between composed and actual embeddings | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Composition Similarity |
| **Composition Similarity** | Cosine similarity between composed vector and actual | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Composition MSE |

---

## 3. Cross-Layer Transformation Metrics

| Metric | Description | Source File | Related Concept |
|--------|-------------|-------------|-----------------|
| **Path Reconstruction Accuracy** | How well composed embedding matches final entity embedding | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Task Alignment Score |
| **Task Alignment Score** | Cosine similarity between composed path embedding and task description embedding | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Path Reconstruction |
| **Subspace Separation** | Orthogonality between architectural layer subspaces (UI, Service, Entity, DB) | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Within-Layer Cohesion |
| **Within-Layer Cohesion** | Average similarity of entities within the same architectural layer | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Subspace Separation |
| **Cross-Layer Transformation Accuracy** | Prediction accuracy for each type of layer transition (UI→Service, Service→Entity, etc.) | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Hit@K |
| **Hit@K (Paths)** | Whether the correct data flow path appears in top-K retrieved paths | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | MRR |

---

## 4. Phenomenological Grounding Metrics

| Metric | Description | Source File | Related Concept |
|--------|-------------|-------------|-----------------|
| **Grounding Accuracy** | % of business terms correctly linked to code identifiers | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Horizon Completeness |
| **Horizon Completeness** | % of actually changed files that fall within the semantic horizon | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Grounding Accuracy |
| **Noematic Precision** | % of grounded "noema" (intentional objects) that match task intent | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Affordance Relevance |
| **Affordance Relevance** | % of suggested actions/operations that were actually performed | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Noematic Precision |
| **Negative Space Accuracy** | % of files excluded by contrastive filtering that were truly irrelevant | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Horizon Completeness |

---

## 5. Philosophical Concepts

| Concept | Description | Source File | Applied As |
|---------|-------------|-------------|------------|
| **Noema** | Object of intention (what the code "is about") | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Identifier cluster + file locations |
| **Noesis** | Act of perceiving/understanding | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Agent's graph traversal |
| **Zuhandenheit** | "Ready-to-hand" - code that works transparently | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Code not in search results |
| **Vorhandenheit** | "Present-at-hand" - code under examination | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Code in search results |
| **Lebenswelt** | "Life-world" - context and experience | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Identifier graph + relations + history |
| **Affordances** | Available operations/actions | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Methods, functions, API endpoints |
| **Symbol Grounding** | Linking abstract terms to concrete code | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Business term → Identifier → Files |
| **Church-Turing Thesis (as design principle)** | Any computable function is achievable by a simple mechanism given sufficient tape (OKG) and time (async agents); theoretical foundation for SLM sufficiency | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) | Replace "model too small" with "tape too short" |
| **Society of Mind (Minsky)** | Intelligence as emergent product of many simple, specialized sub-agents; theoretical basis for the Anthill multi-agent architecture | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) | 10-GPU mining rig as distributed brain |
| **Ding an sich / Noumenon** | Kant's "thing-in-itself" — source code as objective reality, inaccessible directly; agents operate only on the phenomenal OKG representation | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) | Code = noumenon; OKG = phenomenon |
| **Ontological Memoization** | Treating the OKG as a DP memoization table: semantic sub-problems computed once and cached; eliminates redundant re-derivation of architectural knowledge | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) | OKG as DP matrix |
| **Qualia** | Irreducible subjective properties (beauty, foresight) that emerge only above a parameter threshold and cannot be decomposed into explicit rules | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) | Aesthetic design, architectural judgment |
| **Falsificationism (Popper)** | Scientific knowledge advances by attempting to disprove hypotheses; basis for adversarial agent design — the Critic's job is to break the Coder's output | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) | Adversarial Integrity architecture |

---

## 6. Keyword and Entity Mapping

| Concept | Description | Source File | Related Concept |
|---------|-------------|-------------|-----------------|
| **Semantic Coordinates** | Identifiers as coordinates in Word2Vec semantic space | [KEYWORD_INDEXING.md](KEYWORD_INDEXING.md) | Entity-to-File Mapping |
| **Entity-to-File Mapping** | Bidirectional mapping between domain keywords and code files | [KEYWORD_ENTITY_MAPPING.md](KEYWORD_ENTITY_MAPPING.md) | Semantic Coordinates |
| **Keyword Extraction** | Extracting domain terms from code identifiers (camelCase, snake_case splitting) | [KEYWORD_ENTITY_MAPPING.md](KEYWORD_ENTITY_MAPPING.md) | Business Term Grounding |
| **Negative Space** | Identifiers dissimilar to task terms (contrastive learning) | [KEYWORD_INDEXING.md](KEYWORD_INDEXING.md) | Semantic Coordinates |

---

## 6b. Anthill-Specific Metrics

| Metric | Definition | Source File | Target |
|--------|-----------|-------------|--------|
| **PRA@K** (Passport Retrieval Accuracy) | Fraction of top-K Object Passports that correspond to ground-truth changed files | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≥ 0.70 at K=10 |
| **OCS** (OKG Consistency Score) | Fraction of OKG assertions about a changed file that remain logically consistent after the Observer Loop | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≥ 0.95 |
| **MHR** (Memoization Hit Rate) | Fraction of agent sub-queries answered from OKG cache without fresh LLM derivation | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≥ 0.80 (mature OKG) |
| **AHR** (Agent Hallucination Rate) | Fraction of generated code changes containing at least one factual error (wrong signature, non-existent class) | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≤ 0.10 with OKG |
| **CCD** (Cascade Correction Depth) | Average number of Critic-Coder iterations before Auditor approval | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≤ 2.5 (standard tasks) |
| **QBDP** (Qualia Boundary Detection Precision) | Precision of Router classification of tasks requiring large model / human | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≥ 0.85 |
| **QBDR** (Qualia Boundary Detection Recall) | Recall of Qualia task detection (avoids routing too many tasks to Oracle) | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≥ 0.70 |
| **HATF** (Human Attention Time Fraction) | Fraction of total task completion time requiring active human presence | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≤ 0.10 |
| **SDR** (Structural Drift Rate) | Rate of structural quality degradation per code change over N consecutive modifications | [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | ≤ 0.0 (maintain or improve) |

---

## 7. Standard IR Metrics (Used Across Concepts)

| Metric | Description | Source File | Notes |
|--------|-------------|-------------|-------|
| **Precision@K** | Fraction of relevant items in top-K | Multiple | Target: >0.6 |
| **Recall@K** | Fraction of all relevant items found in top-K | Multiple | Target: >0.5 |
| **MRR** | Mean Reciprocal Rank = 1/rank_of_first_relevant | Multiple | Target: >0.5 |
| **NDCG@K** | Normalized Discounted Cumulative Gain | Multiple | Target: >0.7 |
| **MAP** | Mean Average Precision | Multiple | Standard IR metric |

---

## 8. Architectural Approaches

| Approach | Description | Source File |
|----------|-------------|-------------|
| **Two-Phase Reflective Agent** | Phase 1: Reasoning + Search, Phase 2: Reflection + Refinement | [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md) |
| **Three Specialized Agents** | Intent Discovery + Search/Grounding + Synthesis/Reflection | [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md), [FILTERING_CONTEXT_CONCEPT.md](FILTERING_CONTEXT_CONCEPT.md) |
| **Dual MCP Server Architecture** | Separate servers for file-level and task-level embeddings | [DUAL_MCP_SERVER_ARCHITECTURE.md](DUAL_MCP_SERVER_ARCHITECTURE.md) |
| **Dual Collection System** | RECENT (last N tasks) vs ALL (complete history) | [DUAL_SERVER_RAG_EVALUATION.md](DUAL_SERVER_RAG_EVALUATION.md) |
| **Anthill OS** | Distributed cognitive OS: network of SLMs (1B–8B) on commodity GPU mining hardware, coordinated via ontological knowledge graph | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |
| **Ontological Knowledge Graph (OKG)** | RDF-triplet graph as Single Point of Truth for codebase structure; functions as a DP memoization table — computed once, queried by all agents | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |
| **Object Passport** | Minimum-description-length YAML summary of a code artifact (methods, dependencies, architectural tags); replaces raw file reading for context injection | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |
| **Reduction Ladder** | 8-level semantic compression hierarchy: Natural Language → Requirements → Code → Ontology → Passport → Knowledge → Meta-Knowledge → Map of the Unknowable | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |
| **Adversarial Integrity / Crooked Wall Principle** | Architectural adversarialism: each agent validates input against the OKG rather than accepting it from the previous agent; grounded in Popperian falsificationism | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |
| **Kantian Observer Loop** | Code = noumenon (thing-in-itself); OKG = phenomenon (structured perception); Scanner = transcendental filter; post-change OKG consistency check = reality validation | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |
| **Qualia Horizon / Map of the Unknowable** | Explicit meta-cognitive catalog of tasks requiring emergent properties of scale (≥14B parameters): aesthetic design, architectural foresight; routes to Oracle gateway | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |
| **Oracle Gateway** | Privacy-preserving interface to cloud LLMs (Claude, Gemini): sends only Object Passports, never source code; used only for Qualia-boundary tasks | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |
| **SKILL.md** | Per-agent constitution: tool manifest, domain knowledge primer, behavioral constraints, escalation criteria; updated by Researcher agent on failure | [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) |

---

## 9. The 2×2 Trade-off Matrix (SIMARGL)

```
                      Novelty@K
                 (new relationships)
                      HIGH (1.0)
                         ▲
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │     DISRUPTION     │     EVOLUTION      │
    │   New + Cross-     │   New + Within     │
    │   module links     │   module links     │
    │   ⚠️ RISKY         │   ✅ IDEAL         │
    │                    │                    │
LOW ◄────────────────────┼────────────────────►HIGH
(0.0)│                   │                    │(1.0)
    │    STAGNATION      │    MAINTENANCE     │ Structurality@K
    │   Old + Cross-     │   Old + Within     │ (within modules)
    │   module links     │   module links     │
    │   ❌ USELESS       │   🔧 SAFE          │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                         ▼
                      LOW (0.0)
```

---

## 10. File Map

| File | Category | Language |
|------|----------|----------|
| [SIMARGL_concept.md](SIMARGL_concept.md) | Core Metrics | EN |
| [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Embeddings | EN |
| [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Embeddings | EN |
| [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Philosophy | EN |
| [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Implementation | EN |
| [KEYWORD_INDEXING.md](KEYWORD_INDEXING.md) | Indexing | EN |
| [KEYWORD_ENTITY_MAPPING.md](KEYWORD_ENTITY_MAPPING.md) | Indexing | EN |
| [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md) | Architecture | EN |
| [DUAL_MCP_SERVER_ARCHITECTURE.md](DUAL_MCP_SERVER_ARCHITECTURE.md) | Architecture | EN |
| [DUAL_SERVER_RAG_EVALUATION.md](DUAL_SERVER_RAG_EVALUATION.md) | Evaluation | EN |
| [FILTERING_CONTEXT_CONCEPT.md](FILTERING_CONTEXT_CONCEPT.md) | Strategy | EN |
| [FURTHER_RESEARCH_RECOMMENDATION.md](FURTHER_RESEARCH_RECOMMENDATION.md) | Research | EN |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Overview | EN |
| [FINAL_PRODUCT.md](FINAL_PRODUCT.md) | Product Vision | EN |
| [ANTHILL_DISTRIBUTED_COGNITIVE_OS.md](ANTHILL_DISTRIBUTED_COGNITIVE_OS.md) | Distributed AI Architecture | EN |
| [ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md](ANTHILL_CONVERGENCE_AND_EXPERIMENTS.md) | Anthill Lineage + Experiments | EN |
| [1BCODER.md](1BCODER.md) | Human-in-the-loop Anthill Prototype | EN |
| [../exp5/README.md](../exp5/README.md) | Cross-Vocabulary Symbol Grounding experiment | EN |
| [SYMBOL_GROUNDING_ESSAY.md](SYMBOL_GROUNDING_ESSAY.md) | Philosophical essay on symbol grounding in code navigation | EN |
| [ua/](ua/) | Ukrainian translations | UA |

---

**Document Version**: 1.0
**Created**: 2026-02-14
**Project**: SIMARGL (Structural Integrity Metrics for Adaptive Relation Graph Learning)
