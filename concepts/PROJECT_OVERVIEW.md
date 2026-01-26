# SIMARGL Research Project Overview

## Executive Summary

**SIMARGL** (Structural Integrity Metrics for Adaptive Relation Graph Learning) is a comprehensive research project developing an intelligent code navigation and recommendation system. The project addresses a fundamental challenge in software engineering: how to help developers find the right code artifacts (modules, files, functions) to modify when implementing new features or fixing bugs.

The core innovation lies in combining:
- **Advanced embedding techniques** for semantic code understanding
- **Two-phase reflective agents** for intelligent reasoning
- **Phenomenological grounding** to bridge human intention with code semantics
- **Dual-server RAG architecture** balancing historical patterns with recent context

---

## Problem Statement

### The Navigation Challenge

Modern software systems contain millions of lines of code organized into hundreds or thousands of modules. When a developer receives a task like "Fix the authentication timeout issue," they face several challenges:

1. **Where to start?** Which modules are relevant to authentication?
2. **What's related?** Which files typically change together?
3. **What's the context?** What patterns exist in similar past tasks?
4. **What's recent?** What areas are currently under active development?

Traditional approaches (keyword search, static analysis) fail because:
- Code semantics are **compositional** (meaning emerges from combinations)
- Developer intent is **phenomenological** (grounded in human experience)
- Relevance is **contextual** (depends on task, history, and project state)

### The Recommendation Quality Problem

When recommending code changes, systems can fail in two ways:

| Failure Mode | Description | Consequence |
|--------------|-------------|-------------|
| **Too Conservative** | Only recommends well-known, frequently changed files | Misses novel but relevant connections |
| **Too Disruptive** | Recommends many new cross-module connections | Creates spaghetti architecture |

SIMARGL addresses this through **antagonistic metrics** (Novelty vs Structurality) that balance innovation with architectural integrity.

---

## Research Pillars

### Pillar 1: Semantic Code Embeddings

Code understanding requires multiple embedding strategies for different granularities and purposes.

#### 1.1 Compositional Code Embeddings
**Concept**: Functions derive meaning from what they call and use.

```
embedding(calculateTotal) = f(
    embedding(getPrice),
    embedding(applyDiscount),
    embedding(calculateTax)
)
```

**Key Innovation**: Vector arithmetic on code semantics
- `embedding(authenticateUser) - embedding(validateToken) + embedding(validateCookie) ≈ embedding(authenticateWithCookie)`

**Applications**:
- Analogical code search ("find function like X but for Y")
- Semantic diff (what changes in meaning, not just syntax)
- Code completion with semantic awareness

#### 1.2 Cross-Layer Transformation Embeddings
**Concept**: Track how data flows through architectural layers.

```
UI Layer → Service Layer → Repository Layer → Database
   ↓            ↓              ↓                ↓
UserDTO → UserService → UserRepository → users_table
```

**Key Innovation**: Embeddings capture transformation chains, not just individual elements.

**Applications**:
- "Find all code that handles user data from UI to database"
- Impact analysis across architectural boundaries
- Refactoring support for layer restructuring

#### 1.3 Keyword Entity Mapping
**Concept**: Keywords are coordinates in semantic space.

When we identify "RULE" as relevant, we simultaneously:
- **Include**: RuleIndex, RuleUpdater, QualityProfileRules
- **Exclude**: Server, Plugin, DatabaseMigration (negative space)

**Key Innovation**: Contrastive learning for bounded semantic regions.

---

### Pillar 2: Two-Phase Reflective Agent

Intelligent code navigation requires both exploration and critical evaluation.

#### Phase 1: Reasoning
- Analyze task description
- Search for relevant modules, files, and similar tasks
- Generate initial recommendations
- Build context through multiple information sources

#### Phase 2: Reflection
- Critically evaluate Phase 1 results
- Check for missing context
- Verify logical consistency
- Refine recommendations based on deeper analysis

```
Task → [Phase 1: Explore & Reason] → Initial Results
                                          ↓
      [Phase 2: Reflect & Refine] ← Critique
                                          ↓
                                   Final Recommendations
```

**Key Innovation**: Self-improvement loop inspired by human expert reasoning.

---

### Pillar 3: Phenomenological Grounding

Code is not just syntax—it embodies human intentions, actions, and meanings.

#### 3.1 The Symbol Grounding Problem
LLMs manipulate symbols without grounding in lived experience:

| Concept | LLM Understanding | Human Understanding |
|---------|-------------------|---------------------|
| `setTimeout` | "Function that delays execution" | "That frustrating thing that causes race conditions" |
| `NullPointerException` | "Error when dereferencing null" | "The bug that crashed production at 3 AM" |

#### 3.2 Husserlian Framework
- **Intentionality**: Code is always "about" something (functionality, user need)
- **Noema**: The code artifact as experienced (not just text)
- **Noesis**: The act of understanding code (reading with purpose)

#### 3.3 Heideggerian Concepts
- **Zuhandenheit** (readiness-to-hand): Code that "just works"—invisible tools
- **Vorhandenheit** (presence-at-hand): Code that becomes visible when broken

#### 3.4 Code as Performative (Speech Act Theory)
Code doesn't just describe—it **does**:
- `deleteUser(id)` is a **perlocutionary act** with real-world effects
- Understanding code requires understanding its **effects**, not just syntax

**Key Innovation**: Embedding phenomenological concepts into AI-assisted code navigation.

---

### Pillar 4: Dual-Server RAG Architecture

Balance between historical wisdom and current focus.

#### Historical Context Server
- **Coverage**: All historical tasks (9,799+)
- **Strength**: Cross-project patterns, architectural decisions
- **Use case**: "How has authentication been implemented before?"

#### Recent Context Server
- **Coverage**: Last 100 tasks
- **Strength**: Current development focus, active areas
- **Use case**: "What's being worked on now that might be affected?"

#### Fusion Strategies
1. **Weighted Fusion**: Configurable balance (α parameter)
2. **Reciprocal Rank Fusion**: Combines rankings without score normalization
3. **Adaptive Fusion**: Query-type-aware weighting

```
                    Query
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
   Historical                    Recent
    Server                       Server
        ↓                           ↓
        └─────────────┬─────────────┘
                      ↓
               Fusion Layer
                      ↓
              Final Results
```

---

## SIMARGL Metrics Framework

### Core Antagonistic Metrics

#### Novelty@K
**Definition**: Fraction of recommendations that are NEW (not in existing codebase).

```
Novelty@K = |{r ∈ top-K : r ∉ ExistingRelations}| / K
```

- High (→1.0): Recommending mostly new connections (innovative)
- Low (→0.0): Recommending mostly existing connections (safe)

#### Structurality@K
**Definition**: Fraction of recommendations where BOTH source and target belong to the SAME module.

```
Structurality@K = |{r ∈ top-K : module(r.source) == module(r.target)}| / K
```

- High (→1.0): Recommendations stay within modules (preserves structure)
- Low (→0.0): Recommendations cross module boundaries (may disrupt)

### The 2×2 Trade-off Matrix

```
                     Novelty
               LOW           HIGH
           ┌─────────────┬─────────────┐
      HIGH │ MAINTENANCE │  EVOLUTION  │
Struct.    │ Safe, no    │  Ideal:     │
           │ innovation  │  new + local│
           ├─────────────┼─────────────┤
      LOW  │ STAGNATION  │ DISRUPTION  │
           │ No value    │  Risky:     │
           │             │  new + cross│
           └─────────────┴─────────────┘
```

**Goal**: Maximize EVOLUTION zone (high novelty + high structurality).

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     SIMARGL System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Embedding  │    │  Two-Phase  │    │   SIMARGL   │     │
│  │   Engine    │    │    Agent    │    │   Metrics   │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                   ┌────────┴────────┐                       │
│                   │  Dual-Server    │                       │
│                   │  RAG Layer      │                       │
│                   └────────┬────────┘                       │
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         │                  │                  │             │
│  ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐     │
│  │  Historical │    │   Recent    │    │   Keyword   │     │
│  │   Server    │    │   Server    │    │    Index    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    PostgreSQL + pgvector                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Task Input**: Developer describes task in natural language
2. **Keyword Extraction**: Identify semantic coordinates (RULE, SEARCH, etc.)
3. **Embedding Generation**: Create task vector from keywords
4. **Dual-Server Search**: Query both historical and recent context
5. **Fusion**: Combine results with appropriate strategy
6. **Agent Reasoning**: Phase 1 exploration, Phase 2 reflection
7. **Metric Evaluation**: Score recommendations on Novelty/Structurality
8. **Final Output**: Ranked recommendations with explanations

---

## Research Contributions

### Academic Contributions

1. **SIMARGL Metrics**: Novel framework for evaluating code recommendations
2. **Phenomenological Code Understanding**: Bridging philosophy and software engineering
3. **Compositional Code Embeddings**: Vector arithmetic for semantic code operations
4. **Cross-Layer Transformation Tracking**: Embeddings for data flow analysis
5. **Dual-Server RAG**: Balancing historical and recent context

### Practical Contributions

1. **MCP Integration**: Production-ready system via Model Context Protocol
2. **Two-Phase Agent**: Implementable reasoning/reflection architecture
3. **Evaluation Framework**: Comprehensive metrics and testing protocols
4. **Open Source Tools**: Reusable components for code navigation research

---

## Evaluation Strategy

### Offline Evaluation

| Metric | Target | Description |
|--------|--------|-------------|
| MAP@10 | >0.40 | Mean Average Precision |
| MRR | >0.50 | Mean Reciprocal Rank |
| Recall@10 | >0.60 | Coverage of relevant files |
| Evolution@K | >0.35 | New + intra-module recommendations |
| SES | >0.55 | Structural Evolution Score |

### Online Evaluation

| Metric | Target | Description |
|--------|--------|-------------|
| Task Completion | >75% | Developers complete tasks using recommendations |
| Time to First Relevant | <60s | Efficiency of navigation |
| Context Sufficiency | >80% | No additional searches needed |

---

## Concept Documents

This project contains detailed documentation for each research pillar:

| Document | Description |
|----------|-------------|
| `SIMARGL_concept.md` | Core metrics framework (Novelty, Structurality) |
| `TWO_PHASE_REFLECTIVE_AGENT.md` | Agent architecture (Reasoning + Reflection) |
| `COMPOSITIONAL_CODE_EMBEDDINGS.md` | Function-level semantic embeddings |
| `CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md` | Data flow tracking across layers |
| `KEYWORD_ENTITY_MAPPING.md` | Business terms to code mapping |
| `KEYWORD_INDEXING.md` | Semantic coordinates via Word2Vec |
| `DUAL_MCP_SERVER_ARCHITECTURE.md` | Task-based vs file-based search |
| `DUAL_SERVER_RAG_EVALUATION.md` | Evaluation framework and metrics |
| `PHENOMENOLOGICAL_CODE_UNDERSTANDING.md` | Philosophical foundations |
| `SEMANTIC_FINGERPRINT_MCP_SERVER.md` | MCP server implementation guide |

---

## Future Directions

### Short-term (3-6 months)
- [ ] Implement and validate embedding strategies
- [ ] Build two-phase agent prototype
- [ ] Create evaluation benchmark dataset
- [ ] Integrate with IDE plugins (VS Code, JetBrains)

### Medium-term (6-12 months)
- [ ] Fine-tune models on domain-specific codebases
- [ ] Develop feedback learning from user interactions
- [ ] Extend to multi-repository scenarios
- [ ] Publish benchmark and evaluation framework

### Long-term (1-2 years)
- [ ] Explore multimodal embeddings (code + documentation + diagrams)
- [ ] Develop explanation generation for recommendations
- [ ] Build collaborative features (team-aware recommendations)
- [ ] Create industry partnerships for real-world validation

---

## Project Metadata

- **Project Name**: SIMARGL (Structural Integrity Metrics for Adaptive Relation Graph Learning)
- **Institution**: Taras Shevchenko National University of Kyiv
- **Authors**: Stanislav Zholobetskyi, Oleg Andriichuk
- **Version**: 1.0
- **Last Updated**: 2025-01-25
- **Status**: Active Research

---

## References

### Foundational Works

1. Parnas, D.L. (1972). "On the Criteria To Be Used in Decomposing Systems into Modules"
2. Lehman, M.M. (1980). "Programs, Life Cycles, and Laws of Software Evolution"
3. Husserl, E. (1913). "Ideas Pertaining to a Pure Phenomenology"
4. Heidegger, M. (1927). "Being and Time"
5. Austin, J.L. (1962). "How to Do Things with Words"

### Technical References

6. Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
7. Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
8. Feng, Z. et al. (2020). "CodeBERT: A Pre-Trained Model for Programming and Natural Languages"
9. Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

### Related Research

10. Zimmermann, T. et al. (2005). "Mining Version Histories to Guide Software Changes"
11. Castells, P. et al. (2011). "Rank and Relevance in Novelty and Diversity Metrics"
12. Tufano, M. et al. (2019). "An Empirical Study on Learning Bug-Fixing Patches"

---

## Getting Started

1. **Read the concept documents** in order of your interest
2. **Review the SIMARGL metrics** to understand evaluation criteria
3. **Explore the dual-server architecture** for implementation insights
4. **Study the phenomenological foundations** for theoretical depth
5. **Check the MCP server guide** for practical integration

For questions or collaboration, contact the research team.

---

*"Understanding code is not just parsing syntax—it's grasping human intention crystallized in executable form."*
