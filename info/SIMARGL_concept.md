# SIMARGL: Structural Integrity Metrics for Adaptive Relation Graph Learning

## Project Overview

**SIMARGL** is a research framework for evaluating and balancing software change recommendations. The system recommends code relationships (links between modules, files, or functions) that should be modified together when implementing new requirements.

The core innovation is a **pair of antagonistic metrics** that measure the trade-off between:
- **Novelty** — recommending NEW relationships (innovation)
- **Structurality** — preserving EXISTING modular structure (stability)

### Etymology

Named after Simargl (Симаргл) — a Slavic winged deity in the form of a dog who guarded the Tree of Life and carried messages between gods and mortals. His long tail sometimes tangled, causing misdelivered messages — a metaphor for the trade-off between delivering new information (novelty) and maintaining structural order (integrity).

---

## Problem Statement

When a recommendation system suggests code changes, it can make two types of errors:

1. **Too conservative** — only recommends existing, well-known relationships
   - High accuracy but no value (the "broken clock" problem)
   - System stagnates, no innovation

2. **Too disruptive** — recommends many new cross-module relationships
   - High novelty but destroys modular structure
   - Creates "spaghetti code" architecture

**Goal**: Find recommendations that are BOTH novel AND respect the existing modular structure.

---

## Core Metrics

### Novelty@K

**Definition**: The fraction of recommended relationships that are NEW (not in the existing codebase).

```
Novelty@K = |{r ∈ top-K : r ∉ ExistingRelations}| / K
```

- **High value (→1.0)**: System recommends mostly new relationships
- **Low value (→0.0)**: System recommends mostly existing relationships

**Interpretation**:
- Novelty@K = 1.0 means all recommendations are new (possibly risky)
- Novelty@K = 0.0 means all recommendations already exist (no added value)

---

### Structurality@K

**Definition**: The fraction of recommended relationships where BOTH source and target belong to the SAME module.

```
Structurality@K = |{r ∈ top-K : module(r.source) == module(r.target)}| / K
```

- **High value (→1.0)**: Recommendations stay within module boundaries
- **Low value (→0.0)**: Recommendations cross module boundaries

**Interpretation**:
- Structurality@K = 1.0 means all recommendations are intra-module (preserves structure)
- Structurality@K = 0.0 means all recommendations are inter-module (may break structure)

---

## The 2×2 Trade-off Matrix

Combining Novelty and Structurality creates four zones:

```
                          Novelty@K
                     (new relationships)
                          HIGH (1.0)
                             ▲
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        │     DISRUPTION     │     EVOLUTION      │
        │                    │                    │
        │   New + Cross-     │   New + Within     │
        │   module links     │   module links     │
        │                    │                    │
        │   ⚠️ RISKY         │   ✅ IDEAL         │
        │                    │                    │
LOW ◄───┼────────────────────┼────────────────────┼───► HIGH
(0.0)   │                    │                    │    (1.0)
        │    STAGNATION      │    MAINTENANCE     │  Structurality@K
        │                    │                    │  (within modules)
        │   Old + Cross-     │   Old + Within     │
        │   module links     │   module links     │
        │                    │                    │
        │   ❌ USELESS       │   🔧 SAFE          │
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                          LOW (0.0)
```

### Zone Descriptions

| Zone | Novelty | Structurality | Description | Action |
|------|---------|---------------|-------------|--------|
| **EVOLUTION** | High | High | New links that respect module boundaries | ✅ Ideal state — pursue this |
| **DISRUPTION** | High | Low | New links that cross module boundaries | ⚠️ May break architecture |
| **MAINTENANCE** | Low | High | Existing links within modules | 🔧 Safe but no innovation |
| **STAGNATION** | Low | Low | Existing cross-module links | ❌ No value, avoid this |

---

## Derived Metrics

### Zone-Specific Metrics

Each recommendation falls into exactly one zone:

```python
Evolution@K    = |{r ∈ top-K : r is NEW and r is INTRA-module}| / K
Disruption@K   = |{r ∈ top-K : r is NEW and r is INTER-module}| / K
Maintenance@K  = |{r ∈ top-K : r is OLD and r is INTRA-module}| / K
Stagnation@K   = |{r ∈ top-K : r is OLD and r is INTER-module}| / K
```

**Constraint**: Evolution@K + Disruption@K + Maintenance@K + Stagnation@K = 1.0

### Composite Metrics

**Structural Evolution Score (SES)**:
```python
SES = sqrt(Novelty@K × Structurality@K)
```
- Range: [0, 1]
- Penalizes if either metric is low
- Maximum when both are high

**Harmonic Evolution Score (HES)**:
```python
HES = 2 × Novelty@K × Structurality@K / (Novelty@K + Structurality@K)
```
- More sensitive to imbalance than SES

---

## Why Balance Matters

### The Popularity Bias Problem

In real codebases, a few "core" modules are changed very frequently:

```
Module changes distribution:
  CoreUtils (45%)  ████████████████████████████████████
  DataLayer (25%)  ████████████████████
  UI (15%)         ████████████
  Auth (10%)       ████████
  Config (5%)      ████
```

A naive system learns to ALWAYS recommend CoreUtils because:
- It appears in 45% of all changes
- Recommending it gives 45% "accuracy"
- But this is the **"broken clock" problem** — right sometimes, but useless

### The Spaghetti Code Problem

A system optimizing only for novelty might recommend:

```
Task: "Add user authentication"
Recommendations:
  1. PaymentService → AuthController  (NEW, cross-module)
  2. ReportGenerator → UserModel      (NEW, cross-module)
  3. CacheManager → SessionService    (NEW, cross-module)
```

All new links! High novelty! But this creates tight coupling between unrelated modules.

### The Ideal Outcome

A balanced system recommends:

```
Task: "Add user authentication"
Recommendations:
  1. AuthController → UserValidator   (NEW, same auth module) ✅
  2. AuthController → TokenService    (NEW, same auth module) ✅
  3. UserService → AuthController     (NEW, cross-module but logical) ⚠️
```

- New relationships (novelty)
- Mostly within auth module (structurality)
- Cross-module link only where semantically necessary

---

## Implementation Guidelines

### Computing the Metrics

```python
class SIMARGLMetrics:
    def __init__(self, existing_relations: Set[Tuple], module_map: Dict[str, str]):
        """
        existing_relations: Set of (source, target, relation_type) tuples
        module_map: Dict mapping function/file ID to module ID
        """
        self.existing = existing_relations
        self.modules = module_map
    
    def is_novel(self, relation) -> bool:
        return (relation.source, relation.target, relation.type) not in self.existing
    
    def is_intra_module(self, relation) -> bool:
        src_module = self.modules.get(relation.source)
        tgt_module = self.modules.get(relation.target)
        return src_module is not None and src_module == tgt_module
    
    def compute_metrics(self, recommendations: List, k: int) -> Dict:
        top_k = recommendations[:k]
        
        novel = [r for r in top_k if self.is_novel(r)]
        intra = [r for r in top_k if self.is_intra_module(r)]
        
        novelty = len(novel) / k
        structurality = len(intra) / k
        
        evolution = len([r for r in novel if self.is_intra_module(r)]) / k
        disruption = len([r for r in novel if not self.is_intra_module(r)]) / k
        maintenance = len([r for r in top_k if not self.is_novel(r) and self.is_intra_module(r)]) / k
        stagnation = len([r for r in top_k if not self.is_novel(r) and not self.is_intra_module(r)]) / k
        
        ses = (novelty * structurality) ** 0.5
        
        return {
            'Novelty@K': novelty,
            'Structurality@K': structurality,
            'Evolution@K': evolution,
            'Disruption@K': disruption,
            'Maintenance@K': maintenance,
            'Stagnation@K': stagnation,
            'SES': ses
        }
```

### Integration with Recommendation Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Task/Issue     │────►│  Embedding      │────►│  Candidate      │
│  Description    │     │  Model          │     │  Retrieval      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Final         │◄────│  SIMARGL        │◄────│  Raw            │
│  Recommendations│     │  Re-ranking     │     │  Candidates     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Metrics:       │
                        │  - Novelty@K    │
                        │  - Structural@K │
                        │  - Evolution@K  │
                        └─────────────────┘
```

### Temperature-Based Balancing

Use a temperature parameter to control the novelty-structurality balance:

```python
def balanced_score(semantic_score, novelty_score, structural_score, temperature=1.0):
    """
    temperature < 1.0: Favor structure (conservative)
    temperature = 1.0: Balanced
    temperature > 1.0: Favor novelty (explorative)
    """
    novelty_weight = temperature / (1 + temperature)
    structure_weight = 1 / (1 + temperature)
    
    combined = (
        0.5 * semantic_score +
        novelty_weight * novelty_score +
        structure_weight * structural_score
    )
    return combined
```

---

## Evaluation Protocol

### Offline Evaluation

1. Take historical commits where relations changed
2. For each commit:
   - Input: task description + current codebase state
   - Ground truth: actual relations that changed
   - Predictions: top-K recommended relations
3. Compute SIMARGL metrics alongside standard IR metrics (MAP, MRR, Recall@K)

### Reporting Format

```
| Model          | MAP   | MRR   | Novelty@5 | Struct@5 | Evolution@5 | SES   |
|----------------|-------|-------|-----------|----------|-------------|-------|
| Baseline       | 0.45  | 0.52  | 0.20      | 0.80     | 0.15        | 0.40  |
| + Novelty Opt  | 0.38  | 0.44  | 0.85      | 0.30     | 0.25        | 0.51  |
| + SIMARGL      | 0.42  | 0.49  | 0.60      | 0.65     | 0.45        | 0.62  |
```

### Interpretation Guidelines

| Scenario | Diagnosis | Recommendation |
|----------|-----------|----------------|
| High Novelty, Low Structurality | DISRUPTION zone | Lower temperature, add structure penalty |
| Low Novelty, High Structurality | MAINTENANCE zone | Increase temperature, add novelty bonus |
| Low Novelty, Low Structurality | STAGNATION zone | Review model — may be recommending noise |
| High Evolution@K | Ideal state | Model is working well |

---

## Scientific Contribution

### What is Novel in SIMARGL

1. **Structurality@K metric** — measures how recommendations affect modular structure
2. **Antagonistic metric pair** — Novelty vs Structurality as opposing forces
3. **2×2 decomposition** — Evolution/Disruption/Maintenance/Stagnation zones
4. **Application to code change recommendation** — transfer from recommender systems to software evolution

### Key Insight

Existing recommender system research focuses on **user satisfaction** (novelty vs accuracy).
SIMARGL focuses on **system health** (novelty vs structural integrity).

This is a new perspective that bridges:
- Recommender systems theory
- Software architecture principles
- Software evolution research

---

## Related Work and References

### Foundational Works on Software Modularity

1. **Parnas, D.L. (1972)** - "On the Criteria To Be Used in Decomposing Systems into Modules"
   - Foundational paper on information hiding and module design
   - https://dl.acm.org/doi/10.1145/361598.361623

2. **Constantine, L.L. & Yourdon, E. (1979)** - "Structured Design: Fundamentals of a Discipline of Computer Program and Systems Design"
   - Defined coupling and cohesion metrics
   - Book: Prentice-Hall

### Lehman's Laws of Software Evolution

3. **Lehman, M.M. (1980)** - "Programs, Life Cycles, and Laws of Software Evolution"
   - Laws describing how software systems evolve over time
   - https://ieeexplore.ieee.org/document/1456074

4. **Lehman, M.M. & Ramil, J.F. (1997)** - "Metrics and Laws of Software Evolution - The Nineties View"
   - Updated view on software evolution metrics
   - https://ieeexplore.ieee.org/document/637156

5. **Herraiz, I. et al. (2013)** - "The Evolution of the Laws of Software Evolution: A Discussion Based on a Systematic Literature Review"
   - Comprehensive review of Lehman's laws validation
   - https://dl.acm.org/doi/10.1145/2543581.2543595

### Novelty and Diversity in Recommender Systems

6. **Castells, P., Hurley, N., Vargas, S. (2011)** - "Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems"
   - Key paper on novelty-accuracy trade-off in recommendations
   - https://dl.acm.org/doi/10.1145/2043932.2043955

7. **Hurley, N. & Zhang, M. (2011)** - "Novelty and Diversity in Top-N Recommendation - Analysis and Evaluation"
   - Formulates trade-off between diversity and matching quality
   - https://dl.acm.org/doi/10.1145/1944339.1944341

8. **Vargas, S. & Castells, P. (2011)** - "Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems"
   - Workshop on Novelty and Diversity in Recommender Systems
   - https://ceur-ws.org/Vol-816/

9. **Kaminskas, M. & Bridge, D. (2016)** - "Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems"
   - Comprehensive survey on beyond-accuracy metrics
   - https://dl.acm.org/doi/10.1145/2926720

### Software Remodularization and Change Analysis

10. **Candela, I. et al. (2016)** - "Using Cohesion and Coupling for Software Remodularization: Is It Enough?"
    - Empirical study on cohesion/coupling in remodularization
    - https://dl.acm.org/doi/10.1145/2928268

11. **Bavota, G. et al. (2013)** - "Using Structural and Semantic Measures to Improve Software Modularization"
    - Multi-objective approach to remodularization
    - https://link.springer.com/article/10.1007/s10664-012-9226-8

12. **Mitchell, B.S. & Mancoridis, S. (2006)** - "On the Automatic Modularization of Software Systems Using the Bunch Tool"
    - Clustering for automatic modularization
    - https://ieeexplore.ieee.org/document/1599428

### Change Coupling and Co-change Prediction

13. **Zimmermann, T. et al. (2005)** - "Mining Version Histories to Guide Software Changes"
    - Foundational work on co-change mining
    - https://dl.acm.org/doi/10.1145/1062455.1062549

14. **Ying, A.T.T. et al. (2004)** - "Predicting Source Code Changes by Mining Change History"
    - Early work on change prediction from history
    - https://ieeexplore.ieee.org/document/1318072

15. **Mondal, D. et al. (2024)** - "Exploring Evolutionary Coupling in Software Systems"
    - Recent work integrating multiple change relationships
    - ESEM 2024

### Exploration-Exploitation in Code

16. **Tang, H. et al. (2024)** - "Code Repair with LLMs gives an Exploration-Exploitation Tradeoff"
    - Applies bandit algorithms to code refinement
    - https://arxiv.org/abs/2405.17503

### Software Architecture Metrics

17. **Martin, R.C. (2003)** - "Agile Software Development: Principles, Patterns, and Practices"
    - Instability and abstractness metrics
    - Book: Prentice Hall

18. **Lilienthal, C. (2019)** - "Improve Your Architecture with the Modularity Maturity Index"
    - Practical modularity assessment
    - https://www.oreilly.com/library/view/software-architecture-metrics/9781098112226/

---

## Glossary

| Term | Definition |
|------|------------|
| **Relation** | A directed connection between two code elements (function calls, imports, data flow) |
| **Module** | A cohesive unit of code (package, namespace, directory) |
| **Intra-module** | A relation where source and target are in the same module |
| **Inter-module** | A relation where source and target are in different modules |
| **Novelty** | Whether a relation is new (not in current codebase) |
| **Structurality** | Whether a relation respects module boundaries |
| **Evolution** | New + intra-module (ideal) |
| **Disruption** | New + inter-module (risky) |
| **Temperature** | Parameter controlling novelty-structure balance |
| **SES** | Structural Evolution Score — composite metric |

---

## File Metadata

- **Project**: SIMARGL (Structural Integrity Metrics for Adaptive Relation Graph Learning)
- **Version**: 1.0
- **Date**: 2025-01-03
- **Purpose**: Conceptual overview for AI agents and human collaborators
- **Usage**: Provide this file as context when working on SIMARGL-related code
