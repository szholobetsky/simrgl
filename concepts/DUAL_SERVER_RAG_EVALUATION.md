# Dual-Server RAG MCP Architecture: Evaluation & Testing Strategy

## Executive Summary

This document describes a dual-server RAG (Retrieval-Augmented Generation) approach using two separate MCP (Model Context Protocol) servers to balance **historical depth** with **recent context accuracy**. The architecture combines:

1. **Historical Context Server**: Uses `mod_n-all-bge-small` collection (all historical tasks)
2. **Recent Context Server**: Uses `recent-w100-bge-small` collection (last 100 tasks)

This hybrid approach aims to provide comprehensive codebase understanding while maintaining awareness of current development focus.

---

## 1. Architecture Overview

### 1.1 Dual-Server Concept

```
User Query
    ↓
┌─────────────────────────────────────────┐
│     MCP Client (RAG Orchestrator)       │
└─────────────────────────────────────────┘
         ↓                    ↓
    ┌─────────┐          ┌─────────┐
    │ Server 1│          │ Server 2│
    │Historical│         │ Recent  │
    └─────────┘          └─────────┘
         ↓                    ↓
    ┌─────────┐          ┌─────────┐
    │mod_n-all│          │recent-  │
    │bge-small│          │w100     │
    │(ALL)    │          │(LAST100)│
    └─────────┘          └─────────┘
         ↓                    ↓
    ┌───────────────────────────┐
    │   Context Fusion Layer    │
    └───────────────────────────┘
              ↓
         ┌─────────┐
         │   LLM   │
         └─────────┘
```

### 1.2 Collection Characteristics

| Aspect | Historical Server | Recent Server |
|--------|------------------|---------------|
| **Collection** | `mod_n-all-bge-small` | `recent-w100-bge-small` |
| **Task Coverage** | All historical tasks (9,799+) | Last 100 tasks |
| **Strength** | Broad pattern recognition | Current focus awareness |
| **Use Case** | Cross-project patterns, architectural decisions | Active development areas |
| **Update Frequency** | Weekly/monthly | Daily/per-commit |
| **Embedding Model** | bge-small-en-v1.5 | bge-small-en-v1.5 |

### 1.3 Fusion Strategies

Three primary approaches to combine results:

1. **Parallel Retrieval + Score Fusion**
   - Query both servers simultaneously
   - Merge results using weighted scoring
   - Deduplicate and re-rank

2. **Sequential Retrieval**
   - Query historical server first (broad context)
   - Query recent server second (refine)
   - Combine with recency boost

3. **Adaptive Routing**
   - Classify query type (exploratory vs focused)
   - Route to appropriate server or both
   - Dynamic weight adjustment

---

## 2. Evaluation Framework

### 2.1 Core Objectives

The evaluation framework measures:

1. **Relevance**: Are retrieved documents useful for the task?
2. **Coverage**: Do results span necessary codebase areas?
3. **Recency Awareness**: Does the system understand current focus?
4. **Historical Insight**: Does it leverage past patterns?
5. **Diversity**: Are results diverse enough to avoid tunnel vision?

### 2.2 Evaluation Dimensions

#### A. Retrieval Quality Metrics

**Precision@K**
- Measures: Fraction of retrieved items that are relevant
- Formula: `Precision@K = (# relevant items in top-K) / K`
- Apply to: Module, file, and task retrieval
- Target: >0.7 for top-5, >0.5 for top-10

**Recall@K**
- Measures: Fraction of all relevant items retrieved
- Formula: `Recall@K = (# relevant items in top-K) / (total relevant items)`
- Apply to: Each server independently
- Target: >0.6 for combined system

**NDCG@K (Normalized Discounted Cumulative Gain)**
- Measures: Ranking quality with position weighting
- Formula: `NDCG@K = DCG@K / IDCG@K`
- Apply to: Overall retrieval ranking
- Target: >0.75 for top-10

**MRR (Mean Reciprocal Rank)**
- Measures: Position of first relevant result
- Formula: `MRR = 1 / (rank of first relevant item)`
- Apply to: Quick navigation scenarios
- Target: >0.6

#### B. Context Quality Metrics

**Context Relevance Score**
- Measures: Semantic similarity between context and query
- Method: Cosine similarity of aggregated embeddings
- Range: 0.0 to 1.0
- Target: >0.65 average

**Temporal Coverage**
- Measures: Time span of retrieved tasks
- Method: `(max_date - min_date) / total_project_duration`
- Target: >0.3 (covers at least 30% of history)

**Recency Score**
- Measures: Bias towards recent tasks
- Method: `avg(1 - (current_date - task_date) / max_age)`
- Range: 0.0 to 1.0
- Target: >0.4 (balanced with history)

**Diversity Score**
- Measures: Variety in retrieved modules/files
- Method: Unique modules / total retrieved items
- Target: >0.5 (avoid over-concentration)

#### C. Server-Specific Metrics

**Historical Server Contribution**
- Measures: Unique insights from historical data
- Method: Count of items from historical-only (not in recent)
- Target: 30-50% of final context

**Recent Server Contribution**
- Measures: Active development area coverage
- Method: Overlap with last 100 tasks
- Target: 50-70% of final context

**Complementarity Score**
- Measures: How well servers complement each other
- Method: `1 - (Jaccard similarity of results)`
- Target: >0.6 (low overlap = high complementarity)

#### D. End-to-End Task Success Metrics

**Task Completion Accuracy**
- Measures: Did developer complete task using recommendations?
- Method: Manual annotation on test set
- Target: >0.75 success rate

**Time to First Relevant File**
- Measures: Efficiency of navigation
- Method: Seconds until developer opens relevant file
- Target: <60 seconds average

**Context Sufficiency**
- Measures: Did developer need additional searches?
- Method: Binary (sufficient / insufficient)
- Target: >0.8 sufficiency rate

---

## 3. Experimental Design

### 3.1 Test Data Preparation

#### A. Ground Truth Creation

**Option 1: Historical Task Validation**
- Use completed tasks where file changes are known
- Ground truth = files modified in task commits
- Advantage: Objective, large dataset
- Challenge: Assumes file changes = relevance

**Option 2: Expert Annotation**
- Sample 50-100 diverse queries
- Expert developers annotate relevant modules/files
- Advantage: High quality ground truth
- Challenge: Time-intensive, limited scale

**Option 3: Hybrid Approach** (Recommended)
- Core set: 20-30 expert-annotated queries
- Extended set: 100+ historical task validations
- Validation: Cross-check automated with manual samples

#### B. Query Test Set

Create diverse query types:

1. **Exploratory Queries** (30%)
   - "How does authentication work?"
   - "Where is caching implemented?"
   - Target: Historical server should dominate

2. **Focused Queries** (40%)
   - "Fix login bug in UserController"
   - "Add validation to payment endpoint"
   - Target: Recent server should dominate

3. **Cross-cutting Queries** (30%)
   - "Improve performance across the application"
   - "Refactor error handling system-wide"
   - Target: Balanced contribution from both

### 3.2 Baseline Configurations

Compare dual-server against baselines:

1. **Baseline 1: Historical-Only**
   - Single server with `mod_n-all-bge-small`
   - Represents traditional approach

2. **Baseline 2: Recent-Only**
   - Single server with `recent-w100-bge-small`
   - Represents narrow context approach

3. **Baseline 3: Simple Merge**
   - Both collections in single server
   - No specialized fusion logic

4. **Proposed: Dual-Server with Fusion**
   - Two servers with intelligent fusion
   - Various fusion strategies tested

### 3.3 Fusion Strategy Experiments

Test multiple fusion approaches:

#### Experiment 3A: Score Weighting

```python
# Pseudo-code
def weighted_fusion(hist_results, recent_results, alpha=0.5):
    """
    alpha: Weight for recent server (0.0 = all historical, 1.0 = all recent)
    """
    combined = []
    for item in union(hist_results, recent_results):
        score = alpha * recent_score(item) + (1-alpha) * hist_score(item)
        combined.append((item, score))
    return rank_by_score(combined)
```

Test alpha values: [0.2, 0.35, 0.5, 0.65, 0.8]

#### Experiment 3B: Rank Fusion

```python
def reciprocal_rank_fusion(hist_results, recent_results, k=60):
    """
    RRF: Combines rankings without score normalization
    """
    scores = {}
    for rank, item in enumerate(hist_results):
        scores[item] = scores.get(item, 0) + 1/(k + rank)
    for rank, item in enumerate(recent_results):
        scores[item] = scores.get(item, 0) + 1/(k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Test k values: [30, 60, 100]

#### Experiment 3C: Query-Adaptive Fusion

```python
def adaptive_fusion(query, hist_results, recent_results):
    """
    Adjust weights based on query characteristics
    """
    # Classify query
    query_type = classify_query(query)  # exploratory / focused / cross-cutting

    # Adaptive weights
    if query_type == "exploratory":
        alpha = 0.3  # Favor historical
    elif query_type == "focused":
        alpha = 0.7  # Favor recent
    else:
        alpha = 0.5  # Balanced

    return weighted_fusion(hist_results, recent_results, alpha)
```

Requires: Query classifier (can be rule-based or ML-based)

#### Experiment 3D: Diversity-Aware Fusion

```python
def diversity_fusion(hist_results, recent_results, lambda_div=0.3):
    """
    Maximal Marginal Relevance (MMR) inspired fusion
    """
    selected = []
    candidates = merge_results(hist_results, recent_results)

    while len(selected) < top_k:
        best = None
        best_score = -inf

        for item in candidates:
            relevance = item.score
            diversity = min_similarity(item, selected)
            score = lambda_div * relevance + (1-lambda_div) * diversity

            if score > best_score:
                best_score = score
                best = item

        selected.append(best)
        candidates.remove(best)

    return selected
```

Test lambda values: [0.2, 0.4, 0.6]

### 3.4 Ablation Studies

Test impact of each component:

1. **Historical Server Ablation**
   - Run with recent-only
   - Measure drop in cross-cutting query performance

2. **Recent Server Ablation**
   - Run with historical-only
   - Measure drop in focused query performance

3. **Fusion Logic Ablation**
   - Use simple concatenation instead of fusion
   - Measure overall performance degradation

4. **Recency Weighting Ablation**
   - Remove temporal decay in scoring
   - Measure impact on recent task awareness

---

## 4. Implementation Guide

### 4.1 System Architecture

#### Component 1: Dual MCP Servers

**Server 1: Historical Context Server**

```python
# mcp_server_historical.py
import asyncio
from mcp.server import Server
from vector_backends import PostgresBackend

class HistoricalContextServer:
    def __init__(self):
        self.backend = PostgresBackend(
            collection_module='rag_exp_desc_module_modn_all_bge-small',
            collection_file='rag_exp_desc_file_modn_all_bge-small',
            collection_task='task_embeddings_all_bge-small'
        )
        self.server = Server("historical-context-server")

    @self.server.tool()
    async def search_historical(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "hybrid"
    ):
        """Search across all historical tasks for broad patterns"""
        results = await self.backend.search(
            query=query,
            top_k=top_k,
            search_type=search_type
        )
        return {
            'source': 'historical',
            'results': results,
            'total_tasks': 'all',
            'metadata': {
                'collection': 'mod_n-all-bge-small',
                'temporal_coverage': 'full'
            }
        }
```

**Server 2: Recent Context Server**

```python
# mcp_server_recent.py
import asyncio
from mcp.server import Server
from vector_backends import PostgresBackend

class RecentContextServer:
    def __init__(self):
        self.backend = PostgresBackend(
            collection_module='rag_exp_desc_module_recent_w100_bge-small',
            collection_file='rag_exp_desc_file_recent_w100_bge-small',
            collection_task='task_embeddings_recent_w100_bge-small'
        )
        self.server = Server("recent-context-server")

    @self.server.tool()
    async def search_recent(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "hybrid"
    ):
        """Search across recent tasks for current development focus"""
        results = await self.backend.search(
            query=query,
            top_k=top_k,
            search_type=search_type
        )
        return {
            'source': 'recent',
            'results': results,
            'total_tasks': 'last_100',
            'metadata': {
                'collection': 'recent-w100-bge-small',
                'temporal_coverage': 'recent'
            }
        }
```

#### Component 2: Fusion Orchestrator

```python
# fusion_orchestrator.py
import asyncio
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class FusedResult:
    """Combined result from both servers"""
    item_id: str
    item_type: str  # 'module', 'file', 'task'
    score: float
    source: str  # 'historical', 'recent', 'both'
    metadata: Dict

class FusionOrchestrator:
    def __init__(self, hist_client, recent_client, fusion_strategy='weighted'):
        self.hist_client = hist_client
        self.recent_client = recent_client
        self.fusion_strategy = fusion_strategy

    async def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5  # Weight for recent server
    ) -> List[FusedResult]:
        """
        Orchestrate search across both servers and fuse results

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for recent server (0.0-1.0)
        """
        # Parallel retrieval from both servers
        hist_task = asyncio.create_task(
            self.hist_client.search_historical(query, top_k=top_k*2)
        )
        recent_task = asyncio.create_task(
            self.recent_client.search_recent(query, top_k=top_k*2)
        )

        # Wait for both
        hist_results, recent_results = await asyncio.gather(hist_task, recent_task)

        # Fuse results
        if self.fusion_strategy == 'weighted':
            fused = self._weighted_fusion(hist_results, recent_results, alpha)
        elif self.fusion_strategy == 'rrf':
            fused = self._reciprocal_rank_fusion(hist_results, recent_results)
        elif self.fusion_strategy == 'adaptive':
            fused = self._adaptive_fusion(query, hist_results, recent_results)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Return top-k
        return fused[:top_k]

    def _weighted_fusion(
        self,
        hist_results: Dict,
        recent_results: Dict,
        alpha: float
    ) -> List[FusedResult]:
        """Weighted score fusion"""
        scores = {}

        # Score from historical server
        for rank, item in enumerate(hist_results['results']):
            item_key = (item['id'], item['type'])
            hist_score = item.get('score', 1.0 / (rank + 1))
            scores[item_key] = {
                'hist_score': hist_score,
                'recent_score': 0.0,
                'item': item
            }

        # Score from recent server
        for rank, item in enumerate(recent_results['results']):
            item_key = (item['id'], item['type'])
            recent_score = item.get('score', 1.0 / (rank + 1))

            if item_key in scores:
                scores[item_key]['recent_score'] = recent_score
            else:
                scores[item_key] = {
                    'hist_score': 0.0,
                    'recent_score': recent_score,
                    'item': item
                }

        # Compute final scores
        fused_results = []
        for item_key, data in scores.items():
            final_score = (1 - alpha) * data['hist_score'] + alpha * data['recent_score']

            # Determine source
            if data['hist_score'] > 0 and data['recent_score'] > 0:
                source = 'both'
            elif data['hist_score'] > 0:
                source = 'historical'
            else:
                source = 'recent'

            fused_results.append(FusedResult(
                item_id=item_key[0],
                item_type=item_key[1],
                score=final_score,
                source=source,
                metadata=data['item']
            ))

        # Sort by score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    def _reciprocal_rank_fusion(
        self,
        hist_results: Dict,
        recent_results: Dict,
        k: int = 60
    ) -> List[FusedResult]:
        """Reciprocal Rank Fusion"""
        scores = {}

        # RRF scoring from historical
        for rank, item in enumerate(hist_results['results']):
            item_key = (item['id'], item['type'])
            rrf_score = 1.0 / (k + rank + 1)
            scores[item_key] = {
                'score': rrf_score,
                'sources': ['historical'],
                'item': item
            }

        # RRF scoring from recent
        for rank, item in enumerate(recent_results['results']):
            item_key = (item['id'], item['type'])
            rrf_score = 1.0 / (k + rank + 1)

            if item_key in scores:
                scores[item_key]['score'] += rrf_score
                scores[item_key]['sources'].append('recent')
            else:
                scores[item_key] = {
                    'score': rrf_score,
                    'sources': ['recent'],
                    'item': item
                }

        # Convert to FusedResult
        fused_results = []
        for item_key, data in scores.items():
            source = 'both' if len(data['sources']) > 1 else data['sources'][0]
            fused_results.append(FusedResult(
                item_id=item_key[0],
                item_type=item_key[1],
                score=data['score'],
                source=source,
                metadata=data['item']
            ))

        # Sort by score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results

    def _adaptive_fusion(
        self,
        query: str,
        hist_results: Dict,
        recent_results: Dict
    ) -> List[FusedResult]:
        """Query-adaptive fusion"""
        # Classify query (simple rule-based for now)
        query_lower = query.lower()

        # Exploratory keywords
        if any(word in query_lower for word in ['how', 'where', 'what', 'understand', 'architecture']):
            alpha = 0.3  # Favor historical
        # Focused keywords
        elif any(word in query_lower for word in ['fix', 'bug', 'add', 'implement', 'feature']):
            alpha = 0.7  # Favor recent
        # Default
        else:
            alpha = 0.5  # Balanced

        # Use weighted fusion with adaptive alpha
        return self._weighted_fusion(hist_results, recent_results, alpha)
```

#### Component 3: Evaluation Harness

```python
# evaluation_harness.py
import json
import asyncio
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class EvaluationResult:
    """Result of evaluating a single query"""
    query: str
    query_type: str
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    ndcg_at_10: float
    mrr: float
    hist_contribution: float
    recent_contribution: float
    complementarity: float

class EvaluationHarness:
    def __init__(self, orchestrator, ground_truth_path: str):
        self.orchestrator = orchestrator
        self.ground_truth = self._load_ground_truth(ground_truth_path)

    def _load_ground_truth(self, path: str) -> Dict:
        """
        Load ground truth annotations

        Format:
        {
            "query_id_1": {
                "query": "Fix login bug",
                "query_type": "focused",
                "relevant_modules": ["auth", "user"],
                "relevant_files": ["src/auth/Login.java", ...],
                "relevant_tasks": [1234, 1235]
            },
            ...
        }
        """
        with open(path, 'r') as f:
            return json.load(f)

    async def evaluate_query(self, query_id: str) -> EvaluationResult:
        """Evaluate a single query"""
        gt = self.ground_truth[query_id]
        query = gt['query']

        # Run search
        results = await self.orchestrator.search(query, top_k=10)

        # Extract IDs
        retrieved_ids = [r.item_id for r in results]
        relevant_ids = set(gt['relevant_modules'] + gt['relevant_files'] +
                          [str(t) for t in gt['relevant_tasks']])

        # Compute metrics
        precision_5 = self._precision_at_k(retrieved_ids[:5], relevant_ids)
        precision_10 = self._precision_at_k(retrieved_ids[:10], relevant_ids)
        recall_10 = self._recall_at_k(retrieved_ids[:10], relevant_ids)
        ndcg_10 = self._ndcg_at_k(retrieved_ids[:10], relevant_ids)
        mrr = self._mrr(retrieved_ids, relevant_ids)

        # Server contributions
        hist_contrib = sum(1 for r in results if r.source in ['historical', 'both']) / len(results)
        recent_contrib = sum(1 for r in results if r.source in ['recent', 'both']) / len(results)

        # Complementarity (simplified)
        both_count = sum(1 for r in results if r.source == 'both')
        complementarity = 1.0 - (both_count / len(results))

        return EvaluationResult(
            query=query,
            query_type=gt['query_type'],
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            recall_at_10=recall_10,
            ndcg_at_10=ndcg_10,
            mrr=mrr,
            hist_contribution=hist_contrib,
            recent_contribution=recent_contrib,
            complementarity=complementarity
        )

    async def evaluate_all(self) -> List[EvaluationResult]:
        """Evaluate all queries in ground truth"""
        results = []
        for query_id in self.ground_truth.keys():
            result = await self.evaluate_query(query_id)
            results.append(result)
        return results

    def aggregate_results(self, results: List[EvaluationResult]) -> Dict:
        """Aggregate results by query type and overall"""
        # Overall metrics
        overall = {
            'precision_at_5': np.mean([r.precision_at_5 for r in results]),
            'precision_at_10': np.mean([r.precision_at_10 for r in results]),
            'recall_at_10': np.mean([r.recall_at_10 for r in results]),
            'ndcg_at_10': np.mean([r.ndcg_at_10 for r in results]),
            'mrr': np.mean([r.mrr for r in results]),
            'hist_contribution': np.mean([r.hist_contribution for r in results]),
            'recent_contribution': np.mean([r.recent_contribution for r in results]),
            'complementarity': np.mean([r.complementarity for r in results])
        }

        # By query type
        by_type = {}
        for query_type in ['exploratory', 'focused', 'cross-cutting']:
            type_results = [r for r in results if r.query_type == query_type]
            if type_results:
                by_type[query_type] = {
                    'count': len(type_results),
                    'precision_at_5': np.mean([r.precision_at_5 for r in type_results]),
                    'precision_at_10': np.mean([r.precision_at_10 for r in type_results]),
                    'recall_at_10': np.mean([r.recall_at_10 for r in type_results]),
                    'ndcg_at_10': np.mean([r.ndcg_at_10 for r in type_results]),
                    'mrr': np.mean([r.mrr for r in type_results]),
                    'hist_contribution': np.mean([r.hist_contribution for r in type_results]),
                    'recent_contribution': np.mean([r.recent_contribution for r in type_results])
                }

        return {
            'overall': overall,
            'by_query_type': by_type,
            'num_queries': len(results)
        }

    # Metric implementations
    def _precision_at_k(self, retrieved: List, relevant: set) -> float:
        """Precision@K"""
        if not retrieved:
            return 0.0
        relevant_retrieved = sum(1 for item in retrieved if item in relevant)
        return relevant_retrieved / len(retrieved)

    def _recall_at_k(self, retrieved: List, relevant: set) -> float:
        """Recall@K"""
        if not relevant:
            return 0.0
        relevant_retrieved = sum(1 for item in retrieved if item in relevant)
        return relevant_retrieved / len(relevant)

    def _ndcg_at_k(self, retrieved: List, relevant: set) -> float:
        """NDCG@K"""
        # DCG
        dcg = 0.0
        for i, item in enumerate(retrieved):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0

        # IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(retrieved), len(relevant))))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _mrr(self, retrieved: List, relevant: set) -> float:
        """Mean Reciprocal Rank"""
        for i, item in enumerate(retrieved):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save results to JSON"""
        aggregated = self.aggregate_results(results)
        detailed = [asdict(r) for r in results]

        output = {
            'summary': aggregated,
            'detailed_results': detailed
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_path}")
```

### 4.2 Configuration Files

**config_dual_server.py**

```python
"""
Dual-Server RAG Configuration
"""

# Server 1: Historical Context
HISTORICAL_SERVER = {
    'host': 'localhost',
    'port': 5001,
    'collections': {
        'module': 'rag_exp_desc_module_modn_all_bge-small',
        'file': 'rag_exp_desc_file_modn_all_bge-small',
        'task': 'task_embeddings_all_bge-small'
    },
    'metadata': {
        'coverage': 'all_tasks',
        'temporal_range': 'full_history'
    }
}

# Server 2: Recent Context
RECENT_SERVER = {
    'host': 'localhost',
    'port': 5002,
    'collections': {
        'module': 'rag_exp_desc_module_recent_w100_bge-small',
        'file': 'rag_exp_desc_file_recent_w100_bge-small',
        'task': 'task_embeddings_recent_w100_bge-small'
    },
    'metadata': {
        'coverage': 'last_100_tasks',
        'temporal_range': 'recent_window'
    }
}

# Fusion Configuration
FUSION_CONFIG = {
    'default_strategy': 'adaptive',  # 'weighted', 'rrf', 'adaptive'
    'weighted_alpha': 0.5,  # Default weight for recent server
    'rrf_k': 60,  # Reciprocal rank fusion parameter
    'adaptive_rules': {
        'exploratory_alpha': 0.3,  # Favor historical
        'focused_alpha': 0.7,      # Favor recent
        'default_alpha': 0.5       # Balanced
    }
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'ground_truth_path': 'evaluation/ground_truth.json',
    'output_dir': 'evaluation/results',
    'metrics': ['precision', 'recall', 'ndcg', 'mrr'],
    'top_k_values': [5, 10, 20]
}
```

### 4.3 Deployment Steps

1. **Create Recent Collections**

```bash
cd exp3
python create_collections_recent_window.py --window-size 100
```

2. **Start Historical Server**

```bash
cd ragmcp
python mcp_server_historical.py --port 5001
```

3. **Start Recent Server**

```bash
cd ragmcp
python mcp_server_recent.py --port 5002
```

4. **Start Fusion Orchestrator**

```bash
cd ragmcp
python fusion_orchestrator_server.py --hist-port 5001 --recent-port 5002
```

5. **Run Evaluation**

```bash
cd ragmcp
python run_evaluation.py --config config_dual_server.py
```

---

## 5. Testing Protocol

### 5.1 Unit Tests

Test each component independently:

```python
# test_fusion.py
import pytest
from fusion_orchestrator import FusionOrchestrator

class TestFusionOrchestrator:
    def test_weighted_fusion_alpha_0(self):
        """Alpha=0 should return only historical results"""
        # Mock results
        hist_results = {'results': [{'id': 'h1', 'type': 'file', 'score': 0.9}]}
        recent_results = {'results': [{'id': 'r1', 'type': 'file', 'score': 0.9}]}

        orchestrator = FusionOrchestrator(None, None, 'weighted')
        fused = orchestrator._weighted_fusion(hist_results, recent_results, alpha=0.0)

        # Should heavily favor historical
        assert fused[0].item_id == 'h1'

    def test_weighted_fusion_alpha_1(self):
        """Alpha=1 should return only recent results"""
        # Similar test for alpha=1
        pass

    def test_rrf_equal_ranks(self):
        """Items at same rank from both servers should score higher"""
        # Test RRF behavior
        pass
```

### 5.2 Integration Tests

Test server communication:

```python
# test_integration.py
import pytest
import asyncio
from fusion_orchestrator import FusionOrchestrator
from mcp_client import MCPClient

@pytest.mark.asyncio
async def test_dual_server_search():
    """Test search across both servers"""
    hist_client = MCPClient('localhost', 5001)
    recent_client = MCPClient('localhost', 5002)
    orchestrator = FusionOrchestrator(hist_client, recent_client)

    results = await orchestrator.search("fix authentication bug", top_k=10)

    assert len(results) == 10
    assert all(r.score > 0 for r in results)
    # Should have contributions from both servers
    sources = set(r.source for r in results)
    assert 'historical' in sources or 'recent' in sources or 'both' in sources
```

### 5.3 End-to-End Tests

Test complete workflow:

```python
# test_e2e.py
import pytest
from evaluation_harness import EvaluationHarness

@pytest.mark.asyncio
async def test_evaluation_pipeline():
    """Test full evaluation pipeline"""
    # Create mini ground truth
    mini_gt = {
        'q1': {
            'query': 'fix login bug',
            'query_type': 'focused',
            'relevant_files': ['src/auth/Login.java']
        }
    }

    harness = EvaluationHarness(orchestrator, ground_truth=mini_gt)
    results = await harness.evaluate_all()

    assert len(results) == 1
    assert 0.0 <= results[0].precision_at_5 <= 1.0
```

### 5.4 Performance Benchmarks

Measure latency and throughput:

```python
# benchmark.py
import time
import asyncio
from statistics import mean, stdev

async def benchmark_search(orchestrator, queries, num_runs=10):
    """Benchmark search performance"""
    latencies = []

    for _ in range(num_runs):
        for query in queries:
            start = time.time()
            await orchestrator.search(query, top_k=10)
            latency = time.time() - start
            latencies.append(latency)

    return {
        'mean_latency': mean(latencies),
        'std_latency': stdev(latencies),
        'min_latency': min(latencies),
        'max_latency': max(latencies),
        'throughput_qps': len(latencies) / sum(latencies)
    }
```

---

## 6. Practical Usage Examples

### 6.1 Basic Dual-Server Query

```python
# example_basic_query.py
import asyncio
from mcp_client import MCPClient
from fusion_orchestrator import FusionOrchestrator

async def main():
    # Connect to both servers
    hist_client = MCPClient('localhost', 5001)
    recent_client = MCPClient('localhost', 5002)

    # Create orchestrator
    orchestrator = FusionOrchestrator(
        hist_client,
        recent_client,
        fusion_strategy='adaptive'
    )

    # Search
    query = "implement user authentication with OAuth2"
    results = await orchestrator.search(query, top_k=10)

    # Display results
    print(f"\nQuery: {query}\n")
    print("Top 10 Results:")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.item_type}] {result.item_id}")
        print(f"   Score: {result.score:.4f} | Source: {result.source}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
```

### 6.2 Comparing Fusion Strategies

```python
# example_compare_strategies.py
import asyncio
from fusion_orchestrator import FusionOrchestrator

async def compare_strategies(query, hist_client, recent_client):
    """Compare different fusion strategies on same query"""
    strategies = ['weighted', 'rrf', 'adaptive']

    print(f"\nQuery: {query}\n")

    for strategy in strategies:
        orchestrator = FusionOrchestrator(
            hist_client,
            recent_client,
            fusion_strategy=strategy
        )
        results = await orchestrator.search(query, top_k=5)

        print(f"\n=== {strategy.upper()} Strategy ===")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.item_id} ({r.score:.4f}) from {r.source}")
```

### 6.3 Analyzing Server Contributions

```python
# example_analyze_contributions.py
import asyncio
from collections import Counter

async def analyze_contributions(queries, orchestrator):
    """Analyze how often each server contributes to results"""
    all_sources = []

    for query in queries:
        results = await orchestrator.search(query, top_k=10)
        all_sources.extend([r.source for r in results])

    counter = Counter(all_sources)
    total = sum(counter.values())

    print("\nServer Contribution Analysis:")
    print("-" * 40)
    for source, count in counter.items():
        pct = (count / total) * 100
        print(f"{source:15s}: {count:4d} ({pct:5.1f}%)")
```

### 6.4 Running Experiments

```python
# run_experiments.py
import asyncio
import json
from evaluation_harness import EvaluationHarness
from fusion_orchestrator import FusionOrchestrator
from mcp_client import MCPClient

async def run_experiment(experiment_name, config):
    """Run a single experiment configuration"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_name}")
    print(f"{'='*60}")

    # Setup
    hist_client = MCPClient('localhost', 5001)
    recent_client = MCPClient('localhost', 5002)

    orchestrator = FusionOrchestrator(
        hist_client,
        recent_client,
        fusion_strategy=config['fusion_strategy']
    )

    # Evaluate
    harness = EvaluationHarness(orchestrator, config['ground_truth_path'])
    results = await harness.evaluate_all()
    aggregated = harness.aggregate_results(results)

    # Save results
    output_path = f"evaluation/results/{experiment_name}.json"
    harness.save_results(results, output_path)

    # Print summary
    print(f"\nResults Summary:")
    print(f"  Precision@5:  {aggregated['overall']['precision_at_5']:.3f}")
    print(f"  Precision@10: {aggregated['overall']['precision_at_10']:.3f}")
    print(f"  Recall@10:    {aggregated['overall']['recall_at_10']:.3f}")
    print(f"  NDCG@10:      {aggregated['overall']['ndcg_at_10']:.3f}")
    print(f"  MRR:          {aggregated['overall']['mrr']:.3f}")
    print(f"\n  Historical contribution: {aggregated['overall']['hist_contribution']:.1%}")
    print(f"  Recent contribution:     {aggregated['overall']['recent_contribution']:.1%}")

    return aggregated

async def main():
    """Run all experiments"""
    experiments = {
        'baseline_historical_only': {
            'fusion_strategy': 'weighted',
            'alpha': 0.0,  # All historical
            'ground_truth_path': 'evaluation/ground_truth.json'
        },
        'baseline_recent_only': {
            'fusion_strategy': 'weighted',
            'alpha': 1.0,  # All recent
            'ground_truth_path': 'evaluation/ground_truth.json'
        },
        'weighted_balanced': {
            'fusion_strategy': 'weighted',
            'alpha': 0.5,
            'ground_truth_path': 'evaluation/ground_truth.json'
        },
        'rrf_fusion': {
            'fusion_strategy': 'rrf',
            'ground_truth_path': 'evaluation/ground_truth.json'
        },
        'adaptive_fusion': {
            'fusion_strategy': 'adaptive',
            'ground_truth_path': 'evaluation/ground_truth.json'
        }
    }

    all_results = {}
    for exp_name, config in experiments.items():
        result = await run_experiment(exp_name, config)
        all_results[exp_name] = result

    # Compare experiments
    print(f"\n{'='*60}")
    print("Experiment Comparison")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'P@5':>8} {'P@10':>8} {'R@10':>8} {'NDCG':>8}")
    print("-" * 60)

    for exp_name, result in all_results.items():
        overall = result['overall']
        print(f"{exp_name:<25} "
              f"{overall['precision_at_5']:>8.3f} "
              f"{overall['precision_at_10']:>8.3f} "
              f"{overall['recall_at_10']:>8.3f} "
              f"{overall['ndcg_at_10']:>8.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. Monitoring & Logging

### 7.1 Key Metrics to Monitor

**System Health**
- Server uptime and availability
- Query latency (p50, p95, p99)
- Error rates per server
- Memory and CPU usage

**Search Quality**
- Average relevance scores
- Result diversity metrics
- Server contribution ratios
- Fusion strategy effectiveness

**Usage Patterns**
- Query types distribution
- Peak usage times
- Common query patterns
- User satisfaction (if available)

### 7.2 Logging Configuration

```python
# logging_config.py
import logging
from datetime import datetime

def setup_logging():
    """Configure structured logging for dual-server system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/dual_server_{datetime.now():%Y%m%d}.log'),
            logging.StreamHandler()
        ]
    )

    # Create loggers
    logger_hist = logging.getLogger('historical_server')
    logger_recent = logging.getLogger('recent_server')
    logger_fusion = logging.getLogger('fusion_orchestrator')

    return logger_hist, logger_recent, logger_fusion
```

---

## 8. Expected Outcomes & Success Criteria

### 8.1 Quantitative Success Criteria

| Metric | Baseline (Historical-Only) | Target (Dual-Server) | Stretch Goal |
|--------|---------------------------|---------------------|--------------|
| **Precision@10** | 0.50 | 0.65 | 0.75 |
| **Recall@10** | 0.45 | 0.60 | 0.70 |
| **NDCG@10** | 0.60 | 0.75 | 0.85 |
| **MRR** | 0.50 | 0.65 | 0.75 |
| **User Satisfaction** | 3.5/5 | 4.0/5 | 4.5/5 |
| **Time to First Relevant** | 90s | 60s | 30s |

### 8.2 Qualitative Success Criteria

1. **Historical Insight**: System provides relevant historical patterns for cross-cutting concerns
2. **Recent Awareness**: System accurately identifies current development focus areas
3. **Complementarity**: Both servers provide unique value, minimal redundancy
4. **Robustness**: System degrades gracefully if one server is unavailable
5. **Explainability**: Users understand why results are recommended (source transparency)

### 8.3 Hypothesis Validation

**H1: Dual-server improves overall retrieval quality**
- Test: Compare dual-server NDCG@10 vs historical-only baseline
- Success: >15% improvement

**H2: Historical server excels at exploratory queries**
- Test: Compare historical contribution for exploratory vs focused queries
- Success: >70% contribution for exploratory queries

**H3: Recent server excels at focused queries**
- Test: Compare recent contribution for focused vs exploratory queries
- Success: >70% contribution for focused queries

**H4: Adaptive fusion outperforms fixed weighting**
- Test: Compare adaptive vs weighted (α=0.5) strategies
- Success: >10% improvement in NDCG@10

---

## 9. Troubleshooting Guide

### Common Issues

**Issue 1: Low Recent Server Contribution**
- Symptom: <20% of results from recent server
- Possible Causes:
  - Recent collection too small or sparse
  - Alpha weight too low
  - Recent embeddings not updated
- Solution: Check collection size, adjust alpha, verify embedding pipeline

**Issue 2: High Result Overlap (Low Complementarity)**
- Symptom: >70% of results appear in both servers
- Possible Causes:
  - Recent window too large (overlaps with historical)
  - Embedding drift between collections
- Solution: Reduce recent window size, ensure consistent embeddings

**Issue 3: Degraded Performance with Dual Servers**
- Symptom: Slower queries than single server
- Possible Causes:
  - Sequential instead of parallel retrieval
  - Network latency between servers
  - Inefficient fusion logic
- Solution: Verify async calls, optimize fusion algorithm

---

## 10. Future Enhancements

### 10.1 Potential Improvements

1. **Dynamic Window Sizing**
   - Adjust recent window based on project velocity
   - Expand during active development, contract during maintenance

2. **Multi-Granularity Fusion**
   - Different fusion strategies for modules vs files vs tasks
   - Layer-specific weighting

3. **Temporal Decay Functions**
   - Exponential decay for historical results
   - Boost for recent but not too recent (avoid WIP noise)

4. **Feedback Loop**
   - Learn fusion weights from user interactions
   - Personalized alpha per user or project

5. **Three-Server Architecture**
   - Add "medium-term" server (e.g., last 500 tasks)
   - Hierarchical fusion

### 10.2 Research Directions

1. **Query Classification ML Model**
   - Train classifier on query-type labels
   - More nuanced than rule-based approach

2. **Learned Fusion**
   - Train neural fusion model
   - Input: query embedding + retrieval scores
   - Output: fused ranking

3. **Context-Aware Embeddings**
   - Generate query embeddings conditioned on recent development history
   - Personalized to user's recent activity

---

## 11. Conclusion

The dual-server RAG MCP architecture provides a powerful framework for balancing **historical depth** and **recent context awareness** in code navigation and recommendation systems.

**Key Takeaways:**

1. **Complementary Strengths**: Historical server provides broad patterns; recent server provides focused context
2. **Flexible Fusion**: Multiple fusion strategies allow adaptation to different query types
3. **Rigorous Evaluation**: Comprehensive metrics ensure objective assessment of system improvements
4. **Practical Implementation**: Detailed code examples and deployment steps enable rapid prototyping

**Next Steps:**

1. Implement dual MCP servers with separate collections
2. Create ground truth test set (20-30 annotated queries minimum)
3. Run baseline experiments (historical-only, recent-only)
4. Implement and test fusion strategies (weighted, RRF, adaptive)
5. Analyze results and iterate on fusion logic
6. Deploy best-performing configuration
7. Monitor in production and refine based on usage

This evaluation framework provides a solid foundation for systematically improving RAG-based code navigation systems through architectural innovation and rigorous empirical testing.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-03
**Author**: RAG System Research Team
**Status**: Ready for Implementation
