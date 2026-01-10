# Dual MCP Server Architecture: Task-Based and File-Based Semantic Search

## Overview

This document describes an enhanced RAG architecture with **two parallel MCP servers** that provide complementary search capabilities:

1. **MCP Server 1 (Task-Based):** Searches historical task descriptions using semantic similarity
2. **MCP Server 2 (File-Based):** Searches actual code files using direct content embeddings

## Motivation

### Current Limitation

The existing system only searches **task descriptions** from historical data:

```
User Query → Task Embeddings Search → Related Tasks → Extract File References
```

**Problem:** This is indirect - you're finding files through task descriptions, not by actual code content.

### The Gap

When a user asks: *"How is authentication implemented?"*

**Current approach:**
1. Search task descriptions: "implement authentication", "fix login bug"
2. Find tasks that mention authentication
3. Return files referenced in those tasks
4. **Miss**: Files that contain authentication code but were never mentioned in tasks

**Better approach (Dual MCP):**
1. **MCP-1 (Tasks):** Find similar historical tasks
2. **MCP-2 (Files):** Directly search file contents for authentication code
3. **Combine:** Merge results for comprehensive coverage

## Architecture

### Dual MCP Server Design

```
                           User Query
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
        +-----------------------+  +------------------------+
        | MCP Server 1          |  | MCP Server 2           |
        | Task-Based Search     |  | File-Based Search      |
        |-----------------------|  |------------------------|
        | • Historical tasks    |  | • Direct file content  |
        | • Task descriptions   |  | • Code embeddings      |
        | • Solutions           |  | • Cosine similarity    |
        | • Recommendations     |  | • AST/structure aware  |
        +-----------------------+  +------------------------+
                    |                       |
                    v                       v
           Task Results (T)          File Results (F)
                    |                       |
                    +----------+------------+
                               |
                               v
                    +----------------------+
                    | Result Fusion Layer  |
                    | • Merge & dedupe     |
                    | • Rank by relevance  |
                    | • Combine scores     |
                    +----------------------+
                               |
                               v
                    Combined Context → LLM
```

### Component Details

#### MCP Server 1: Task-Based Search (Existing)

**Data Sources:**
- Historical task descriptions (9,799 tasks)
- Task outcomes and solutions
- File references from tasks
- User feedback and ratings

**Embedding Target:**
- Task descriptions (natural language)
- Problem statements
- Solution summaries

**Search Method:**
```sql
-- PostgreSQL with pgvector
SELECT task_id, description, files_involved, similarity
FROM task_embeddings
ORDER BY embedding <=> query_embedding
LIMIT k;
```

**Strengths:**
- Learns from historical solutions
- Understands problem-solution patterns
- Captures user intent better
- Includes context about why code exists

**Weaknesses:**
- Indirect file discovery
- Misses files not mentioned in tasks
- Dependent on task quality
- No direct code understanding

#### MCP Server 2: File-Based Search (NEW)

**Data Sources:**
- All source code files (12,532 files)
- File content with code embeddings
- AST structure (optional enhancement)
- Function/class signatures

**Embedding Target:**
- Raw file content
- Code chunks (functions, classes)
- Comments and docstrings
- Combined code + documentation

**Search Method:**
```python
# Direct cosine similarity on file embeddings
def search_files_by_content(query: str, top_k: int = 10):
    # Embed the query
    query_embedding = model.encode(query)  # bge-small

    # Get all file embeddings from vector DB
    file_embeddings = load_file_embeddings()

    # Compute cosine similarity
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        file_embeddings
    )

    # Return top-k most similar files
    top_indices = similarities.argsort()[0][-top_k:][::-1]
    return [(file_ids[i], similarities[0][i]) for i in top_indices]
```

**Strengths:**
- Direct code content search
- Finds all relevant files
- No dependency on task history
- Better for new/uncommon queries

**Weaknesses:**
- No historical context
- Doesn't know why code exists
- No solution patterns
- Raw code may be noisy

## Detailed Design: MCP Server 2

### Data Indexing Pipeline

```python
# File indexing workflow
class FileIndexingPipeline:
    """Index all source code files for semantic search"""

    def __init__(self, embedding_model='BAAI/bge-small-en-v1.5'):
        self.model = SentenceTransformer(embedding_model)
        self.db = PostgreSQLVectorDB()

    async def index_repository(self, repo_path: str):
        """Index all files in repository"""

        # 1. Discover all code files
        files = self._discover_source_files(repo_path)
        print(f"Found {len(files)} source files")

        # 2. Process each file
        for file_path in files:
            await self._index_file(file_path)

    async def _index_file(self, file_path: str):
        """Index a single file"""

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract metadata
        metadata = self._extract_metadata(file_path, content)

        # Chunking strategies (choose one or combine)
        chunks = self._chunk_file(content, strategy='hybrid')

        # Generate embeddings for each chunk
        for chunk in chunks:
            embedding = self.model.encode(chunk.text)

            # Store in vector database
            await self.db.insert_file_embedding(
                file_path=file_path,
                chunk_id=chunk.id,
                chunk_text=chunk.text,
                chunk_type=chunk.type,  # function, class, module
                embedding=embedding,
                metadata=metadata
            )

    def _chunk_file(self, content: str, strategy: str):
        """
        Chunk file content for embedding.

        Strategies:
        - 'whole': Entire file as one chunk
        - 'fixed': Fixed-size chunks (512 tokens)
        - 'semantic': Function/class boundaries
        - 'hybrid': Combination of semantic + sliding window
        """
        if strategy == 'whole':
            return [Chunk(id=0, text=content, type='file')]

        elif strategy == 'semantic':
            # Parse AST and extract functions/classes
            return self._semantic_chunking(content)

        elif strategy == 'hybrid':
            # Semantic chunks + overlap for context
            return self._hybrid_chunking(content)
```

### Chunking Strategies

**Option 1: Whole File Embedding**
```python
# Pros: Maximum context, simple
# Cons: Large files dilute relevance, token limits
embedding = model.encode(entire_file_content)
```

**Option 2: Fixed-Size Chunks**
```python
# Pros: Consistent sizes, handles large files
# Cons: May split logical units (functions)
chunks = split_text(content, chunk_size=512, overlap=50)
```

**Option 3: Semantic Chunking (RECOMMENDED)**
```python
# Pros: Respects code structure, better relevance
# Cons: More complex, AST parsing needed
import ast

def semantic_chunking(python_code: str):
    """Chunk by functions and classes"""
    tree = ast.parse(python_code)
    chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # Extract function/class with docstring
            chunk_text = ast.get_source_segment(python_code, node)
            chunks.append(Chunk(
                text=chunk_text,
                type=type(node).__name__,
                name=node.name,
                line_start=node.lineno,
                line_end=node.end_lineno
            ))

    return chunks
```

**Option 4: Hybrid Approach (BEST)**
```python
def hybrid_chunking(content: str):
    """Combine semantic units with overlapping windows"""

    # 1. Extract semantic units (functions, classes)
    semantic_chunks = semantic_chunking(content)

    # 2. For each semantic chunk, create embedding
    # 3. Also create sliding window chunks for coverage
    window_chunks = sliding_window(content, size=512, overlap=100)

    # 4. Return both types
    return semantic_chunks + window_chunks
```

### Vector Database Schema

```sql
-- PostgreSQL schema for file-based search
CREATE EXTENSION vector;

CREATE TABLE file_embeddings (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT,  -- For change detection
    chunk_id INTEGER,
    chunk_text TEXT,
    chunk_type TEXT,  -- 'file', 'function', 'class', 'window'
    chunk_metadata JSONB,  -- {line_start, line_end, name, etc.}
    embedding vector(384),  -- bge-small dimension
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast similarity search
CREATE INDEX ON file_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for file path lookups
CREATE INDEX idx_file_path ON file_embeddings(file_path);

-- Index for chunk type filtering
CREATE INDEX idx_chunk_type ON file_embeddings(chunk_type);
```

### MCP Server 2 API

```python
# mcp_server_files.py
class MCPFileSearchServer:
    """MCP Server for direct file content search"""

    def __init__(self):
        self.db = PostgreSQLVectorDB()
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    @mcp_tool
    async def search_files_by_content(
        self,
        query: str,
        top_k: int = 10,
        file_types: List[str] = None,
        chunk_type: str = None
    ) -> List[FileMatch]:
        """
        Search files by direct content similarity.

        Args:
            query: User's search query
            top_k: Number of results to return
            file_types: Filter by file extension ['.py', '.js', etc.]
            chunk_type: Filter by chunk type ['function', 'class', etc.]

        Returns:
            List of file matches with similarity scores
        """
        # Embed the query
        query_embedding = self.model.encode(query)

        # Build SQL query
        sql = """
            SELECT
                file_path,
                chunk_id,
                chunk_text,
                chunk_type,
                chunk_metadata,
                1 - (embedding <=> %s::vector) as similarity
            FROM file_embeddings
            WHERE 1=1
        """

        params = [query_embedding.tolist()]

        # Apply filters
        if file_types:
            sql += " AND file_path LIKE ANY(%s)"
            params.append([f"%{ext}" for ext in file_types])

        if chunk_type:
            sql += " AND chunk_type = %s"
            params.append(chunk_type)

        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding.tolist(), top_k])

        # Execute search
        results = await self.db.execute(sql, params)

        # Format results
        return [
            FileMatch(
                file_path=r['file_path'],
                chunk_text=r['chunk_text'],
                chunk_type=r['chunk_type'],
                similarity=r['similarity'],
                metadata=r['chunk_metadata']
            )
            for r in results
        ]

    @mcp_tool
    async def search_functions(
        self,
        query: str,
        top_k: int = 5
    ) -> List[FunctionMatch]:
        """Search for specific functions"""
        return await self.search_files_by_content(
            query=query,
            top_k=top_k,
            chunk_type='function'
        )

    @mcp_tool
    async def search_classes(
        self,
        query: str,
        top_k: int = 5
    ) -> List[ClassMatch]:
        """Search for specific classes"""
        return await self.search_files_by_content(
            query=query,
            top_k=top_k,
            chunk_type='class'
        )
```

## Result Fusion Strategy

### Combining Results from Both MCP Servers

```python
class DualMCPFusionLayer:
    """Combine results from task-based and file-based search"""

    async def search(
        self,
        query: str,
        top_k: int = 10,
        task_weight: float = 0.5,
        file_weight: float = 0.5
    ) -> FusedResults:
        """
        Execute parallel search on both MCP servers and fuse results.

        Args:
            query: User query
            top_k: Total results to return
            task_weight: Weight for task-based results (0-1)
            file_weight: Weight for file-based results (0-1)
        """
        # Parallel search on both servers
        task_results, file_results = await asyncio.gather(
            self.mcp_task.search_similar_tasks(query, top_k=top_k*2),
            self.mcp_file.search_files_by_content(query, top_k=top_k*2)
        )

        # Fusion strategies
        fused = self._fuse_results(
            task_results,
            file_results,
            strategy='weighted_hybrid',
            task_weight=task_weight,
            file_weight=file_weight
        )

        return fused[:top_k]

    def _fuse_results(
        self,
        task_results: List[TaskMatch],
        file_results: List[FileMatch],
        strategy: str,
        task_weight: float,
        file_weight: float
    ) -> List[FusedMatch]:
        """
        Fuse results using various strategies.

        Strategies:
        - 'weighted_hybrid': Weighted average of scores
        - 'rrf': Reciprocal Rank Fusion
        - 'cascade': File results first, then task context
        - 'vote': Count occurrences across both sources
        """

        if strategy == 'weighted_hybrid':
            return self._weighted_fusion(
                task_results, file_results, task_weight, file_weight
            )
        elif strategy == 'rrf':
            return self._reciprocal_rank_fusion(task_results, file_results)
        elif strategy == 'cascade':
            return self._cascade_fusion(task_results, file_results)
        elif strategy == 'vote':
            return self._voting_fusion(task_results, file_results)

    def _weighted_fusion(
        self,
        task_results: List[TaskMatch],
        file_results: List[FileMatch],
        task_weight: float,
        file_weight: float
    ) -> List[FusedMatch]:
        """Weighted average of similarity scores"""

        # Build file -> score mapping
        file_scores = {}

        # Add task-based scores
        for task in task_results:
            for file_path in task.files_involved:
                if file_path not in file_scores:
                    file_scores[file_path] = {
                        'task_score': 0,
                        'file_score': 0,
                        'task_context': []
                    }
                file_scores[file_path]['task_score'] = max(
                    file_scores[file_path]['task_score'],
                    task.similarity
                )
                file_scores[file_path]['task_context'].append(task)

        # Add file-based scores
        for file in file_results:
            if file.file_path not in file_scores:
                file_scores[file.file_path] = {
                    'task_score': 0,
                    'file_score': 0,
                    'task_context': []
                }
            file_scores[file.file_path]['file_score'] = max(
                file_scores[file.file_path]['file_score'],
                file.similarity
            )

        # Compute weighted scores
        fused = []
        for file_path, scores in file_scores.items():
            combined_score = (
                task_weight * scores['task_score'] +
                file_weight * scores['file_score']
            )

            fused.append(FusedMatch(
                file_path=file_path,
                combined_score=combined_score,
                task_score=scores['task_score'],
                file_score=scores['file_score'],
                source='both' if scores['task_score'] > 0 and scores['file_score'] > 0 else
                       'task' if scores['task_score'] > 0 else 'file',
                task_context=scores['task_context']
            ))

        # Sort by combined score
        fused.sort(key=lambda x: x.combined_score, reverse=True)
        return fused

    def _reciprocal_rank_fusion(
        self,
        task_results: List[TaskMatch],
        file_results: List[FileMatch],
        k: int = 60
    ) -> List[FusedMatch]:
        """
        Reciprocal Rank Fusion (RRF)

        RRF Score = sum(1 / (k + rank_i)) for all sources

        Good for combining ranked lists without score calibration.
        """
        file_rrf_scores = defaultdict(float)

        # Task-based ranks
        for rank, task in enumerate(task_results, 1):
            for file_path in task.files_involved:
                file_rrf_scores[file_path] += 1 / (k + rank)

        # File-based ranks
        for rank, file in enumerate(file_results, 1):
            file_rrf_scores[file.file_path] += 1 / (k + rank)

        # Convert to FusedMatch objects
        fused = [
            FusedMatch(
                file_path=file_path,
                combined_score=rrf_score,
                source='rrf'
            )
            for file_path, rrf_score in file_rrf_scores.items()
        ]

        fused.sort(key=lambda x: x.combined_score, reverse=True)
        return fused
```

## Evaluation Metrics

### 1. Individual MCP Server Metrics

#### MCP Server 1 (Task-Based) Metrics

**Precision@K:**
```python
def precision_at_k(retrieved_files: List[str], relevant_files: List[str], k: int) -> float:
    """
    Precision@K = (# relevant files in top-K) / K
    """
    top_k = retrieved_files[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_files))
    return relevant_in_top_k / k
```

**Recall@K:**
```python
def recall_at_k(retrieved_files: List[str], relevant_files: List[str], k: int) -> float:
    """
    Recall@K = (# relevant files in top-K) / (total # relevant files)
    """
    top_k = retrieved_files[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_files))
    return relevant_in_top_k / len(relevant_files)
```

**Mean Reciprocal Rank (MRR):**
```python
def mrr(retrieved_files: List[str], relevant_files: List[str]) -> float:
    """
    MRR = 1 / rank of first relevant file
    """
    for rank, file in enumerate(retrieved_files, 1):
        if file in relevant_files:
            return 1 / rank
    return 0.0
```

**Task Coverage:**
```python
def task_coverage(query: str, task_results: List[TaskMatch]) -> float:
    """
    Does task history cover this type of query?
    """
    if len(task_results) == 0:
        return 0.0

    avg_similarity = np.mean([t.similarity for t in task_results])
    return avg_similarity
```

#### MCP Server 2 (File-Based) Metrics

**Direct Hit Rate:**
```python
def direct_hit_rate(file_results: List[FileMatch], relevant_files: List[str]) -> float:
    """
    % of relevant files found directly (not through tasks)
    """
    found_files = [f.file_path for f in file_results]
    hits = len(set(found_files) & set(relevant_files))
    return hits / len(relevant_files) if relevant_files else 0.0
```

**Content Relevance Score:**
```python
def content_relevance(file_results: List[FileMatch]) -> float:
    """
    Average similarity score of retrieved chunks
    """
    if not file_results:
        return 0.0
    return np.mean([f.similarity for f in file_results])
```

**Chunk Quality:**
```python
def chunk_quality(file_results: List[FileMatch], ground_truth_chunks: List[str]) -> float:
    """
    How well do retrieved chunks match expected code segments?
    """
    retrieved_chunks = [f.chunk_text for f in file_results]

    # Measure overlap/similarity with ground truth chunks
    scores = []
    for gt_chunk in ground_truth_chunks:
        max_sim = max([
            jaccard_similarity(gt_chunk, ret_chunk)
            for ret_chunk in retrieved_chunks
        ])
        scores.append(max_sim)

    return np.mean(scores)
```

### 2. Fusion Metrics

**Coverage Improvement:**
```python
def coverage_improvement(
    task_only_files: Set[str],
    file_only_files: Set[str],
    fused_files: Set[str],
    relevant_files: Set[str]
) -> Dict[str, float]:
    """
    Measure how much fusion improves coverage
    """
    return {
        'task_only_recall': len(task_only_files & relevant_files) / len(relevant_files),
        'file_only_recall': len(file_only_files & relevant_files) / len(relevant_files),
        'fused_recall': len(fused_files & relevant_files) / len(relevant_files),
        'improvement': (len(fused_files & relevant_files) -
                       max(len(task_only_files & relevant_files),
                           len(file_only_files & relevant_files))) / len(relevant_files)
    }
```

**Complementarity Score:**
```python
def complementarity_score(
    task_results: List[str],
    file_results: List[str]
) -> float:
    """
    Measure how complementary the two sources are.

    Score = (# files found by only one source) / (# total unique files)

    High score = sources are complementary
    Low score = sources are redundant
    """
    task_files = set(task_results)
    file_files = set(file_results)

    only_task = task_files - file_files
    only_file = file_files - task_files
    unique_contribution = len(only_task) + len(only_file)
    total_unique = len(task_files | file_files)

    return unique_contribution / total_unique if total_unique > 0 else 0.0
```

**Ranking Quality (NDCG):**
```python
def ndcg_at_k(
    ranked_files: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain

    Measures ranking quality considering graded relevance.
    """
    def dcg(scores):
        return sum(
            (2**score - 1) / np.log2(idx + 2)
            for idx, score in enumerate(scores)
        )

    # Actual DCG
    actual_scores = [relevance_scores.get(f, 0.0) for f in ranked_files[:k]]
    actual_dcg = dcg(actual_scores)

    # Ideal DCG
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

### 3. End-to-End Metrics

**User Satisfaction Metrics:**
```python
class UserSatisfactionMetrics:
    """Metrics based on user feedback"""

    def accuracy_score(self, user_rating: int) -> float:
        """
        User rates answer accuracy (1-5)
        """
        return user_rating / 5.0

    def completeness_score(self, user_feedback: Dict) -> float:
        """
        Did the response include all necessary files?
        """
        return 1.0 if user_feedback['all_files_found'] else 0.5

    def efficiency_score(self, num_results: int, relevant_results: int) -> float:
        """
        Are results efficient (not too many irrelevant)?
        """
        if num_results == 0:
            return 0.0
        return relevant_results / num_results
```

**Response Quality:**
```python
def response_quality_score(
    llm_response: str,
    ground_truth: str,
    retrieved_context: List[str]
) -> Dict[str, float]:
    """
    Measure overall response quality
    """
    return {
        'answer_similarity': semantic_similarity(llm_response, ground_truth),
        'context_utilization': context_usage_rate(llm_response, retrieved_context),
        'hallucination_rate': check_hallucinations(llm_response, retrieved_context),
        'code_accuracy': validate_code_suggestions(llm_response)
    }
```

## Testing Methodology

### Test Dataset Creation

```python
class EvaluationDataset:
    """Create evaluation dataset for testing dual MCP"""

    def __init__(self):
        self.test_queries = []

    def create_test_set(self, n_queries: int = 100):
        """
        Create diverse test queries with ground truth.

        Query Types:
        1. Historical queries (should favor task-based)
        2. New/uncommon queries (should favor file-based)
        3. Hybrid queries (need both sources)
        4. Specific code queries (class/function names)
        5. Conceptual queries (design patterns, architectures)
        """

        # Type 1: Historical queries (from existing task DB)
        historical = self._sample_historical_tasks(n=20)

        # Type 2: New queries (manually created)
        new_queries = [
            ("How is caching implemented?", ["cache.py", "redis_client.py"]),
            ("Where is rate limiting handled?", ["middleware/rate_limit.py"]),
            ("Show me WebSocket connection logic", ["websocket_handler.py"]),
            # ... more
        ]

        # Type 3: Hybrid queries
        hybrid = [
            ("How to add a new API endpoint?", [
                # Expected from tasks: patterns, examples
                # Expected from files: actual endpoint implementations
            ]),
            # ... more
        ]

        # Type 4: Specific code queries
        specific = [
            ("Find the UserAuthentication class", ["auth/user_auth.py"]),
            ("Where is validate_token function?", ["utils/token_validator.py"]),
            # ... more
        ]

        # Type 5: Conceptual queries
        conceptual = [
            ("How is the repository pattern used?", [
                # Multiple files implementing the pattern
            ]),
            # ... more
        ]

        self.test_queries = historical + new_queries + hybrid + specific + conceptual
        return self.test_queries

    def _sample_historical_tasks(self, n: int):
        """Sample queries from task history with known good results"""
        # Query task DB for diverse, successful tasks
        pass
```

### A/B Testing Framework

```python
class ABTestFramework:
    """Compare different retrieval strategies"""

    async def run_experiment(
        self,
        test_queries: List[TestQuery],
        variants: List[str]
    ) -> ExperimentResults:
        """
        Test different configurations:

        Variants:
        - A: Task-based only (baseline)
        - B: File-based only
        - C: Fusion (50/50 weight)
        - D: Fusion (70/30 task/file)
        - E: Fusion (30/70 task/file)
        - F: RRF fusion
        """

        results = defaultdict(list)

        for query in test_queries:
            for variant in variants:
                # Run search with variant configuration
                retrieved = await self._search_with_config(query, variant)

                # Evaluate
                metrics = self._evaluate(
                    retrieved=retrieved,
                    ground_truth=query.relevant_files
                )

                results[variant].append(metrics)

        return self._summarize_results(results)

    def _evaluate(self, retrieved: List[str], ground_truth: List[str]):
        """Compute all metrics for one query"""
        return {
            'precision@5': precision_at_k(retrieved, ground_truth, 5),
            'precision@10': precision_at_k(retrieved, ground_truth, 10),
            'recall@5': recall_at_k(retrieved, ground_truth, 5),
            'recall@10': recall_at_k(retrieved, ground_truth, 10),
            'mrr': mrr(retrieved, ground_truth),
            'ndcg@10': ndcg_at_k(retrieved, ground_truth, 10)
        }
```

### Component Isolation Testing

```python
class ComponentIsolationTest:
    """Test impact of each component separately"""

    async def test_task_mcp_only(self, queries: List[str]):
        """Baseline: Only task-based search"""
        results = []
        for query in queries:
            task_results = await self.mcp_task.search(query)
            files = self._extract_files_from_tasks(task_results)
            results.append(files)
        return results

    async def test_file_mcp_only(self, queries: List[str]):
        """Only file-based search"""
        results = []
        for query in queries:
            file_results = await self.mcp_file.search(query)
            results.append(file_results)
        return results

    async def test_ablation(self, queries: List[str]):
        """
        Ablation study: Remove components one at a time

        Configurations:
        1. Full system (both MCPs)
        2. Remove task MCP (only files)
        3. Remove file MCP (only tasks)
        4. Remove fusion (simple concatenation)
        """
        configs = {
            'full': lambda q: self.dual_search(q, fusion=True),
            'no_task': lambda q: self.file_only(q),
            'no_file': lambda q: self.task_only(q),
            'no_fusion': lambda q: self.dual_search(q, fusion=False)
        }

        results = {}
        for name, search_fn in configs.items():
            config_results = []
            for query in queries:
                retrieved = await search_fn(query)
                config_results.append(retrieved)
            results[name] = config_results

        return results
```

### Statistical Significance Testing

```python
def statistical_comparison(
    baseline_metrics: List[float],
    experiment_metrics: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Test if improvement is statistically significant.

    Uses paired t-test (same queries evaluated on both systems)
    """
    from scipy import stats

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(experiment_metrics, baseline_metrics)

    # Effect size (Cohen's d)
    mean_diff = np.mean(experiment_metrics) - np.mean(baseline_metrics)
    pooled_std = np.sqrt(
        (np.std(baseline_metrics)**2 + np.std(experiment_metrics)**2) / 2
    )
    cohens_d = mean_diff / pooled_std

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'cohens_d': cohens_d,
        'effect_size': 'small' if abs(cohens_d) < 0.5 else
                      'medium' if abs(cohens_d) < 0.8 else 'large',
        'mean_improvement': mean_diff,
        'improvement_pct': (mean_diff / np.mean(baseline_metrics)) * 100
    }
```

## Expected Results & Analysis

### Hypothesis: When Each MCP Excels

#### Task-Based MCP (Server 1) Expected to Excel:

**Query Types:**
- "How do I implement X?" (pattern-based)
- "Best practices for Y" (requires context)
- "Fix bug in Z" (similar to historical tasks)
- Queries matching historical task descriptions

**Metrics:**
- Higher precision for common development tasks
- Better contextual understanding
- Provides "why" (not just "where")

**Example:**
```
Query: "How to add pagination to API endpoints?"

Task MCP finds:
- Task #1234: "Implement pagination for user list endpoint"
  → Shows pattern used before
  → Includes test examples
  → Explains pagination strategy

File MCP finds:
- Files containing "pagination" keyword
  → May include irrelevant uses
  → No context on preferred pattern
```

#### File-Based MCP (Server 2) Expected to Excel:

**Query Types:**
- Specific code element searches
- New/uncommon functionality
- "Where is X defined?"
- Code that exists but wasn't frequently tasked

**Metrics:**
- Higher recall (finds all relevant files)
- Better for exploratory queries
- Discovers code not in task history

**Example:**
```
Query: "Find all classes that inherit from BaseModel"

Task MCP:
- Limited to tasks that mentioned BaseModel
- May miss newer subclasses

File MCP:
- Directly searches code content
- Finds all actual subclasses
- More complete coverage
```

### Expected Fusion Benefits

```python
# Expected metrics comparison
Expected Results = {
    'Task-Only (Baseline)': {
        'Precision@10': 0.72,
        'Recall@10': 0.58,
        'Coverage': 0.64,
        'User Satisfaction': 3.8/5
    },
    'File-Only': {
        'Precision@10': 0.68,
        'Recall@10': 0.71,
        'Coverage': 0.82,
        'User Satisfaction': 3.5/5
    },
    'Dual MCP (Fused)': {
        'Precision@10': 0.78,  # +8% vs baseline
        'Recall@10': 0.79,     # +36% vs baseline
        'Coverage': 0.89,      # +39% vs baseline
        'User Satisfaction': 4.3/5  # +13% vs baseline
    }
}
```

### Complementarity Analysis

```python
def analyze_complementarity(test_results: Dict):
    """
    Analyze how often each MCP finds unique results.

    Expected distribution:
    - 30% files found by both (core codebase)
    - 20% files found only by task MCP (historical context)
    - 50% files found only by file MCP (comprehensive coverage)
    """

    for query_result in test_results:
        task_files = set(query_result['task_mcp'])
        file_files = set(query_result['file_mcp'])

        both = task_files & file_files
        only_task = task_files - file_files
        only_file = file_files - task_files

        print(f"Query: {query_result['query']}")
        print(f"  Both: {len(both)} ({len(both)/len(task_files | file_files)*100:.1f}%)")
        print(f"  Only Task: {len(only_task)}")
        print(f"  Only File: {len(only_file)}")
        print(f"  Complementarity: {(len(only_task) + len(only_file)) / len(task_files | file_files):.2f}")
```

## Implementation Checklist

### Phase 1: File Indexing
- [ ] Implement file discovery (all source files)
- [ ] Choose chunking strategy (recommend: hybrid)
- [ ] Set up bge-small embedding model
- [ ] Create PostgreSQL vector table
- [ ] Index all 12,532 files
- [ ] Verify embedding quality with sample queries

### Phase 2: MCP Server 2
- [ ] Implement `MCPFileSearchServer` class
- [ ] Add cosine similarity search
- [ ] Implement filtering (file type, chunk type)
- [ ] Add specialized search methods (functions, classes)
- [ ] Test with sample queries

### Phase 3: Fusion Layer
- [ ] Implement result fusion logic
- [ ] Add weighted combination
- [ ] Add RRF fusion option
- [ ] Tune fusion weights
- [ ] Test with diverse queries

### Phase 4: Evaluation
- [ ] Create test dataset (100+ queries)
- [ ] Implement evaluation metrics
- [ ] Run A/B tests
- [ ] Perform ablation study
- [ ] Statistical significance testing
- [ ] Document results

### Phase 5: Integration with Reflective Agent
- [ ] Update agent to use both MCPs
- [ ] Add logic to decide when to use which server
- [ ] Include dual MCP in reasoning phase
- [ ] Update reflection to assess both sources
- [ ] Modify response format to show source attribution

## Conclusion

The dual MCP server architecture provides:

1. **Comprehensive Coverage**: Task-based search for historical context + file-based search for direct code access

2. **Complementary Strengths**:
   - Task MCP: Historical patterns, contextual understanding
   - File MCP: Complete coverage, direct code search

3. **Measurable Impact**: Clear metrics to evaluate each component's contribution

4. **Flexibility**: Can adjust fusion weights based on query type

5. **Robustness**: If one MCP fails or has poor coverage, the other provides backup

### Key Metrics to Watch

**Primary Success Metric**: Recall@10 improvement over baseline
**Target**: +25% recall improvement with dual MCP fusion

**Secondary Metrics**:
- Precision@10 (maintain or improve)
- User satisfaction scores
- Complementarity score (expect 0.6-0.7)
- Coverage improvement for new query types

This architecture transforms the RAG system from task-history-dependent to comprehensive code search, while maintaining the valuable historical context that task descriptions provide.
