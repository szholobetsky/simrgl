# Compositional Code Embeddings: Vector Arithmetic for Program Structure

## Executive Summary

This document explores a novel approach to representing program composition through **algebraic operations on function embeddings**. The core hypothesis: if function `d()` is composed of functions `a()`, `b()`, and `c()` (e.g., `d() = a(b() + c())`), then the embedding vector of `d()` can be derived from the embedding vectors of its components through specific vector operations.

This extends beyond cosine similarity to investigate whether **vector arithmetic in embedding space can encode the compositional and transformational relationships between program entities**, similar to how word2vec captures semantic relationships through operations like "king - man + woman = queen."

---

## 1. Motivation and Core Idea

### 1.1 The Limitation of Cosine Similarity

Current approaches to code embeddings primarily rely on **cosine similarity** to measure semantic relatedness:

```python
similarity(a, b) = cos(θ) = (a · b) / (||a|| × ||b||)
```

**Problem**: Cosine similarity captures **resemblance** but not **composition** or **transformation**.

**Example**:
```python
def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

def calculate(a, b, c):
    return multiply(add(a, b), c)  # (a + b) * c
```

Using cosine similarity:
- `cos_sim(calculate, add)` might be 0.65
- `cos_sim(calculate, multiply)` might be 0.68
- But this doesn't tell us that `calculate` **composes** `add` and `multiply`

### 1.2 The Compositional Hypothesis

**Key Insight**: Function composition in code might be representable as vector operations in embedding space.

**Hypothesis**:
```
If d() calls a(b() + c()), then:
embedding(d) ≈ f(embedding(a), embedding(b), embedding(c))

where f is some vector operation (addition, multiplication, concatenation, etc.)
```

**Analogies**:
- **Word2Vec**: `king - man + woman ≈ queen` ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781))
- **Code2Vec**: Path embeddings aggregate to function embeddings ([Alon et al., 2019](https://dl.acm.org/doi/10.1145/3290353))
- **Compositional Semantics**: Sentence meaning = f(word meanings) ([research on compositional distributional semantics](https://direct.mit.edu/coli/article/51/1/139/124463/Compositionality-and-Sentence-Meaning-Comparing))

### 1.3 From File/Module to Entity-Level Granularity

Current RAG approaches operate at **file or module level**. This proposal shifts to **entity-level** (functions, classes, methods) with explicit modeling of **relationships**:

```
Old: Module A → Module B (cosine similarity)
New: Function a() → Function b() → Function c() (compositional relationships)
```

This aligns with the DUAL_MCP_SERVER_ARCHITECTURE.md concept: moving from coarse-grained to fine-grained semantic understanding.

---

## 2. Theoretical Framework

### 2.1 Types of Program Relationships

#### A. Compositional Relationships (Function Calls)

**Pattern**: Function `d` calls function `a`
```python
def d():
    result = a()  # direct call
    return result
```

**Vector Hypothesis**:
```
embedding(d) ≈ α × embedding(a) + β × context(d)
```

Where `α` and `β` are learned weights.

#### B. Sequential Composition (Chaining)

**Pattern**: Function `d` chains functions `a`, then `b`, then `c`
```python
def d(x):
    y = a(x)
    z = b(y)
    return c(z)
```

**Vector Hypothesis**:
```
embedding(d) ≈ embedding(a) + shift₁ + embedding(b) + shift₂ + embedding(c)
```

Or with order encoding:
```
embedding(d) ≈ w₁⊙embedding(a) + w₂⊙embedding(b) + w₃⊙embedding(c)
```

Where `⊙` is element-wise multiplication and `w₁, w₂, w₃` encode position.

#### C. Parallel Composition (Independent Operations)

**Pattern**: Function `d` combines results from `b` and `c`
```python
def d(x):
    result_b = b(x)
    result_c = c(x)
    return result_b + result_c
```

**Vector Hypothesis**:
```
embedding(d) ≈ embedding(b) + embedding(c) + merge_bias
```

#### D. Higher-Order Composition (Functional Application)

**Pattern**: Function `a` transforms the output of combined functions
```python
def d(x):
    return a(b(x) + c(x))  # a ∘ (b + c)
```

**Vector Hypothesis** (Multiplicative Composition):
```
embedding(d) ≈ embedding(a) ⊗ (embedding(b) + embedding(c))
```

Where `⊗` could be:
- Element-wise multiplication (Hadamard product)
- Outer product followed by dimensionality reduction
- Learned bilinear transformation: `W × embedding(a) × (embedding(b) + embedding(c))`

### 2.2 Vector Operations Taxonomy

| Operation | Math | Code Example | Semantic Meaning |
|-----------|------|--------------|------------------|
| **Addition** | `v_d = v_a + v_b` | `d() = a() + b()` | Merging/combining |
| **Weighted Sum** | `v_d = α·v_a + β·v_b` | `d() { mostly a(), some b() }` | Dominant component |
| **Hadamard Product** | `v_d = v_a ⊙ v_b` | `d() = a(b())` | Filtering/transformation |
| **Concatenation** | `v_d = [v_a; v_b]` | `d() { a(); b(); }` | Sequential execution |
| **Bilinear Form** | `v_d = v_a^T W v_b` | `d() = a() interacts with b()` | Complex interaction |
| **Circular Convolution** | `v_d = v_a ★ v_b` | Role binding (HRR) | Structured composition |

### 2.3 Mathematical Models from Literature

#### A. Translation-Based (Inspired by TransE for Knowledge Graphs)

From knowledge graph embeddings ([Bordes et al., 2013](https://www.hindawi.com/journals/sp/2018/6325635/)):

```
If "d calls a", then: embedding(d) + relation_calls ≈ embedding(a)
```

Applied to code:
```
embedding(caller) + relation_type ≈ embedding(callee)
```

#### B. Tensor Decomposition

For complex multi-way relationships:
```
score(d, relation, a, b, c) = Σᵢ Wᵢ × embedding(d)ᵢ × relation_vectorᵢ × embedding(a)ᵢ × ...
```

#### C. Neural Compositional Models

Learn composition function via neural network:
```python
def compose(embedding_a, embedding_b, embedding_c, relation_type):
    # Neural network that learns how to combine
    hidden = MLP([embedding_a, embedding_b, embedding_c, relation_type])
    composed = output_layer(hidden)
    return composed
```

This is more expressive but less interpretable than algebraic operations.

---

## 3. Hypotheses and Research Questions

### 3.1 Primary Hypotheses

#### H1: Additive Composition for Parallel Calls
```
If d() = b() + c(), then:
embedding(d) ≈ embedding(b) + embedding(c)

Metric: ||embedding(d) - (embedding(b) + embedding(c))|| < threshold
```

#### H2: Multiplicative Composition for Function Application
```
If d() = a(b()), then:
embedding(d) ≈ embedding(a) ⊙ embedding(b)

Metric: cos_sim(embedding(d), embedding(a) ⊙ embedding(b)) > baseline
```

#### H3: Weighted Composition Based on Call Frequency
```
If d() calls a() three times and b() once, then:
embedding(d) ≈ 0.75 × embedding(a) + 0.25 × embedding(b)

Metric: Correlation between call frequency and weight magnitude
```

#### H4: Sequential Encoding via Positional Weighting
```
If d() executes a(), then b(), then c(), then:
embedding(d) ≈ w₁⊙embedding(a) + w₂⊙embedding(b) + w₃⊙embedding(c)

where w₁, w₂, w₃ encode execution order
```

#### H5: Business Domain Alignment
```
If function f() implements business concept C, then:
cos_sim(embedding(f), embedding(business_description_C)) > cos_sim(embedding(f), embedding(unrelated_concept))

Metric: Domain-specific semantic similarity
```

### 3.2 Secondary Research Questions

**RQ1: Embedding Quality**
- Do different embedding models (CodeBERT, GraphCodeBERT, CodeT5) exhibit compositional properties?
- Are AST-based embeddings more compositional than token-based?

**RQ2: Granularity**
- At what level does compositionality emerge? (Statement, function, class, module?)
- Are local compositions (within a function) more predictable than global ones?

**RQ3: Domain Transfer**
- Do compositional patterns learned from one codebase transfer to another?
- Are business-domain embeddings (from task descriptions) composable with code embeddings?

**RQ4: Noise and Robustness**
- How do variable names, comments, and code style affect compositional properties?
- Can we isolate semantic composition from syntactic variation?

---

## 4. Metrics and Evaluation

### 4.1 Direct Reconstruction Metrics

#### Mean Squared Error (MSE) in Embedding Space
```python
def composition_mse(d_actual, a, b, c, operation='add'):
    """
    Measure how well composed embedding matches actual embedding
    """
    if operation == 'add':
        d_composed = embedding(a) + embedding(b) + embedding(c)
    elif operation == 'multiply':
        d_composed = embedding(a) * (embedding(b) + embedding(c))
    # ... other operations

    mse = np.mean((embedding(d_actual) - d_composed) ** 2)
    return mse
```

**Target**: MSE < baseline (random composition)

#### Cosine Similarity to Composed Vector
```python
def composition_similarity(d_actual, composed_vector):
    """
    Does the composed vector point in the same direction?
    """
    return cosine_similarity(embedding(d_actual), composed_vector)
```

**Target**: Similarity > 0.7 (threshold may vary)

### 4.2 Ranking-Based Metrics

#### Rank of True Composed Function
```python
def composition_rank(d_actual, all_functions, a, b, c):
    """
    Given composed vector, rank all functions by similarity.
    What rank is the actual function d?
    """
    composed = compose_operation(a, b, c)

    similarities = [
        (f, cosine_similarity(composed, embedding(f)))
        for f in all_functions
    ]
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Find rank of d_actual
    rank = next(i for i, (f, _) in enumerate(ranked, 1) if f == d_actual)
    return rank
```

**Target**: Rank in top-K (e.g., top-10)

**Mean Reciprocal Rank (MRR)**:
```python
MRR = (1 / N) × Σ (1 / rank_i)
```

**Target**: MRR > 0.5

#### Precision@K
```python
def precision_at_k(composed_vector, ground_truth_functions, all_functions, k=10):
    """
    Of the top-K most similar to composed vector, how many are ground truth?
    """
    similarities = [
        (f, cosine_similarity(composed_vector, embedding(f)))
        for f in all_functions
    ]
    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    top_k_functions = [f for f, _ in top_k]

    hits = len(set(top_k_functions) & set(ground_truth_functions))
    return hits / k
```

**Target**: Precision@5 > 0.6

### 4.3 Algebraic Structure Metrics

#### Closure Property
```
If d = a(b()) and e = c(b()), does:
d - a + c ≈ e?
```

```python
def test_closure(d, a, b, c, e):
    """
    Test algebraic closure: substitution should work
    """
    # d uses a and b, e uses c and b
    # Can we go from d to e by swapping a for c?
    transformed = embedding(d) - embedding(a) + embedding(c)
    similarity = cosine_similarity(transformed, embedding(e))
    return similarity
```

**Target**: Similarity > random baseline

#### Commutative Property (for parallel operations)
```
If d() = a() + b() (parallel, no dependency), then:
embedding(a) + embedding(b) ≈ embedding(b) + embedding(a)
```

This should hold by vector addition. But does it hold semantically?

```python
def test_commutativity(a, b):
    """
    For parallel functions, order shouldn't matter
    """
    ab = embedding(a) + embedding(b)
    ba = embedding(b) + embedding(a)
    return np.allclose(ab, ba)  # Should be True for addition
```

#### Associative Property (for chaining)
```
If d() = a(b(c())), then:
(a ⊙ b) ⊙ c ≈ a ⊙ (b ⊙ c)?
```

Test if composition operation is associative.

### 4.4 Business Domain Alignment Metrics

#### Concept Similarity Score
```python
def business_concept_alignment(function_embedding, concept_description):
    """
    How well does function align with business concept?
    """
    concept_embedding = embed_text(concept_description)  # Using same model
    return cosine_similarity(function_embedding, concept_embedding)
```

#### Domain Transfer Accuracy
```python
def domain_transfer_test(code_functions, business_descriptions, test_pairs):
    """
    Given business description, can we find the right function?
    """
    correct = 0
    for business_desc, true_function in test_pairs:
        desc_embedding = embed_text(business_desc)

        # Find most similar function
        similarities = [
            (f, cosine_similarity(desc_embedding, embedding(f)))
            for f in code_functions
        ]
        predicted = max(similarities, key=lambda x: x[1])[0]

        if predicted == true_function:
            correct += 1

    return correct / len(test_pairs)
```

**Target**: Accuracy > 0.5 (better than random)

---

## 5. Experimental Design

### 5.1 Dataset Preparation

#### Phase 1: Synthetic Dataset (Controlled)

Create synthetic functions with known composition patterns:

```python
# Example synthetic dataset
def a(x):
    return x + 10

def b(x):
    return x * 2

def c(x):
    return x - 5

# Test compositions
def d1(x):
    return a(x)  # Simple call: d1 = a

def d2(x):
    return a(b(x))  # Nested: d2 = a ∘ b

def d3(x):
    return a(x) + b(x)  # Parallel: d3 = a + b

def d4(x):
    return a(b(x) + c(x))  # Complex: d4 = a ∘ (b + c)
```

**Advantages**:
- Ground truth known
- Control over composition patterns
- Can test specific hypotheses in isolation

**Dataset Size**:
- 100 base functions
- 500 composed functions (various patterns)
- 50 levels of composition depth

#### Phase 2: Real-World Dataset (Apache Flink)

Use existing codebase (Flink) with:
- Call graph extraction
- AST parsing for composition patterns
- Historical task data for business domain alignment

**Data Extraction**:
```python
class CompositionDataExtractor:
    def extract_compositions(self, codebase_path):
        """
        Extract function composition patterns from real code
        """
        compositions = []

        for file in glob_python_files(codebase_path):
            ast_tree = parse_ast(file)

            for function in extract_functions(ast_tree):
                calls = extract_function_calls(function)

                compositions.append({
                    'function': function.name,
                    'calls': calls,
                    'pattern': classify_pattern(calls),  # sequential, parallel, nested
                    'depth': composition_depth(calls)
                })

        return compositions
```

**Expected Patterns in Flink**:
- ~12,000 Java files
- ~50,000 methods
- Various composition depths (1-10 levels)

#### Phase 3: Business Domain Dataset

Map task descriptions to code:

```python
# From DUAL_MCP_SERVER_ARCHITECTURE
task_to_code_mapping = {
    'task_id': 'FLINK-1234',
    'description': 'Implement time window aggregation for streaming data',
    'business_concepts': ['windowing', 'aggregation', 'streaming'],
    'code_entities': [
        'WindowOperator.java',
        'AggregateFunction.java',
        'TimeWindow.java'
    ]
}
```

### 5.2 Embedding Models to Test

| Model | Type | Strengths | Weaknesses |
|-------|------|-----------|------------|
| **CodeBERT** | Transformer (token-based) | Pre-trained, general | May miss structural info |
| **GraphCodeBERT** | Graph + Transformer | Captures data flow | Complex, slower |
| **CodeT5** | Encoder-decoder | Good for generation | Large model size |
| **Code2Vec** | AST path-based | Explicit structure | Limited context |
| **BGE-small** (current) | General embedding | Fast, lightweight | Not code-specific |

**Experiment**: Test compositional properties across all models.

**Hypothesis**: AST-based models (Code2Vec, GraphCodeBERT) will show stronger compositional properties than token-based models.

### 5.3 Experimental Protocols

#### Experiment 1: Simple Composition Recovery

**Setup**:
1. Embed all base functions (a, b, c, ...)
2. For each composed function d, compute composed embedding using different operations
3. Measure distance to actual embedding(d)

**Operations to Test**:
- Addition: `v_a + v_b`
- Weighted addition: `α·v_a + β·v_b`
- Hadamard product: `v_a ⊙ v_b`
- Concatenation + projection: `W[v_a; v_b]`
- Learned composition: `MLP(v_a, v_b)`

**Evaluation**:
```python
for operation in ['add', 'multiply', 'weighted', 'concat', 'learned']:
    for composed_function in dataset:
        components = composed_function.get_components()
        composed_vec = apply_operation(components, operation)

        mse = np.mean((composed_vec - embedding(composed_function)) ** 2)
        cos_sim = cosine_similarity(composed_vec, embedding(composed_function))

        results[operation].append({'mse': mse, 'cos_sim': cos_sim})
```

**Success Criterion**: At least one operation shows MSE < baseline (random composition) with p < 0.05.

#### Experiment 2: Composition Prediction Task

**Task**: Given embeddings of components, predict which function is the composition.

**Setup**:
1. For a composed function d = compose(a, b, c)
2. Compute composed vector v_composed = f(v_a, v_b, v_c)
3. Rank all functions by similarity to v_composed
4. Check if d is in top-K

**Baselines**:
- Random ranking
- Cosine similarity to average(v_a, v_b, v_c)
- TF-IDF based code search

**Metrics**:
- MRR (Mean Reciprocal Rank)
- Precision@K (K = 1, 5, 10)
- Recall@K

#### Experiment 3: Algebraic Structure Tests

**Test 1: Substitution**
```
If d = a(b(x)) and e = c(b(x)), then:
embedding(d) - embedding(a) + embedding(c) ≈ embedding(e)
```

**Test 2: Commutativity** (for parallel operations)
```
If d = merge(a, b) and e = merge(b, a), then:
embedding(d) ≈ embedding(e)
```

**Test 3: Transitivity** (for chains)
```
If d = a(b(x)) and e = b(c(x)) and f = a(b(c(x))), then:
Does the composition algebra hold?
```

**Evaluation**: Count how many algebraic properties hold (similarity > threshold).

#### Experiment 4: Business Domain Alignment

**Task**: Map business concepts to code functions using compositional embeddings.

**Setup**:
1. Embed task descriptions using same model
2. Embed code functions
3. For composed functions, test if composition aligns with business composition

**Example**:
```
Business Task: "Aggregate windowed streaming data"
→ Concepts: [windowing, aggregation, streaming]

Expected Code Composition:
window_function + aggregate_function + stream_function

Test: Does the composed embedding align with task embedding?
```

**Metric**:
```python
alignment_score = cosine_similarity(
    embed_text("Aggregate windowed streaming data"),
    embedding(window_fn) + embedding(agg_fn) + embedding(stream_fn)
)
```

**Success**: Alignment score > similarity to random function combinations.

#### Experiment 5: Cross-Project Transfer

**Task**: Do compositional patterns learned on Project A work on Project B?

**Setup**:
1. Learn composition weights on Apache Flink
2. Test on Apache Spark (similar domain) and Django (different domain)
3. Measure transfer accuracy

**Hypothesis**: Compositional patterns transfer better within the same domain.

### 5.4 Ablation Studies

#### Ablation 1: Remove Composition, Use Only Cosine Similarity
Measure performance drop when using only similarity vs. composition.

#### Ablation 2: Different Embedding Dimensions
Test if compositionality emerges more in higher dimensions (128 vs 384 vs 768).

#### Ablation 3: Code Style Normalization
- Test with variable names preserved
- Test with variable names anonymized
- Test with comments removed

Does normalization improve compositional properties?

---

## 6. Implementation Roadmap

### 6.1 Phase 1: Data Collection (Weeks 1-2)

**Tasks**:
- [ ] Extract call graphs from Flink codebase
- [ ] Parse AST to identify composition patterns
- [ ] Create synthetic dataset with known compositions
- [ ] Generate ground truth labels

**Deliverables**:
- `compositions.json`: List of (function, components, pattern) tuples
- `synthetic_dataset.py`: Code generator for synthetic examples

### 6.2 Phase 2: Embedding Generation (Week 3)

**Tasks**:
- [ ] Set up CodeBERT, GraphCodeBERT, Code2Vec
- [ ] Embed all functions in dataset
- [ ] Store embeddings in vector database (PostgreSQL + pgvector)

**Deliverables**:
- Embedding database with metadata
- Embedding visualization (t-SNE, UMAP)

### 6.3 Phase 3: Composition Experiments (Weeks 4-5)

**Tasks**:
- [ ] Implement vector operations (add, multiply, weighted, concat)
- [ ] Run Experiment 1 (composition recovery)
- [ ] Run Experiment 2 (prediction task)
- [ ] Compute metrics (MSE, cosine similarity, MRR, P@K)

**Deliverables**:
- Results tables comparing operations
- Visualizations of composed vs. actual embeddings

### 6.4 Phase 4: Algebraic Properties (Week 6)

**Tasks**:
- [ ] Test algebraic properties (substitution, commutativity, associativity)
- [ ] Measure how often properties hold
- [ ] Identify patterns where they break down

**Deliverables**:
- Algebraic property report
- Edge cases and failure modes documentation

### 6.5 Phase 5: Business Domain Integration (Week 7)

**Tasks**:
- [ ] Embed task descriptions from historical data
- [ ] Test business-code alignment
- [ ] Measure domain transfer accuracy

**Deliverables**:
- Business-code alignment scores
- Examples of successful/failed mappings

### 6.6 Phase 6: Integration with RAG System (Week 8)

**Tasks**:
- [ ] Integrate compositional search into MCP server
- [ ] Implement composition-aware ranking
- [ ] Update dual MCP architecture to use compositional features

**Deliverables**:
- Enhanced MCP server with compositional search
- Comparative evaluation vs. cosine-similarity-only baseline

---

## 7. Expected Results and Discussion

### 7.1 Expected Findings

#### Strong Compositional Properties Expected For:
- **Simple function calls**: `d() = a()` → `v_d ≈ v_a`
- **Parallel operations**: `d() = a() + b()` → `v_d ≈ v_a + v_b`
- **Weighted combinations**: Dominant function should have higher weight

#### Weak Compositional Properties Expected For:
- **Deep nesting**: `d() = a(b(c(d(e()))))` → Composition may degrade
- **Complex control flow**: Conditionals, loops may break simple algebra
- **Context-dependent behavior**: Side effects, state changes

#### Model Comparison:
- **Code2Vec, GraphCodeBERT**: Likely to show stronger compositional properties (structure-aware)
- **CodeBERT, CodeT5**: May show compositional properties but less pronounced
- **BGE-small (general)**: Baseline; may lack code-specific compositional structure

### 7.2 Potential Applications

#### Application 1: Enhanced Code Search

Instead of:
```
"Find functions similar to X"
```

Enable:
```
"Find functions that compose A and B in a specific way"
composed_query = embedding(A) ⊙ embedding(B)
results = search_similar(composed_query)
```

#### Application 2: Refactoring Recommendations

Detect refactoring opportunities:
```python
# If many functions have similar composition pattern:
pattern = embedding(a) + embedding(b)

# Find all functions close to this pattern
candidates = find_similar(pattern, threshold=0.8)

# Suggest: Extract common composition into helper function
```

#### Application 3: Business-Code Traceability

Map business requirements to code:
```python
business_req = "Calculate tax on discounted price"
# Compose relevant concepts
composed = embedding("tax") + embedding("discount") + embedding("price")

# Find implementing functions
implementing_functions = search_code(composed)
```

#### Application 4: Semantic Code Completion

Predict next function call:
```python
# User has called a() and b()
# What should c be?
current_context = embedding(a) + embedding(b)

# Find functions that often appear in this composition
suggestions = complete_composition(current_context)
```

### 7.3 Limitations and Challenges

#### Challenge 1: Embedding Space Noise
Real embeddings contain noise; perfect algebraic properties unlikely.

**Mitigation**: Use thresholds, statistical significance tests.

#### Challenge 2: Polysemy (Function Overloading)
Same function name, different behavior in different contexts.

**Mitigation**: Context-aware embeddings, separate embeddings per usage context.

#### Challenge 3: Implicit Dependencies
Not all dependencies are explicit function calls (e.g., shared state, global variables).

**Mitigation**: Use data flow analysis, augment with runtime traces.

#### Challenge 4: Scalability
Large codebases (50K+ functions) → expensive pairwise comparisons.

**Mitigation**: Index embeddings in vector databases, use approximate nearest neighbor search.

#### Challenge 5: Evaluation Ground Truth
Hard to define "correct" composition in real code.

**Mitigation**: Use multiple evaluation metrics, human validation on samples.

---

## 8. Related Work and Scientific Foundation

### 8.1 Compositional Semantics in NLP

**Word2Vec and Semantic Arithmetic**:
- [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781): "king - man + woman = queen"
- Demonstrates that semantic relationships can be encoded as vector operations
- [Reproducing and learning new algebraic operations on word embeddings](https://ar5iv.labs.arxiv.org/html/1702.05624)

**Compositional Distributional Semantics**:
- [MIT Press: Compositionality and Sentence Meaning](https://direct.mit.edu/coli/article/51/1/139/124463/Compositionality-and-Sentence-Meaning-Comparing)
- How sentence embeddings compose from word embeddings
- [The Role of Syntax in Vector Space Models](https://www.researchgate.net/publication/270878336_The_Role_of_Syntax_in_Vector_Space_Models_of_Compositional_Semantics)

**Grounded Compositional Semantics**:
- [Grounded learning for compositional vector semantics](https://arxiv.org/html/2401.06808v1)
- Building vector representations and combining them compositionally

### 8.2 Code Embeddings

**Code2Vec**:
- [Alon et al., 2019 - ACM POPL](https://dl.acm.org/doi/10.1145/3290353)
- [arXiv version](https://arxiv.org/pdf/1803.09473)
- AST path-based embeddings with attention-weighted composition
- Shows that code embeddings can capture semantic analogies

**Code Embedding Comprehensive Guide**:
- [Unite.AI Guide to Code Embedding](https://www.unite.ai/code-embedding-a-comprehensive-guide/)
- Code embeddings capture semantic and functional relationships
- Unlike traditional methods, embeddings capture semantic relationships between code parts

**Improvements to Code2Vec**:
- [ScienceDirect: Generating path vectors using RNN](https://www.sciencedirect.com/science/article/abs/pii/S0167404823002328)
- Enhancements to compositional path aggregation

### 8.3 Knowledge Graphs and Relation Embeddings

**Knowledge Graph Embeddings**:
- [Ontotext: What Are Knowledge Graph Embeddings?](https://www.ontotext.com/knowledgehub/fundamentals/what-are-knowledge-graph-embeddings/)
- Vector representations capture semantic meaning and structure
- Enable prediction of missing relationships

**TransE and Translation-Based Models**:
- [Knowledge Graph Representation via Similarity-Based Embedding](https://www.hindawi.com/journals/sp/2018/6325635/)
- If relation (h, r, t) exists, then: `h + r ≈ t`
- Applied to knowledge graphs, could apply to code call graphs

**Comprehensive Survey**:
- [ACM Survey: Knowledge Graph Embedding from Representation Spaces](https://dl.acm.org/doi/10.1145/3643806)
- Taxonomies of embedding approaches and composition methods

### 8.4 Neural Program Synthesis

**Learning Compositional Rules**:
- [Neural Program Synthesis - NeurIPS](https://www.cs.princeton.edu/~bl8144/papers/NyeEtAl2020NeurIPS.pdf)
- [arXiv version](https://ar5iv.labs.arxiv.org/html/2003.05562)
- Programs have compositional nature; can be reasoned about through parts

**Graph Neural Networks for Programs**:
- [Guiding Genetic Programming with GNNs](https://arxiv.org/html/2411.05820v1)
- GNNs access syntactic and semantic information in program graphs
- Can model compositional structure

**ExeDec - Compositional Generalization**:
- [OpenReview: Execution Decomposition](https://openreview.net/forum?id=oTRwljRgiv)
- Measures whether models trained on simpler subtasks can solve complex tasks
- Compositional generalization in program synthesis

### 8.5 Vector Arithmetic in LLMs

**In-Context Vector Arithmetic**:
- [ICML 2025: Provable In-Context Vector Arithmetic](https://icml.cc/virtual/2025/poster/45998)
- LLMs form internal "task vectors" and solve tasks using vector addition
- Transformers retrieve task vectors through attention mechanisms

**Embedding Math**:
- [BERT Embedding Math](https://go281.user.srcf.net/blog/research/embedding-math/)
- Exploration of mathematical operations in embedding spaces

---

## 9. Metrics Summary Table

| Metric Category | Specific Metric | Formula | Target | Interpretation |
|-----------------|-----------------|---------|--------|----------------|
| **Reconstruction** | MSE | `mean((v_actual - v_composed)²)` | < baseline | How close is composed to actual? |
| | Cosine Similarity | `(a·b)/(‖a‖‖b‖)` | > 0.7 | Directional similarity |
| **Ranking** | MRR | `(1/N) Σ(1/rank_i)` | > 0.5 | Average rank quality |
| | Precision@5 | `relevant_in_top5 / 5` | > 0.6 | Top-5 accuracy |
| | Recall@10 | `found_in_top10 / total_relevant` | > 0.6 | Coverage in top-10 |
| **Algebraic** | Substitution Test | `cos_sim(v_d - v_a + v_c, v_e)` | > 0.6 | Algebraic closure |
| | Commutativity | `‖v_a + v_b - v_b - v_a‖` | ≈ 0 | Order independence |
| **Domain** | Business Alignment | `cos_sim(task_embed, code_embed)` | > 0.5 | Task-code mapping |
| | Transfer Accuracy | `correct_matches / total_test` | > 0.5 | Cross-project transfer |

---

## 10. Conclusion and Future Directions

### 10.1 Summary

This document proposes a research agenda to investigate **compositional properties of code embeddings through vector arithmetic**. The core hypothesis is that function composition in programs can be represented as algebraic operations on embedding vectors, extending beyond simple cosine similarity to capture **transformational and compositional relationships**.

### 10.2 Key Contributions

1. **Theoretical Framework**: Taxonomy of composition types and corresponding vector operations
2. **Evaluation Metrics**: Comprehensive metrics for measuring compositional properties
3. **Experimental Protocols**: Rigorous experiments on synthetic and real-world data
4. **Practical Applications**: Enhanced code search, refactoring, and business-code traceability

### 10.3 Future Research Directions

#### Direction 1: Learned Composition Functions
Instead of fixed operations (add, multiply), learn optimal composition functions:
```python
composition_network = NeuralCompositionModel()
v_composed = composition_network(v_a, v_b, relation_type)
```

#### Direction 2: Temporal Dynamics
Extend to temporal code evolution:
```
If function f at time t₁ evolves to f' at time t₂, can we predict:
embedding(f') ≈ embedding(f) + evolution_vector
```

#### Direction 3: Multi-Modal Composition
Combine code embeddings with:
- Documentation embeddings
- Execution trace embeddings
- User interaction embeddings

#### Direction 4: Causal Composition
Move from correlation (cosine similarity) to causation:
```
"Function a causes effect in function b"
vs.
"Functions a and b are similar"
```

Use causal inference methods (do-calculus) on embedding spaces.

#### Direction 5: Hierarchical Composition
Model compositions at multiple levels:
- Statement level
- Function level
- Class level
- Module level

Investigate how composition properties propagate across levels.

---

## 11. References and Further Reading

### Core Papers

1. **Word2Vec and Semantic Arithmetic**
   - [Mikolov et al., 2013 - Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)
   - [Reproducing and learning new algebraic operations on word embeddings](https://ar5iv.labs.arxiv.org/html/1702.05624)

2. **Code2Vec**
   - [Alon et al., 2019 - code2vec: Learning Distributed Representations of Code (ACM POPL)](https://dl.acm.org/doi/10.1145/3290353)
   - [arXiv version](https://arxiv.org/pdf/1803.09473)
   - [GitHub Implementation](https://github.com/tech-srl/code2vec)

3. **Compositional Semantics**
   - [Compositionality and Sentence Meaning - MIT Press](https://direct.mit.edu/coli/article/51/1/139/124463/Compositionality-and-Sentence-Meaning-Comparing)
   - [Grounded learning for compositional vector semantics](https://arxiv.org/html/2401.06808v1)

4. **Knowledge Graph Embeddings**
   - [Knowledge Graph Embeddings Overview - Ontotext](https://www.ontotext.com/knowledgehub/fundamentals/what-are-knowledge-graph-embeddings/)
   - [ACM Survey on Knowledge Graph Embeddings](https://dl.acm.org/doi/10.1145/3643806)

5. **Neural Program Synthesis**
   - [Learning Compositional Rules via Neural Program Synthesis - NeurIPS](https://www.cs.princeton.edu/~bl8144/papers/NyeEtAl2020NeurIPS.pdf)
   - [arXiv version](https://ar5iv.labs.arxiv.org/html/2003.05562)

6. **Vector Arithmetic in Modern LLMs**
   - [ICML 2025 - Provable In-Context Vector Arithmetic](https://icml.cc/virtual/2025/poster/45998)

### Practical Resources

- [Code Embedding: A Comprehensive Guide - Unite.AI](https://www.unite.ai/code-embedding-a-comprehensive-guide/)
- [word2vec-style vector arithmetic on docs embeddings](https://technicalwriting.dev/embeddings/arithmetic/index.html)
- [Semantic Embedding Overview - EmergentMind](https://www.emergentmind.com/topics/semantic-embedding)

### Related Concepts Documents

- `SIMARGL_concept.md` - Structural metrics for code recommendations
- `DUAL_MCP_SERVER_ARCHITECTURE.md` - File-level vs task-level embeddings
- `TWO_PHASE_REFLECTIVE_AGENT.md` - Agent reasoning with compositional understanding

---

## Appendix A: Code Examples

### A.1 Embedding Composition Implementation

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Callable

class CompositionExperiment:
    """Test compositional properties of code embeddings"""

    def __init__(self, model_name='BAAI/bge-small-en-v1.5'):
        self.model = SentenceTransformer(model_name)
        self.embeddings_cache = {}

    def embed_function(self, function_code: str) -> np.ndarray:
        """Generate embedding for a function"""
        if function_code not in self.embeddings_cache:
            self.embeddings_cache[function_code] = self.model.encode(function_code)
        return self.embeddings_cache[function_code]

    def compose_add(self, *functions: str) -> np.ndarray:
        """Additive composition"""
        embeddings = [self.embed_function(f) for f in functions]
        return np.sum(embeddings, axis=0)

    def compose_multiply(self, func_a: str, func_b: str) -> np.ndarray:
        """Hadamard product composition"""
        emb_a = self.embed_function(func_a)
        emb_b = self.embed_function(func_b)
        return emb_a * emb_b

    def compose_weighted(self, functions: List[str], weights: List[float]) -> np.ndarray:
        """Weighted composition"""
        embeddings = [self.embed_function(f) for f in functions]
        weights = np.array(weights) / np.sum(weights)  # Normalize
        return np.sum([w * e for w, e in zip(weights, embeddings)], axis=0)

    def test_composition(
        self,
        composed_function: str,
        components: List[str],
        operation: str = 'add'
    ) -> Dict[str, float]:
        """
        Test if composed_function ≈ compose(components)

        Returns:
            Dictionary with MSE and cosine similarity
        """
        actual_embedding = self.embed_function(composed_function)

        if operation == 'add':
            predicted_embedding = self.compose_add(*components)
        elif operation == 'multiply':
            predicted_embedding = self.compose_multiply(components[0], components[1])
        elif operation == 'weighted':
            # For this example, weight by call frequency (would need real data)
            weights = [1.0] * len(components)
            predicted_embedding = self.compose_weighted(components, weights)

        # Compute metrics
        mse = np.mean((actual_embedding - predicted_embedding) ** 2)
        cos_sim = np.dot(actual_embedding, predicted_embedding) / (
            np.linalg.norm(actual_embedding) * np.linalg.norm(predicted_embedding)
        )

        return {
            'mse': mse,
            'cosine_similarity': cos_sim,
            'operation': operation
        }

# Example usage
experiment = CompositionExperiment()

# Define functions
func_a = """
def add(x, y):
    return x + y
"""

func_b = """
def multiply(x, y):
    return x * y
"""

func_composed = """
def calculate(x, y, z):
    return multiply(add(x, y), z)
"""

# Test composition
result = experiment.test_composition(
    composed_function=func_composed,
    components=[func_a, func_b],
    operation='add'
)

print(f"MSE: {result['mse']:.4f}")
print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")
```

### A.2 Ranking Experiment

```python
def composition_ranking_experiment(
    experiment: CompositionExperiment,
    all_functions: List[str],
    composed_func: str,
    components: List[str],
    operation: str = 'add'
) -> int:
    """
    Rank all functions by similarity to composed vector.
    Return rank of the actual composed function.
    """
    # Create composed vector
    if operation == 'add':
        composed_vector = experiment.compose_add(*components)
    # ... other operations

    # Compute similarity to all functions
    similarities = []
    for func in all_functions:
        func_embedding = experiment.embed_function(func)
        similarity = np.dot(composed_vector, func_embedding) / (
            np.linalg.norm(composed_vector) * np.linalg.norm(func_embedding)
        )
        similarities.append((func, similarity))

    # Rank by similarity
    ranked = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Find rank of composed_func
    for rank, (func, _) in enumerate(ranked, 1):
        if func == composed_func:
            return rank

    return len(all_functions) + 1  # Not found
```

---

**Document Version**: 1.0
**Date**: 2026-01-03
**Author**: Research Team
**Status**: Ready for Experimentation

