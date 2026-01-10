# Keyword Indexing: Semantic Coordinates for Code Understanding

## The Breakthrough Insight

### Keywords as Semantic Coordinates

When we recognize **RULE** as a keyword, we simultaneously recognize:
- **Positive Space**: X is RULE
- **Negative Space**: X is NOT(SERVER, PLUGIN, USER_DETAILS, ...)

In human cognition, statistics, and everyday reasoning, we **bound entities through inversion**. Understanding what something IS requires understanding what it IS NOT.

**Keywords are coordinates in the semantic space** - they define position in a multi-dimensional "bag of words" that maps the entire system.

### The Core Problem (Revisited)

Current RAG approach:
```
Task → Vector Search → Files (no structure)
```

What we need:
```
Task → Keyword Extraction → Keyword Coordinates → Bounded Semantic Region → Files
```

## The Proposed Solution: Keyword Vector Indexing

### Step 1: Train Word2Vec on Code

**Key Insight**: Code contains identifiers, not natural language. If we train Word2Vec on the codebase:
- Every identifier (`Rule`, `Index`, `Quality`, `Profile`) gets its own vector
- Programming keywords (`class`, `public`, `extends`) get vectors
- These vectors capture **co-occurrence patterns** in the code

**Example**:

```python
# Corpus: All code files in SonarQube
corpus = [
    "public class RuleIndex extends BaseIndex",
    "public class RuleUpdater implements RuleService",
    "public class QualityProfileRules",
    "private RuleRepository ruleRepository",
    ...
]

# Train Word2Vec
model = Word2Vec(corpus, vector_size=300, window=5, min_count=3)

# Results:
model.most_similar("Rule")
# [('Index', 0.87), ('Quality', 0.82), ('Profile', 0.79), ('Repository', 0.75)]

model.most_similar("Server")
# [('Client', 0.84), ('Request', 0.81), ('Handler', 0.78), ('Service', 0.72)]
```

**What This Gives Us**:
- `Rule` and `Index` are close in semantic space (often co-occur)
- `Server` and `Client` are close (infrastructure concepts)
- `Rule` and `Server` are distant (different domains)

### Step 2: Extract Keywords from Task Descriptions

**Using LLM or NLP**:

```python
task = "Cannot search for words longer than 15 characters in rule descriptions"

# LLM extraction
keywords = llm.extract_keywords(task)
# Result: ["search", "rule", "description", "character"]

# Rank by importance
ranked_keywords = rank_by_tf_idf(keywords, task_corpus)
# Result: [("rule", 0.92), ("search", 0.87), ("description", 0.65), ("character", 0.42)]
```

**For Each Keyword**: Get its vector from the code-trained Word2Vec model

```python
# Get vectors for extracted keywords
keyword_vectors = {
    "rule": model.wv["Rule"],      # Vector in 300-dim space
    "search": model.wv["Search"],
    "description": model.wv["Description"]
}
```

### Step 3: Keyword as Coordinate System

**Semantic Space Visualization**:

```
Dimension 1: RULE ←→ USER
Dimension 2: INDEX ←→ UI
Dimension 3: SERVER ←→ CLIENT
...
Dimension 300: ...
```

Each keyword is a **coordinate** (point) in this 300-dimensional space.

**Task Vector = Weighted Sum of Keyword Vectors**:

```python
task_vector = (
    0.92 * keyword_vectors["rule"] +
    0.87 * keyword_vectors["search"] +
    0.65 * keyword_vectors["description"]
)
```

This creates a **task coordinate** in the same semantic space as all code identifiers.

### Step 4: Negative Space Understanding (Contrastive Learning)

**The Critical Insight**: When we identify RULE, we eliminate non-RULE regions.

```python
# Positive: Task is about RULE
positive_keywords = ["Rule", "RuleIndex", "RuleUpdater"]

# Negative: Task is NOT about these (high distance in vector space)
negative_keywords = find_distant_keywords(
    positive=["Rule"],
    all_keywords=code_vocabulary,
    threshold=0.3  # Cosine similarity < 0.3
)
# Result: ["Server", "Plugin", "UserDetails", "DatabaseMigration", ...]
```

**This Creates a Bounded Region**:
```
Semantic Space:
  [RULE region]
      ↓
  Files: RuleIndex.java, RuleUpdater.java, QualityProfileRules.java
      ↓
  NOT: ServerPlugin.java, UserDetailsService.java
```

**Benefit**: Even if vector search accidentally returns `ServerPlugin.java` (high similarity by chance), we can **filter it out** because "Server" is in the negative space of "Rule".

### Step 5: Create Keyword Vector Index

**New PostgreSQL Table**:

```sql
CREATE TABLE vectors.keyword_index (
    id SERIAL PRIMARY KEY,
    keyword VARCHAR(255) NOT NULL,
    vector vector(300),  -- pgvector extension
    frequency INT,       -- How often keyword appears in code
    importance FLOAT,    -- TF-IDF or other metric
    related_files TEXT[], -- Files containing this keyword
    related_keywords TEXT[], -- Co-occurring keywords
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_keyword_vector ON vectors.keyword_index
USING ivfflat (vector vector_cosine_ops);

CREATE INDEX idx_keyword_name ON vectors.keyword_index(keyword);
```

**Populate Index**:

```python
# Extract all identifiers from code
identifiers = extract_identifiers_from_code()
# ['Rule', 'Index', 'Quality', 'Profile', 'Server', 'Plugin', ...]

# Get vectors from Word2Vec model
for identifier in identifiers:
    if identifier in model.wv:
        vector = model.wv[identifier]
        frequency = count_occurrences(identifier, codebase)
        related_files = find_files_with_identifier(identifier)

        insert_keyword(
            keyword=identifier,
            vector=vector,
            frequency=frequency,
            related_files=related_files
        )
```

### Step 6: Keyword-Based Context Search

**New Search Flow**:

```python
async def search_keywords(task_description: str, top_k: int = 5):
    """
    Search for most relevant KEYWORDS, not files directly
    """

    # Step 1: Extract keywords from task
    task_keywords = extract_keywords_llm(task_description)
    # ["rule", "search", "description"]

    # Step 2: Get task vector (weighted sum)
    task_vector = compute_task_vector(task_keywords, word2vec_model)

    # Step 3: Vector search in keyword_index
    cursor.execute("""
        SELECT keyword, vector <=> %s AS distance, related_files
        FROM vectors.keyword_index
        ORDER BY vector <=> %s
        LIMIT %s
    """, (task_vector, task_vector, top_k))

    # Result:
    # [
    #   ("Rule", 0.12, ["RuleIndex.java", "RuleUpdater.java"]),
    #   ("Index", 0.18, ["RuleIndex.java", "ElasticSearchIndex.java"]),
    #   ("Search", 0.23, ["SearchService.java", "RuleIndex.java"]),
    # ]

    return results
```

**Then Use Keywords to Guide File Search**:

```python
# Step 4: Get files from matched keywords
primary_keyword = "Rule"  # Closest keyword
primary_files = get_files_for_keyword(primary_keyword)

# Step 5: Boost files that match primary keyword
file_search_results = vector_search_files(task_description, top_k=20)

for file in file_search_results:
    if file in primary_files:
        file['score'] += 0.5  # Strong boost

# Step 6: Filter out negative space
negative_keywords = get_negative_keywords(primary_keyword, threshold=0.3)
# ["Server", "Plugin", "Database"]

for file in file_search_results:
    if any(neg_kw in file['path'] for neg_kw in negative_keywords):
        file['score'] -= 0.3  # Penalize
```

## Academic Foundation

This approach is strongly supported by recent research:

### 1. Word2Vec on Source Code

**Key Finding**: Word embeddings trained on code capture semantic relationships between identifiers.

> "Word2vec on source code can learn semantic meaning of code identifiers. Similar types of tokens cluster together in the representation."

**Research**:
- [Word2Vec on source code: Semantic meaning of code (Medium, 2018)](https://medium.com/@amarbudhiraja/word2vec-on-source-code-semantic-meaning-of-code-and-its-beautiful-implications-cb34cafc4a58)
- [A Literature Study of Embeddings on Source Code (ArXiv, 2019)](https://arxiv.org/abs/1904.03061) - Comprehensive survey of code embeddings
- [IdBench: A benchmark for evaluating embeddings of identifiers (GitHub)](https://github.com/sola-st/IdBench)
- [Python2Vec: Word Embeddings for Source Code (Lab41, 2017)](https://gab41.lab41.org/python2vec-word-embeddings-for-source-code-3d14d030fe8f)

### 2. Word Embeddings for Software Engineering Domain

**Key Finding**: Domain-specific embeddings trained on software artifacts outperform general-purpose models.

> "Pre-trained word embeddings based on word2vec, trained over 15GB of Stack Overflow posts, capture software engineering domain knowledge."

**Research**:
- [Word Embeddings for the Software Engineering Domain (IEEE MSR, 2018)](https://ieeexplore.ieee.org/document/8595174)
- [Word embeddings for the software engineering domain (ACM, 2018)](https://dl.acm.org/doi/10.1145/3196398.3196448)
- [EmbSE: A Word Embeddings Model for SE Domain (ResearchGate, 2019)](https://www.researchgate.net/publication/334748845_EmbSE_A_Word_Embeddings_Model_Oriented_Towards_Software_Engineering_Domain)
- [GitHub: Word Embeddings for Software Engineering](https://github.com/SibaMishra/Word-Embeddings-for-Software-Engineering-Domain)

### 3. Semantic Spaces and Coordinates

**Key Finding**: Words can be viewed as coordinates in high-dimensional semantic space where similarity is distance.

> "Vectors are coordinates of points (individual words) in a high-dimensional semantic space. Semantic similarity is a matter of distance between points."

**Research**:
- [An intuitive introduction to text embeddings (Stack Overflow Blog, 2023)](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)
- [Mobilizing Conceptual Spaces: Word Embeddings in Organization Science (INFORMS, 2023)](https://pubsonline.informs.org/doi/10.1287/orsc.2023.1686)
- [Graph-based exploration of semantic spaces (Applied Network Science, 2019)](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0228-y)
- [Producing high-dimensional semantic spaces (Behavior Research Methods, 2007)](https://link.springer.com/article/10.3758/BF03204766)

### 4. Keyword Extraction with Embeddings

**Key Finding**: BERT/word embeddings enable automatic keyword extraction from text.

> "KeyBERT leverages BERT embeddings to create keywords most similar to a document in the semantic space."

**Research**:
- [KeyBERT: Minimal keyword extraction with BERT (GitHub)](https://github.com/MaartenGr/KeyBERT)
- [Using Word Embeddings to Enhance Keyword Identification (Springer, 2015)](https://link.springer.com/content/pdf/10.1007/978-3-319-19548-3_21.pdf)
- [Deep learning embeddings for keyphrase extraction: A literature review (Springer, 2024)](https://link.springer.com/article/10.1007/s10115-024-02164-w)

### 5. Contrastive Learning and Negative Sampling

**Key Finding**: Understanding positive concepts requires negative samples to bound the semantic region.

> "Contrastive learning benefits from hard negative samples that help distinguish target entities from similar but incorrect ones."

**Research**:
- [Contrastive Learning with Hard Negative Entities for Entity Set Expansion (ACM SIGIR, 2022)](https://dl.acm.org/doi/10.1145/3477495.3531954)
- [Domain generalization by class-aware negative sampling (ScienceDirect, 2022)](https://www.sciencedirect.com/science/article/pii/S2666651022000195)
- [Contrastive Learning with Hard Negative Samples (ArXiv, 2020)](https://arxiv.org/abs/2010.04592)
- [Uncertainty-Aware Contrastive Learning for NER (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124003976)

## Implementation Details

### Training Word2Vec on Code

```python
import os
import re
from gensim.models import Word2Vec
from pathlib import Path

def tokenize_code_file(file_path):
    """Extract tokens from a code file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split on non-alphanumeric, preserve camelCase
    tokens = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', content)

    # Filter short tokens and numbers
    tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]

    return tokens

def build_code_corpus(repo_path):
    """Build corpus from all code files"""
    corpus = []

    for file_path in Path(repo_path).rglob("*.java"):
        tokens = tokenize_code_file(file_path)
        if tokens:
            corpus.append(tokens)

    return corpus

# Train model
corpus = build_code_corpus("path/to/sonarqube")
model = Word2Vec(
    sentences=corpus,
    vector_size=300,
    window=5,           # Context window
    min_count=3,        # Ignore rare words
    workers=4,
    sg=1,              # Skip-gram (better for rare words)
    negative=10,       # Negative sampling
    epochs=10
)

# Save model
model.save("sonarqube_word2vec.model")
```

### Extracting Keywords from Task Descriptions

```python
from keybert import KeyBERT
import spacy

# Option 1: Using KeyBERT (BERT-based)
kw_model = KeyBERT()

def extract_keywords_bert(task_description):
    keywords = kw_model.extract_keywords(
        task_description,
        keyphrase_ngram_range=(1, 2),  # Unigrams and bigrams
        stop_words='english',
        top_n=5
    )
    return [kw for kw, score in keywords]

# Option 2: Using spaCy (NLP-based)
nlp = spacy.load("en_core_web_sm")

def extract_keywords_spacy(task_description):
    doc = nlp(task_description)

    # Extract nouns and proper nouns
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ['NOUN', 'PROPN'] and len(token) > 2
    ]

    return list(set(keywords))

# Option 3: Using LLM (best quality)
def extract_keywords_llm(task_description):
    prompt = f"""
    Extract the 3-5 most important technical keywords from this task description.
    Return ONLY the keywords, one per line, in lowercase.

    Task: {task_description}

    Keywords:
    """

    response = llm.generate(prompt)
    keywords = [kw.strip() for kw in response.split('\n') if kw.strip()]
    return keywords
```

### Mapping Keywords to Code Vectors

```python
def map_keywords_to_vectors(task_keywords, word2vec_model):
    """Map extracted keywords to code space vectors"""

    keyword_vectors = {}

    for keyword in task_keywords:
        # Try exact match
        if keyword in word2vec_model.wv:
            keyword_vectors[keyword] = word2vec_model.wv[keyword]
        else:
            # Try capitalized (class names)
            capitalized = keyword.capitalize()
            if capitalized in word2vec_model.wv:
                keyword_vectors[keyword] = word2vec_model.wv[capitalized]
            else:
                # Try finding similar words
                try:
                    similar = word2vec_model.wv.most_similar(keyword, topn=1)
                    if similar[0][1] > 0.7:  # High similarity
                        keyword_vectors[keyword] = word2vec_model.wv[similar[0][0]]
                except:
                    pass

    return keyword_vectors

# Example
task = "Cannot search for words longer than 15 characters in rule descriptions"
keywords = extract_keywords_llm(task)  # ["rule", "search", "description"]
vectors = map_keywords_to_vectors(keywords, model)

# Result:
# {
#   "rule": array([0.23, -0.15, 0.87, ..., 0.34]),  # 300-dim
#   "search": array([0.12, 0.45, -0.23, ..., 0.67]),
#   "description": array([-0.34, 0.12, 0.56, ..., -0.12])
# }
```

### Computing Task Vector

```python
import numpy as np

def compute_task_vector(keyword_vectors, importance_weights=None):
    """
    Compute task vector as weighted sum of keyword vectors
    """

    if not keyword_vectors:
        return None

    if importance_weights is None:
        # Equal weights
        importance_weights = {kw: 1.0 for kw in keyword_vectors}

    # Normalize weights
    total_weight = sum(importance_weights.values())
    normalized_weights = {
        kw: w/total_weight
        for kw, w in importance_weights.items()
    }

    # Weighted sum
    task_vector = np.zeros(300)  # Assuming 300-dim vectors

    for keyword, vector in keyword_vectors.items():
        weight = normalized_weights.get(keyword, 0)
        task_vector += weight * vector

    # Normalize to unit length
    task_vector = task_vector / np.linalg.norm(task_vector)

    return task_vector

# Example with importance weights from TF-IDF
importance_weights = {
    "rule": 0.92,
    "search": 0.87,
    "description": 0.65
}

task_vector = compute_task_vector(keyword_vectors, importance_weights)
```

### Finding Negative Space (Contrastive Keywords)

```python
def find_negative_keywords(positive_keywords, word2vec_model, threshold=0.3):
    """
    Find keywords in negative space (dissimilar to positive keywords)
    """

    # Get all vocabulary
    all_keywords = list(word2vec_model.wv.index_to_key)

    # Compute average similarity to positive keywords
    negative_keywords = []

    for candidate in all_keywords:
        if candidate in positive_keywords:
            continue

        similarities = []
        for pos_kw in positive_keywords:
            try:
                sim = word2vec_model.wv.similarity(candidate, pos_kw)
                similarities.append(sim)
            except:
                continue

        if similarities:
            avg_similarity = np.mean(similarities)

            # If dissimilar (low similarity), it's in negative space
            if avg_similarity < threshold:
                negative_keywords.append((candidate, avg_similarity))

    # Sort by dissimilarity (lowest similarity first)
    negative_keywords.sort(key=lambda x: x[1])

    return [kw for kw, sim in negative_keywords[:50]]  # Top 50

# Example
positive = ["Rule", "Index"]
negative = find_negative_keywords(positive, model, threshold=0.3)
# Result: ["Server", "Client", "Database", "Migration", "Plugin", ...]
```

### Keyword-Guided File Ranking

```python
async def search_files_with_keywords(
    task_description: str,
    word2vec_model,
    top_k: int = 10
):
    """
    Enhanced file search using keyword coordinates
    """

    # Step 1: Extract keywords from task
    task_keywords = extract_keywords_llm(task_description)

    # Step 2: Map to vectors
    keyword_vectors = map_keywords_to_vectors(task_keywords, word2vec_model)

    # Step 3: Compute task vector
    task_vector = compute_task_vector(keyword_vectors)

    # Step 4: Search keyword index
    keyword_results = await search_keyword_index(task_vector, top_k=5)
    # [("Rule", 0.12, ["RuleIndex.java", ...]), ("Search", 0.18, ...), ...]

    # Step 5: Get primary keyword
    primary_keyword = keyword_results[0][0]  # "Rule"
    primary_files = keyword_results[0][2]    # ["RuleIndex.java", ...]

    # Step 6: Find negative space
    negative_keywords = find_negative_keywords(
        positive_keywords=[primary_keyword],
        word2vec_model=word2vec_model,
        threshold=0.3
    )

    # Step 7: Traditional vector search
    file_results = await vector_search_files(task_description, top_k=20)

    # Step 8: Apply keyword boosting and filtering
    for file in file_results:
        # Boost files matching primary keyword
        if file['path'] in primary_files:
            file['score'] += 0.5

        # Penalize files in negative space
        for neg_kw in negative_keywords:
            if neg_kw.lower() in file['path'].lower():
                file['score'] -= 0.3
                break

    # Step 9: Re-rank and filter
    file_results.sort(key=lambda x: x['score'], reverse=True)

    # Step 10: Return with keyword context
    return {
        'files': file_results[:top_k],
        'primary_keywords': [kw for kw, _, _ in keyword_results],
        'negative_keywords': negative_keywords[:10],
        'keyword_context': f"Task is about {primary_keyword}, NOT about {', '.join(negative_keywords[:3])}"
    }
```

## Integration with Existing RAG Pipeline

### Enhanced Phase 1: Reasoning with Keyword Coordinates

```python
async def phase1_with_keyword_indexing(
    task_description: str,
    word2vec_model,
    ...
):
    """
    Phase 1 enhanced with keyword coordinate system
    """

    # NEW: Keyword extraction and mapping
    task_keywords = extract_keywords_llm(task_description)
    keyword_vectors = map_keywords_to_vectors(task_keywords, word2vec_model)
    task_vector = compute_task_vector(keyword_vectors)

    # NEW: Search keyword index
    keyword_results = await search_keyword_index(task_vector, top_k=5)
    primary_keyword = keyword_results[0][0]

    # NEW: Find negative space
    negative_keywords = find_negative_keywords([primary_keyword], word2vec_model)

    # EXISTING: Vector searches (now keyword-guided)
    module_results = await search_modules_with_boost(
        task_description,
        primary_keyword=primary_keyword,
        negative_keywords=negative_keywords
    )

    file_results = await search_files_with_boost(
        task_description,
        primary_keyword=primary_keyword,
        negative_keywords=negative_keywords
    )

    task_results = await search_tasks(task_description)

    # Return with keyword context
    return Phase1Output(
        task_description=task_description,
        primary_keywords=[kw for kw, _, _ in keyword_results],
        negative_keywords=negative_keywords[:10],
        keyword_context=build_keyword_context(keyword_results, negative_keywords),
        module_scores=module_results,
        file_scores=file_results,
        similar_tasks=task_results,
        selected_files=select_top_files(file_results, top_k=10)
    )
```

### Enhanced LLM Prompt with Keyword Context

```python
prompt = f"""
# Task
{task_description}

# Semantic Coordinate Analysis (Keyword Indexing)

## Primary Domain Keywords (Positive Space)
The task is primarily about these concepts:
{format_primary_keywords(primary_keywords)}

Example:
- **RULE** (similarity: 0.92)
  * Found in 15 files: RuleIndex.java, RuleUpdater.java, RuleCreator.java
  * Co-occurs with: Index, Quality, Profile
  * This is a CORE domain entity representing coding rules

## Excluded Concepts (Negative Space)
The task is NOT about these concepts:
{format_negative_keywords(negative_keywords)}

Example:
- Server, Client, Database, Plugin, Migration (low similarity < 0.3)
- These are infrastructure/different domain concepts

## Bounded Semantic Region
Based on keyword coordinates, focus on files in:
- RULE + SEARCH intersection
- RULE + DESCRIPTION intersection
- Exclude: SERVER, PLUGIN regions

# Relevant Files (Ranked by Keyword-Boosted Similarity)
{format_files_with_keyword_boost(file_results)}

Your task:
1. Consider the primary domain keywords to understand task context
2. Focus on files in the positive semantic space
3. Avoid files in the negative space
4. Select 5-10 most relevant files for detailed analysis
"""
```

## Performance Benefits

### 1. Precision Improvement

**Before (Vector Search Only)**:
```
Task: "Fix rule search performance"

Top 10 Results:
1. RuleIndex.java (0.87)           ✓ Correct
2. ServerSearchService.java (0.85) ✗ Wrong domain (Server)
3. QualitySearch.java (0.83)       ~ Partially relevant
4. RuleUpdater.java (0.81)         ✓ Correct
5. PluginSearch.java (0.80)        ✗ Wrong domain (Plugin)
...

Precision@5: 40% (2/5 correct)
```

**After (Keyword-Guided)**:
```
Task: "Fix rule search performance"

Keywords: ["rule", "search", "performance"]
Primary: RULE (not SERVER, not PLUGIN)
Negative: [Server, Client, Plugin, Database]

Top 10 Results (after boosting/filtering):
1. RuleIndex.java (0.87 + 0.5) = 1.37          ✓ Correct
2. RuleUpdater.java (0.81 + 0.5) = 1.31        ✓ Correct
3. QualitySearch.java (0.83)                   ~ Partially relevant
4. RuleRepository.java (0.75 + 0.5) = 1.25     ✓ Correct
5. ServerSearchService.java (0.85 - 0.3) = 0.55 ✗ Penalized
...

Precision@5: 80% (4/5 correct)
```

**Improvement**: 40% → 80% precision

### 2. Disambiguation

**Ambiguous Task**: "Update index configuration"

Without keywords:
- Could be database index
- Could be search index (Elasticsearch)
- Could be rule index
- No way to distinguish

With keyword coordinates:
```python
keywords = extract_keywords("Update rule index configuration for better search")
# ["rule", "index", "configuration", "search"]

# Compute distances
distance_to_rule_index = 0.15      # Close
distance_to_db_index = 0.67        # Far
distance_to_search_index = 0.22    # Close

# Decision: RULE + SEARCH index, not database index
```

### 3. Cross-Domain Tasks

**Task**: "Migrate rule quality profile data from old server to new server"

Multiple domains: RULE, QUALITY_PROFILE, SERVER, MIGRATION

Keyword extraction identifies:
- Primary: RULE (0.92), QUALITY_PROFILE (0.89)
- Secondary: MIGRATION (0.76), SERVER (0.65)

Result: Focus on files at intersection:
- `RuleQualityProfile.java` (RULE ∩ QUALITY_PROFILE)
- `ProfileMigration.java` (QUALITY_PROFILE ∩ MIGRATION)
- NOT: Generic server files, database migration utilities

## Handling Edge Cases

### Case 1: Keyword Not in Code Vocabulary

**Problem**: Task mentions "authentication" but code uses "auth"

**Solution**: Use Word2Vec similarity to find closest match

```python
def find_closest_keyword(task_keyword, word2vec_model, threshold=0.7):
    """Find closest code keyword for task keyword"""
    try:
        # Try direct lookup
        if task_keyword in word2vec_model.wv:
            return task_keyword

        # Try variations
        variations = [
            task_keyword.capitalize(),
            task_keyword.upper(),
            task_keyword[:4],  # Abbreviation
        ]

        for var in variations:
            if var in word2vec_model.wv:
                return var

        # Find most similar
        similar = word2vec_model.wv.most_similar(task_keyword, topn=3)
        if similar[0][1] > threshold:
            return similar[0][0]

    except:
        pass

    return None

# Example
find_closest_keyword("authentication", model)
# Returns: "Auth" (similarity: 0.85)
```

### Case 2: Multiple Meanings

**Problem**: "Index" could mean:
- Array index
- Search index
- Database index
- Rule index

**Solution**: Use context (co-occurring keywords)

```python
def disambiguate_keyword(keyword, context_keywords, word2vec_model):
    """Disambiguate keyword using context"""

    # Get all code instances of keyword
    candidates = [
        "RuleIndex",
        "DatabaseIndex",
        "SearchIndex",
        "ArrayIndex"
    ]

    # Score each candidate by similarity to context
    scores = {}
    for candidate in candidates:
        if candidate not in word2vec_model.wv:
            continue

        # Average similarity to context keywords
        sims = []
        for ctx in context_keywords:
            if ctx in word2vec_model.wv:
                sim = word2vec_model.wv.similarity(candidate, ctx)
                sims.append(sim)

        if sims:
            scores[candidate] = np.mean(sims)

    # Return best match
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]

    return keyword

# Example
disambiguate_keyword(
    keyword="Index",
    context_keywords=["Rule", "Quality", "Search"],
    word2vec_model=model
)
# Returns: "RuleIndex" (high similarity to Rule, Quality)
```

### Case 3: New Keywords (Not in Training)

**Problem**: New feature added, keyword "GraphQL" not in Word2Vec model

**Solution**: Use fallback to BERT/sentence embeddings

```python
from sentence_transformers import SentenceTransformer

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_keyword_vector_hybrid(keyword, word2vec_model, bert_model):
    """Get vector using Word2Vec or fallback to BERT"""

    # Try Word2Vec first (trained on code)
    if keyword in word2vec_model.wv:
        return word2vec_model.wv[keyword]

    # Fallback to BERT (general knowledge)
    bert_vector = bert_model.encode(keyword)

    # Project BERT vector to Word2Vec space (optional: train projection)
    # For now, return BERT vector with lower confidence
    return bert_vector, 0.5  # 50% confidence
```

## Evaluation Metrics

### 1. Keyword Extraction Quality

```python
# Ground truth: manually labeled keywords for 100 tasks
ground_truth = {
    "SONAR-12345": ["Rule", "Index", "Search"],
    "SONAR-12346": ["Quality", "Profile", "Export"],
    ...
}

# Predicted keywords
predicted = {
    "SONAR-12345": extract_keywords_llm(task_12345),
    ...
}

# Metrics
precision = len(predicted ∩ ground_truth) / len(predicted)
recall = len(predicted ∩ ground_truth) / len(ground_truth)
f1 = 2 * (precision * recall) / (precision + recall)
```

### 2. Keyword Coordinate Accuracy

```python
# For each task, check if primary keyword matches expected domain
def evaluate_keyword_accuracy(test_set):
    correct = 0

    for task_id, expected_domain in test_set.items():
        task = get_task(task_id)
        keywords = extract_keywords(task)
        primary = search_keyword_index(keywords)[0]

        if primary == expected_domain:
            correct += 1

    return correct / len(test_set)
```

### 3. File Ranking Improvement

```python
# Compare ranking with and without keyword boosting
def evaluate_ranking(tasks):
    without_keywords = []
    with_keywords = []

    for task in tasks:
        # Baseline: vector search only
        baseline_files = vector_search_files(task)
        baseline_mrr = compute_mrr(baseline_files, ground_truth[task.id])
        without_keywords.append(baseline_mrr)

        # With keywords: boosted search
        enhanced_files = search_files_with_keywords(task)
        enhanced_mrr = compute_mrr(enhanced_files, ground_truth[task.id])
        with_keywords.append(enhanced_mrr)

    improvement = np.mean(with_keywords) - np.mean(without_keywords)
    return improvement

# Mean Reciprocal Rank
def compute_mrr(ranked_files, ground_truth_files):
    for rank, file in enumerate(ranked_files, start=1):
        if file in ground_truth_files:
            return 1.0 / rank
    return 0
```

## Related Research and Citations

### Code Embeddings

1. **Word2Vec on Source Code**
   - [Word2Vec on source code: Semantic meaning (Medium, 2018)](https://medium.com/@amarbudhiraja/word2vec-on-source-code-semantic-meaning-of-code-and-its-beautiful-implications-cb34cafc4a58)
   - [A Literature Study of Embeddings on Source Code (ArXiv, 2019)](https://arxiv.org/abs/1904.03061)
   - [Python2Vec: Word Embeddings for Source Code (Lab41, 2017)](https://gab41.lab41.org/python2vec-word-embeddings-for-source-code-3d14d030fe8f)

2. **Domain-Specific Software Embeddings**
   - [Word Embeddings for the Software Engineering Domain (IEEE MSR, 2018)](https://ieeexplore.ieee.org/document/8595174)
   - [Word embeddings for SE domain (ACM, 2018)](https://dl.acm.org/doi/10.1145/3196398.3196448)
   - [EmbSE: Word Embeddings Model for SE (ResearchGate, 2019)](https://www.researchgate.net/publication/334748845_EmbSE_A_Word_Embeddings_Model_Oriented_Towards_Software_Engineering_Domain)

3. **Identifier Embeddings Benchmark**
   - [IdBench: Benchmark for identifier embeddings (GitHub)](https://github.com/sola-st/IdBench)
   - [source2vec: Embeddings for various languages (GitHub)](https://github.com/Jur1cek/source2vec)

### Semantic Spaces

4. **Semantic Coordinates**
   - [An intuitive introduction to text embeddings (Stack Overflow, 2023)](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)
   - [Mobilizing Conceptual Spaces: Word Embeddings (INFORMS, 2023)](https://pubsonline.informs.org/doi/10.1287/orsc.2023.1686)
   - [Graph-based exploration of semantic spaces (Springer, 2019)](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0228-y)

5. **High-Dimensional Semantic Spaces**
   - [Producing high-dimensional semantic spaces (Springer, 2007)](https://link.springer.com/article/10.3758/BF03204766)
   - [Mined semantic analysis: Concept space model (ResearchGate, 2018)](https://www.researchgate.net/publication/322512205_Mined_semantic_analysis_A_new_concept_space_model_for_semantic_representation_of_textual_data)

### Keyword Extraction

6. **Embedding-Based Keyword Extraction**
   - [KeyBERT: Keyword extraction with BERT (GitHub)](https://github.com/MaartenGr/KeyBERT)
   - [Using Word Embeddings for Keyword Identification (Springer, 2015)](https://link.springer.com/content/pdf/10.1007/978-3-319-19548-3_21.pdf)
   - [Deep learning for keyphrase extraction: Review (Springer, 2024)](https://link.springer.com/article/10.1007/s10115-024-02164-w)

### Contrastive Learning

7. **Negative Sampling and Contrastive Learning**
   - [Contrastive Learning with Hard Negative Entities (ACM SIGIR, 2022)](https://dl.acm.org/doi/10.1145/3477495.3531954)
   - [Domain generalization by negative sampling (ScienceDirect, 2022)](https://www.sciencedirect.com/science/article/pii/S2666651022000195)
   - [Contrastive Learning with Hard Negative Samples (ArXiv, 2020)](https://arxiv.org/abs/2010.04592)
   - [Uncertainty-Aware Contrastive Learning for NER (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124003976)

## Implementation Roadmap

### Phase 1: Train Word2Vec Model (1-2 days)
1. Extract all code files from repository
2. Tokenize code (preserve identifiers, split camelCase)
3. Train Word2Vec model (vector_size=300, window=5)
4. Evaluate on identifier similarity benchmark
5. Save model for reuse

### Phase 2: Build Keyword Index (2-3 days)
1. Extract all unique identifiers from code
2. Get vectors from Word2Vec model
3. Compute importance scores (frequency, TF-IDF)
4. Create PostgreSQL `keyword_index` table
5. Populate with identifier vectors
6. Create vector similarity index (ivfflat)

### Phase 3: Integrate Keyword Extraction (1-2 days)
1. Implement keyword extraction (LLM + NLP)
2. Test on historical tasks
3. Validate extracted keywords against manual labels
4. Tune extraction parameters

### Phase 4: Enhance RAG Pipeline (2-3 days)
1. Add keyword search to Phase 1
2. Implement keyword boosting for file ranking
3. Implement negative space filtering
4. Update LLM prompts with keyword context
5. Test end-to-end

### Phase 5: Evaluation (1-2 days)
1. Benchmark on 50-100 historical tasks
2. Measure precision, recall, MRR
3. Compare with baseline (no keywords)
4. Iterate based on results

**Total estimated time**: 7-12 days

## Conclusion

The **Keyword Indexing** approach transforms RAG from a "random village names" approach to a proper **coordinate system** with:

1. **Semantic Coordinates**: Keywords are points in high-dimensional space trained on actual code
2. **Positive Space**: Primary keywords identify what the task IS about
3. **Negative Space**: Contrastive keywords identify what the task IS NOT about
4. **Bounded Regions**: Keywords define semantic boundaries for focused search
5. **Integration**: Keywords link tasks, files, and modules through shared vector space

This approach is **strongly supported by academic research** in:
- Code embeddings (Word2Vec on source code)
- Semantic spaces (coordinates in vector space)
- Keyword extraction (BERT-based methods)
- Contrastive learning (negative sampling)

By treating keywords as **semantic coordinates**, we give the LLM a **map of the system** rather than disconnected files, dramatically improving its ability to reason about code even with limited capacity.

---

## Sources

### Code Embeddings
- [Word2Vec on source code: Semantic meaning (Medium, 2018)](https://medium.com/@amarbudhiraja/word2vec-on-source-code-semantic-meaning-of-code-and-its-beautiful-implications-cb34cafc4a58)
- [A Literature Study of Embeddings on Source Code (ArXiv, 2019)](https://arxiv.org/abs/1904.03061)
- [Python2Vec: Word Embeddings for Source Code (Lab41, 2017)](https://gab41.lab41.org/python2vec-word-embeddings-for-source-code-3d14d030fe8f)
- [Word Embeddings for the Software Engineering Domain (IEEE MSR, 2018)](https://ieeexplore.ieee.org/document/8595174)
- [Word embeddings for SE domain (ACM, 2018)](https://dl.acm.org/doi/10.1145/3196398.3196448)
- [EmbSE: Word Embeddings Model for SE (ResearchGate, 2019)](https://www.researchgate.net/publication/334748845_EmbSE_A_Word_Embeddings_Model_Oriented_Towards_Software_Engineering_Domain)
- [IdBench: Benchmark for identifier embeddings (GitHub)](https://github.com/sola-st/IdBench)

### Semantic Spaces
- [An intuitive introduction to text embeddings (Stack Overflow, 2023)](https://stackoverflow.blog/2023/11/09/an-intuitive-introduction-to-text-embeddings/)
- [Mobilizing Conceptual Spaces: Word Embeddings (INFORMS, 2023)](https://pubsonline.informs.org/doi/10.1287/orsc.2023.1686)
- [Graph-based exploration of semantic spaces (Springer, 2019)](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0228-y)
- [Producing high-dimensional semantic spaces (Springer, 2007)](https://link.springer.com/article/10.3758/BF03204766)

### Keyword Extraction
- [KeyBERT: Keyword extraction with BERT (GitHub)](https://github.com/MaartenGr/KeyBERT)
- [Using Word Embeddings for Keyword Identification (Springer, 2015)](https://link.springer.com/content/pdf/10.1007/978-3-319-19548-3_21.pdf)
- [Deep learning for keyphrase extraction: Review (Springer, 2024)](https://link.springer.com/article/10.1007/s10115-024-02164-w)

### Contrastive Learning
- [Contrastive Learning with Hard Negative Entities (ACM SIGIR, 2022)](https://dl.acm.org/doi/10.1145/3477495.3531954)
- [Domain generalization by negative sampling (ScienceDirect, 2022)](https://www.sciencedirect.com/science/article/pii/S2666651022000195)
- [Contrastive Learning with Hard Negative Samples (ArXiv, 2020)](https://arxiv.org/abs/2010.04592)
- [Uncertainty-Aware Contrastive Learning for NER (ScienceDirect, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124003976)
