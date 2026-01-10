# Keyword-Based Entity Mapping for Improved RAG Context

## The Core Problem

### Current Limitation: Disconnected Context

The current RAG implementation suffers from **poor predictive ability** because:

1. **RECENT vs ALL searches return non-intersecting results** - Different collections produce completely different file/module/task sets with no overlap
2. **No structural understanding** - The LLM receives a "tangled" list of potentially relevant files without understanding their relationships
3. **No project map** - The system lacks a high-level understanding of where entities and concepts exist in the codebase
4. **Weak supervision** - A small local LLM cannot resolve complex relationships from unstructured, disconnected input

### The Analogy

> **Current approach**: "Create a path from Paris to London and here are random village names that might be somewhere between them"
>
> **Needed approach**: "Here's a map showing Paris, London, the major cities (keywords/entities), and the roads connecting them"

## The Solution: Keyword-Centric Entity Mapping

### Key Insight from Human Understanding

When reading a task description like:

```
"Cannot search for words longer than 15 characters in rule descriptions"
```

A human immediately recognizes **"RULE"** as the keyword because:

1. **RULE is an entity** - A core domain concept/storage element in the system
2. **RULE appears in multiple places** - Not a one-off occurrence
3. **RULE is embedded in code structure** - Visible in file paths, class names, method names

The system needs to automatically identify and map these **business domain terms** to **code locations**.

## Academic Foundation

This idea is well-supported by extensive research in software engineering:

### 1. Domain Concept Extraction from Source Code

**Key Finding**: Program identifiers contain domain concepts that can be extracted using NLP techniques.

> "Program understanding involves mapping domain concepts to the code elements that implement them. Identifier names contain relevant clues to rediscover the mapping and make it available to programmers."

**Research**:
- [Extraction of domain concepts from the source code (ScienceDirect, 2014)](https://www.sciencedirect.com/science/article/pii/S0167642314004419)
- [Towards the Extraction of Domain Concepts from the Identifiers (IEEE, 2011)](https://ieeexplore.ieee.org/document/6079777/)
- [On the Use of Domain Terms in Source Code (ResearchGate, 2008)](https://www.researchgate.net/publication/4349667_On_the_Use_of_Domain_Terms_in_Source_Code)

### 2. Concept Location and Feature Location

**Key Finding**: Information retrieval techniques can map natural language concepts to source code locations.

> "Concept location identifies parts of a software system that implement a specific concept that originates from the problem or the solution domain."

**Research**:
- [An information retrieval approach to concept location in source code (IEEE, 2003)](https://ieeexplore.ieee.org/document/1374321/)
- [Feature location in source code: A taxonomy and survey (Wiley, 2013)](https://onlinelibrary.wiley.com/doi/full/10.1002/smr.567)
- [Text Retrieval Approaches for Concept Location in Source Code (Springer, 2013)](https://link.springer.com/chapter/10.1007/978-3-642-36054-1_5)
- [Concept location using formal concept analysis and information retrieval (ACM TOSEM, 2012)](https://dl.acm.org/doi/10.1145/2377656.2377660)

### 3. Requirements Traceability

**Key Finding**: Semantic analysis can bridge the gap between business requirements and code entities.

> "Conventional traceability methods, relying on textual similarities between requirements and code, often suffer from low precision due to the semantic gap between high-level natural language requirements and the syntactic nature of code."

**Research**:
- [An empirical study on the importance of source code entities for requirements traceability (Springer, 2015)](https://link.springer.com/article/10.1007/s10664-014-9315-y)
- [Recovering semantic traceability between requirements and design (Springer, 2019)](https://link.springer.com/article/10.1007/s11334-019-00330-w)
- [On the role of semantics in automated requirements tracing (Springer, 2014)](https://link.springer.com/article/10.1007/s00766-013-0199-y)
- [Constructing Traceability Links between Software Requirements and Source Code Based on Neural Networks (MDPI, 2023)](https://www.mdpi.com/2227-7390/11/2/315)

### 4. The Vocabulary Problem

**Key Finding**: The biggest challenge is that developers and requirements use different vocabulary for the same concepts.

> "The Vocabulary Problem in Human-System Communication is recognized as a fundamental challenge in this area."

## Implementation Approach

### Phase 1: Keyword Extraction from Code Structure

**For Legacy Code (Normal Identifiers)**

Extract keywords from file paths and identifiers:

```
Input: server/sonar-server/src/main/java/org/sonar/server/rule/index/RuleIndex.java

Extracted Keywords:
- RULE (appears in path: "rule/index/RuleIndex.java")
- INDEX (appears in path and class name)
- SERVER (appears in path)

Keyword Score:
- RULE: High (appears 2x in path, in class name)
- INDEX: Medium (appears in path segment and class name)
- SERVER: Low (common infrastructure term)
```

**Algorithm**:
1. Tokenize all file paths in the repository
2. Extract identifiers from file names, package names, class names
3. Count occurrences of each token across the codebase
4. Filter common words (src, main, java, org, etc.)
5. Rank tokens by:
   - Frequency (appears in multiple locations)
   - Structural depth (deeper = more specific domain term)
   - Naming patterns (PascalCase, camelCase boundaries)

**Example for SonarQube**:

```python
# Extract from paths
paths = [
    "server/sonar-server/src/main/java/org/sonar/server/rule/index/RuleIndex.java",
    "server/sonar-server/src/main/java/org/sonar/server/rule/RuleUpdater.java",
    "server/sonar-server/src/main/java/org/sonar/server/rule/RuleCreator.java",
    "server/sonar-webserver/src/main/java/org/sonar/server/qualityprofile/QProfileRules.java"
]

# Extract tokens
keywords = extract_keywords(paths)
# Result:
# {
#   'RULE': {'count': 15, 'paths': [...], 'importance': 0.92},
#   'INDEX': {'count': 8, 'paths': [...], 'importance': 0.67},
#   'QUALITY_PROFILE': {'count': 12, 'paths': [...], 'importance': 0.85},
#   ...
# }
```

### Phase 2: Keyword Extraction from Task Descriptions

**From JIRA/Task History**

For each task that modified a file, extract business terms:

```sql
-- Get all tasks that modified RuleIndex.java
SELECT task_name, title, description
FROM vectors.rawdata
WHERE path LIKE '%RuleIndex.java%'
```

**NLP Pipeline**:
1. Extract nouns and noun phrases from task titles/descriptions
2. Remove stopwords
3. Identify recurring terms across tasks that touched the same file
4. Map: `File → Business Terms`

**Example**:

```
File: server/sonar-server/.../rule/RuleUpdater.java

Tasks that modified it:
- SONAR-12345: "Fix rule parameter validation"
- SONAR-12346: "Add custom rule support"
- SONAR-12401: "Update rule severity handling"

Extracted business terms:
- "rule" (appears 3/3 tasks)
- "parameter" (appears 1/3 tasks)
- "validation" (appears 1/3 tasks)
- "custom" (appears 1/3 tasks)
- "severity" (appears 1/3 tasks)

Primary keyword: RULE (100% occurrence)
Secondary keywords: parameter, severity, validation
```

### Phase 3: Entity-to-Identifier Mapping

**Create bidirectional mapping**:

```json
{
  "entities": {
    "RULE": {
      "business_terms": ["rule", "coding rule", "quality rule", "custom rule"],
      "files": [
        "server/sonar-server/.../rule/RuleIndex.java",
        "server/sonar-server/.../rule/RuleUpdater.java",
        "server/sonar-server/.../rule/RuleCreator.java"
      ],
      "modules": [
        "server/sonar-server/src/main/java/org/sonar/server/rule"
      ],
      "importance": 0.92,
      "centrality": "core",
      "related_entities": ["QUALITY_PROFILE", "ISSUE", "SEVERITY"]
    },
    "QUALITY_PROFILE": {
      "business_terms": ["quality profile", "profile", "rule profile"],
      "files": [...],
      "importance": 0.85,
      "related_entities": ["RULE", "LANGUAGE"]
    }
  }
}
```

### Phase 4: Handling Obfuscated Code

**Challenge**: What if identifiers are renamed to `identifier0001.java`, `identifier0002.java`?

**Solution**: Reverse mapping from tasks to identifiers

```python
def extract_business_terms_for_obfuscated_code():
    """
    When code is obfuscated, use task history to recover business semantics
    """

    # Step 1: Get all tasks that modified identifier0001.java
    tasks = get_tasks_for_file("identifier0001.java")

    # Step 2: Extract all nouns/terms from task titles and descriptions
    all_terms = []
    for task in tasks:
        terms = extract_nouns(task.title + " " + task.description)
        all_terms.extend(terms)

    # Step 3: Find most common terms (TF-IDF)
    term_scores = compute_tfidf(all_terms)

    # Step 4: Top term becomes the business name
    business_name = term_scores[0]  # e.g., "Rule"

    # Step 5: Create mapping
    mapping = {
        "identifier0001.java": {
            "inferred_business_term": "Rule",
            "confidence": 0.87,
            "based_on_tasks": [task.id for task in tasks]
        }
    }

    return mapping
```

**Example**:

```
File: identifier0001.java

Tasks:
- TASK-001: "Add rule validation logic"
- TASK-015: "Fix rule parameter bug"
- TASK-027: "Update rule severity"
- TASK-042: "Improve rule indexing performance"

Term extraction:
- "rule" appears 4/4 times (100%)
- "validation" appears 1/4 times
- "parameter" appears 1/4 times
- "severity" appears 1/4 times

Inference: identifier0001.java → "RULE entity" (confidence: 95%)
```

## Enhanced RAG Pipeline with Entity Map

### Current Flow (Poor Supervision)

```
Task: "Cannot search for words longer than 15 characters in rule descriptions"
    ↓
Vector search → 10 modules (RECENT) + 10 modules (ALL) = 20 modules
    ↓
Vector search → 15 files (RECENT) + 15 files (ALL) = 30 files
    ↓
Vector search → 5 tasks (RECENT) + 5 tasks (ALL) = 10 tasks
    ↓
LLM receives: 60 disconnected items
    ↓
LLM struggles to understand relationships
```

### Enhanced Flow (Keyword-Guided Supervision)

```
Task: "Cannot search for words longer than 15 characters in rule descriptions"
    ↓
Keyword extraction: ["RULE", "SEARCH", "DESCRIPTION", "CHARACTER"]
    ↓
Entity map lookup:
    - RULE → 12 core files, 3 modules, related to [QUALITY_PROFILE, INDEX]
    - SEARCH → 8 files, related to [INDEX, ELASTICSEARCH]
    - DESCRIPTION → 5 files, related to [RULE, METADATA]
    ↓
Priority ranking:
    1. Files in RULE entity (high relevance)
    2. Files in intersection of RULE + SEARCH
    3. Files related to RULE.INDEX
    ↓
Vector search (constrained):
    - Boost files matching RULE entity (+0.3 to similarity)
    - Boost files in RULE ∩ SEARCH (+0.5 to similarity)
    ↓
LLM receives:
    - "This task involves RULE entity (core domain concept)"
    - "RULE entity files: RuleIndex.java, RuleUpdater.java, ..."
    - "RULE is related to QUALITY_PROFILE and INDEX entities"
    - "Top relevant files ranked by entity match + vector similarity"
    ↓
LLM has structural context and can reason about relationships
```

## Implementation Steps

### Step 1: Build Entity Map from Existing Codebase

**Script**: `build_entity_map.py`

```python
import os
import re
from collections import defaultdict
import psycopg2

def extract_identifiers_from_paths():
    """Extract all identifiers from file paths in rawdata"""
    conn = psycopg2.connect(...)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT path FROM vectors.rawdata")
    paths = cursor.fetchall()

    # Tokenize paths
    identifier_counts = defaultdict(int)
    identifier_files = defaultdict(list)

    for (path,) in paths:
        tokens = tokenize_path(path)
        for token in tokens:
            if is_valid_identifier(token):
                identifier_counts[token] += 1
                identifier_files[token].append(path)

    # Filter and rank
    entities = {}
    for identifier, count in identifier_counts.items():
        if count >= 3:  # Appears in at least 3 files
            entities[identifier] = {
                'count': count,
                'files': identifier_files[identifier],
                'importance': compute_importance(identifier, count)
            }

    return entities

def tokenize_path(path):
    """Split path into tokens"""
    # Split by /, \, .
    parts = re.split(r'[/\\.]', path)

    tokens = []
    for part in parts:
        # Split camelCase: RuleIndex → [Rule, Index]
        words = re.findall(r'[A-Z][a-z]*|[a-z]+', part)
        tokens.extend([w.upper() for w in words if len(w) > 2])

    return tokens

def is_valid_identifier(token):
    """Filter out common noise words"""
    noise = ['SRC', 'MAIN', 'JAVA', 'ORG', 'COM', 'TEST', 'IMPL', 'UTIL']
    return token not in noise and len(token) >= 3
```

### Step 2: Extract Business Terms from Task History

**Script**: `extract_business_terms.py`

```python
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def extract_terms_from_tasks(file_path):
    """Get business terms from all tasks that modified a file"""

    # Get all tasks
    cursor.execute("""
        SELECT task_name, message
        FROM vectors.rawdata
        WHERE path = %s
    """, (file_path,))

    tasks = cursor.fetchall()

    # Extract nouns
    all_nouns = []
    for task_name, message in tasks:
        doc = nlp(message)
        nouns = [token.lemma_.lower() for token in doc if token.pos_ == 'NOUN']
        all_nouns.extend(nouns)

    # Count occurrences
    term_counts = Counter(all_nouns)

    # Return top terms
    return term_counts.most_common(10)
```

### Step 3: Create Entity-to-File Index in PostgreSQL

**New table**: `entity_map`

```sql
CREATE TABLE vectors.entity_map (
    id SERIAL PRIMARY KEY,
    entity_name VARCHAR(255) NOT NULL,
    business_terms TEXT[],  -- Array of related terms
    file_paths TEXT[],      -- Array of file paths
    module_paths TEXT[],    -- Array of module paths
    importance FLOAT,       -- 0.0 to 1.0
    centrality VARCHAR(50), -- 'core', 'feature', 'util', 'infrastructure'
    related_entities TEXT[], -- Array of related entity names
    confidence FLOAT,        -- 0.0 to 1.0 for obfuscated code
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_entity_name ON vectors.entity_map(entity_name);
CREATE INDEX idx_file_paths ON vectors.entity_map USING GIN(file_paths);
```

### Step 4: Integrate Entity Map into RAG Pipeline

**Modified Phase 1**: Add keyword-based boosting

```python
async def phase1_reasoning_with_entity_map(task_description):
    # Extract keywords from task description
    keywords = extract_keywords_nlp(task_description)

    # Lookup entities
    entity_boosts = {}
    for keyword in keywords:
        entity = lookup_entity(keyword)
        if entity:
            for file in entity['files']:
                entity_boosts[file] = entity_boosts.get(file, 0) + 0.3

    # Run vector search
    module_results = vector_search_modules(task_description, top_k=10)
    file_results = vector_search_files(task_description, top_k=15)

    # Apply entity boost to scores
    for file in file_results:
        if file['path'] in entity_boosts:
            file['score'] += entity_boosts[file['path']]

    # Re-rank
    file_results = sorted(file_results, key=lambda x: x['score'], reverse=True)

    # Add entity context to LLM prompt
    entity_context = build_entity_context(keywords, entity_map)

    return {
        'files': file_results,
        'entity_context': entity_context
    }
```

**Enhanced LLM Prompt**:

```python
prompt = f"""
# Task
{task_description}

# Identified Domain Entities (Keywords)
{entity_context}

Example:
- RULE: Core domain entity found in 15 files
  * Primary files: RuleIndex.java, RuleUpdater.java, RuleCreator.java
  * Related entities: QUALITY_PROFILE, INDEX, ISSUE
  * This is a core business concept representing coding rules

# Relevant Modules (structure context)
{modules_text}

# Relevant Files (ranked by entity match + semantic similarity)
{files_text}

Your task: Select the 5-10 most important files, considering:
1. Which files belong to the identified domain entities
2. Which files are at the intersection of multiple relevant entities
3. Semantic similarity scores
"""
```

## Benefits of This Approach

### 1. Structured Context (The "Map")

Instead of random villages, LLM receives:
- **Entities**: Core domain concepts (RULE, QUALITY_PROFILE, INDEX)
- **Relationships**: How entities connect (RULE related to INDEX)
- **Geography**: Where each entity lives in the codebase

### 2. Improved Precision

- **Keyword boosting**: Files matching domain entities get higher priority
- **Disambiguation**: When "rule" appears in task, system knows it refers to RULE entity
- **Filtering**: Can filter out false positives that have high vector similarity but wrong entity

### 3. Better LLM Reasoning

With entity map, even a small LLM can:
- Understand "RULE is a core concept, so RuleIndex.java is highly relevant"
- Reason "Task mentions 'rule descriptions', so I need RULE entity + DESCRIPTION fields"
- Infer "If RULE relates to INDEX, check ElasticSearch configuration"

### 4. Handling Code Evolution

- **Refactoring**: If RuleIndex.java is renamed to RuleFinder.java, entity map updates automatically
- **Obfuscation**: Can recover business terms from historical tasks
- **New features**: Entity map grows as new domain concepts appear

### 5. Cross-Project Transfer

Once you build entity extraction pipeline:
- Apply to any Java/Python/C# project
- Works for legacy code with good identifiers
- Works for obfuscated code using task history
- Can even work for projects with poor documentation

## Validation and Evaluation

### Metrics

1. **Entity Extraction Quality**
   - Precision: % of extracted keywords that are real domain entities
   - Recall: % of actual domain entities that were extracted
   - F1 Score

2. **Traceability Quality**
   - % of task descriptions where correct files are in top-5 after entity boosting
   - Mean Reciprocal Rank (MRR)
   - Precision@K

3. **LLM Performance**
   - With vs without entity map
   - Recommendation quality (human eval)
   - File selection accuracy

### Test Cases

```python
test_cases = [
    {
        "task": "Fix rule parameter validation",
        "expected_entity": "RULE",
        "expected_files": ["RuleUpdater.java", "RuleParameters.java"]
    },
    {
        "task": "Improve quality profile export performance",
        "expected_entity": "QUALITY_PROFILE",
        "expected_files": ["QProfileExporter.java", "ProfileBackuper.java"]
    },
    {
        "task": "Add support for longer search terms in rule descriptions",
        "expected_entities": ["RULE", "SEARCH", "DESCRIPTION"],
        "expected_files": ["RuleIndex.java", "RuleDoc.java"]
    }
]
```

## Related Work and Citations

### Key Papers

1. **Domain Concept Extraction**
   - [Extraction of domain concepts from the source code](https://www.sciencedirect.com/science/article/pii/S0167642314004419) - Automatic extraction using NLP
   - [From source code identifiers to natural language terms](https://www.sciencedirect.com/science/article/abs/pii/S0164121214002179) - Converting identifiers to domain vocabulary

2. **Concept Location**
   - [An information retrieval approach to concept location in source code](https://ieeexplore.ieee.org/document/1374321/) - IR-based concept location
   - [Feature location in source code: A taxonomy and survey](https://onlinelibrary.wiley.com/doi/full/10.1002/smr.567) - Comprehensive survey

3. **Requirements Traceability**
   - [An empirical study on the importance of source code entities for requirements traceability](https://link.springer.com/article/10.1007/s10664-014-9315-y) - Role of code entities
   - [Constructing Traceability Links between Software Requirements and Source Code Based on Neural Networks](https://www.mdpi.com/2227-7390/11/2/315) - Neural approaches

### Research Areas

- **Software Feature Location**: Finding where features are implemented
- **Concept Location**: Mapping domain concepts to code
- **Domain Vocabulary Extraction**: Extracting business terms from code
- **Requirements-to-Code Traceability**: Linking requirements to implementation
- **Program Comprehension**: Understanding software structure and semantics
- **The Vocabulary Problem**: Bridging terminology gaps between stakeholders

## Next Steps

1. **Prototype**: Build entity map for SonarQube codebase
2. **Evaluate**: Test on 50-100 historical tasks
3. **Integrate**: Add entity boosting to Phase 1
4. **Iterate**: Refine keyword extraction based on results
5. **Scale**: Apply to other projects

## Conclusion

The keyword-based entity mapping approach provides the **"map"** that current RAG lacks. By extracting domain entities from code structure and task history, we can:

- Give LLM **structural context** instead of random files
- **Boost relevant files** based on entity matching
- **Handle obfuscated code** by recovering business terms from tasks
- **Improve precision** through keyword-guided search
- **Enable better reasoning** even with small LLMs

This approach is **well-supported by academic research** and addresses the core problem: providing supervision and structure to help LLM navigate the codebase like a developer with domain knowledge.

---

## Sources

- [Extraction of domain concepts from the source code (ScienceDirect, 2014)](https://www.sciencedirect.com/science/article/pii/S0167642314004419)
- [From source code identifiers to natural language terms (ScienceDirect, 2014)](https://www.sciencedirect.com/science/article/abs/pii/S0164121214002179)
- [On the Use of Domain Terms in Source Code (ResearchGate, 2008)](https://www.researchgate.net/publication/4349667_On_the_Use_of_Domain_Terms_in_Source_Code)
- [Towards the Extraction of Domain Concepts from the Identifiers (IEEE, 2011)](https://ieeexplore.ieee.org/document/6079777/)
- [An information retrieval approach to concept location in source code (IEEE, 2003)](https://ieeexplore.ieee.org/document/1374321/)
- [Text Retrieval Approaches for Concept Location in Source Code (Springer, 2013)](https://link.springer.com/chapter/10.1007/978-3-642-36054-1_5)
- [Concept location using formal concept analysis and information retrieval (ACM TOSEM, 2012)](https://dl.acm.org/doi/10.1145/2377656.2377660)
- [Feature location in source code: A taxonomy and survey (Wiley, 2013)](https://onlinelibrary.wiley.com/doi/full/10.1002/smr.567)
- [An empirical study on the importance of source code entities for requirements traceability (Springer, 2015)](https://link.springer.com/article/10.1007/s10664-014-9315-y)
- [Recovering semantic traceability between requirements and design (Springer, 2019)](https://link.springer.com/article/10.1007/s11334-019-00330-w)
- [On the role of semantics in automated requirements tracing (Springer, 2014)](https://link.springer.com/article/10.1007/s00766-013-0199-y)
- [Constructing Traceability Links between Software Requirements and Source Code Based on Neural Networks (MDPI, 2023)](https://www.mdpi.com/2227-7390/11/2/315)
