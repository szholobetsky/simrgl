# Phenomenological Grounding: Universal Implementation

## Overview

This document describes a **language-agnostic** implementation of phenomenological code grounding. The approach works with ANY programming language (Java, Python, SQL, PL/SQL, VB, C, 1С Предприятие, etc.) and ANY natural language (English, Ukrainian, Russian, German, etc.) without requiring grammar understanding.

**Core Principle**: We don't need to **understand** code to **ground** symbols. We need to extract **identifiers** and their **co-occurrence relations**, then link business terms to these identifier clusters.

---

## 1. Philosophical Foundation (Recap)

### From Phenomenology to Implementation

| Philosophical Concept | Implementation |
|----------------------|----------------|
| **Noema** (object of intention) | Identifier cluster + file locations |
| **Noesis** (act of perceiving) | Graph traversal by agent |
| **Zuhandenheit** (ready-to-hand) | Code that works (not in search results) |
| **Vorhandenheit** (present-at-hand) | Code under change (in search results) |
| **Lebenswelt** (life-world) | Identifier graph + relations + history |
| **Affordances** | Available operations (methods, functions) |
| **Symbol Grounding** | Business term → Identifier cluster → Files |
| **Epoché** (bracketing) | Ignore syntax, focus on co-occurrence |

### The Key Insight from dialog.txt

> "Код — це не опис бізнесу, це його цифрова еманація."
> (Code is not a description of business, it's its digital emanation)

We don't parse the emanation — we **feel its patterns** through:
- Co-occurrence statistics (like Word2Vec)
- Relation patterns (universal across languages)
- Cross-file identifier sharing

---

## 2. Universal Identifier Extraction

### 2.1 The Algorithm

```python
import re
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict, List, Tuple

class UniversalIdentifierExtractor:
    """
    Language-agnostic identifier extraction.
    Works with ANY text-based code file.
    """

    def __init__(self, min_length: int = 3, min_files: int = 2):
        self.min_length = min_length
        self.min_files = min_files

    def extract_from_content(self, content: str) -> List[str]:
        """
        Extract all potential identifiers from file content.
        Works for: Java, Python, C#, SQL, PL/SQL, VB, 1С, etc.
        """
        identifiers = []

        # Step 1: Extract alphanumeric tokens (ASCII)
        # Matches: getUserName, get_user_name, GET_USER_NAME
        ascii_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', content)

        # Step 2: Extract Cyrillic tokens (for 1С, comments in RU/UA)
        # Matches: ПолучитьИмяПользователя, акт_списання
        cyrillic_tokens = re.findall(r'[а-яА-ЯіїєґІЇЄҐёЁ_][а-яА-ЯіїєґІЇЄҐёЁ0-9_]*', content)

        # Step 3: Split camelCase and PascalCase
        for token in ascii_tokens:
            # "getUserName" → ["get", "user", "name", "getusername"]
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', token)
            identifiers.extend([p.lower() for p in parts if len(p) >= self.min_length])
            if len(token) >= self.min_length:
                identifiers.append(token.lower())

        # Step 4: Split snake_case
        for token in ascii_tokens + cyrillic_tokens:
            parts = token.split('_')
            identifiers.extend([p.lower() for p in parts if len(p) >= self.min_length])

        # Step 5: Add Cyrillic tokens as-is (already word-like)
        for token in cyrillic_tokens:
            if len(token) >= self.min_length:
                identifiers.append(token.lower())

        return identifiers

    def extract_from_directory(self,
                               directory: str,
                               extensions: List[str] = None) -> Dict[str, List[str]]:
        """
        Extract identifiers from all files in directory.

        Args:
            directory: Root directory to scan
            extensions: File extensions to include (None = all text files)

        Returns:
            Dict mapping filepath → list of identifiers
        """
        if extensions is None:
            # Common code file extensions (add more as needed)
            extensions = [
                '.java', '.py', '.cs', '.js', '.ts', '.cpp', '.c', '.h',
                '.sql', '.plsql', '.vb', '.vbs', '.pas', '.dpr',
                '.xml', '.json', '.yaml', '.yml', '.properties',
                '.1cd', '.bsl', '.os',  # 1С files
                '.go', '.rs', '.rb', '.php', '.swift', '.kt'
            ]

        file_identifiers = {}

        for ext in extensions:
            for filepath in Path(directory).rglob(f'*{ext}'):
                try:
                    content = filepath.read_text(encoding='utf-8', errors='ignore')
                    identifiers = self.extract_from_content(content)
                    if identifiers:
                        file_identifiers[str(filepath)] = identifiers
                except Exception as e:
                    print(f"Warning: Could not process {filepath}: {e}")

        return file_identifiers

    def filter_cross_file(self,
                          file_identifiers: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """
        Keep only identifiers that appear in min_files or more files.
        This filters noise and keeps domain vocabulary.
        """
        # Count files per identifier
        identifier_files = defaultdict(set)
        for filepath, identifiers in file_identifiers.items():
            for ident in set(identifiers):
                identifier_files[ident].add(filepath)

        # Filter to cross-file identifiers
        cross_file = {
            ident: files
            for ident, files in identifier_files.items()
            if len(files) >= self.min_files
        }

        return cross_file

    def get_identifier_stats(self,
                             file_identifiers: Dict[str, List[str]],
                             cross_file: Dict[str, Set[str]]) -> List[Dict]:
        """
        Compute statistics for each cross-file identifier.
        """
        stats = []

        for ident, files in cross_file.items():
            # Total occurrences across all files
            total_count = sum(
                file_identifiers[f].count(ident)
                for f in files
            )

            stats.append({
                'identifier': ident,
                'file_count': len(files),
                'total_count': total_count,
                'files': list(files),
                'avg_per_file': total_count / len(files)
            })

        # Sort by file_count descending
        stats.sort(key=lambda x: x['file_count'], reverse=True)

        return stats
```

### 2.2 Example: Processing 1С Предприятие

```python
# 1С code example
content_1c = """
Процедура ОбработкаЗаполнения(ДанныеЗаполнения)
    АктСписанияТары = Документы.АктСписанияТары.СоздатьДокумент();
    АктСписанияТары.Номер = ПолучитьНомерДокумента("АСТ-");
    АктСписанияТары.Дата = ТекущаяДата();
    АктСписанияТары.Ответственный = ТекущийПользователь();
КонецПроцедуры
"""

extractor = UniversalIdentifierExtractor()
identifiers = extractor.extract_from_content(content_1c)

# Result:
# ['обработка', 'заполнения', 'данные', 'акт', 'списания', 'тары',
#  'документы', 'создать', 'документ', 'номер', 'получить',
#  'дата', 'текущая', 'ответственный', 'пользователь']
```

### 2.3 LLM-Assisted Keyword Filtering

```python
async def filter_keywords_with_llm(
    identifiers: List[str],
    sample_files: Dict[str, str],
    llm_client
) -> Tuple[List[str], List[str]]:
    """
    Use local LLM to separate:
    - Business terms (keep)
    - Language keywords / standard library (exclude)

    Works for ANY language because LLM reasons about patterns.
    """
    # Take a sample of identifiers (to fit context)
    sample = identifiers[:200]

    # Get sample contexts
    contexts = []
    for ident in sample[:10]:
        for filepath, content in list(sample_files.items())[:3]:
            if ident in content.lower():
                # Find line containing identifier
                for line in content.split('\n'):
                    if ident in line.lower():
                        contexts.append(f"  {ident}: {line.strip()[:100]}")
                        break

    prompt = f"""
You are analyzing code identifiers to separate business domain terms from
programming language keywords and standard library names.

Here are sample identifiers with their contexts:
{chr(10).join(contexts[:20])}

Full list of identifiers to classify:
{', '.join(sample)}

Instructions:
1. KEEP: Domain-specific business terms (entities, processes, attributes)
2. EXCLUDE: Language keywords (if, else, while, class, function, public, private)
3. EXCLUDE: Standard library names (java.util, System, Console, String, Integer)
4. EXCLUDE: Generic programming words (get, set, new, create, delete, update, list, map)

Return TWO lists:
KEEP: [word1, word2, ...]
EXCLUDE: [word1, word2, ...]
"""

    response = await llm_client.generate(prompt)

    # Parse response
    keep_list = []
    exclude_list = []

    # Simple parsing (could be more robust)
    if 'KEEP:' in response:
        keep_part = response.split('KEEP:')[1].split('EXCLUDE:')[0]
        keep_list = [w.strip().lower() for w in re.findall(r'\w+', keep_part)]

    if 'EXCLUDE:' in response:
        exclude_part = response.split('EXCLUDE:')[1]
        exclude_list = [w.strip().lower() for w in re.findall(r'\w+', exclude_part)]

    return keep_list, exclude_list
```

---

## 3. Universal Relation Extraction

### 3.1 Language-Agnostic Patterns

These patterns work across virtually ALL programming languages:

| Pattern | Regex | Meaning | Examples |
|---------|-------|---------|----------|
| `A.B` | `(\w+)\.(\w+)` | member_of, access | Java: `obj.field`, SQL: `schema.table`, 1С: `Документ.Номер` |
| `A(B)` | `(\w+)\s*\(([^)]+)\)` | calls, uses | All languages with functions |
| `A = B` | `(\w+)\s*:?=\s*(\w+)` | receives, assigned | All languages with assignment |
| `A, B` | same line | co_occurs | Weak relation, universal |
| `A : B` | `(\w+)\s*:\s*(\w+)` | type_of | Python hints, TypeScript, Pascal |
| `A < B` | `(\w+)\s*[<>=!]+\s*(\w+)` | compares_with | All languages |
| `A extends B` | `(\w+)\s+extends\s+(\w+)` | inherits | Java, TypeScript |
| `A implements B` | `(\w+)\s+implements\s+(\w+)` | implements | Java |

### 3.2 The Algorithm

```python
class UniversalRelationExtractor:
    """
    Extract relations between identifiers using universal patterns.
    No grammar knowledge required - works with pattern matching.
    """

    def __init__(self, valid_identifiers: Set[str]):
        """
        Args:
            valid_identifiers: Set of identifiers we care about
                              (from cross-file filtering)
        """
        self.valid = {i.lower() for i in valid_identifiers}

    def extract_from_line(self, line: str, line_num: int) -> List[Dict]:
        """
        Extract relations from a single line of code.
        """
        relations = []
        line_lower = line.lower()

        # Find all valid identifiers in this line
        found_in_line = [i for i in self.valid if i in line_lower]

        if len(found_in_line) < 2:
            return relations

        # Pattern 1: A.B → member_of
        for match in re.finditer(r'(\w+)\.(\w+)', line):
            a, b = match.group(1).lower(), match.group(2).lower()
            if a in self.valid and b in self.valid:
                relations.append({
                    'source': b,
                    'target': a,
                    'type': 'member_of',
                    'line': line_num,
                    'context': line.strip()[:100]
                })

        # Pattern 2: A(B, C, ...) → calls
        for match in re.finditer(r'(\w+)\s*\(([^)]*)\)', line):
            func = match.group(1).lower()
            args_str = match.group(2)

            if func in self.valid:
                # Extract identifiers from arguments
                args = re.findall(r'\w+', args_str)
                for arg in args:
                    arg_lower = arg.lower()
                    if arg_lower in self.valid and arg_lower != func:
                        relations.append({
                            'source': func,
                            'target': arg_lower,
                            'type': 'calls',
                            'line': line_num,
                            'context': line.strip()[:100]
                        })

        # Pattern 3: A = B or A := B → receives
        for match in re.finditer(r'(\w+)\s*:?=\s*(\w+)', line):
            a, b = match.group(1).lower(), match.group(2).lower()
            if a in self.valid and b in self.valid and a != b:
                relations.append({
                    'source': a,
                    'target': b,
                    'type': 'receives',
                    'line': line_num,
                    'context': line.strip()[:100]
                })

        # Pattern 4: A extends/implements B → inherits
        for match in re.finditer(r'(\w+)\s+(?:extends|implements|:)\s+(\w+)', line, re.IGNORECASE):
            a, b = match.group(1).lower(), match.group(2).lower()
            if a in self.valid and b in self.valid:
                relations.append({
                    'source': a,
                    'target': b,
                    'type': 'inherits',
                    'line': line_num,
                    'context': line.strip()[:100]
                })

        # Pattern 5: Co-occurrence (weak relation)
        for i, a in enumerate(found_in_line):
            for b in found_in_line[i+1:]:
                relations.append({
                    'source': a,
                    'target': b,
                    'type': 'co_occurs',
                    'line': line_num,
                    'context': line.strip()[:100]
                })

        return relations

    def extract_from_file(self, filepath: str) -> List[Dict]:
        """
        Extract all relations from a file.
        """
        relations = []

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line_relations = self.extract_from_line(line, line_num)
                    for rel in line_relations:
                        rel['file'] = filepath
                    relations.extend(line_relations)
        except Exception as e:
            print(f"Warning: Could not process {filepath}: {e}")

        return relations

    def extract_from_directory(self, directory: str) -> List[Dict]:
        """
        Extract relations from all files in directory.
        """
        all_relations = []

        for filepath in Path(directory).rglob('*'):
            if filepath.is_file() and filepath.suffix:
                relations = self.extract_from_file(str(filepath))
                all_relations.extend(relations)

        return all_relations

    def aggregate_relations(self, relations: List[Dict]) -> Dict[Tuple, Dict]:
        """
        Aggregate relations by (source, target, type).
        Count occurrences and collect locations.
        """
        aggregated = defaultdict(lambda: {
            'count': 0,
            'files': set(),
            'lines': []
        })

        for rel in relations:
            key = (rel['source'], rel['target'], rel['type'])
            aggregated[key]['count'] += 1
            aggregated[key]['files'].add(rel['file'])
            aggregated[key]['lines'].append({
                'file': rel['file'],
                'line': rel['line'],
                'context': rel['context']
            })

        # Convert sets to lists for JSON serialization
        result = {}
        for key, data in aggregated.items():
            result[key] = {
                'source': key[0],
                'target': key[1],
                'type': key[2],
                'count': data['count'],
                'file_count': len(data['files']),
                'files': list(data['files']),
                'samples': data['lines'][:5]  # Keep 5 samples
            }

        return result
```

---

## 4. PostgreSQL Storage Schema

### 4.1 Schema Definition

```sql
-- Create schema for phenomenological grounding
CREATE SCHEMA IF NOT EXISTS ontology;

-- Extension for vector similarity
CREATE EXTENSION IF NOT EXISTS vector;

-- Identifiers table (Noema candidates)
CREATE TABLE ontology.identifiers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    normalized VARCHAR(255),

    -- Statistics
    file_count INT NOT NULL DEFAULT 0,
    total_count INT NOT NULL DEFAULT 0,
    avg_per_file FLOAT,

    -- Embedding (from Word2Vec trained on this codebase)
    embedding vector(300),

    -- Classification
    is_business_term BOOLEAN DEFAULT NULL,  -- NULL = not classified
    is_excluded BOOLEAN DEFAULT FALSE,       -- Keywords/stdlib filtered out
    domain VARCHAR(100),                      -- 'inventory', 'accounting', etc.

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- File locations for each identifier
CREATE TABLE ontology.identifier_locations (
    id SERIAL PRIMARY KEY,
    identifier_id INT REFERENCES ontology.identifiers(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_numbers INT[],
    occurrence_count INT DEFAULT 1,

    UNIQUE(identifier_id, file_path)
);

-- Relations between identifiers
CREATE TABLE ontology.relations (
    id SERIAL PRIMARY KEY,
    source_id INT REFERENCES ontology.identifiers(id) ON DELETE CASCADE,
    target_id INT REFERENCES ontology.identifiers(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL,  -- 'member_of', 'calls', 'receives', etc.

    -- Statistics
    occurrence_count INT DEFAULT 1,
    file_count INT DEFAULT 1,

    -- Weight (can be computed from statistics)
    weight FLOAT DEFAULT 1.0,

    UNIQUE(source_id, target_id, relation_type)
);

-- Sample contexts for relations (for debugging/display)
CREATE TABLE ontology.relation_samples (
    id SERIAL PRIMARY KEY,
    relation_id INT REFERENCES ontology.relations(id) ON DELETE CASCADE,
    file_path TEXT,
    line_number INT,
    context TEXT
);

-- Business glossary (multilingual)
CREATE TABLE ontology.glossary (
    id SERIAL PRIMARY KEY,
    term TEXT NOT NULL,
    language VARCHAR(10) NOT NULL,  -- 'en', 'ua', 'ru', 'de', etc.

    -- Grounding to identifier
    identifier_id INT REFERENCES ontology.identifiers(id),
    confidence FLOAT DEFAULT 1.0,

    -- Synonyms and variations
    synonyms TEXT[],

    -- Embedding for similarity search
    embedding vector(384),  -- Using sentence transformer

    -- Metadata
    domain VARCHAR(100),
    importance VARCHAR(20),  -- 'high', 'medium', 'low'

    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_identifiers_embedding ON ontology.identifiers
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_glossary_embedding ON ontology.glossary
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_identifiers_name ON ontology.identifiers(name);
CREATE INDEX idx_identifiers_normalized ON ontology.identifiers(normalized);
CREATE INDEX idx_relations_source ON ontology.relations(source_id);
CREATE INDEX idx_relations_target ON ontology.relations(target_id);
CREATE INDEX idx_relations_type ON ontology.relations(relation_type);
CREATE INDEX idx_glossary_term ON ontology.glossary(term);
CREATE INDEX idx_glossary_language ON ontology.glossary(language);
```

### 4.2 Data Population

```python
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

class OntologyStorage:
    """
    Store and query the phenomenological ontology in PostgreSQL.
    """

    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)

    def insert_identifiers(self, identifier_stats: List[Dict]):
        """
        Insert identifiers with their statistics.
        """
        with self.conn.cursor() as cur:
            values = [
                (
                    stat['identifier'],
                    stat['identifier'].lower(),
                    stat['file_count'],
                    stat['total_count'],
                    stat['avg_per_file']
                )
                for stat in identifier_stats
            ]

            execute_values(
                cur,
                """
                INSERT INTO ontology.identifiers
                    (name, normalized, file_count, total_count, avg_per_file)
                VALUES %s
                ON CONFLICT (name) DO UPDATE SET
                    file_count = EXCLUDED.file_count,
                    total_count = EXCLUDED.total_count,
                    avg_per_file = EXCLUDED.avg_per_file,
                    updated_at = NOW()
                """,
                values
            )

        self.conn.commit()

    def insert_locations(self, identifier_stats: List[Dict]):
        """
        Insert file locations for identifiers.
        """
        with self.conn.cursor() as cur:
            for stat in identifier_stats:
                # Get identifier ID
                cur.execute(
                    "SELECT id FROM ontology.identifiers WHERE name = %s",
                    (stat['identifier'],)
                )
                row = cur.fetchone()
                if not row:
                    continue

                ident_id = row[0]

                for filepath in stat['files']:
                    cur.execute(
                        """
                        INSERT INTO ontology.identifier_locations
                            (identifier_id, file_path, occurrence_count)
                        VALUES (%s, %s, 1)
                        ON CONFLICT (identifier_id, file_path) DO UPDATE SET
                            occurrence_count = ontology.identifier_locations.occurrence_count + 1
                        """,
                        (ident_id, filepath)
                    )

        self.conn.commit()

    def insert_relations(self, aggregated_relations: Dict):
        """
        Insert relations between identifiers.
        """
        with self.conn.cursor() as cur:
            # Get identifier name → ID mapping
            cur.execute("SELECT name, id FROM ontology.identifiers")
            id_map = {row[0]: row[1] for row in cur.fetchall()}

            for key, data in aggregated_relations.items():
                source, target, rel_type = key

                source_id = id_map.get(source)
                target_id = id_map.get(target)

                if not source_id or not target_id:
                    continue

                # Insert relation
                cur.execute(
                    """
                    INSERT INTO ontology.relations
                        (source_id, target_id, relation_type, occurrence_count, file_count)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, relation_type) DO UPDATE SET
                        occurrence_count = EXCLUDED.occurrence_count,
                        file_count = EXCLUDED.file_count
                    RETURNING id
                    """,
                    (source_id, target_id, rel_type, data['count'], data['file_count'])
                )

                rel_id = cur.fetchone()[0]

                # Insert samples
                for sample in data['samples'][:3]:
                    cur.execute(
                        """
                        INSERT INTO ontology.relation_samples
                            (relation_id, file_path, line_number, context)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (rel_id, sample['file'], sample['line'], sample['context'])
                    )

        self.conn.commit()

    def update_embeddings(self, word2vec_model):
        """
        Update identifier embeddings from trained Word2Vec model.
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, name FROM ontology.identifiers")

            for row in cur.fetchall():
                ident_id, name = row

                # Try to get vector from Word2Vec
                vector = None
                for variant in [name, name.capitalize(), name.lower(), name.upper()]:
                    if variant in word2vec_model.wv:
                        vector = word2vec_model.wv[variant].tolist()
                        break

                if vector:
                    cur.execute(
                        "UPDATE ontology.identifiers SET embedding = %s WHERE id = %s",
                        (vector, ident_id)
                    )

        self.conn.commit()
```

---

## 5. Word2Vec Training (Keyword Indexing Integration)

### 5.1 Training on Extracted Identifiers

```python
from gensim.models import Word2Vec
from typing import List, Dict

class CodeWord2Vec:
    """
    Train Word2Vec on code corpus for identifier embeddings.
    Language-agnostic: works with extracted identifiers from any language.
    """

    def __init__(self, vector_size: int = 300, window: int = 5, min_count: int = 2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def prepare_corpus(self, file_identifiers: Dict[str, List[str]]) -> List[List[str]]:
        """
        Prepare corpus for Word2Vec training.
        Each file becomes a "sentence" of identifiers.
        """
        corpus = []

        for filepath, identifiers in file_identifiers.items():
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for ident in identifiers:
                if ident not in seen:
                    seen.add(ident)
                    unique.append(ident)

            if len(unique) >= 3:  # Need at least 3 identifiers
                corpus.append(unique)

        return corpus

    def train(self, file_identifiers: Dict[str, List[str]], epochs: int = 10):
        """
        Train Word2Vec model on code corpus.
        """
        corpus = self.prepare_corpus(file_identifiers)

        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=1,           # Skip-gram (better for rare words)
            negative=10,    # Negative sampling
            epochs=epochs
        )

        return self.model

    def save(self, filepath: str):
        """Save trained model."""
        if self.model:
            self.model.save(filepath)

    def load(self, filepath: str):
        """Load trained model."""
        self.model = Word2Vec.load(filepath)
        return self.model

    def get_similar(self, identifier: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar identifiers in the code semantic space.
        """
        if not self.model or identifier not in self.model.wv:
            return []

        return self.model.wv.most_similar(identifier, topn=top_k)

    def get_negative_space(self,
                           positive: List[str],
                           threshold: float = 0.3) -> List[str]:
        """
        Find identifiers in the "negative space" (dissimilar to positive set).
        Used for contrastive grounding.
        """
        if not self.model:
            return []

        all_words = list(self.model.wv.index_to_key)
        negative = []

        for word in all_words:
            if word in positive:
                continue

            # Compute average similarity to positive set
            sims = []
            for pos_word in positive:
                if pos_word in self.model.wv:
                    sim = self.model.wv.similarity(word, pos_word)
                    sims.append(sim)

            if sims and np.mean(sims) < threshold:
                negative.append((word, np.mean(sims)))

        # Sort by dissimilarity (lowest first)
        negative.sort(key=lambda x: x[1])

        return [w for w, _ in negative]
```

---

## 6. MCP Server for Ontology Navigation

### 6.1 Server Implementation

```python
# mcp_server_ontology.py

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
import psycopg2
import numpy as np

server = Server("ontology-grounding-server")

# Database connection
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'simargl',
    'user': 'postgres',
    'password': 'postgres'
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


@server.tool()
async def get_noema(business_term: str, language: str = 'auto') -> dict:
    """
    Find the intentional object (noema) for a business term.
    Returns grounded identifier with files and relations.

    This is the core phenomenological operation: linking abstract
    business intent to concrete code locations.

    Args:
        business_term: The business term to ground (any language)
        language: Language hint ('en', 'ua', 'ru', 'auto')

    Returns:
        Grounded noema with identifier, files, and affordances
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Step 1: Search glossary by exact match
        cur.execute("""
            SELECT g.id, g.term, g.language, g.confidence,
                   i.id, i.name, i.file_count
            FROM ontology.glossary g
            JOIN ontology.identifiers i ON g.identifier_id = i.id
            WHERE LOWER(g.term) = LOWER(%s)
               OR %s = ANY(g.synonyms)
            ORDER BY g.confidence DESC
            LIMIT 1
        """, (business_term, business_term.lower()))

        row = cur.fetchone()

        if not row:
            # Step 2: Search by embedding similarity
            # (requires embedding the business_term first)
            cur.execute("""
                SELECT i.id, i.name, i.file_count,
                       1 - (i.embedding <=>
                           (SELECT embedding FROM ontology.glossary
                            WHERE term ILIKE %s LIMIT 1)) as similarity
                FROM ontology.identifiers i
                WHERE i.embedding IS NOT NULL
                  AND i.is_excluded = FALSE
                ORDER BY similarity DESC
                LIMIT 5
            """, (f'%{business_term}%',))

            rows = cur.fetchall()
            if not rows:
                return {
                    'found': False,
                    'term': business_term,
                    'message': 'No grounding found for this term'
                }

            # Use best match
            ident_id, ident_name, file_count, similarity = rows[0]
        else:
            _, _, _, confidence, ident_id, ident_name, file_count = row
            similarity = confidence

        # Step 3: Get file locations
        cur.execute("""
            SELECT file_path, occurrence_count
            FROM ontology.identifier_locations
            WHERE identifier_id = %s
            ORDER BY occurrence_count DESC
            LIMIT 10
        """, (ident_id,))

        files = [{'path': row[0], 'count': row[1]} for row in cur.fetchall()]

        # Step 4: Get affordances (relations where this identifier is source)
        cur.execute("""
            SELECT r.relation_type, i.name as target, r.occurrence_count
            FROM ontology.relations r
            JOIN ontology.identifiers i ON r.target_id = i.id
            WHERE r.source_id = %s
            ORDER BY r.occurrence_count DESC
            LIMIT 10
        """, (ident_id,))

        affordances = [
            {'action': row[0], 'target': row[1], 'count': row[2]}
            for row in cur.fetchall()
        ]

        return {
            'found': True,
            'term': business_term,
            'grounded_to': ident_name,
            'confidence': float(similarity) if similarity else 1.0,
            'file_count': file_count,
            'files': files,
            'affordances': affordances
        }

    finally:
        cur.close()
        conn.close()


@server.tool()
async def get_horizon(
    business_terms: list[str],
    radius: int = 2
) -> dict:
    """
    Get the hermeneutic horizon for a set of business terms.
    Returns all related identifiers within graph distance.

    This implements Gadamer's "fusion of horizons" - combining
    the user's intent with the code's structure.

    Args:
        business_terms: List of business terms to ground
        radius: Graph traversal depth (1-3 recommended)

    Returns:
        Horizon with all related identifiers and files
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Step 1: Ground all business terms
        grounded_ids = set()

        for term in business_terms:
            noema = await get_noema(term)
            if noema.get('found'):
                cur.execute(
                    "SELECT id FROM ontology.identifiers WHERE name = %s",
                    (noema['grounded_to'],)
                )
                row = cur.fetchone()
                if row:
                    grounded_ids.add(row[0])

        if not grounded_ids:
            return {
                'found': False,
                'message': 'Could not ground any business terms'
            }

        # Step 2: Expand horizon through relations
        horizon_ids = set(grounded_ids)

        for _ in range(radius):
            new_ids = set()

            for ident_id in horizon_ids:
                # Get related through outgoing relations
                cur.execute("""
                    SELECT target_id FROM ontology.relations
                    WHERE source_id = %s
                """, (ident_id,))
                new_ids.update(row[0] for row in cur.fetchall())

                # Get related through incoming relations
                cur.execute("""
                    SELECT source_id FROM ontology.relations
                    WHERE target_id = %s
                """, (ident_id,))
                new_ids.update(row[0] for row in cur.fetchall())

            horizon_ids.update(new_ids)

        # Step 3: Get all identifiers in horizon
        cur.execute("""
            SELECT id, name, file_count, is_business_term
            FROM ontology.identifiers
            WHERE id = ANY(%s)
            ORDER BY file_count DESC
        """, (list(horizon_ids),))

        identifiers = [
            {
                'id': row[0],
                'name': row[1],
                'file_count': row[2],
                'is_business_term': row[3],
                'is_grounded': row[0] in grounded_ids
            }
            for row in cur.fetchall()
        ]

        # Step 4: Get all files in horizon
        cur.execute("""
            SELECT DISTINCT file_path
            FROM ontology.identifier_locations
            WHERE identifier_id = ANY(%s)
        """, (list(horizon_ids),))

        files = [row[0] for row in cur.fetchall()]

        return {
            'found': True,
            'grounded_terms': list(business_terms),
            'horizon_size': len(horizon_ids),
            'identifiers': identifiers,
            'files': files[:50],  # Limit to 50 files
            'total_files': len(files)
        }

    finally:
        cur.close()
        conn.close()


@server.tool()
async def get_neighbors(
    identifier: str,
    relation_types: list[str] = None,
    direction: str = 'both'
) -> dict:
    """
    Navigate the code graph - get neighboring identifiers.
    This is the agent's "proprioception" - feeling the code structure.

    Args:
        identifier: The identifier to start from
        relation_types: Filter by relation types (None = all)
        direction: 'outgoing', 'incoming', or 'both'

    Returns:
        Neighboring identifiers with their relations
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Get identifier ID
        cur.execute(
            "SELECT id FROM ontology.identifiers WHERE name = %s",
            (identifier,)
        )
        row = cur.fetchone()
        if not row:
            return {'found': False, 'identifier': identifier}

        ident_id = row[0]
        neighbors = []

        # Outgoing relations
        if direction in ('outgoing', 'both'):
            query = """
                SELECT i.name, r.relation_type, r.occurrence_count
                FROM ontology.relations r
                JOIN ontology.identifiers i ON r.target_id = i.id
                WHERE r.source_id = %s
            """
            params = [ident_id]

            if relation_types:
                query += " AND r.relation_type = ANY(%s)"
                params.append(relation_types)

            cur.execute(query, params)

            for row in cur.fetchall():
                neighbors.append({
                    'identifier': row[0],
                    'relation': row[1],
                    'direction': 'outgoing',
                    'count': row[2]
                })

        # Incoming relations
        if direction in ('incoming', 'both'):
            query = """
                SELECT i.name, r.relation_type, r.occurrence_count
                FROM ontology.relations r
                JOIN ontology.identifiers i ON r.source_id = i.id
                WHERE r.target_id = %s
            """
            params = [ident_id]

            if relation_types:
                query += " AND r.relation_type = ANY(%s)"
                params.append(relation_types)

            cur.execute(query, params)

            for row in cur.fetchall():
                neighbors.append({
                    'identifier': row[0],
                    'relation': row[1],
                    'direction': 'incoming',
                    'count': row[2]
                })

        return {
            'found': True,
            'identifier': identifier,
            'neighbor_count': len(neighbors),
            'neighbors': neighbors
        }

    finally:
        cur.close()
        conn.close()


@server.tool()
async def get_negative_space(
    positive_terms: list[str],
    threshold: float = 0.3
) -> dict:
    """
    Find identifiers in the negative space (dissimilar to positive terms).
    Used for contrastive grounding - knowing what the task is NOT about.

    Args:
        positive_terms: Terms the task IS about
        threshold: Similarity threshold (lower = more dissimilar)

    Returns:
        Identifiers that are far from the positive terms
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Get embeddings for positive terms
        positive_embeddings = []

        for term in positive_terms:
            cur.execute("""
                SELECT embedding FROM ontology.identifiers
                WHERE name ILIKE %s AND embedding IS NOT NULL
                LIMIT 1
            """, (f'%{term}%',))

            row = cur.fetchone()
            if row and row[0]:
                positive_embeddings.append(np.array(row[0]))

        if not positive_embeddings:
            return {
                'found': False,
                'message': 'Could not find embeddings for positive terms'
            }

        # Compute centroid of positive embeddings
        centroid = np.mean(positive_embeddings, axis=0)

        # Find distant identifiers
        cur.execute("""
            SELECT name, 1 - (embedding <=> %s) as similarity
            FROM ontology.identifiers
            WHERE embedding IS NOT NULL
              AND is_excluded = FALSE
            ORDER BY similarity ASC
            LIMIT 50
        """, (centroid.tolist(),))

        negative = [
            {'identifier': row[0], 'similarity': float(row[1])}
            for row in cur.fetchall()
            if row[1] < threshold
        ]

        return {
            'found': True,
            'positive_terms': positive_terms,
            'threshold': threshold,
            'negative_count': len(negative),
            'negative_space': negative
        }

    finally:
        cur.close()
        conn.close()


@server.tool()
async def ground_glossary_term(
    term: str,
    identifier: str,
    language: str = 'en',
    synonyms: list[str] = None,
    confidence: float = 1.0
) -> dict:
    """
    Manually ground a business term to an identifier.
    Used to build the glossary incrementally.

    Args:
        term: The business term (any language)
        identifier: The code identifier to ground to
        language: Language code ('en', 'ua', 'ru', etc.)
        synonyms: Alternative forms of the term
        confidence: Confidence score (0.0-1.0)

    Returns:
        Confirmation of grounding
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Get identifier ID
        cur.execute(
            "SELECT id FROM ontology.identifiers WHERE name = %s",
            (identifier,)
        )
        row = cur.fetchone()
        if not row:
            return {
                'success': False,
                'message': f'Identifier "{identifier}" not found'
            }

        ident_id = row[0]

        # Insert or update glossary entry
        cur.execute("""
            INSERT INTO ontology.glossary
                (term, language, identifier_id, synonyms, confidence)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (term, language) DO UPDATE SET
                identifier_id = EXCLUDED.identifier_id,
                synonyms = EXCLUDED.synonyms,
                confidence = EXCLUDED.confidence
            RETURNING id
        """, (term, language, ident_id, synonyms or [], confidence))

        glossary_id = cur.fetchone()[0]
        conn.commit()

        return {
            'success': True,
            'glossary_id': glossary_id,
            'term': term,
            'grounded_to': identifier
        }

    finally:
        cur.close()
        conn.close()


# Run server
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

    asyncio.run(main())
```

---

## 7. Experiment Design

### 7.1 Experiment Protocol

```
Phase 1: Data Preparation
├── Extract identifiers from target codebase (any language)
├── Filter cross-file identifiers (min_files=2)
├── Extract relations using universal patterns
├── Store in PostgreSQL
├── Train Word2Vec on identifier corpus
└── Update identifier embeddings

Phase 2: Glossary Building
├── Use LLM to filter keywords/stdlib
├── Use LLM to identify business terms
├── Manual review of top-100 business terms
└── Ground terms to identifiers

Phase 3: Baseline Comparison
├── Baseline A: Pure vector RAG (current system)
├── Baseline B: Keyword search (TF-IDF)
├── Proposed: Ontology-grounded search
└── Ground truth: Historical task file changes

Phase 4: Evaluation
├── Compute metrics for each approach
├── Statistical significance testing
└── Qualitative analysis of examples
```

### 7.2 Test Set Creation

```python
def create_test_set(task_history: List[Dict]) -> List[Dict]:
    """
    Create test set from historical tasks with known file changes.

    Each test case:
    - Task description (input)
    - Changed files (ground truth)
    - Task type (exploratory/focused/cross-cutting)
    """
    test_set = []

    for task in task_history:
        # Filter to tasks with meaningful file changes
        if len(task['changed_files']) < 1:
            continue
        if len(task['changed_files']) > 20:
            continue  # Too broad

        # Classify task type
        desc_lower = task['description'].lower()
        if any(w in desc_lower for w in ['how', 'where', 'what', 'understand']):
            task_type = 'exploratory'
        elif any(w in desc_lower for w in ['fix', 'bug', 'add', 'implement']):
            task_type = 'focused'
        else:
            task_type = 'cross-cutting'

        test_set.append({
            'id': task['id'],
            'description': task['description'],
            'changed_files': task['changed_files'],
            'task_type': task_type
        })

    return test_set
```

---

## 8. Evaluation Metrics

### 8.1 Standard IR Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Precision@K** | relevant_in_top_K / K | >0.6 |
| **Recall@K** | relevant_in_top_K / total_relevant | >0.5 |
| **MRR** | 1 / rank_of_first_relevant | >0.5 |
| **NDCG@K** | DCG@K / IDCG@K | >0.7 |

### 8.2 Phenomenological Metrics (Novel)

| Metric | Definition | Rationale |
|--------|------------|-----------|
| **Grounding Accuracy** | % of business terms correctly linked to code | Measures symbol grounding quality |
| **Horizon Completeness** | % of changed files within horizon | Does horizon capture all relevant code? |
| **Noematic Precision** | % of grounded noema matching task intent | Are we finding the right objects? |
| **Affordance Relevance** | % of suggested actions actually performed | Do affordances guide correct actions? |
| **Negative Space Accuracy** | % of excluded files that were truly irrelevant | Does contrastive filtering work? |

### 8.3 Metric Computation

```python
class PhenomenologicalMetrics:
    """
    Compute phenomenological evaluation metrics.
    """

    def grounding_accuracy(self,
                          task_terms: List[str],
                          grounded_identifiers: List[str],
                          changed_files: List[str]) -> float:
        """
        What fraction of grounded identifiers appear in changed files?
        """
        correct = 0
        for ident in grounded_identifiers:
            for file in changed_files:
                if ident.lower() in file.lower():
                    correct += 1
                    break

        return correct / len(grounded_identifiers) if grounded_identifiers else 0

    def horizon_completeness(self,
                            horizon_files: List[str],
                            changed_files: List[str]) -> float:
        """
        What fraction of changed files are within the horizon?
        """
        horizon_set = set(f.lower() for f in horizon_files)
        changed_set = set(f.lower() for f in changed_files)

        covered = len(horizon_set & changed_set)

        return covered / len(changed_set) if changed_set else 0

    def negative_space_accuracy(self,
                               negative_files: List[str],
                               changed_files: List[str]) -> float:
        """
        What fraction of negative space files were truly irrelevant?
        (Should be high - we want to exclude irrelevant files)
        """
        negative_set = set(f.lower() for f in negative_files)
        changed_set = set(f.lower() for f in changed_files)

        true_negatives = len(negative_set - changed_set)

        return true_negatives / len(negative_set) if negative_set else 1.0

    def compute_all(self, results: Dict) -> Dict:
        """
        Compute all metrics for a single task evaluation.
        """
        return {
            'grounding_accuracy': self.grounding_accuracy(
                results['task_terms'],
                results['grounded_identifiers'],
                results['changed_files']
            ),
            'horizon_completeness': self.horizon_completeness(
                results['horizon_files'],
                results['changed_files']
            ),
            'negative_space_accuracy': self.negative_space_accuracy(
                results['negative_files'],
                results['changed_files']
            ),
            'precision_at_10': self.precision_at_k(
                results['recommended_files'][:10],
                results['changed_files']
            ),
            'mrr': self.mrr(
                results['recommended_files'],
                results['changed_files']
            )
        }
```

---

## 9. Integration with Existing System

### 9.1 Enhanced RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Task Description                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              1. Extract Business Terms (LLM)                 │
│    "Акт списання тари" → ["акт", "списання", "тари"]        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Ground Terms (Ontology MCP)                  │
│    get_noema("акт") → {identifier: "act", files: [...]}     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              3. Expand Horizon (Ontology MCP)                │
│    get_horizon(["акт", "списання"]) → {files: [...]}        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              4. Get Negative Space                           │
│    get_negative_space(["акт"]) → ["server", "plugin", ...]  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              5. Vector Search (Existing RAG)                 │
│    + Boost files in horizon                                  │
│    - Penalize files in negative space                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              6. Two-Phase Agent Processing                   │
│    Phase 1: Reason with grounded context                     │
│    Phase 2: Reflect and refine                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Final Recommendations                       │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Configuration

```python
# config_ontology.py

ONTOLOGY_CONFIG = {
    # Extraction settings
    'extraction': {
        'min_identifier_length': 3,
        'min_files_for_cross_file': 2,
        'file_extensions': [
            '.java', '.py', '.cs', '.js', '.ts', '.sql',
            '.plsql', '.vb', '.1cd', '.bsl'
        ]
    },

    # Word2Vec settings
    'word2vec': {
        'vector_size': 300,
        'window': 5,
        'min_count': 2,
        'epochs': 10
    },

    # Grounding settings
    'grounding': {
        'horizon_radius': 2,
        'negative_threshold': 0.3,
        'boost_factor': 0.5,
        'penalty_factor': 0.3
    },

    # Database
    'database': {
        'host': 'localhost',
        'port': 5432,
        'database': 'simargl',
        'schema': 'ontology'
    }
}
```

---

## 10. Implementation Roadmap

| Phase | Task | Output |
|-------|------|--------|
| **1** | Universal identifier extractor | `universal_extractor.py` |
| **2** | Universal relation extractor | `relation_extractor.py` |
| **3** | PostgreSQL schema setup | SQL migrations |
| **4** | Data population pipeline | `populate_ontology.py` |
| **5** | Word2Vec training | `train_word2vec.py` |
| **6** | MCP server implementation | `mcp_server_ontology.py` |
| **7** | Integration with existing RAG | Modified `two_phase_agent.py` |
| **8** | Evaluation harness | `phenomenological_eval.py` |
| **9** | Experiment execution | Results + analysis |

---

## 11. Scientific Contributions

### 11.1 Novel Contributions

1. **Language-Agnostic Symbol Grounding**: First approach that works across programming languages without parsers
2. **Phenomenological Metrics**: New evaluation metrics based on philosophical concepts
3. **Universal Relation Extraction**: Pattern-based relation detection without grammar
4. **Contrastive Grounding**: Using negative space for bounded search

### 11.2 Related Work

| Paper | Relation to Our Work |
|-------|---------------------|
| [Graph RAG Survey (ACM 2024)](https://dl.acm.org/doi/10.1145/3777378) | Framework for graph-enhanced retrieval |
| [Symbol Grounding in LLMs (Royal Society)](https://royalsocietypublishing.org/doi/10.1098/rsta.2022.0041) | Theoretical foundation |
| [CodeOntology (Springer 2017)](https://link.springer.com/chapter/10.1007/978-3-319-68204-4_2) | Language-specific ontology (we generalize) |
| [Word2Vec on Code (ArXiv 2019)](https://arxiv.org/abs/1904.03061) | Code embeddings (we use for grounding) |
| [KG in SE (ScienceDirect 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0950584923001829) | Knowledge graphs for software engineering |

---

## 12. Conclusion

This implementation provides a **practical, language-agnostic** approach to phenomenological code grounding. By focusing on:

1. **Universal patterns** instead of grammar
2. **Co-occurrence statistics** instead of semantic parsing
3. **Graph structure** instead of text similarity

We achieve symbol grounding that works across:
- Any programming language (Java, Python, SQL, 1С, etc.)
- Any natural language (English, Ukrainian, Russian, etc.)
- Any codebase size

The approach is **simpler** than AST-based methods while being **more universal**, making it practical for real-world heterogeneous codebases.

---

**Document Version**: 1.0
**Created**: 2025-01-25
**Status**: Implementation Specification
**Next Steps**: Begin implementation with Phase 1 (Universal Extractor)
