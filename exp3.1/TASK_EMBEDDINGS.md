# Task Embeddings Collection

## Overview

The task embeddings collection stores individual task embeddings (title + description) for each task in your database. This is different from module/file embeddings which are aggregated centroids.

## Purpose

Task embeddings enable:

1. **Task-to-Task Similarity Search**: Find similar historical tasks
2. **Module Recreation**: Rebuild module embeddings from historical task data
3. **Historical Analysis**: See what similar problems were solved before
4. **Learning from History**: Discover how past tasks were addressed

## Creating Task Embeddings

### Option 1: Using Current Backend (Config)

**Windows:**
```cmd
cd exp3
create_tasks.bat
```

**Linux:**
```bash
cd exp3
chmod +x create_tasks.sh
./create_tasks.sh
```

This uses the backend specified in `config.py` (VECTOR_BACKEND).

### Option 2: Explicitly Choose Backend

**For Qdrant:**
```bash
python create_task_collection.py --backend qdrant
```

**For PostgreSQL:**
```bash
# Windows
create_tasks_postgres.bat

# Linux
./create_tasks_postgres.sh

# Or directly
python create_task_collection.py --backend postgres
```

### With Different Model

```bash
python create_task_collection.py --model bge-large --backend postgres
```

## What Gets Created

### Qdrant Collection

- **Name**: `task_embeddings_all_bge-small`
- **Type**: Qdrant collection
- **Vectors**: ~9,799 individual task embeddings (depending on your database)
- **Payload**:
  - `task_id`: Task ID
  - `task_name`: Task name
  - `title`: Task title (truncated to 500 chars)
  - `description`: Task description (truncated to 500 chars)
  - `has_comments`: Boolean indicating if task has comments

### PostgreSQL Table

- **Table**: `vectors.task_embeddings_all_bge_small`
- **Columns**:
  - `id`: Auto-increment primary key
  - `path`: Task ID (stored as text)
  - `type`: Always 'task'
  - `vector`: 384-dimensional vector (bge-small)
  - `metadata`: JSONB with task_name, title, description, has_comments
- **Indexes**:
  - HNSW index on vector column for fast similarity search
  - B-tree index on path (task ID) for lookups

## Usage

### 1. Search Similar Tasks in Gradio UI

1. Launch Gradio UI:
   ```bash
   cd ragmcp
   python gradio_ui.py
   ```

2. Go to the **"Task Search"** tab

3. Enter a task description

4. View similar historical tasks with similarity scores

### 2. Programmatic Search (Python)

```python
from sentence_transformers import SentenceTransformer
from vector_backends import get_vector_backend
import config

# Initialize
backend = get_vector_backend(config.VECTOR_BACKEND)
backend.connect()
model = SentenceTransformer(config.EMBEDDING_MODEL)

# Search for similar tasks
query = "Fix memory leak in connection pool"
query_vector = model.encode(query)

results = backend.search(
    collection_name=config.COLLECTION_TASK,
    query_vector=query_vector,
    top_k=10
)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. Task {result['path']}: {result.get('task_name', 'Unknown')}")
    print(f"   Similarity: {result['score']:.4f}")
    print(f"   Title: {result.get('title', '')}")
    print(f"   Description: {result.get('description', '')[:200]}...")
    print()
```

### 3. Module Recreation from Tasks

You can use task embeddings to recreate module embeddings dynamically:

```python
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_backends import get_vector_backend
import config

# 1. Load task-to-module mapping from database
conn = sqlite3.connect(config.DB_PATH)
query = """
SELECT t.ID, t.NAME, r.PATH
FROM TASK t
JOIN RAWDATA r ON t.NAME = r.TASK_NAME
WHERE r.PATH LIKE 'somemodule/%'
"""
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Get task embeddings
backend = get_vector_backend(config.VECTOR_BACKEND)
backend.connect()

task_vectors = []
for task_id in df['ID'].unique():
    # Fetch task vector
    # (This requires querying by ID - implementation depends on backend)
    pass

# 3. Compute module centroid
module_vector = np.mean(task_vectors, axis=0)

# 4. Use module vector for similarity search
```

## Configuration

### Add to config.py

Both `exp3/config.py` and `ragmcp/config.py` should have:

```python
# Task Embeddings Collection Name
COLLECTION_TASK = 'task_embeddings_all_bge-small'
```

For different models:
```python
COLLECTION_TASK = 'task_embeddings_all_bge-large'  # if using bge-large
```

## Comparison: Task vs Module/File Embeddings

| Feature | Task Embeddings | Module/File Embeddings |
|---------|----------------|----------------------|
| **Granularity** | Individual tasks | Aggregated by module/file |
| **Count** | ~9,799 (one per task) | 64 modules / 63,069 files |
| **Vector** | Direct from task text | Centroid of related tasks |
| **Use Case** | Find similar tasks | Find relevant code locations |
| **Precision** | Exact task match | Approximate module match |
| **Speed** | Fast (fewer vectors) | Moderate (many file vectors) |
| **Purpose** | Historical analysis | Code navigation |

## Use Cases

### 1. Find Similar Historical Tasks

**Scenario**: You have a new task "Fix SSL certificate validation error"

**Action**: Search task collection

**Result**: Discover tasks like:
- Task #5234: "SSL handshake fails with self-signed certificates"
- Task #6789: "Certificate validation error in HTTPS connections"
- Task #7123: "Fix SSL/TLS configuration for secure connections"

**Benefit**: See how similar problems were solved before

### 2. Module Recreation

**Scenario**: Module embeddings need to be rebuilt for updated tasks

**Action**:
1. Get all tasks for a module
2. Fetch their task embeddings
3. Compute new centroid
4. Update module collection

**Benefit**: Keep module embeddings up-to-date without recomputing from scratch

### 3. Task Clustering

**Scenario**: Analyze project history to find patterns

**Action**: Run clustering on task embeddings

**Result**: Discover:
- Common bug categories
- Feature themes
- Maintenance patterns

**Benefit**: Better project planning and resource allocation

### 4. Quality Analysis

**Scenario**: Compare task descriptions quality

**Action**: Analyze embedding similarity within clusters

**Result**: Identify:
- Well-described vs poorly-described tasks
- Duplicate or redundant tasks
- Task description patterns

**Benefit**: Improve task documentation standards

## Performance

### Creation Time

- **CPU (8 cores)**: 10-15 minutes for ~10,000 tasks
- **GPU (P106-100)**: 3-5 minutes for ~10,000 tasks

### Storage Size

- **Qdrant**: ~40 MB for 10,000 tasks (384-dim vectors)
- **PostgreSQL**: ~50 MB (including indexes and metadata)

### Search Speed

- **Qdrant**: < 50ms for top-10 search
- **PostgreSQL + HNSW**: < 100ms for top-10 search

## Maintenance

### Update Task Embeddings

When new tasks are added:

```bash
# Recreate entire collection (recommended)
python create_task_collection.py --backend qdrant

# Or for PostgreSQL
python create_task_collection.py --backend postgres
```

The script will delete the existing collection and recreate it.

### Incremental Updates

For incremental updates (add only new tasks), you'll need to:

1. Query database for new tasks since last update
2. Generate embeddings for new tasks only
3. Upsert to collection (without deleting)

This is not implemented in the default script but can be customized.

## Troubleshooting

### Collection Not Found

**Error**: `task_embeddings_all_bge-small` not found

**Solution**:
```bash
cd exp3
python create_task_collection.py
```

### Metadata Missing in Results

**Qdrant**: Make sure payload is being returned (should be automatic)

**PostgreSQL**: Verify metadata column exists:
```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'task_embeddings_all_bge_small';
```

If missing, recreate the collection.

### Slow Search

**Qdrant**: Increase memory limit in qdrant-compose.yml

**PostgreSQL**:
1. Verify HNSW index exists:
   ```sql
   \d vectors.task_embeddings_all_bge_small
   ```

2. Rebuild index if needed:
   ```sql
   REINDEX INDEX vectors.task_embeddings_all_bge_small_vector_idx;
   ```

3. Analyze table:
   ```sql
   ANALYZE vectors.task_embeddings_all_bge_small;
   ```

## Advanced Usage

### Custom Similarity Threshold

```python
results = backend.search(collection_name, query_vector, top_k=50)

# Filter by similarity threshold
filtered = [r for r in results if r['score'] >= 0.7]
```

### Batch Search

```python
queries = [
    "Fix memory leak",
    "Add authentication",
    "Improve performance"
]

for query in queries:
    vector = model.encode(query)
    results = backend.search(collection_name, vector, top_k=5)
    print(f"\nResults for '{query}':")
    for r in results:
        print(f"  - {r['path']}: {r['score']:.4f}")
```

### Export Task Vectors

For external analysis (e.g., t-SNE visualization):

```python
# Get all task vectors (Qdrant example)
from qdrant_client import QdrantClient

client = QdrantClient(host='localhost', port=6333)
collection = client.scroll(
    collection_name='task_embeddings_all_bge-small',
    limit=10000,
    with_payload=True,
    with_vectors=True
)

# Extract vectors and metadata
task_ids = []
vectors = []
titles = []

for point in collection[0]:
    task_ids.append(point.payload['task_id'])
    vectors.append(point.vector)
    titles.append(point.payload['title'])

# Save to numpy array
import numpy as np
np.save('task_vectors.npy', np.array(vectors))
```

## Integration with ragmcp

The task search is already integrated in Gradio UI:

1. **Tab**: "Task Search"
2. **Function**: `search_tasks()`
3. **Config**: Uses `COLLECTION_TASK` from config.py

No additional setup needed beyond creating the collection.

---

**Last Updated**: 2025-12-22
**Version**: 1.0
