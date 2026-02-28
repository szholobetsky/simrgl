# Experiment 4: Embedding Architecture Families for Task-to-Code Retrieval

> Continuation of exp3. Goal: compare three embedding architecture families across three project datasets to understand which paradigm best predicts relevant code modules from Jira task descriptions.

---

## Research Goal

**RQ: Does embedding architecture family matter for task-to-code retrieval, and does the answer generalise across different codebases?**

Previous experiments (exp2, exp3) established that sentence-transformer bi-encoders (bge-large) outperform classical approaches (FastText, GloVe, BERT). Exp4 investigates whether newer architectural paradigms — ModernBERT and LLM-based embeddings — push the ceiling further, and whether this finding holds across multiple real-world projects.

---

## Architecture Families Under Comparison

### Family 0: Classical Transformers / Sentence-Transformers ✅ DONE (exp2 + exp3)
> Bi-encoder: one fixed-size vector per document, cosine similarity.

Models already evaluated:
- Word2Vec, FastText, GloVe (exp2)
- BERT-base, RoBERTa-base, CodeBERT (exp2)
- MPNet, MS-MARCO-distilbert (exp2)
- **BGE-large-en-v1.5** — best result, MAP=0.371 (Sonar), MAP=0.361 (Flask) in exp2
- **BGE-small-en-v1.5, BGE-m3** (exp3, Sonar only)

Reference baseline for exp4: **bge-large-en-v1.5** (MAP=0.371 Sonar / MAP=0.361 Flask).

---

### Family 1: ModernBERT Family 🆕
> Encoder-only architecture rebuilt from scratch with modern techniques: RoPE positional encoding, Flash Attention, alternating local/global attention, 8192-token context window. Trained on code data alongside text.

Key advantage over Family 0: longer context (8192 vs 512 tokens) captures full Jira descriptions + architecture improvements improve semantic quality.

**Proposed models:**

| Model | HF ID | Dims | Max Tokens | Params | Notes |
|-------|-------|------|-----------|--------|-------|
| ModernBERT-embed-base | `nomic-ai/modernbert-embed-base` | 768 | 8192 | 149M | RAG-tuned, Matryoshka |
| ModernBERT-embed-large | `nomic-ai/modernbert-embed-large` | 1024 | 8192 | 395M | Larger variant |
| ColBERT v2 | `colbert-ir/colbertv2.0` | 128/tok | 512 | 110M | Late interaction |
| answerai-colbert-small | `answerdotai/answerai-colbert-small-v1` | 96/tok | 512 | 33M | Compact ColBERT |

**ColBERT note**: ColBERT uses **late interaction** — it stores one vector per token rather than one vector per document. At query time it computes MaxSim between query tokens and document tokens. This is fundamentally different from centroid-based aggregation and is expected to handle exact code term matches (module names, file names) better.

> **Aggregation strategy does NOT apply to ColBERT** — it has its own MaxSim scoring. This is a pipeline difference that needs to be reflected in the implementation.

---

### Family 2: LLM-Based Family 🆕
> Decoder-only large language models repurposed as bi-encoders. Much larger parameter count, richer semantic representations. Supports **instruction-tuned retrieval** — you can prefix the query with a task description to guide the embedding.

Key advantage: instruction prefix allows the model to understand retrieval intent:
```
Query prefix:    "Represent this Jira task for retrieving relevant code modules: {text}"
Document prefix: "Represent this code module based on historical task descriptions: {text}"
```

**Proposed models (via sentence-transformers + Ollama for 8B):**

| Model | HF ID | Dims | Max Tokens | Params | Memory | MTEB Retrieval |
|-------|-------|------|-----------|--------|--------|----------------|
| Qwen3-Embedding-0.6B | `Qwen/Qwen3-Embedding-0.6B` | 1024 | 32768 | 0.6B | 1.1GB | 64.65 |
| Qwen3-Embedding-4B | `Qwen/Qwen3-Embedding-4B` | 2560 | 32768 | 4B | 7.7GB | 69.60 |
| Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | 4096 | 32768 | 8B | ~4-5GB Q4 | 70.88 |
| jina-embeddings-v5-text-small | `jinaai/jina-embeddings-v5-text-small` | 1024 | 32768 | 0.6B | 1.1GB | 64.88 |
| multilingual-e5-large-instruct | `intfloat/multilingual-e5-large-instruct` | 1024 | 514 | 0.56B | 1.1GB | 57.12 |
| gte-Qwen2-1.5B-instruct | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 8960 | 32768 | 1.5B | 6.8GB | 60.78 |

> **Qwen3-Embedding-8B via Ollama**: Run `ollama pull qwen3-embedding:8b` and call the `/api/embeddings` endpoint. Ollama applies 4-bit quantization (~4-5GB VRAM), making 8B models feasible on 6GB GPU. Requires a small adapter in the ETL pipeline.

---

## What We Already Know — Fixed Variables for exp4

Based on exp2 and exp3 analysis, the following dimensions are **settled and should NOT be re-investigated**:

### ✅ Aggregation Strategy — IGNORE (except ColBERT)

From exp2, MAP values are **identical across all strategies** (avg, sum, median, weighted_avg, cluster) for every model tested (FastText, GloVe, BERT, RoBERTa, CodeBERT, MPNet, BGE-large). Only Word2Vec showed minor variance (~0.01).

**Conclusion**: For standard bi-encoders, centroid aggregation strategy does not affect the result. Use `avg` as default for all Family 0 and Family 2 models.

For **ColBERT (Family 1)**: aggregation strategy is irrelevant by design — ColBERT's MaxSim replaces the entire concept.

### ✅ Source Variant (title / desc / comments)

From exp3: `desc` (title + description) consistently outperforms `title` alone. `comments` adds noise and decreases performance.

**Conclusion**: Use `desc` only for all exp4 runs.

### ✅ Window Size (w100 / w1000 / all)

From exp3:
- `w100` is best for `recent` split (high MAP due to temporal proximity)
- `w1000` is best for `modn` split
- `all` is consistently worse

**Conclusion**: The temporal window effect is understood. Fix to **`w1000`** for exp4 as it gives the most stable and representative results.

### ⚠️ Split Strategy (recent vs modn) — FIX TO ONE

From exp3, `recent` gives inflated MAP for module retrieval (~0.80) because the test tasks are temporally close to training tasks and tend to touch the same modules. `modn` (uniform sampling) gives a harder, more representative evaluation (~0.45).

**Conclusion**: Use **`modn`** split as the primary evaluation. It is harder, more representative, and avoids the temporal proximity inflation. Report `recent` optionally for comparison with exp3 numbers.

### ✅ Granularity (file vs module)

From exp3: module-level retrieval consistently achieves higher MAP than file-level. This was the original research question (RQ1) and is answered.

**Conclusion**: Report **both** file and module, but focus analysis on module-level as primary metric.

---

## Multi-Dataset Evaluation ⭐ NEW IN EXP4

**This is a critical addition.** All previous experiments used only the **SonarQube** dataset. Results from a single project may reflect dataset-specific characteristics rather than general model quality.

Exp4 will run all model families on **three datasets**:

| Dataset | Project | Domain | Size (tasks) | Status |
|---------|---------|--------|-------------|--------|
| **Sonar** | SonarQube | Java static analysis tool | ~9,799 | ✅ Available (`sonar.db`) |
| **Kafka** | Apache Kafka | Distributed messaging | TBD | ✅ Available |
| **Spark** | Apache Spark | Big data processing | TBD | ✅ Available |

**Why this matters**:
- SonarQube: large enterprise Java codebase with many modules
- Kafka: distributed systems, different module structure and vocabulary
- Spark: data engineering domain, different task description style

A model family that wins on all three datasets provides much stronger evidence than a single-project result. Cross-project consistency is a key contribution for research credibility.

---

## Experiment Matrix

Fixed parameters: `source=desc`, `window=w1000`, `split=modn`, `target=module+file`

| ID | Family | Model | Dataset |
|----|--------|-------|---------|
| F0-BL | Sentence-Transformer | bge-large-en-v1.5 (baseline, from exp3) | Sonar |
| F1-MB-base | ModernBERT | nomic-ai/modernbert-embed-base | Sonar, Kafka, Spark |
| F1-MB-large | ModernBERT | nomic-ai/modernbert-embed-large | Sonar, Kafka, Spark |
| F1-CB | ColBERT | colbert-ir/colbertv2.0 | Sonar, Kafka, Spark |
| F2-Q3-0.6 | LLM-based | Qwen3-Embedding-0.6B | Sonar, Kafka, Spark |
| F2-Q3-4B | LLM-based | Qwen3-Embedding-4B | Sonar, Kafka, Spark |
| F2-Q3-8B | LLM-based | Qwen3-Embedding-8B (Ollama) | Sonar, Kafka, Spark |
| F2-Jina | LLM-based | jina-embeddings-v5-text-small | Sonar, Kafka, Spark |
| F2-e5-inst | LLM-based | multilingual-e5-large-instruct | Sonar, Kafka, Spark |

Total: ~25 experiment configurations (some Family 0 baselines reuse exp3 Sonar results).

---

## Setup

### Code Base

Copy exp3 as starting point:

```bash
cp -r exp3/ exp4/
cd exp4/
```

Key files to adapt:
- `config.py` — add new model definitions, fix source/window/split defaults
- `etl_pipeline.py` — add Ollama embedding adapter for Family 2 (8B models)
- `run_experiments.py` — add ColBERT MaxSim scoring for Family 1
- `experiment_ui.py` — extend results dashboard for multi-dataset view

### New Dependencies

```bash
pip install colbert-ai          # ColBERT late interaction
pip install ollama              # Ollama Python client (for 8B models)
pip install modernbert          # if not already via transformers
```

### Datasets

```
exp4/data/
├── sonar.db     # SonarQube (existing)
├── kafka.db     # Apache Kafka (new)
└── spark.db     # Apache Spark (new)
```

---

## Expected Outcomes

| Family | Hypothesis | Rationale |
|--------|-----------|-----------|
| ModernBERT | Beats bge-large by 5-15% MAP | 8192 token window captures full descriptions; better architecture |
| ColBERT | Best file-level precision | Token-level MaxSim captures exact module/file name matches |
| Qwen3-0.6B | Beats bge-large with same VRAM | 75% better MTEB Retrieval score at identical memory footprint |
| Qwen3-8B | Best overall MAP | Largest model, richest semantic understanding, instruction-tuned |
| Cross-project | Rankings consistent across datasets | If model quality generalises, ranking should be stable across Sonar/Kafka/Spark |

---

## Key Architectural Differences Summary

```
Family 0 — Sentence-Transformer (bi-encoder)
  Task description → [BERT encoder] → single vector
  Code module      → [BERT encoder] → single vector (centroid of task embeddings)
  Score: cosine(query_vec, doc_vec)

Family 1 — ColBERT (late interaction)
  Task description → [BERT encoder] → N token vectors
  Code module      → [BERT encoder] → M token vectors
  Score: sum over query tokens of max(cosine with all doc tokens)  ← MaxSim

Family 2 — LLM-based (decoder as encoder)
  "Instruct: {task}\n{description}" → [Qwen3-8B] → single vector
  Code module                       → [Qwen3-8B] → single vector
  Score: cosine(query_vec, doc_vec)
  + Instruction prefix guides the embedding toward retrieval semantics
```

---

## References

- ModernBERT: [Introducing ModernBERT](https://huggingface.co/blog/modernbert) — Answer.AI, December 2024
- ColBERT: [Late Interaction Overview](https://weaviate.io/blog/late-interaction-overview) — Weaviate, 2025
- LLM2Vec: [LLM2Vec paper](https://arxiv.org/abs/2404.05961) — McGill NLP, 2024
- Qwen3-Embedding: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) — MTEB Rank #3, 2025
- MTEB Leaderboard: [mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Exp2 results: `../info/exp2_result/` (Sonar + Flask datasets)
- Exp3 results: `../info/LAST/` (Sonar, bge-small/large/m3)

---

## Implementation Notes for Coding Session

> This section is written for the developer (or AI assistant) implementing exp4 code.
> Read this before touching any file.

### Step 0: Starting Point

Copy exp3 as the base:
```bash
cp -r ../exp3/* .
```

Files to **keep unchanged**: `utils.py`, `checkpoint_manager.py`, `gpu_utils.py`, `experiment_ui.py` (extend later), `backup_restore_qdrant.py`, `docker-compose.yml`.

Files to **modify**: `config.py`, `etl_pipeline.py`, `run_experiments.py`, `vector_backends.py`, `run_comprehensive_experiments.py`.

Files to **create new**: `embedders.py` (embedding backend abstraction), `colbert_pipeline.py` (ColBERT-specific ETL + search).

---

### Step 1: config.py Changes

The current `config.py` has `EMBEDDING_MODELS` as a flat dict and `DB_PATH` as a single string. Exp4 needs:

**1a. Add model family field to every model entry:**
```python
EMBEDDING_MODELS = {
    # --- Family 0: Sentence-Transformers baseline (from exp3) ---
    'bge-large': {
        'name': 'BAAI/bge-large-en-v1.5',
        'dim': 1024,
        'family': 'sentence_transformer',
        'batch_size': 32,
        'trust_remote_code': False,
        'instruction_query': None,   # no instruction prefix
        'instruction_doc': None,
    },
    # --- Family 1: ModernBERT ---
    'modernbert-base': {
        'name': 'nomic-ai/modernbert-embed-base',
        'dim': 768,
        'family': 'modernbert',
        'batch_size': 32,
        'trust_remote_code': True,
        'instruction_query': 'search_query: ',   # nomic prompt prefix
        'instruction_doc':   'search_document: ',
    },
    'modernbert-large': {
        'name': 'nomic-ai/modernbert-embed-large',
        'dim': 1024,
        'family': 'modernbert',
        'batch_size': 16,
        'trust_remote_code': True,
        'instruction_query': 'search_query: ',
        'instruction_doc':   'search_document: ',
    },
    # --- Family 1: ColBERT ---
    'colbert': {
        'name': 'colbert-ir/colbertv2.0',
        'dim': 128,              # per-token dim (not single vector)
        'family': 'colbert',     # triggers separate pipeline
        'batch_size': 32,
        'trust_remote_code': False,
        'instruction_query': None,
        'instruction_doc': None,
    },
    # --- Family 2: LLM-based ---
    'qwen3-0.6b': {
        'name': 'Qwen/Qwen3-Embedding-0.6B',
        'dim': 1024,
        'family': 'llm',
        'batch_size': 16,
        'trust_remote_code': True,
        'instruction_query': 'query',    # prompt_name for SentenceTransformer
        'instruction_doc':   'passage',
    },
    'qwen3-4b': {
        'name': 'Qwen/Qwen3-Embedding-4B',
        'dim': 2560,
        'family': 'llm',
        'batch_size': 4,
        'trust_remote_code': True,
        'instruction_query': 'query',
        'instruction_doc':   'passage',
    },
    'qwen3-8b-ollama': {
        'name': 'qwen3-embedding:8b',   # Ollama model name
        'dim': 4096,
        'family': 'ollama',             # triggers Ollama backend
        'batch_size': 1,                # Ollama is called one-by-one via REST
        'trust_remote_code': False,
        'instruction_query': None,
        'instruction_doc': None,
    },
    'jina-v5-small': {
        'name': 'jinaai/jina-embeddings-v5-text-small',
        'dim': 1024,
        'family': 'llm',
        'batch_size': 16,
        'trust_remote_code': True,
        'instruction_query': None,
        'instruction_doc': None,
    },
    'e5-instruct': {
        'name': 'intfloat/multilingual-e5-large-instruct',
        'dim': 1024,
        'family': 'llm',
        'batch_size': 16,
        'trust_remote_code': False,
        # e5-instruct uses a text prefix, not prompt_name
        'instruction_query': 'Instruct: Retrieve relevant code modules for this software task\nQuery: ',
        'instruction_doc': None,   # documents don't use a prefix for e5
    },
}
```

**1b. Add multi-dataset config:**
```python
DATASETS = {
    'sonar': {
        'db_path': 'data/sonar.db',
        'description': 'SonarQube — Java static analysis tool',
    },
    'kafka': {
        'db_path': 'data/kafka.db',
        'description': 'Apache Kafka — distributed messaging',
    },
    'spark': {
        'db_path': 'data/spark.db',
        'description': 'Apache Spark — big data processing',
    },
}
```

**1c. Fix default parameters:**
```python
DEFAULT_SOURCE   = 'desc'    # was a variable before; now hardcoded
DEFAULT_WINDOW   = 'w1000'   # settled by exp3
DEFAULT_SPLIT    = 'modn'    # settled by exp3
BATCH_SIZE       = 32        # overridden per model by model_config['batch_size']
OLLAMA_HOST      = 'http://localhost:11434'
```

**1d. Output directory:**
```python
RESULTS_DIR = 'experiment_results'
# Output file will be: experiment_results/results_{model_key}_{dataset_key}.csv
# Combined file:       experiment_results/comprehensive_results.csv
# Schema must match exp3 comprehensive_results.csv plus columns: dataset, family
```

---

### Step 2: embedders.py (NEW FILE)

Create a unified embedding interface so `etl_pipeline.py` and `run_experiments.py` don't need to know which backend they're talking to.

```python
# embedders.py
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

class BaseEmbedder:
    def encode_queries(self, texts: list) -> np.ndarray: ...
    def encode_documents(self, texts: list) -> np.ndarray: ...

class SentenceTransformerEmbedder(BaseEmbedder):
    """Covers family=sentence_transformer, modernbert, llm (all via HuggingFace)."""

    def __init__(self, model_config: dict):
        self.cfg = model_config
        self.model = SentenceTransformer(
            model_config['name'],
            trust_remote_code=model_config.get('trust_remote_code', False)
        )
        self.batch_size = model_config.get('batch_size', 32)

    def _apply_instruction(self, texts: list, instruction: str) -> list:
        """
        Instruction handling differs by family:
        - modernbert: prepend string prefix ("search_query: {text}")
        - llm (Qwen3): use prompt_name kwarg, NOT string prefix
        - llm (e5-instruct): prepend string prefix
        See encode_queries for dispatch logic.
        """
        if instruction is None:
            return texts
        return [instruction + t for t in texts]

    def encode_queries(self, texts: list) -> np.ndarray:
        cfg = self.cfg
        instr = cfg.get('instruction_query')
        family = cfg.get('family')

        if family == 'llm' and instr in ('query', 'passage'):
            # Qwen3-style: uses prompt_name parameter
            return self.model.encode(texts, prompt_name=instr,
                                     batch_size=self.batch_size,
                                     normalize_embeddings=True)
        else:
            # modernbert / e5-instruct / sentence_transformer: string prefix
            texts = self._apply_instruction(texts, instr)
            return self.model.encode(texts, batch_size=self.batch_size,
                                     normalize_embeddings=True)

    def encode_documents(self, texts: list) -> np.ndarray:
        cfg = self.cfg
        instr = cfg.get('instruction_doc')
        family = cfg.get('family')

        if family == 'llm' and instr in ('query', 'passage'):
            return self.model.encode(texts, prompt_name=instr,
                                     batch_size=self.batch_size,
                                     normalize_embeddings=True)
        else:
            texts = self._apply_instruction(texts, instr)
            return self.model.encode(texts, batch_size=self.batch_size,
                                     normalize_embeddings=True)


class OllamaEmbedder(BaseEmbedder):
    """Covers family=ollama. Calls local Ollama REST API."""

    def __init__(self, model_config: dict, host: str = 'http://localhost:11434'):
        self.model_name = model_config['name']   # e.g. "qwen3-embedding:8b"
        self.host = host
        # batch_size ignored; Ollama is always called one text at a time

    def _embed_one(self, text: str) -> list:
        resp = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def encode_queries(self, texts: list) -> np.ndarray:
        return np.array([self._embed_one(t) for t in texts])

    def encode_documents(self, texts: list) -> np.ndarray:
        return np.array([self._embed_one(t) for t in texts])


def get_embedder(model_config: dict, ollama_host: str = 'http://localhost:11434') -> BaseEmbedder:
    """Factory — returns the right embedder based on model family."""
    family = model_config.get('family', 'sentence_transformer')
    if family == 'ollama':
        return OllamaEmbedder(model_config, host=ollama_host)
    else:
        # sentence_transformer, modernbert, llm — all via SentenceTransformer
        return SentenceTransformerEmbedder(model_config)
```

**Important**: ColBERT (`family='colbert'`) is NOT handled here — it has its own pipeline in `colbert_pipeline.py`.

---

### Step 3: etl_pipeline.py Changes

The existing `ETLPipeline.__init__` creates `self.model = SentenceTransformer(...)` directly. Change this to use the embedder factory.

**Changes needed:**
1. Add `dataset_key` parameter to `__init__` — sets `config.DB_PATH` from `config.DATASETS[dataset_key]['db_path']`
2. Replace `self.model = SentenceTransformer(...)` with `self.embedder = get_embedder(self.model_config)`
3. Replace all `self.model.encode(texts, ...)` calls with `self.embedder.encode_documents(texts)` for training task encoding
4. The `generate_embeddings()` method aggregates per-module — keep centroid (`np.mean`) as the only strategy (aggregation strategy is settled)
5. Use `BATCH_SIZE` from model config (`self.model_config['batch_size']`) not global config

Signature change:
```python
# OLD
def __init__(self, split_strategy='recent', test_size=None, model_key=None, backend_type=None):

# NEW
def __init__(self, split_strategy='modn', test_size=None, model_key=None,
             backend_type=None, dataset_key='sonar'):
    ...
    dataset_cfg = config.DATASETS[dataset_key]
    self.db_path = dataset_cfg['db_path']
    self.dataset_key = dataset_key
    self.embedder = get_embedder(self.model_config, config.OLLAMA_HOST)
```

---

### Step 4: run_experiments.py Changes

The `ExperimentRunner` encodes query texts with `self.model.encode(...)`. Change to use embedder.

```python
# OLD: query_vector = self.model.encode([query_text])[0]
# NEW: query_vector = self.embedder.encode_queries([query_text])[0]
```

Also add `dataset_key` parameter and load the correct test_set file:
```python
# test_set file is now named per dataset + split:
# e.g. test_set_modn_sonar.json, test_set_modn_kafka.json
test_set_file = f"test_set_{split_strategy}_{dataset_key}.json"
```

The output CSV must add two columns: `dataset` and `family`. Schema:
```
model, family, dataset, split_strategy, experiment_id, source, target, window,
MAP, MRR, P@1, R@1, P@3, R@3, P@5, R@5, P@10, R@10
```
This is backwards-compatible with exp3 (just two extra columns at the start).

---

### Step 5: colbert_pipeline.py (NEW FILE)

ColBERT is fundamentally different — it stores token-level vectors, not a single centroid per module. The retrieval strategy for this project should be:

**Index**: Each training task description → individual ColBERT document (with its module/file label stored as metadata).

**Query**: New task → find top-K most similar training tasks by MaxSim → aggregate scores by module (sum of top scores per module) → return ranked modules.

This mirrors how exp3 works conceptually (tasks → aggregate to modules) but uses MaxSim instead of cosine similarity on centroids.

**Library to use: `pylate`** (simpler than raw `colbert-ai`, integrates with HuggingFace):
```bash
pip install pylate
```

Skeleton structure:
```python
# colbert_pipeline.py
from pylate import models, indexes, retrieve
import numpy as np

class ColBERTPipeline:
    def __init__(self, model_key='colbert', dataset_key='sonar', split_strategy='modn'):
        self.model = models.ColBERT("colbert-ir/colbertv2.0")
        self.dataset_key = dataset_key
        self.split_strategy = split_strategy
        # Index stored on disk: colbert_index_{dataset_key}_{split_strategy}/
        self.index_path = f"colbert_index_{dataset_key}_{split_strategy}"

    def build_index(self, train_tasks_df, rawdata_df, target='module'):
        """
        Encode all training task texts with ColBERT.
        Store token embeddings + module/file labels.
        """
        # Build (task_text, module_label) pairs from training set
        # Use pylate to encode and store on disk
        ...

    def search(self, query_text: str, top_k: int = 10) -> list:
        """
        1. Encode query with ColBERT
        2. Find top-N similar tasks by MaxSim (N >> top_k, e.g. N=200)
        3. Aggregate task scores by module: score(module) = sum of top-3 task scores
        4. Return ranked module list
        """
        ...

    def evaluate(self, test_set: list, target='module', top_k=10) -> list:
        """Run search for all test tasks, return results in same format as ExperimentRunner."""
        ...
```

Note: ColBERT index building is slow but only done once per dataset+split. Results go into the same output CSV format as the standard pipeline.

---

### Step 6: run_comprehensive_experiments.py Changes

The existing `run_comprehensive_experiments.py` loops over `models × strategies`. For exp4 it should loop over `models × datasets`, with fixed `strategy=modn`, `source=desc`, `window=w1000`.

Key loop structure:
```python
for dataset_key in args.datasets:          # sonar, kafka, spark
    for model_key in args.models:          # all model keys
        if model_config['family'] == 'colbert':
            # run ColBERTPipeline separately
            run_colbert(model_key, dataset_key)
        else:
            # run standard ETL + ExperimentRunner
            run_standard(model_key, dataset_key, strategy='modn')
```

Checkpoint file should be keyed by `(model_key, dataset_key)` pair to allow resumption.

---

### Step 7: Known Issues & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| OOM on bge-m3 | BATCH_SIZE=32 too large for 6GB VRAM | Set `batch_size: 4` in model config |
| Qwen3 requires trust_remote_code | Non-standard pooling code | Set `trust_remote_code=True` in SentenceTransformer() |
| Qwen3 prompt_name not found | Old sentence-transformers version | `pip install -U sentence-transformers` (need ≥2.7) |
| Ollama model not found | Model not pulled | Run `ollama pull qwen3-embedding:8b` before ETL |
| Ollama timeout on long texts | Default timeout too short | Set `timeout=120` in requests.post() |
| pylate/colbert GPU OOM | ColBERT stores many token vectors | Use `model.encode(..., batch_size=8)` |
| Jina-v5 needs trust_remote_code | Custom pooling | Set `trust_remote_code=True` |
| ModernBERT prompt format | Uses prefix strings not prompt_name | Prepend `"search_query: "` / `"search_document: "` to text |
| e5-instruct prompt format | Uses text prefix not prompt_name | Prepend `"Instruct: ...\nQuery: "` to query texts only; documents have no prefix |

---

### Step 8: Validation Before Full Run

Run this quick sanity check before launching the full experiment:
```bash
# 1. Test standard embedder (Qwen3-0.6B)
python -c "
from embedders import get_embedder
from config import EMBEDDING_MODELS
e = get_embedder(EMBEDDING_MODELS['qwen3-0.6b'])
vecs = e.encode_queries(['Fix authentication bug in login module'])
print('Shape:', vecs.shape)  # expect (1, 1024)
"

# 2. Test Ollama embedder (must have Ollama running + model pulled)
python -c "
from embedders import get_embedder
from config import EMBEDDING_MODELS
e = get_embedder(EMBEDDING_MODELS['qwen3-8b-ollama'])
vecs = e.encode_queries(['Fix authentication bug'])
print('Shape:', vecs.shape)  # expect (1, 4096)
"

# 3. Test ColBERT pipeline
python -c "
from pylate import models
m = models.ColBERT('colbert-ir/colbertv2.0')
q = m.encode(['test query'], is_query=True)
print('ColBERT token vecs shape:', q[0].shape)  # expect (N_tokens, 128)
"

# 4. Test multi-dataset loading
python -c "
import sqlite3, config
for ds_key, ds_cfg in config.DATASETS.items():
    conn = sqlite3.connect(ds_cfg['db_path'])
    n = conn.execute('SELECT COUNT(*) FROM TASK').fetchone()[0]
    conn.close()
    print(f'{ds_key}: {n} tasks')
"
```

---

### Step 9: File Structure After Implementation

```
exp4/
├── config.py                      # Modified: families, datasets, fixed params
├── embedders.py                   # NEW: SentenceTransformerEmbedder + OllamaEmbedder
├── colbert_pipeline.py            # NEW: ColBERT index + search + evaluate
├── etl_pipeline.py                # Modified: uses embedders.py, dataset_key param
├── run_experiments.py             # Modified: uses embedders.py, dataset_key param
├── run_comprehensive_experiments.py  # Modified: loops over datasets, ColBERT branch
├── vector_backends.py             # Unchanged
├── utils.py                       # Unchanged
├── checkpoint_manager.py          # Unchanged
├── gpu_utils.py                   # Unchanged
├── experiment_ui.py               # Minor: add dataset filter to UI
├── data/
│   ├── sonar.db
│   ├── kafka.db
│   └── spark.db
├── experiment_results/
│   ├── comprehensive_results.csv  # All results (compatible with exp3 schema + dataset/family cols)
│   ├── results_{model}_{dataset}.csv
│   └── checkpoint.json
└── colbert_index_{dataset}_{split}/  # ColBERT on-disk index (auto-created)
```
