# Semantic Code Analysis with Embeddings

> AI Course Capstone Project - Semantic similarity analysis for code-to-text matching using multiple embedding models

## Overview

This project implements a semantic analysis system that matches code modules with textual descriptions using various embedding techniques. It analyzes Git commit data from Jira tasks to build vector representations of code modules, enabling semantic similarity searches.

**Key Features:**
- Multiple embedding models: Word2Vec, FastText, GloVe, BERT, and LLM-based embeddings
- Bradford ranking and HHI (Herfindahl-Hirschman Index) for term filtering
- Train/test split methodology for unbiased evaluation
- Multiple distance metrics (cosine, euclidean, manhattan)
- Statistical evaluation with MAP, MRR, and Recall@K metrics

## Quick Start

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd claude_11092025
   ```

2. **Install dependencies:**
   ```bash
   # Minimal installation (Word2Vec only)
   pip install -r requirements_minimal.txt

   # Or full installation
   # Windows:
   pip install -r requirements_windows.txt

   # Linux/macOS:
   pip install -r requirements_linux.txt
   ```

3. **Prepare your database:**
   - Place your Flink database at `data/flink.db`
   - Or update `DATABASE_PATH` in `properties.py`

### Running the Analysis

**Complete pipeline:**
```bash
# 1. Extract terms and build module structure
python main.py

# 2. Create embeddings and vectors
python vectorizer.py

# 3. Evaluate similarity performance
python statistical_evaluator.py --count 200
```

**Test similarity search:**
```bash
# Create a task description in task.txt, then:
python test.py task.txt 10
```

## Configuration

Edit `properties.py` to configure the system:

### Embedding Model Selection
```python
VECTORISER_MODEL = 'fast_text'  # Options: 'own', 'fast_text', 'glove', 'bert', 'llm'
CLEAR_EMBEDDINGS = True         # Set to True for first run with new model
VECTOR_DIMENSION = 100          # Vector size (model-dependent)
```

### Model Options

| Model | Description | Best For |
|-------|-------------|----------|
| `'own'` | Custom Word2Vec trained on your data | Domain-specific vocabulary |
| `'fast_text'` | Pretrained FastText embeddings | Handling out-of-vocabulary words |
| `'glove'` | Pretrained GloVe embeddings | General-purpose, fast |
| `'bert'` | BERT transformer embeddings | High-quality contextual embeddings |
| `'llm'` | Sentence transformers / API models | State-of-the-art performance |

### Experimental Parameters

```python
# Evaluation settings
MODULE_VECTOR_STRATEGY = 'sum'           # avg, sum, median, weighted_avg, cluster
DISTANCE_METRICS = ['cosine']            # cosine, euclidean, manhattan
NORMALIZE_VECTORS = True                 # Normalize to unit length

# Train/Test split
EXCLUDE_TEST_TASKS_FROM_MODEL = True     # Proper evaluation methodology
PREPROCESS_TEST_TASK = False             # Filter test text using HHI
```

## Project Structure

```
.
├── main.py                    # Extract terms and build module database
├── vectorizer.py              # Create embeddings and vectors
├── statistical_evaluator.py   # Evaluate performance metrics
├── test.py                    # Interactive similarity testing
├── baseVectorizer.py          # Abstract base for vectorizers
├── vectorizer_factory.py      # Factory pattern for model selection
├── task_selector.py           # Train/test split utilities
├── properties.py              # Configuration file
├── requirements*.txt          # Dependency files
└── data/                      # Database directory
    └── flink.db              # Your project database

```

## Evaluation Metrics

The system evaluates performance using:

- **MAP (Mean Average Precision)**: Overall ranking quality
- **MRR (Mean Reciprocal Rank)**: Quality of first relevant result
- **Recall@K**: Fraction of relevant items in top K results

## Example Workflow

### 1. Initial Setup with Word2Vec
```python
# properties.py
VECTORISER_MODEL = 'own'
CLEAR_EMBEDDINGS = True
```
```bash
python main.py && python vectorizer.py && python statistical_evaluator.py --count 200
```

### 2. Compare with FastText
```python
# properties.py
VECTORISER_MODEL = 'fast_text'
CLEAR_EMBEDDINGS = True  # System auto-detects model change
```
```bash
python vectorizer.py && python statistical_evaluator.py --count 200
```

### 3. Test Different Aggregation Strategies
```python
# properties.py
MODULE_VECTOR_STRATEGY = 'weighted_avg'  # Try weighted average
CLEAR_EMBEDDINGS = False                  # Reuse embeddings
```
```bash
python statistical_evaluator.py --count 200
```

## Database Schema

The system creates the following tables with `SIMRGL_` prefix:

- `SIMRGL_TERMS`: Extracted terms with frequency counts
- `SIMRGL_TERM_RANK`: Bradford zones and HHI metrics
- `SIMRGL_MODULES`: Code modules (root directories)
- `SIMRGL_FILES`: Files mapped to modules
- `SIMRGL_TASK_VECTOR`: Task embeddings
- `SIMRGL_MODULE_VECTOR`: Module embeddings
- `SIMRGL_EMBEDDING_METADATA`: Model configuration tracking

## Research Questions

This capstone project explores:

1. **Which embedding model performs best** for code-to-text semantic similarity?
2. **How does train/test split methodology** affect evaluation results?
3. **Which distance metric** works best for different embedding types?
4. **How do different aggregation strategies** affect module representation quality?
5. **What is the performance vs computational cost trade-off** across models?

## Requirements

**Core:**
- Python 3.8+
- NumPy, Pandas, SciPy, scikit-learn
- tqdm, requests

**Model-specific:**
- Word2Vec: `gensim`
- FastText: `fasttext` or `fasttext-wheel`
- BERT: `torch`, `transformers`
- LLM: `sentence-transformers`

## Troubleshooting

**Database not found:**
- Ensure `data/flink.db` exists or update `DATABASE_PATH` in `properties.py`

**Out of memory:**
- Reduce batch sizes: `BERT_BATCH_SIZE`, `LLM_BATCH_SIZE`
- Use smaller models or CPU-only mode
- Process in smaller chunks

**Slow performance:**
- Start with Word2Vec or FastText for development
- Use GPU for BERT/LLM models if available
- Set `CLEAR_EMBEDDINGS = False` to reuse existing embeddings

## Performance Tips

1. **First run**: Use `CLEAR_EMBEDDINGS = True` with your chosen model
2. **Subsequent experiments**: Set `CLEAR_EMBEDDINGS = False` to reuse embeddings
3. **Model comparison**: System auto-detects model changes and regenerates as needed
4. **Parameter tuning**: Reuse embeddings when only changing evaluation parameters

## License

Educational project for AI course capstone.

## Author

Created as capstone project for internal IT company AI course (2025)
