# Full Comprehensive RAG Experiment Plan

## Overview

This experiment evaluates different combinations of:
- **Models**: Embedding models for semantic similarity
- **Strategies**: Train/test split strategies
- **Sources**: Input fields from task descriptions
- **Targets**: Granularity of code units (file vs module)
- **Windows**: Historical context window size

## Experiment Matrix

### Models (6)
| Model | Dimensions | Description |
|-------|------------|-------------|
| `bge-small` | 384 | BAAI/bge-small-en-v1.5 - Fast, baseline |
| `bge-large` | 1024 | BAAI/bge-large-en-v1.5 - High quality |
| `gte-large` | 1024 | Alibaba-NLP/gte-large-en-v1.5 - Alternative large |
| `bge-m3` | 1024 | BAAI/bge-m3 - Multilingual, large |
| `gte-qwen2` | 1536 | Alibaba-NLP/gte-Qwen2-1.5B-instruct - 1.5B params |
| `nomic-embed` | 768 | nomic-ai/nomic-embed-text-v1.5 - Long context |

### Split Strategies (2)
| Strategy | Description | Test Set Selection |
|----------|-------------|-------------------|
| `recent` | Temporal split | Most recent 200 tasks |
| `modn` | Uniform sampling | Every Nth task (200 total) |

### Source Variants (3)
| Source | Fields Used | Purpose |
|--------|-------------|---------|
| `title` | TITLE only | Minimal context |
| `desc` | TITLE + DESCRIPTION | Standard context |
| `comments` | TITLE + DESCRIPTION + COMMENTS | Full context including discussions |

### Target Variants (2)
| Target | Granularity | Example |
|--------|-------------|---------|
| `file` | Individual files | `src/utils/parser.py` |
| `module` | Module directories | `src/utils/` |

### Window Variants (3)
| Window | Training Size | Purpose |
|--------|---------------|---------|
| `w100` | Last 100 tasks | Very recent context |
| `w1000` | Last 1000 tasks | Recent context |
| `all` | All training tasks (~9599) | Full historical context |

## Total Experiments

```
6 models × 2 strategies × 3 sources × 2 targets × 3 windows = 216 experiments
```

## Example Combinations

1. **bge-small_recent_title_file_w100**
   - Embeddings: bge-small (384 dim)
   - Split: Recent 200 tasks as test
   - Input: Task titles only
   - Target: File-level retrieval
   - Training: Last 100 tasks

2. **gte-qwen2_modn_comments_module_all**
   - Embeddings: gte-qwen2 (1536 dim, 1.5B params)
   - Split: Uniformly sampled 200 tasks as test
   - Input: Full task context (title+desc+comments)
   - Target: Module-level retrieval
   - Training: All ~9599 tasks

## Metrics Collected

For each experiment, we compute:

| Metric | Description |
|--------|-------------|
| **MAP** | Mean Average Precision |
| **MRR** | Mean Reciprocal Rank |
| **P@k** | Precision at k (k=1,3,5,10) |
| **R@k** | Recall at k (k=1,3,5,10) |

## Research Questions

1. **Model Comparison**: Which embedding model provides best retrieval?
2. **Strategy Impact**: Does temporal (recent) or uniform (modN) split affect performance?
3. **Context Value**: Do comments improve retrieval over title+description?
4. **Granularity**: File-level vs module-level retrieval effectiveness?
5. **History Size**: Does more training data (larger window) improve results?

## Execution Plan

### Phase 1: ETL Pipeline (per combination)
1. Load tasks from SQLite database (9,799 tasks, 469,836 commits)
2. Create train/test split based on strategy
3. Apply time window filter to training data
4. Generate embeddings for training tasks using source variant
5. Aggregate embeddings by target variant (file or module centroids)
6. Upload vectors to PostgreSQL collection

### Phase 2: Evaluation (per combination)
1. Load test set (200 tasks)
2. Generate embeddings for test queries
3. Search PostgreSQL collection (k=10)
4. Compute precision/recall metrics
5. Save results to CSV

## Resource Requirements

### Compute
- **GPU**: NVIDIA P106-100 (6GB VRAM) or similar
- **CPU**: Multi-core recommended for centroid computation
- **RAM**: 16GB+ recommended

### Storage
- **Database**: ~2GB (sonar.db)
- **Vectors**: ~50-100MB per collection (216 collections total = ~10-20GB)
- **Models**: ~1-5GB per embedding model (6 models = ~6-15GB)
- **Results**: ~10-50MB

### Time Estimates (with GPU)
- **bge-small**: ~5-8 minutes per combination
- **bge-large/gte-large**: ~8-12 minutes per combination
- **bge-m3/gte-qwen2/nomic-embed**: ~12-20 minutes per combination

**Total estimated time**: 6-10 hours

## Output Files

### Results
- `experiment_results/comprehensive_results.csv` - All experiments combined
- `experiment_results/results_{model}.csv` - Per-model results
- `experiment_results/checkpoint.json` - Resume state

### Test Sets
- `experiment_results/test_set_recent_{model}.json`
- `experiment_results/test_set_modn_{model}.json`

### PostgreSQL Collections (216 total)
Format: `rag_exp_{source}_{target}_{window}_{strategy}_{model}`

Examples:
- `rag_exp_title_file_w100_recent_bge_small`
- `rag_exp_comments_module_all_modn_gte_qwen2`

### Logs
- `experiment_full_run.log` - Complete execution log
- `experiment_run.log` - Current run log (for monitoring)

## Usage

### Start Full Experiment
```bash
cd /home/stzh/Projects/simrgl/exp3
./run_full_experiment.sh
```

### Resume After Interruption
The checkpoint system automatically resumes from the last completed experiment:
```bash
./run_full_experiment.sh
```

### Monitor Progress
```bash
# Watch log
tail -f experiment_full_run.log

# Check checkpoint
cat experiment_results/checkpoint.json | jq '.completed_etl | length'

# View intermediate results
cat experiment_results/comprehensive_results.csv | wc -l
```

### Analyze Results
```bash
# Best configurations by MAP
head -1 experiment_results/comprehensive_results.csv
tail -n +2 experiment_results/comprehensive_results.csv | sort -t',' -k7 -rn | head -20

# Compare strategies
grep "recent" experiment_results/comprehensive_results.csv > recent_results.csv
grep "modn" experiment_results/comprehensive_results.csv > modn_results.csv

# Compare models
for model in bge-small bge-large gte-large bge-m3 gte-qwen2 nomic-embed; do
    echo "=== $model ==="
    grep "$model" experiment_results/comprehensive_results.csv | \
        awk -F',' '{sum+=$7; count++} END {print "Average MAP:", sum/count}'
done
```

## Backup & Recovery

### Automatic Backup
After successful completion, `run_full_experiment.sh` automatically creates a PostgreSQL backup.

### Manual Backup
```bash
./backup_postgres.sh
```

### Restore from Backup
```bash
./restore_postgres.sh backups/exp3_vectors_backup_TIMESTAMP.sql
```

## Troubleshooting

### Out of Memory
- Reduce batch size in `config.py`: `BATCH_SIZE = 8`
- Process models sequentially instead of all at once

### GPU Errors
- Verify GPU: `nvidia-smi`
- Check PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Fallback to CPU: Set `CUDA_VISIBLE_DEVICES=""`

### Interrupted Experiment
The checkpoint system saves progress. Simply restart:
```bash
./run_full_experiment.sh
```

### PostgreSQL Issues
```bash
# Restart container
podman-compose down
podman-compose up -d

# Check logs
podman logs exp3-postgres-1
```

## Expected Results

Based on preliminary test (bge-small):

| Strategy | Typical MAP Range | Typical MRR Range |
|----------|-------------------|-------------------|
| recent | 0.001 - 0.01 | 0.002 - 0.02 |
| modn | 0.02 - 0.10 | 0.08 - 0.20 |

**Note**: modN strategy typically performs better due to more diverse training examples.

## Post-Experiment Analysis

After completion, analyze:
1. **Model ranking**: Which models consistently perform best?
2. **Source impact**: Does comment inclusion help specific models?
3. **Window effects**: Diminishing returns of larger windows?
4. **Target preference**: File vs module retrieval trade-offs?
5. **Strategy robustness**: Performance consistency across splits?

Generate visualizations:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('experiment_results/comprehensive_results.csv')

# Model comparison
df.groupby('model')['MAP'].mean().sort_values().plot(kind='barh')
plt.title('Average MAP by Model')
plt.savefig('model_comparison.png')

# Source comparison
df.groupby('source')['MAP'].mean().sort_values().plot(kind='barh')
plt.title('Average MAP by Source')
plt.savefig('source_comparison.png')
```
