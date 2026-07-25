# Comprehensive Experiment Guide

## Overview

This guide explains how to run the comprehensive RAG experiments with **resume capability**, **GPU memory management**, and support for **both split strategies** (recent & modN).

## New Features ✨

### 1. **Resumable Experiments** 🔄
- Automatically saves progress after each variant completes
- Can resume after power outage or crash
- Prompts you to choose: resume or start fresh

### 2. **GPU Memory Management** 🎮
- Automatic GPU cache clearing between models
- Adaptive batch size based on model size
- Fallback to CPU for very large models
- Prevents CUDA out of memory errors

### 3. **Both Split Strategies** 📊
- Runs experiments for **both** `recent` and `modN` strategies
- Results include split_strategy column
- Separate test sets for each strategy

### 4. **PostgreSQL Integration** 🐘
- Uses PostgreSQL + pgvector (more stable than Qdrant)
- Automatic backup after experiments complete
- Easy restore from backups

### 5. **Comprehensive Results** 📈
- Combined results CSV with all models and strategies
- Per-model result files
- Detailed logging and progress tracking

## Quick Start

### 1. Install Dependencies

```bash
cd /home/stzh/Projects/simrgl/exp3

# Install Python packages
pip install -r requirements.txt
```

### 2. Start PostgreSQL

```bash
./start_postgres.sh
```

Wait for PostgreSQL to be ready (~10 seconds).

### 3. Run Comprehensive Experiments

**Option A: Use defaults (recommended for first run)**

```bash
./run_comprehensive_experiment.sh
```

This will run experiments for:
- Models: `bge-small`, `bge-large`, `gte-large`
- Strategies: `recent`, `modN`
- Sources: `title`, `desc`, `comments`
- Targets: `file`, `module`
- Windows: `w100`, `w1000`, `all`

**Total variants**: 3 models × 2 strategies × 3 sources × 2 targets × 3 windows = **108 experiments**

**Option B: Customize models**

```bash
./run_comprehensive_experiment.sh --models bge-small bge-large
```

**Option C: Python directly (advanced)**

```bash
python run_comprehensive_experiments.py \
    --models bge-small bge-large \
    --strategies recent modn \
    --backend postgres
```

### 4. Monitor Progress

The script will show:
- Current variant being processed
- Progress (e.g., "Progress: 15/108 completed, 1 failed")
- GPU memory usage
- Estimated time remaining (coming soon)

### 5. Resume After Interruption

If the experiment is interrupted (power outage, crash, etc.):

```bash
./run_comprehensive_experiment.sh
```

You'll be prompted:

```
CHECKPOINT STATUS
==========================================================================
Created: 2025-12-28T10:30:00
Last Updated: 2025-12-28T12:45:00

Completed ETL: 24
Completed Experiments: 24
Failed ETL: 1
Failed Experiments: 0

Current Model: bge-large
Current Strategy: modn
==========================================================================

Resume from checkpoint? (y/n):
```

Choose:
- `y` - Resume from where it left off
- `n` - Start fresh (deletes checkpoint)

## Results

### Output Files

```
experiment_results/
├── comprehensive_results.csv         # All results (all models, all strategies)
├── results_bge-small.csv             # Per-model results
├── results_bge-large.csv
├── results_gte-large.csv
├── checkpoint.json                   # Resume checkpoint
├── backups/
│   └── exp3_vectors_backup_*.sql    # PostgreSQL backups
└── test_set_recent_bge-small.json   # Test sets (per strategy & model)
    test_set_modn_bge-small.json
    ...
```

### Results CSV Format

```csv
model,split_strategy,experiment_id,source,target,window,MAP,MRR,P@1,R@1,P@3,R@3,P@5,R@5,P@10,R@10
bge-small,recent,title_file_all_recent,title,file,all,0.0234,0.0456,0.0120,0.0164,0.0135,0.0245,0.0142,0.0298,0.0150,0.0345
bge-small,modn,title_file_all_modn,title,file,all,0.0198,0.0398,0.0105,0.0142,0.0118,0.0212,0.0125,0.0268,0.0132,0.0312
bge-large,recent,desc_module_w1000_recent,desc,module,w1000,0.0312,0.0523,0.0145,0.0178,0.0156,0.0289,0.0164,0.0342,0.0172,0.0398
...
```

### Key Columns

- `model`: Embedding model used (bge-small, bge-large, etc.)
- `split_strategy`: Split strategy (recent or modn)
- `source`: Text source (title, desc, comments)
- `target`: Target granularity (file, module)
- `window`: Historical window (w100, w1000, all)
- `MAP`: Mean Average Precision
- `MRR`: Mean Reciprocal Rank
- `P@K` / `R@K`: Precision/Recall at K

## GPU Memory Management

### Automatic Features

1. **Cache Clearing**: GPU cache is cleared after each model
2. **Garbage Collection**: Python garbage collector runs between variants
3. **Adaptive Batch Size**: Large models use smaller batches (8 instead of 32)
4. **CPU Fallback**: Very large models can fall back to CPU

### Model Memory Requirements

| Model | GPU Memory | Batch Size | Device |
|-------|-----------|------------|---------|
| bge-small | ~0.5 GB | 32 | GPU |
| bge-large | ~1.5 GB | 32 | GPU |
| bge-m3 | ~2.0 GB | 8 | GPU (adaptive) |
| gte-qwen2 | ~3.0 GB | 8 | GPU/CPU (fallback) |
| nomic-embed | ~1.2 GB | 32 | GPU |

### Manual Control

If you encounter OOM errors:

1. **Reduce batch size** in `config.py`:
   ```python
   BATCH_SIZE = 8  # Default: 32
   ```

2. **Force CPU mode** for specific models:
   ```python
   # In gpu_utils.py, update should_use_cpu() for your model
   ```

## PostgreSQL Backup & Restore

### Backup

Automatic backup after experiments:
```bash
./backup_postgres.sh
```

Output: `experiment_results/backups/exp3_vectors_backup_YYYYMMDD_HHMMSS.sql`

### Restore

```bash
# Restore latest backup
./restore_postgres.sh

# Restore specific backup
./restore_postgres.sh experiment_results/backups/exp3_vectors_backup_20251228_120000.sql
```

## Checkpoint Management

### Checkpoint File

Location: `experiment_results/checkpoint.json`

Contains:
- Completed ETL variants
- Completed experiment variants
- Failed variants (with error messages)
- Current progress (model, strategy)

### Clear Checkpoint

To start completely fresh:

```bash
rm experiment_results/checkpoint.json
./run_comprehensive_experiment.sh
```

Or use the `--no-resume` flag:

```bash
./run_comprehensive_experiment.sh --no-resume
```

## Troubleshooting

### PostgreSQL Not Starting

```bash
# Check if container exists
podman ps -a | grep semantic_vectors_db

# Remove old container
podman rm -f semantic_vectors_db

# Start fresh
./start_postgres.sh
```

### CUDA Out of Memory

**Error**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Solutions**:

1. Reduce batch size in `config.py`:
   ```python
   BATCH_SIZE = 8
   ```

2. Run only small models first:
   ```bash
   ./run_comprehensive_experiment.sh --models bge-small
   ```

3. Use CPU for large models (edit `gpu_utils.py`):
   ```python
   # Force CPU for gte-qwen2
   if 'qwen2' in model_name.lower():
       return True  # Use CPU
   ```

### Missing Dependencies

**Error**: `ImportError: No module named 'einops'`

**Solution**:
```bash
pip install einops psycopg2-binary torch
```

### Collection Not Found

**Error**: `Collection rag_exp_desc_file_all_recent_bge_small not found`

**Cause**: ETL failed for that variant

**Solution**:
1. Check checkpoint for errors
2. Resume will automatically retry failed variants
3. Or delete checkpoint and start fresh

### Slow Performance

**If experiments are very slow**:

1. **Check GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Reduce parallel workers** in `run_experiments.py`:
   ```python
   with ThreadPoolExecutor(max_workers=5):  # Default: 10
   ```

3. **Use GPU** (not CPU):
   - Ensure PyTorch CUDA is installed:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```

## Advanced Configuration

### Custom Experiment Variants

Edit `config.py` to customize:

```python
# Add new source variant
SOURCE_VARIANTS['custom'] = {
    'name': 'CUSTOM',
    'description': 'Title + Description + Custom Field',
    'fields': ['TITLE', 'DESCRIPTION', 'CUSTOM_FIELD']
}

# Add new window variant
WINDOW_VARIANTS['w500'] = {
    'name': 'NEAREST 500',
    'description': 'Last 500 tasks',
    'size': 500
}
```

Then run:
```bash
python run_comprehensive_experiments.py --sources custom --windows w500
```

### Running Subset of Experiments

**Only recent strategy**:
```bash
python run_comprehensive_experiments.py --strategies recent
```

**Only specific combinations**:
```bash
python run_comprehensive_experiments.py \
    --models bge-small \
    --strategies recent \
    --sources desc \
    --targets file \
    --windows all
```

This runs only 1 experiment (instead of 108).

## Performance Estimates

Based on a machine with:
- CPU: 8 cores
- GPU: NVIDIA P106-100 (6GB)
- Database: PostgreSQL on SSD

### Time Estimates

| Phase | Time per Variant | Total (108 variants) |
|-------|------------------|----------------------|
| ETL | 5-10 min | 9-18 hours |
| Experiments | 2-5 min | 3.6-9 hours |
| **Total** | **7-15 min** | **12-27 hours** |

### Optimization Tips

1. **Use GPU**: 5-10× faster than CPU
2. **Increase batch size**: Faster, but needs more GPU memory
3. **Reduce variants**: Test with 1-2 models first
4. **Resume capability**: Critical for long experiments

## FAQ

**Q: Can I run this on CPU-only machine?**
A: Yes, but it will be 5-10× slower. Models will automatically fallback to CPU.

**Q: How much disk space needed?**
A: ~5-10 GB for PostgreSQL database + embeddings.

**Q: Can I pause and resume?**
A: Yes! The checkpoint system automatically saves progress. Just Ctrl+C to pause, and re-run to resume.

**Q: What if a variant fails?**
A: Failed variants are logged in the checkpoint. When you resume, they will be retried automatically.

**Q: Can I run multiple models in parallel?**
A: Not recommended - GPU memory conflicts. Run sequentially (default behavior).

**Q: How to compare results across models?**
A: Use `comprehensive_results.csv` which includes all models. Sort by MAP or MRR to find best performers.

## Next Steps

1. **View Results Interactively**:
   ```bash
   streamlit run experiment_ui.py
   ```

2. **Analyze Best Configurations**:
   ```bash
   # Show top 10 configurations by MAP
   head -1 experiment_results/comprehensive_results.csv > top10.csv
   tail -n +2 experiment_results/comprehensive_results.csv | sort -t, -k7 -rn | head -10 >> top10.csv
   cat top10.csv
   ```

3. **Deploy Best Model**:
   - Copy best performing collections to production
   - Use in `../ragmcp/` for MCP server or local agent

## Support

If you encounter issues:

1. Check the checkpoint file for error messages
2. Review experiment logs
3. Try with a smaller subset first (1-2 models)
4. Ensure GPU drivers and CUDA are properly installed

---

**Last Updated**: 2025-12-28
**Version**: 2.0 (Comprehensive Experiments with Resume)
