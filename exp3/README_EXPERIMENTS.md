# RAG Experiment Scripts - Quick Reference

## Available Scripts

### 1. Preview Experiment Plan
**Script:** `./preview_experiment.sh`
**Purpose:** Shows what will be run without executing
**Output:** Experiment matrix, resource estimates, sample IDs

```bash
./preview_experiment.sh
```

### 2. Full Experiment (216 experiments)
**Script:** `./run_full_experiment.sh`
**Configuration:**
- 6 models: bge-small, bge-large, gte-large, bge-m3, gte-qwen2, nomic-embed
- 2 strategies: recent, modn
- 3 sources: title, desc, comments
- 2 targets: file, module
- 3 windows: w100, w1000, all

**Time:** 6-10 hours with GPU
**Storage:** ~20-35GB

```bash
./run_full_experiment.sh
```

### 3. Subset Experiment (72 experiments)
**Script:** `./run_subset_experiment.sh`
**Configuration:**
- 2 models: bge-small, bge-large (test before full run)
- All other parameters same as full experiment

**Time:** 2-3 hours with GPU
**Storage:** ~5-10GB

```bash
./run_subset_experiment.sh
```

### 4. Custom Experiment
**Script:** Direct Python call with specific parameters

```bash
source venv_py312/bin/activate

python run_comprehensive_experiments.py \
  --models bge-small \
  --strategies recent modn \
  --sources desc \
  --targets file \
  --windows all \
  --backend postgres
```

## Recommended Workflow

### First Time Setup
1. **Read the documentation:**
   ```bash
   cat VENV_SETUP.md
   cat EXPERIMENT_PLAN.md
   ```

2. **Preview the plan:**
   ```bash
   ./preview_experiment.sh
   ```

3. **Test with single configuration** (already done):
   ```bash
   # You already completed this test successfully!
   # Results: experiment_results/comprehensive_results.csv
   ```

### Before Full Run

4. **Run subset experiment** (recommended):
   ```bash
   ./run_subset_experiment.sh
   ```

   This tests with 2 models (bge-small + bge-large) across all combinations to:
   - Verify all configurations work
   - Check disk space usage
   - Estimate actual runtime
   - Validate PostgreSQL stability

5. **Review subset results:**
   ```bash
   # Check results
   cat experiment_results/comprehensive_results.csv

   # Verify PostgreSQL collections
   podman exec exp3-postgres-1 psql -U postgres -d semantic_vectors -c "\dt vectors.*"

   # Check disk usage
   df -h .
   ```

### Full Run

6. **Run full experiment:**
   ```bash
   ./run_full_experiment.sh
   ```

7. **Monitor progress:**
   ```bash
   # In another terminal
   tail -f experiment_full_run.log

   # Check checkpoint
   cat experiment_results/checkpoint.json | jq

   # Count completed experiments
   tail -n +2 experiment_results/comprehensive_results.csv | wc -l
   ```

## After Restart/Interruption

If your system restarts or the experiment is interrupted:

1. **Reactivate environment:**
   ```bash
   cd /home/stzh/Projects/simrgl/exp3
   source venv_py312/bin/activate
   ```

2. **Restart PostgreSQL:**
   ```bash
   podman-compose up -d
   ```

3. **Resume experiment** (automatically continues from checkpoint):
   ```bash
   ./run_full_experiment.sh
   ```

## Monitoring & Management

### Check Progress
```bash
# Completed experiments
cat experiment_results/checkpoint.json | jq '.completed_etl | length'

# Failed experiments
cat experiment_results/checkpoint.json | jq '.failed_etl'

# Current results count
tail -n +2 experiment_results/comprehensive_results.csv | wc -l
```

### View Results
```bash
# Top configurations by MAP
head -1 experiment_results/comprehensive_results.csv
tail -n +2 experiment_results/comprehensive_results.csv | sort -t',' -k7 -rn | head -20

# Compare strategies
echo "=== Recent Strategy ==="
grep "recent" experiment_results/comprehensive_results.csv | \
  awk -F',' '{sum+=$7; count++} END {print "Average MAP:", sum/count}'

echo "=== ModN Strategy ==="
grep "modn" experiment_results/comprehensive_results.csv | \
  awk -F',' '{sum+=$7; count++} END {print "Average MAP:", sum/count}'
```

### Manage PostgreSQL
```bash
# List collections
podman exec exp3-postgres-1 psql -U postgres -d semantic_vectors -c "\dt vectors.*"

# Collection row counts
podman exec exp3-postgres-1 psql -U postgres -d semantic_vectors -c \
  "SELECT table_name, pg_size_pretty(pg_total_relation_size('vectors.' || table_name))
   FROM information_schema.tables
   WHERE table_schema = 'vectors';"

# Backup
./backup_postgres.sh

# Restore
./restore_postgres.sh backups/exp3_vectors_backup_TIMESTAMP.sql
```

### GPU Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Troubleshooting

### Experiment Fails
```bash
# Check last error
tail -100 experiment_full_run.log | grep -A 5 ERROR

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check PostgreSQL
podman exec exp3-postgres-1 psql -U postgres -d semantic_vectors -c "SELECT version();"
```

### Out of Disk Space
```bash
# Check usage
df -h .

# Clean old checkpoints (after backup!)
rm -f experiment_results/checkpoint.json.backup.*

# Remove specific collections
podman exec exp3-postgres-1 psql -U postgres -d semantic_vectors -c \
  "DROP TABLE IF EXISTS vectors.rag_exp_old_collection;"
```

### Resume After Changes
```bash
# Start fresh (clear checkpoint)
python run_comprehensive_experiments.py \
  --models bge-small \
  --strategies recent modn \
  --sources desc \
  --targets file \
  --windows all \
  --backend postgres \
  --no-resume
```

## File Organization

```
exp3/
├── run_full_experiment.sh          # Main script (216 experiments)
├── run_subset_experiment.sh        # Subset script (72 experiments)
├── preview_experiment.sh           # Preview plan
├── backup_postgres.sh              # Backup collections
├── restore_postgres.sh             # Restore collections
├── run_comprehensive_experiments.py # Python runner
├── etl_pipeline.py                 # ETL logic
├── run_experiments.py              # Evaluation logic
├── config.py                       # Configuration
├── venv_py312/                     # Python 3.12 + PyTorch 2.3.1
├── experiment_results/             # Results directory
│   ├── comprehensive_results.csv   # All results
│   ├── results_*.csv               # Per-model results
│   ├── checkpoint.json             # Resume state
│   └── test_set_*.json             # Test sets
├── backups/                        # PostgreSQL backups
├── VENV_SETUP.md                   # Environment setup guide
├── EXPERIMENT_PLAN.md              # Detailed experiment plan
└── README_EXPERIMENTS.md           # This file
```

## Quick Commands Cheat Sheet

```bash
# Activate environment
source venv_py312/bin/activate

# Preview plan
./preview_experiment.sh

# Run subset (2 models)
./run_subset_experiment.sh

# Run full (6 models)
./run_full_experiment.sh

# Monitor
tail -f experiment_full_run.log

# Check progress
cat experiment_results/checkpoint.json | jq

# View results
head experiment_results/comprehensive_results.csv

# Backup
./backup_postgres.sh

# GPU status
nvidia-smi
```

## Expected Output

After successful completion:

```
experiment_results/
├── comprehensive_results.csv       # 217 lines (header + 216 results)
├── results_bge-small.csv           # 36 experiments
├── results_bge-large.csv           # 36 experiments
├── results_gte-large.csv           # 36 experiments
├── results_bge-m3.csv              # 36 experiments
├── results_gte-qwen2.csv           # 36 experiments
├── results_nomic-embed.csv         # 36 experiments
├── test_set_recent_*.json          # 6 test sets
├── test_set_modn_*.json            # 6 test sets
└── checkpoint.json                 # Final state

backups/
└── exp3_vectors_backup_TIMESTAMP.sql  # ~5-10GB
```

## Questions?

See detailed documentation:
- **Setup:** `VENV_SETUP.md`
- **Experiment design:** `EXPERIMENT_PLAN.md`
- **Configuration:** Check `config.py`
