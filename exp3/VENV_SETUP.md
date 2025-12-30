# Python 3.12 Virtual Environment Setup

## Quick Start After Restart

### 1. Activate the Virtual Environment
```bash
cd /home/stzh/Projects/simrgl/exp3
source venv_py312/bin/activate
```

You should see `(venv_py312)` in your terminal prompt.

### 2. Verify GPU Support
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.3.1+cu118
CUDA available: True
GPU: NVIDIA P106-100
```

### 3. Check PostgreSQL Container
```bash
podman ps | grep postgres
```

If not running, start it:
```bash
cd /home/stzh/Projects/simrgl/exp3
podman-compose up -d
```

Verify connection:
```bash
podman exec -it exp3-postgres-1 psql -U postgres -d semantic_vectors -c "SELECT version();"
```

### 4. Run Experiments

#### Test with bge-small (both strategies):
```bash
python run_comprehensive_experiments.py \
  --models bge-small \
  --strategies recent modn \
  --sources desc \
  --targets file \
  --windows all \
  --backend postgres
```

#### Full experiment (all models):
```bash
python run_comprehensive_experiments.py \
  --models bge-small bge-large gte-large \
  --strategies recent modn \
  --sources desc \
  --targets file \
  --windows all \
  --backend postgres
```

#### Resume interrupted experiment:
```bash
# Checkpoint is automatically loaded - just run the same command
python run_comprehensive_experiments.py \
  --models bge-small \
  --strategies recent modn \
  --sources desc \
  --targets file \
  --windows all \
  --backend postgres
```

#### Start fresh (ignore checkpoint):
```bash
python run_comprehensive_experiments.py \
  --models bge-small \
  --strategies recent modn \
  --sources desc \
  --targets file \
  --windows all \
  --backend postgres \
  --no-resume
```

### 5. Monitor Progress

#### Check experiment log:
```bash
tail -f experiment_run.log
```

#### Check checkpoint status:
```bash
cat experiment_results/checkpoint.json
```

#### View results:
```bash
ls -lh experiment_results/
cat experiment_results/comprehensive_results.csv
```

### 6. Deactivate Virtual Environment
```bash
deactivate
```

## Environment Details

- **Python Version:** 3.12.12
- **PyTorch Version:** 2.3.1+cu118
- **CUDA Support:** 11.8 (compatible with sm_61 / P106-100 GPU)
- **Vector Backend:** PostgreSQL with pgvector
- **Database:** ../data/sonar.db (9,799 tasks, 469,836 commits)

## Troubleshooting

### GPU not detected:
```bash
# Check NVIDIA driver
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### PostgreSQL connection fails:
```bash
# Restart container
podman-compose down
podman-compose up -d

# Check logs
podman logs exp3-postgres-1
```

### Module not found errors:
```bash
# Ensure venv is activated
source venv_py312/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Out of memory errors:
The experiment uses adaptive batch sizing and GPU memory management.
Check available GPU memory:
```bash
nvidia-smi
```

If still failing, reduce batch size in `config.py`:
```python
BATCH_SIZE = 8  # Default is 32
```

## Important Files

- `venv_py312/` - Python 3.12 virtual environment
- `experiment_results/checkpoint.json` - Resume state
- `experiment_results/comprehensive_results.csv` - Final metrics
- `experiment_run.log` - Execution log
- `config.py` - Configuration settings
- `run_comprehensive_experiments.py` - Main experiment runner

## Backup & Restore PostgreSQL

### Create backup:
```bash
./backup_postgres.sh
```

### Restore from backup:
```bash
./restore_postgres.sh backups/exp3_vectors_backup_TIMESTAMP.sql
```
