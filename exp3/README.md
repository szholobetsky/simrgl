# RAG Research Experiment: Task-to-Code Retrieval

A comprehensive RAG (Retrieval-Augmented Generation) system for answering research questions about task-to-code retrieval effectiveness.

## 🎯 Research Questions

This experiment systematically investigates:

1. **RQ1: Granularity Impact** - File-level vs. Module-level retrieval
2. **RQ2: Semantic Density** - Task titles vs. descriptions
3. **RQ3: Noise Tolerance** - Impact of including comments
4. **RQ4: Temporal Dynamics** - Recent history vs. full history

## 🔄 Vector Database Backends

This project supports **two vector database backends**:

### Qdrant (Default for Research)
- **Use for**: Research experiments, model comparisons
- **Advantages**: Easy Docker setup, built-in backup/restore
- **Setup**: See sections below

### PostgreSQL + pgvector (Production Ready)
- **Use for**: Production deployments, MCP server integration
- **Advantages**: Single database, backup-friendly, MCP protocol support
- **Setup**: See `../ragmcp/` folder for complete setup
- **Features**:
  - 27 module embeddings
  - 12,532 file embeddings
  - 9,799 task embeddings
  - MCP server for Claude Desktop / VS Code integration
  - Local offline AI agent (CLI + Web)

**To use PostgreSQL backend:**
```bash
# Create task embeddings in PostgreSQL
python create_task_collection.py --backend postgres

# Use with MCP server or local agent
cd ../ragmcp
./start_local_agent.bat
```

See `../ragmcp/README.md` for complete PostgreSQL backend documentation.

## 📁 Project Structure

```
├── config.py                  # Centralized configuration
├── utils.py                   # Utility functions and metrics
├── etl_pipeline.py            # Data processing and embedding generation
├── run_experiments.py         # Experiment execution and evaluation
├── experiment_ui.py           # Streamlit web interface
├── backup_restore_qdrant.py   # Qdrant backup/restore utility
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Qdrant vector database setup
├── start.bat / start.sh       # Automated full workflow (Windows/Linux)
├── quick_start.bat/sh         # Quick UI launch (Windows/Linux)
├── rerun_experiments.bat      # Re-run experiments without ETL (Windows)
├── backup_qdrant.bat          # Backup Qdrant data (Windows)
├── restore_qdrant.bat         # Restore Qdrant data (Windows)
├── BIG_EXPERIMENT.sh          # Multi-model experiment (Linux)
├── sonar.db                   # Source database (9,799 tasks, 469,836 commits)
└── README.md                  # This file
```

## 🚀 Easy Start (Automated)

### One-Command Execution

**Windows (with Podman):**
```cmd
start.bat
```

**Linux/Mac (with Podman):**
```bash
./start.sh
```

These scripts will automatically:
1. Clear old Qdrant data
2. Start Qdrant
3. Run ETL pipeline (~20-30 min)
4. Run all experiments (~10-15 min)
5. Launch the Streamlit UI

**Just view results (after experiments complete):**
```cmd
quick_start.bat    # Windows
./quick_start.sh   # Linux/Mac
```

**Note:** The ETL pipeline and experiments take 30-40 minutes to complete. Keep your computer plugged in during long-running experiments.

---

## 🔧 Manual Setup (Step-by-Step)

### Prerequisites

- Python 3.8+
- Docker or Podman (for Qdrant)
- ~4GB disk space for embeddings
- SQLite database with RAWDATA and TASK tables (see [Database Requirements](#database-requirements))

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant Vector Database

```bash
docker-compose up -d
```

Qdrant will be available at `http://localhost:6333`

### 3. Run ETL Pipeline

Process data and generate embeddings for all experiment variants:

```bash
# Using "recent" split strategy (recommended)
python etl_pipeline.py --split_strategy recent

# Or using "modn" split strategy (uniform sampling)
python etl_pipeline.py --split_strategy modn
```

This will:
- Split data into train (9,599 tasks) and test (200 tasks)
- Generate embeddings for 3 source variants × 2 target variants × 3 time windows
- Create 18 Qdrant collections
- Save `test_set.json` with ground truth

**Estimated time:** 20-30 minutes (depends on hardware)

### 4. Run Experiments

Evaluate all 18 experiment combinations:

```bash
python run_experiments.py --split_strategy recent
```

This will:
- Query each test task against all collection variants
- Calculate MAP, MRR, P@K, R@K metrics
- Save results to `experiment_results.csv`

**Estimated time:** 10-15 minutes

### 5. View Results

Launch the interactive Streamlit UI:

```bash
streamlit run experiment_ui.py
```

Open your browser to `http://localhost:8501`

## 📊 Experiment Variants

### Source Variants (RQ2 & RQ3)
- **title**: Task title only
- **desc**: Task title + description
- **comments**: Task title + description + comments

### Target Variants (RQ1)
- **file**: Individual file granularity
- **module**: Root folder/module granularity

### Window Variants (RQ4)
- **w100**: Last 100 tasks before test set
- **w1000**: Last 1000 tasks before test set
- **all**: All available training tasks

### Split Strategies
- **recent**: Test on 200 most recent tasks
- **modn**: Test on uniformly sampled tasks

**Total Combinations:** 3 × 2 × 3 = 18 experiments per split strategy

## 🔧 Advanced Usage

### Run Specific Variants

```bash
# Only process title and desc sources
python etl_pipeline.py --sources title desc

# Only evaluate file-level targets
python run_experiments.py --targets file

# Only test recent time windows
python run_experiments.py --windows w100 w1000
```

### Custom Test Set Size

```bash
python etl_pipeline.py --test_size 500
python run_experiments.py --split_strategy recent
```

### Custom Output File

```bash
python run_experiments.py --output my_results.csv
```

## 🧪 Multi-Model Experiments

Compare different embedding models to find the best one for your use case.

### Available Models

| Key | Model | Dimensions | Description |
|-----|-------|------------|-------------|
| `bge-small` | BAAI/bge-small-en-v1.5 | 384 | Default, fast, lightweight |
| `bge-large` | BAAI/bge-large-en-v1.5 | 1024 | Better quality |
| `bge-m3` | BAAI/bge-m3 | 1024 | Multilingual, long context |
| `gte-qwen2` | Alibaba-NLP/gte-Qwen2-1.5B-instruct | 1536 | Qwen-based, high quality |
| `nomic-embed` | nomic-ai/nomic-embed-text-v1.5 | 768 | Good quality, efficient |
| `gte-large` | thenlper/gte-large | 1024 | Strong on technical text |
| `e5-large` | intfloat/e5-large-v2 | 1024 | Microsoft, strong general |

### Run Single Model Experiment

```bash
# Run ETL with a specific model
python etl_pipeline.py --model bge-large --split_strategy recent

# Run experiments with that model
python run_experiments.py --model bge-large --split_strategy recent
```

### Run All Models (Linux)

The `BIG_EXPERIMENT.sh` script runs experiments for multiple models automatically:

```bash
# Run all default models (bge-small, bge-large, bge-m3, gte-qwen2, nomic-embed)
./BIG_EXPERIMENT.sh

# Or run specific models only
./BIG_EXPERIMENT.sh bge-small bge-large gte-qwen2
```

**What it does:**
1. Starts Qdrant (clears old data)
2. For each model:
   - Runs ETL pipeline
   - Runs experiments
   - Backs up Qdrant to `qdrant_snapshots_<model>/`
   - Saves results to `experiment_results/results_<model>.csv`
3. Combines all results into `all_models_results.csv`

**Output files:**
```
├── all_models_results.csv           # Combined results with model column
├── experiment_results/
│   ├── results_bge-small.csv        # Per-model results
│   ├── results_bge-large.csv
│   ├── log_bge-small.log            # Per-model logs
│   └── log_bge-large.log
├── qdrant_snapshots_bge-small/      # Qdrant backup per model
├── qdrant_snapshots_bge-large/
└── ...
```

### Restore and Re-run a Specific Model

If you need to re-run experiments for a model without redoing ETL:

```bash
# Restore the model's Qdrant data
python backup_restore_qdrant.py --action restore --input qdrant_snapshots_bge-large

# Re-run experiments only
python run_experiments.py --model bge-large --output experiment_results/results_bge-large.csv
```

## 💾 Backup and Restore

### Backup Qdrant Data

```bash
# Windows
backup_qdrant.bat

# Linux/Manual
python backup_restore_qdrant.py --action backup --output qdrant_snapshots
```

### Restore Qdrant Data

```bash
# Windows
restore_qdrant.bat

# Linux/Manual
python backup_restore_qdrant.py --action restore --input qdrant_snapshots
```

### Deploy to Another Machine

1. Copy these files to the new machine:
   - `qdrant_snapshots/` folder (or model-specific `qdrant_snapshots_<model>/`)
   - `test_set.json`
   - All Python files and `requirements.txt`

2. Start Qdrant on the new machine:
   ```bash
   docker-compose up -d
   ```

3. Restore the data:
   ```bash
   python backup_restore_qdrant.py --action restore --input qdrant_snapshots
   ```

4. Run experiments or view UI:
   ```bash
   python run_experiments.py --split_strategy recent
   streamlit run experiment_ui.py
   ```

## 📈 Understanding Results

### Metrics

- **MAP (Mean Average Precision)**: Overall ranking quality across all relevant items
- **MRR (Mean Reciprocal Rank)**: Quality of the first relevant result
- **P@K**: Precision at rank K (what fraction of top-K are relevant?)
- **R@K**: Recall at rank K (what fraction of relevant items are in top-K?)

Default K values: 1, 3, 5, 10

### Example Results Table

| experiment_id | source | target | window | MAP | MRR | P@10 | R@10 |
|--------------|--------|--------|--------|-----|-----|------|------|
| title_file_all_recent | title | file | all | 0.0234 | 0.0456 | 0.0120 | 0.0345 |
| desc_file_all_recent | desc | file | all | 0.0312 | 0.0523 | 0.0145 | 0.0398 |

### Answering Research Questions

Compare results to answer each RQ:

1. **RQ1**: Compare `*_file_*` vs `*_module_*` rows
2. **RQ2**: Compare `title_*` vs `desc_*` rows
3. **RQ3**: Compare `desc_*` vs `comments_*` rows
4. **RQ4**: Compare `*_w100_*` vs `*_w1000_*` vs `*_all_*` rows

## 🖥️ Web Interface Features

The Streamlit UI provides three modes:

### 1. Results Dashboard
- View all experiment results
- Compare variant performance
- Analyze research question findings

### 2. Interactive Search
- Enter custom task descriptions
- Search using different configurations
- See ranked code file/module results

### 3. Research Questions
- View RQ definitions and hypotheses
- Understand methodology
- Learn about evaluation metrics

## 🐛 Troubleshooting

### Qdrant Connection Error

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart
```

### Collection Not Found

The ETL pipeline must complete before running experiments. Collections are named:
```
rag_exp_{source}_{target}_{window}_{split}
```

Example: `rag_exp_title_file_all_recent`

### Low Performance Metrics

This is expected! Task-to-code retrieval is challenging:
- Many tasks touch the same files
- Task descriptions are often vague
- Code changes span multiple unrelated files

Typical MAP scores: 0.01 - 0.05 (1-5%)

### Out of Memory

Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # Default: 32
```

## 📝 Configuration

Edit `config.py` to customize:

- Database path
- Qdrant connection
- Embedding model (default: `BAAI/bge-small-en-v1.5`)
- Test set size (default: 200)
- Top-K values for metrics
- Batch sizes

## 🔬 Extending the Experiment

### Add New Source Variant

```python
# In config.py
SOURCE_VARIANTS['custom'] = {
    'name': 'CUSTOM',
    'description': 'My custom variant',
    'fields': ['TITLE', 'CUSTOM_FIELD']
}
```

### Add New Time Window

```python
# In config.py
WINDOW_VARIANTS['w500'] = {
    'name': 'NEAREST 500',
    'description': 'Last 500 tasks',
    'size': 500
}
```

Then re-run ETL and experiments.

## 📚 Implementation Details

### Embedding Models
- **Default Model**: BAAI/bge-small-en-v1.5 (384 dimensions)
- **Type**: Sentence-BERT transformers
- **Multiple models supported**: See [Multi-Model Experiments](#-multi-model-experiments) section

### Vector Aggregation
- **Strategy**: Centroid (mean vector)
- **Rationale**: Each file/module is represented by the average of all task embeddings that touched it

### Evaluation Protocol
1. Split tasks chronologically (or uniform sampling)
2. Generate embeddings for training tasks only
3. Aggregate by file/module (excluding test tasks)
4. Query test tasks against aggregated vectors
5. Calculate metrics using ground truth files

## 💾 Database Requirements

This experiment requires a SQLite database populated with the following tables:

### Required Tables

**TASK Table:**
- `ID` (INTEGER): Task identifier
- `NAME` (TEXT): Task key (e.g., JIRA-123)
- `TITLE` (TEXT): Task title/summary
- `DESCRIPTION` (TEXT): Task description
- `COMMENTS` (TEXT): All task comments

**RAWDATA Table:**
- `ID` (INTEGER): Commit identifier
- `TASK_NAME` (TEXT): Associated task key
- `PATH` (TEXT): File path from commit

### Database Setup

You can create the database using the CodeXplorer data gathering tool from the `refactor` folder. See the main project README for instructions.

Alternatively, configure `config.py` to point to your own database:
```python
DB_PATH = 'path/to/your/database.db'
```

## 📝 Version Control

The following files are excluded from git (see `.gitignore`):

### Generated Data
- `test_set.json` - Generated test set
- `experiment_results.csv` - Experiment results
- `*_model_comparison_summary.csv` - Model comparison CSVs

### Logs & Temporary Files
- `*.log` - Log files
- `conversation_log.txt` - Session logs
- All `log_*.txt` and `log_*.md` files

### Qdrant Data
- `qdrant_storage/` - Qdrant database storage
- `qdrant_snapshots*/` - Backup snapshots

### Note on Database Files
The source database (`sonar.db`) is tracked by default. If your database is large or contains sensitive data, add it to `.gitignore`:
```
sonar.db
*.db
```

## 📄 License

This is a research experiment. Use for academic purposes.

## 🙏 Acknowledgments

- Qdrant for vector database
- Sentence-Transformers for embeddings
- Streamlit for UI framework
