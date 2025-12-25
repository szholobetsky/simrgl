# CodeXplorer Research Experiments

A comprehensive collection of experiments investigating task-to-code prediction using different machine learning and information retrieval approaches.

## 🎯 Research Goal

**Can we automatically predict which code modules/files should be modified based on a task description?**

This research explores different approaches to linking natural language task descriptions (from issue trackers like Jira) to relevant code artifacts, helping developers quickly locate relevant code for new tasks.

---

## 📊 Experiments Overview

| Experiment | Approach | Accuracy (MAP@10) | Runtime | Status | Recommendation |
|------------|----------|-------------------|---------|--------|----------------|
| **exp0** | TF-IDF | 0.5-1.5% | 4-48 hours | ❌ Failed | Historical only |
| **exp1** | Statistical Analysis | N/A (exploratory) | Fast | ✅ Complete | For insights |
| **exp3** | Dense Embeddings (BERT) | 2.3-3.5% | 20-40 min | ✅ Best | **Use this** |
| **ragmcp** | MCP Server + Local AI Agent | N/A (production tool) | Real-time | ✅ Production | **Deployable** |

### Evolution Timeline

```
exp0 (TF-IDF)
    ↓ [Failed: too slow, poor results]
exp1 (Statistical Analysis)
    ↓ [Insights: term distributions, module relationships]
exp3 (Embeddings + RAG)
    ↓ [Success: 2-3× better accuracy, 100× faster]
ragmcp (MCP + Local Agent)
    ✓ [Production: MCP server, offline AI agent, PostgreSQL backend]
```

---

## 📁 Experiment Details

### Experiment 0: TF-IDF Approach ❌ Not Recommended

**Location**: `exp0/`

**Approach**: Traditional TF-IDF (Term Frequency-Inverse Document Frequency) scoring to rank code modules based on term overlap with task descriptions.

**Key Scripts**:
- `taskTokenizer.py` - Tokenize task descriptions
- `TFIDF_module_token.py` - Calculate TF-IDF scores per module
- `tfidfFast.py` - Faster sklearn-based implementation

**Results**:
- ❌ **Accuracy**: 0.5-1.5% MAP@10
- ⏱️ **Runtime**: 4-48 hours for full dataset
- 💾 **Memory**: High (large sparse matrices)

**Why it failed**:
- Cannot understand semantics (synonyms, context)
- Too sparse (most term-module pairs are zero)
- Linear scoring insufficient for complex relationships
- Computationally inefficient

**When to read**:
- Understanding why traditional IR fails for code
- Establishing baseline for comparison
- Research transparency and methodology documentation

📖 **Full documentation**: `exp0/README.md`

---

### Experiment 1: Statistical Analysis ✅ Exploratory

**Location**: `exp1/`

**Approach**: Statistical analysis of term distributions, module hierarchies, and co-occurrence patterns.

**Key Scripts**:
- `title_term.py` - Extract terms from task titles
- `module_task.py` - Build hierarchical file/folder structure
- `term_rank.py` - Calculate term ranking metrics (HHI, composite index)
- `interlink.py` - Co-occurrence matrices for terms and files

**Outputs**:
- `TITLE_TERM` - Unique vocabulary
- `MODULE` - Hierarchical file structure
- `TERM_RANK` - Term specificity metrics
- `TERM_LINKS` - Term co-occurrence patterns
- `FILE_LINKS` - File co-modification patterns

**Insights gained**:
- Distribution of general vs. specific terms
- Module coupling through shared tasks
- Term concentration metrics (HHI)
- File modification patterns

**Use cases**:
- Understanding codebase structure
- Identifying module boundaries
- Finding coupled components
- Vocabulary analysis

**When to use**:
- Research into code structure and terminology
- Identifying refactoring opportunities
- Understanding domain vocabulary

📖 **Full documentation**: `exp1/README.md`

---

### Experiment 3: Embedding-Based RAG ✅ **RECOMMENDED**

**Location**: `exp3/`

**Approach**: Retrieval-Augmented Generation (RAG) using sentence transformers (BERT-based embeddings) and vector similarity search.

**Key Components**:
- **Embeddings**: BAAI/bge-small-en-v1.5 (and other models)
- **Vector DB**: Qdrant for similarity search
- **Aggregation**: Centroid-based file/module embeddings
- **UI**: Streamlit for interactive exploration

**Key Scripts**:
- `etl_pipeline.py` - Data processing and embedding generation
- `run_experiments.py` - Systematic evaluation
- `experiment_ui.py` - Interactive web interface
- `backup_restore_qdrant.py` - Vector DB management

**Research Questions Investigated**:

| RQ | Question | Finding |
|----|----------|---------|
| RQ1 | File vs Module granularity | Module-level better recall, file-level better precision |
| RQ2 | Title vs Description | Descriptions provide better semantic signal |
| RQ3 | Impact of comments | Comments add noise, decrease performance |
| RQ4 | Recent vs full history | Recent history reduces obsolete associations |

**Results**:
- ✅ **Accuracy**: 2.3-3.5% MAP@10 (2-3× better than TF-IDF)
- ⚡ **Runtime**: 20-40 minutes (10-100× faster than TF-IDF)
- 🎯 **MRR**: 4.5-6.0% (first result quality)
- 📈 **Recall@10**: 3.4-4.5%

**Features**:
- Multiple embedding models support
- Configurable experiment variants
- Interactive search interface
- Comprehensive evaluation metrics
- Easy backup/restore

**When to use**:
- **Production task-to-code recommendation**
- Research on semantic code search
- Comparing embedding models
- Understanding modern RAG systems

📖 **Full documentation**: `exp3/README.md`

---

### Production Tool: MCP Server + Local AI Agent ✅ **PRODUCTION READY**

**Location**: `ragmcp/`

**Approach**: Production-ready deployment of exp3 results using Model Context Protocol (MCP) and local LLM integration.

**Key Components**:
- **MCP Server**: PostgreSQL-based semantic search server
- **Local AI Agent**: 100% offline coding assistant (CLI + Web)
- **Vector DB**: PostgreSQL + pgvector (27 modules, 12,532 files, 9,799 tasks)
- **LLM**: Ollama with qwen2.5-coder for local reasoning
- **Backup/Restore**: Full data backup and recovery

**Key Scripts**:
- `mcp_server_postgres.py` - MCP server for Claude Desktop / VS Code
- `local_agent.py` - CLI offline agent
- `local_agent_web.py` - Web interface (Gradio)
- `backup_data_from_postgree.bat/sh` - PostgreSQL backup
- `test_mcp_server.py` - Server testing

**Features**:
- ✅ **100% Offline**: No cloud services, no API keys needed
- ✅ **Privacy-First**: All data stays on your machine
- ✅ **Free Forever**: No subscriptions, no costs
- ✅ **Multi-Interface**: CLI, Web, Claude Desktop, VS Code
- ✅ **Historical Context**: 9,799 task embeddings for similarity search
- ✅ **MCP Protocol**: Standard integration with AI tools

**MCP Tools Available**:
1. `search_modules` - Find relevant code modules (top-K)
2. `search_files` - Find relevant files (top-K)
3. `search_similar_tasks` - Find historical similar tasks
4. `get_collections_info` - Collection statistics

**Interfaces**:

**CLI Mode:**
```bash
cd ragmcp/
./start_local_agent.bat    # Windows
./start_local_agent.sh     # Linux/Mac
```

**Web Interface:**
```bash
cd ragmcp/
./start_local_agent_web.bat    # Windows
# Open: http://127.0.0.1:7861
```

**Claude Desktop Integration:**
- Configure MCP server in Claude Desktop settings
- Use semantic search directly in Claude
- See `ragmcp/MCP_SETUP_GUIDE.md` for setup

**VS Code Integration:**
- Use with Cline or Continue extensions
- Terminal integration
- Task configuration
- See `ragmcp/LOCAL_AGENT_GUIDE.md` for details

**Example Query:**
```
> Fix authentication bug in login module

Agent provides:
- Relevant modules: server, sonar-server
- Relevant files: SimpleSessionsContainer.js, cookies.ts
- Similar tasks: SONAR-8493, SONAR-11066, SONAR-16181
- AI recommendations based on historical context
```

**Performance**:
- ⚡ **Search**: <100ms per query
- 🤖 **LLM**: 5-10 seconds (after first query)
- 💾 **Storage**: 112 MB (PostgreSQL)
- 🔄 **Backup**: Full backup in 30 seconds

**Comparison with Cloud Solutions**:

| Feature | Local Agent | Claude Desktop | GitHub Copilot |
|---------|------------|----------------|----------------|
| Offline | ✅ Yes | ❌ No | ❌ No |
| Cost | ✅ Free | ❌ $20/month | ❌ $10/month |
| Privacy | ✅ 100% local | ❌ Cloud | ❌ Cloud |
| Codebase Search | ✅ Yes (MCP) | ✅ Yes (MCP) | ❌ Limited |
| Historical Tasks | ✅ Yes (9,799) | ❌ No | ❌ No |

**When to use**:
- **Production deployment** of task-to-code recommendations
- **Privacy-sensitive** projects (no data leaves your machine)
- **Offline development** environments
- **Integration** with Claude Desktop or VS Code
- **Free alternative** to cloud AI assistants

📖 **Full documentation**: `ragmcp/README.md`
📖 **Setup guides**: `ragmcp/MCP_SETUP_GUIDE.md`, `ragmcp/LOCAL_AGENT_GUIDE.md`

---

## 🚀 Quick Start Guide

### For Production Use (Recommended - Offline AI Agent)

If you want a deployable, offline AI coding assistant:

```bash
cd ragmcp/

# 1. Ensure PostgreSQL is running
podman start semantic_vectors_db

# 2. Ensure Ollama is running
ollama serve

# 3. Start the local agent (CLI)
./start_local_agent.bat    # Windows
./start_local_agent.sh     # Linux/Mac

# Or start web interface
./start_local_agent_web.bat    # Windows
# Open: http://127.0.0.1:7861
```

**What you get:**
- 100% offline AI coding assistant
- Semantic search across 12,532 files and 9,799 historical tasks
- LLM-powered recommendations
- No cloud services, no costs
- Privacy-first (all data stays local)

See `ragmcp/README.md` for complete documentation.

### For Research & Experiments

If you want to run research experiments or compare embedding models:

```bash
cd exp3/

# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant database
docker-compose up -d

# 3. Run the full pipeline (automated)
./start.sh    # Linux/Mac
start.bat     # Windows

# 4. Access the UI at http://localhost:8501
```

**Alternative**: Just view results without running experiments:
```bash
./quick_start.sh    # Linux/Mac
quick_start.bat     # Windows
```

### For Research & Analysis

If you want to understand codebase structure:

```bash
cd exp1/

# Update database path in each script
# Then run in order:
python title_term.py
python module_task.py
python term_rank.py
python interlink.py
```

### For Historical Comparison

If you want to understand why TF-IDF failed:

```bash
cd exp0/

# Read the warnings first!
# Then run the fast version:
python tfidfFast.py
```

---

## 📋 Prerequisites

### Required for All Experiments

1. **SQLite Database** with tables:
   - `RAWDATA` - Git commit data
   - `TASK` - Jira task data

   Create using the data gathering tool in `../../data_gathering/refactor/`

2. **Python 3.8+** with pip

### Experiment-Specific

**exp0**:
- pandas, numpy, nltk, scikit-learn, tqdm

**exp1**:
- tqdm only

**exp3**:
- pandas, numpy, sentence-transformers, qdrant-client, streamlit, tqdm
- Docker/Podman for Qdrant database

---

## 📊 Performance Comparison

### Accuracy Metrics

| Metric | exp0 (TF-IDF) | exp3 (Embeddings) | Improvement |
|--------|---------------|-------------------|-------------|
| MAP@10 | 0.5-1.5% | 2.3-3.5% | **2-3× better** |
| MRR | 0.8-2.0% | 4.5-6.0% | **2.5-3× better** |
| P@10 | 0.3-1.2% | 1.2-1.8% | **3-4× better** |
| R@10 | 1.0-2.5% | 3.4-4.5% | **2-3× better** |

### Runtime Performance

| Task | exp0 | exp3 | Speed-up |
|------|------|------|----------|
| ETL Pipeline | 4-48 hours | 20-30 min | **10-100× faster** |
| Evaluation | N/A | 10-15 min | - |
| Query (single) | Slow | <100ms | **100-1000× faster** |

### Resource Usage

| Resource | exp0 | exp1 | exp3 |
|----------|------|------|------|
| Memory | High (sparse matrices) | Low | Medium (embeddings) |
| Disk | Medium | Low | High (vectors) |
| CPU | Very high | Low | Medium |
| GPU | No | No | Optional (speeds up) |

---

## 🔬 Research Questions & Findings

### RQ1: What granularity works best?

**Experiment**: exp3, File vs Module targets

**Finding**:
- **Module-level**: Better recall (finds more relevant code)
- **File-level**: Better precision (fewer false positives)
- **Recommendation**: Use module-level for exploration, file-level for precise changes

### RQ2: How much task information is needed?

**Experiment**: exp3, Title vs Description vs Comments

**Finding**:
- **Title**: Concise but limited information
- **Description**: Best balance of semantic richness and signal-to-noise
- **Comments**: Too noisy, decreases performance
- **Recommendation**: Use Title + Description

### RQ3: Does semantic understanding matter?

**Experiment**: exp0 (term matching) vs exp3 (embeddings)

**Finding**:
- TF-IDF (term overlap): 0.5-1.5% MAP@10
- Embeddings (semantic): 2.3-3.5% MAP@10
- **Answer**: Yes, 2-3× improvement with semantic understanding

### RQ4: Is historical context important?

**Experiment**: exp3, Recent vs Full history

**Finding**:
- Recent history (1000 tasks): Better for evolving codebases
- Full history: More data but includes obsolete associations
- **Recommendation**: Use recent history for active projects

---

## 📚 Research Methodology

### Data Collection
1. **Git Repository**: Extract all commits with file changes
2. **Issue Tracker**: Fetch task descriptions from Jira
3. **Linking**: Match commit messages to task IDs

### Evaluation Protocol
1. **Split**:
   - Recent: Last 200 tasks for testing
   - ModN: Uniform sampling across history
2. **Training**: Build embeddings/TF-IDF from remaining tasks
3. **Querying**: For each test task, retrieve top-K code artifacts
4. **Metrics**: Calculate MAP, MRR, P@K, R@K against ground truth

### Ground Truth
- Files touched by commits associated with each task
- Assumption: Developers knew which files to modify

---

## 🗂️ Project Structure

```
simrgl/
├── README.md                    # This file
├── exp0/                        # TF-IDF experiment (historical)
│   ├── README.md               # Detailed exp0 documentation
│   ├── QUICKSTART.md           # Quick start guide
│   ├── config.py               # Configuration
│   ├── taskTokenizer.py        # Tokenization
│   ├── TFIDF_module_token.py   # Module-level TF-IDF
│   ├── tfidfFast.py            # Fast sklearn TF-IDF
│   └── chainTfidfFast.py       # Word groups TF-IDF
├── exp1/                        # Statistical analysis
│   ├── README.md               # Detailed exp1 documentation
│   ├── config.py               # Configuration
│   ├── title_term.py           # Term extraction
│   ├── module_task.py          # Module hierarchy
│   ├── term_rank.py            # Term metrics
│   └── interlink.py            # Co-occurrence analysis
├── exp3/                        # Embedding-based RAG (research)
│   ├── README.md               # Comprehensive exp3 documentation
│   ├── config.py               # Configuration
│   ├── etl_pipeline.py         # Data processing
│   ├── run_experiments.py      # Evaluation
│   ├── experiment_ui.py        # Streamlit UI
│   ├── utils.py                # Helper functions
│   ├── docker-compose.yml      # Qdrant setup
│   ├── start.sh/bat            # Automated pipeline
│   └── quick_start.sh/bat      # UI launcher
└── ragmcp/                      # Production MCP server + Local AI agent
    ├── README.md               # Comprehensive documentation
    ├── README_UA.md            # Ukrainian version
    ├── mcp_server_postgres.py  # PostgreSQL MCP server
    ├── local_agent.py          # CLI offline agent
    ├── local_agent_web.py      # Web interface (Gradio)
    ├── gradio_ui.py            # Original Gradio UI
    ├── llm_integration.py      # LLM integration
    ├── backup_data_from_postgree.bat/sh  # Backup scripts
    ├── restore_data_to_postgree.bat/sh   # Restore scripts
    ├── start_local_agent.bat/sh          # CLI launchers
    ├── start_local_agent_web.bat/sh      # Web launchers
    ├── MCP_SETUP_GUIDE.md      # MCP integration guide
    ├── LOCAL_AGENT_GUIDE.md    # Local agent guide
    └── READY_TO_USE.md         # Quick start summary
```

---

## 📖 Getting Started Workflow

### Step 1: Gather Data (One-time)
```bash
cd ../../data_gathering/refactor/

# Configure your settings
vim config.py

# Run data gathering
python main.py
```

This creates a SQLite database with RAWDATA and TASK tables.

### Step 2: Run Analysis (Optional)
```bash
cd ../../capestone/claude11/simrgl/exp1/

# Generate insights about your codebase
python title_term.py
python module_task.py
python term_rank.py
python interlink.py
```

### Step 3: Build Prediction System
```bash
cd ../exp3/

# Run the full pipeline
./start.sh    # Linux/Mac
start.bat     # Windows
```

### Step 4: Use the System
- Open browser to `http://localhost:8501`
- Enter task descriptions
- Get ranked code file/module recommendations

---

## 🔗 Related Projects

### Production Tool: MCP Server + Local AI Agent
**Location**: `ragmcp/`

Production deployment of semantic search:
- MCP server for Claude Desktop / VS Code
- Local offline AI agent (CLI + Web)
- PostgreSQL + pgvector backend
- 100% offline, free, privacy-first
- Backup/restore capabilities
- See `ragmcp/README.md` for details

### Data Gathering Tool
**Location**: `../../data_gathering/refactor/`

Creates the database used by all experiments:
- Extracts Git commits
- Fetches Jira task details
- Links commits to tasks

### Python Scripts
**Location**: `../../python/`

Original experimental scripts (before refactoring):
- Legacy code for historical reference
- Many scripts migrated to exp0, exp1, exp3

---

## 📊 Publications & References

### Related Work

**Code Search**:
- Lv, F., et al. (2015). "CodeHow: Effective code search based on API understanding and extended Boolean model"
- Ye, X., et al. (2016). "Learning to rank relevant files for bug reports"

**Embeddings for Code**:
- Feng, Z., et al. (2020). "CodeBERT: A Pre-Trained Model for Programming and Natural Languages"
- Husain, H., et al. (2019). "CodeSearchNet Challenge"

**TF-IDF Limitations**:
- Manning, C. D., et al. (2008). "Introduction to Information Retrieval"
- Allamanis, M., et al. (2018). "A Survey of Machine Learning for Big Code"

---

## 🤝 Contributing

This research is part of an academic project. Improvements welcome:

1. **exp3**: Enhancements to the embedding approach
2. **Documentation**: Clarifications and examples
3. **New experiments**: Novel approaches to task-to-code linking

Please do NOT spend time optimizing exp0 - it's a fundamental limitation of the approach.

---

## 📄 License

This is academic research code. Use for educational and research purposes.

---

## 🎓 Conclusion

### What We Learned

1. **Semantic understanding is crucial**: Traditional term matching (TF-IDF) is insufficient
2. **Embeddings work better**: 2-3× improvement over TF-IDF
3. **Speed matters**: Fast iteration enables better research
4. **Context helps**: Recent history better than full history
5. **Granularity trades**: File vs module has precision/recall trade-off

### Recommended Path

**For Production Use:**
1. ✅ **Use ragmcp** (MCP server + local AI agent) - Deployable, offline, free
2. 📖 **Read ragmcp/README.md** for setup instructions

**For Research:**
1. ✅ **Use exp3** (embedding-based RAG) - Best accuracy, experiments
2. 📊 **Reference exp1** (for codebase insights) - Statistical analysis
3. ❌ **Avoid exp0** (TF-IDF too slow and inaccurate) - Historical only

### Future Directions

- Fine-tuned code-specific language models (CodeBERT, GraphCodeBERT)
- Graph-based approaches (code structure graphs)
- Multi-modal learning (code + commit messages + documentation)
- Active learning for low-resource projects
- Transfer learning across projects

---

## 📧 Contact & Support

For questions about:
- **ragmcp (Production)**: See `ragmcp/README.md` for deployment guide
- **exp3 (Research)**: See `exp3/README.md` for comprehensive guide
- **exp1 (Analysis)**: See `exp1/README.md` for usage instructions
- **exp0 (Historical)**: See `exp0/README.md` for detailed documentation
- **Data gathering**: See `../../data_gathering/refactor/README.md`

---

**Built with**: Python • SQLite • Sentence Transformers • Qdrant • Streamlit

**Research conducted**: 2024-2025
