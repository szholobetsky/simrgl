# Experimental Programme Complex for Automated Task-to-Code Retrieval Research

## Abstract Summary

This section describes the experimental programme complex designed to investigate the problem of automated code navigation — the task of predicting which source code modules and files a developer should modify when implementing or resolving a given software task. The complex evolved through three experimental generations (*exp2*, *exp3*) before reaching its current form. The final complex comprises two major interacting subsystems: a **data gathering infrastructure** (*codeXplorer*) and a **retrieval experiment framework** (*exp3*). Together they constitute a complete, reproducible pipeline from raw version-control history to evaluated information-retrieval metrics. A critical engineering milestone in this evolution was the migration of vector storage from an embedded SQLite-based solution through an in-memory vector database (Qdrant) to a fully persistent PostgreSQL backend — a progression driven by concrete operational constraints encountered during multi-day experiment runs.

---

## 1. Overview and Research Context

Modern software projects encompass tens of thousands of source files distributed across dozens of architectural modules. A developer who receives a task described in natural language (e.g., a Jira issue) must first navigate this space to identify the relevant files — a cognitive effort that can consume a significant portion of development time. The goal of the experimental complex presented here is to study whether this navigation can be automated by learning associations between historical task descriptions and the code artefacts touched in their resolution.

The experimental complex is organised around a single central research question: *Given the natural language description of a software task, can an information retrieval system surface the relevant source files with sufficient precision and recall to be practically useful?* Four orthogonal dimensions of this question are investigated:

- **RQ1 — Retrieval granularity:** Should the system operate at the level of individual files or at the level of coarse-grained modules?
- **RQ2 — Semantic density of queries:** Does including the full task description improve over using only the task title?
- **RQ3 — Noise tolerance:** Does appending user comments to the query hurt or help retrieval quality?
- **RQ4 — Temporal window:** Is recent history more informative than the full project lifetime?

The programme complex is built on real data from a large-scale industrial open-source project (Apache SonarQube) comprising 9,799 resolved Jira issues linked to 469,836 commit–file pairs across 12,532 distinct source files. This dataset provides both the training signal for the retrieval models and a held-out test set for objective evaluation.

---

## 2. Structural Organisation of the Complex

The experimental programme complex consists of the following principal components, whose relationships are captured in **Diagram 1**.

> **[Diagram 1 — High-Level Architecture of the Experimental Complex]**
> *What to draw:* A horizontal block diagram with three vertical layers. The top layer shows external data sources: a Git repository on the left and a Jira project management system on the right. The middle layer shows the two main software subsystems: the codeXplorer data ingestion tool (centre-left) and the exp3 retrieval experiment framework (centre-right), connected by a SQLite database icon between them. The bottom layer shows outputs: an experiment results CSV, a Streamlit web dashboard, and an archived vector database. Arrows indicate data flow direction. Colour-code each layer distinctly.
> *Why this diagram is needed:* It gives the reader an immediate visual map of all components and their interactions before the detailed description begins, preventing disorientation when individual subsystems are discussed in isolation.

The components and their roles are:

| Component | Location | Role |
|---|---|---|
| codeXplorer | `codeXplorer/` | Data collection and enrichment pipeline |
| Intermediate store | `data/sonar.db` | Shared SQLite database (task ↔ code mapping) |
| exp3 | `exp3/` | Embedding-based retrieval and evaluation framework |
| Vector database | PostgreSQL / Qdrant | Persistent storage for dense vectors |
| Results | `experiment_results.csv` | Aggregated evaluation metrics |
| Web interface | Streamlit | Interactive exploration of results |

---

## 3. Data Collection Infrastructure: codeXplorer

### 3.1 Purpose and Design Principles

The *codeXplorer* tool transforms a version-controlled software project into a structured, queryable dataset of task-to-code associations. Its design follows three principles: (1) **source independence** — it works with any Git repository and supports multiple Jira access strategies; (2) **completeness** — it captures the full project history including file-level diffs; (3) **extensibility** — connectors are interchangeable plug-ins.

### 3.2 Pipeline Stages

The tool executes five sequential stages illustrated in **Diagram 2**.

> **[Diagram 2 — codeXplorer Data Collection Pipeline]**
> *What to draw:* A vertical flowchart with five numbered boxes connected by downward arrows. Box 1: "Database Initialisation — create RAWDATA and TASK tables". Box 2: "Commit Extraction — iterate all commits on the target branch, extract SHA, author, date, message, file path, diff (GitConnector)". Box 3: "Task ID Extraction — apply regex to commit messages, populate TASK_NAME field (TaskExtractor)". Box 4: "Task Enrichment — fetch title, description, comments from Jira for each unique task ID (TaskFetcher + connector strategy)". Box 5: "Unified Dataset — SQLite database ready for downstream experiments". On the right side, show a branching diagram for the Jira connector strategies (API / HTML / Selenium) feeding into Box 4. Use dashed borders for the connector alternatives.
> *Why this diagram is needed:* The pipeline has a clear sequential dependency structure that prose alone communicates poorly; the flowchart makes the stage ordering and the connector branching immediately legible.

**Stage 1 — Database Initialisation.** The `DatabaseManager` class creates two relational tables in a local SQLite file. The `RAWDATA` table stores one row per commit–file pair (columns: `SHA`, `AUTHOR_NAME`, `CMT_DATE`, `MESSAGE`, `PATH`, `DIFF`, `TASK_NAME`). The `TASK` table stores one row per unique issue (`NAME`, `TITLE`, `DESCRIPTION`, `COMMENTS`). B-tree indexes on `TASK.ID` and `TASK.NAME` support efficient join operations in later stages.

**Stage 2 — Commit Extraction.** The `GitConnector` class iterates over all commits on the configured branch using the `gitpython` library. For each commit, it iterates over all file diffs relative to the parent commit, extracting the modified file path and the unified diff text. Records are inserted in batches for performance. A `test_mode` flag limits processing to 100,000 records during development.

**Stage 3 — Task ID Extraction.** The `TaskExtractor` class scans each commit message with a configurable regular expression. The default pattern recognises identifiers in the formats `PROJECT-123` (appearing at the start of the message) and `[PROJECT-123]` (appearing in brackets). Extracted identifiers are written back to the `RAWDATA.TASK_NAME` column and deduplicated into the `TASK` table.

**Stage 4 — Task Enrichment.** The `TaskFetcher` class queries the Jira instance for the title, description, and comment thread of each task. Three interchangeable connector strategies are provided: a REST API connector (fastest; requires credentials), an HTML-parsing connector (works with public instances), and a Selenium browser-automation connector (for JavaScript-heavy deployments). A configurable rate limiter (default: 50 requests per minute) prevents throttling.

**Stage 5 — Unified Dataset.** Upon completion, the SQLite database encodes a directed bipartite relation: each task in the `TASK` table is linked to one or more `RAWDATA` rows representing the files modified in its resolution. For the SonarQube dataset this relation contains 469,836 edges.

### 3.3 Database Schema and Task–Code Relation

The fundamental data model is a many-to-one relationship between commit–file pairs and tasks. One task may span multiple commits; each commit touches multiple files; therefore a single task is associated with a set of file paths. This set constitutes the **ground truth** used for evaluation. The schema is shown in **Diagram 3**.

> **[Diagram 3 — Database Entity-Relationship Diagram]**
> *What to draw:* A standard ER diagram with two entity boxes. Left box: `TASK` with attributes (PK) `ID`, `NAME`, `TITLE`, `DESCRIPTION`, `COMMENTS`. Right box: `RAWDATA` with attributes (PK) `ID`, `SHA`, `AUTHOR_NAME`, `CMT_DATE`, `MESSAGE`, `PATH`, `DIFF`, (FK) `TASK_NAME`. Connect them with a crow's-foot notation line showing 1-to-many (one TASK, many RAWDATA rows). Add a small annotation box showing the dataset statistics: 9,799 tasks, 469,836 commit-file pairs, 12,532 unique files, 27 modules.
> *Why this diagram is needed:* The logical structure of the data store is the basis for understanding how ground truth is derived. Readers unfamiliar with the schema may otherwise confuse commit counts with task counts or file counts.

---

## 4. Retrieval Experiment Framework: exp3

### 4.1 Design Philosophy

The *exp3* framework implements a **Retrieval-Augmented Generation (RAG)** approach in which task descriptions are encoded as dense vectors by pre-trained transformer models and matched against a vector index of code artefacts. The experiment is designed as a **full-factorial study**: every combination of four independent variables is evaluated, yielding a comprehensive measurement grid.

The four independent variables and their levels are:

- **Source variant** (query construction): `title` | `desc` | `comments`
- **Target variant** (index granularity): `file` | `module`
- **Window variant** (training recency): `w100` | `w1000` | `all`
- **Split strategy** (train/test partitioning): `recent` | `modn`

This gives 3 × 2 × 3 × 2 = **36 base experiment configurations** per embedding model. With seven embedding models evaluated, the total experiment space contains **252 configurations**. The factorial structure is illustrated in **Diagram 4**.

> **[Diagram 4 — Factorial Experiment Design Matrix]**
> *What to draw:* A three-dimensional matrix or a nested table. The outermost axis is the embedding model (7 rows: bge-small, bge-large, bge-m3, gte-qwen2, gte-large, nomic-embed, e5-large). The second axis is the source variant (3 column groups: title / desc / comments). Within each group, two sub-columns for target variant (file / module). The rows within each model block are the window variants (w100, w1000, all) × split strategies (recent, modn). Each cell represents one experiment run and contains the configuration identifier used in the vector database collection name. Use alternating row colours to separate the three window levels. Highlight the baseline configuration (bge-small, desc, file, w1000, recent) with a border.
> *Why this diagram is needed:* The sheer number of configurations makes it difficult to appreciate the systematic coverage of the experiment without a visual summary. The matrix also directly communicates to the reader that there are no gaps — every meaningful combination is evaluated.

### 4.2 Embedding Models

Seven pre-trained sentence transformer models were selected to cover a range of architectural choices and training objectives:

| Model Key | HuggingFace Identifier | Dimension | Characteristics |
|---|---|---|---:|
| bge-small | BAAI/bge-small-en-v1.5 | 384 | Fast baseline |
| bge-large | BAAI/bge-large-en-v1.5 | 1024 | Higher capacity BGE |
| bge-m3 | BAAI/bge-m3 | 1024 | Multilingual |
| gte-qwen2 | Alibaba-NLP/gte-Qwen2-1.5B | 1536 | Largest, highest quality |
| gte-large | thenlper/gte-large | 1024 | Technical text optimised |
| nomic-embed | nomic-ai/nomic-embed-text-v1.5 | 768 | Efficient mid-size |
| e5-large | intfloat/e5-large-v2 | 1024 | Microsoft E5 family |

All models are used in inference mode only; no fine-tuning is performed on the task corpus. This tests the **zero-shot transferability** of general-purpose sentence embeddings to the code-navigation domain.

### 4.3 ETL Pipeline

The `ETLPipeline` class (`etl_pipeline.py`) prepares the vector index for a given experiment configuration. Its six steps are shown in **Diagram 5**.

> **[Diagram 5 — ETL Pipeline Internal Data Flow]**
> *What to draw:* A horizontal swimlane diagram with two swimlanes: "Data Processing" (top) and "Storage" (bottom). In the Data Processing lane, show six labelled boxes connected left-to-right by arrows: (1) Load TASK + RAWDATA from SQLite → (2) Create Train/Test Split (200 test tasks) → (3) Apply Time Window (w100/w1000/all filter on training set) → (4) Combine Text Fields per source variant → (5) Batch-encode with transformer model → (6) Aggregate vectors per file or module (centroid). In the Storage lane, show two cylindrical database icons: left = SQLite (feeds into step 1), right = Vector DB (receives output of step 6). Draw a vertical arrow from step 2 downward labelled "test_set.json — ground truth" pointing to a document icon below the swimlane. This document icon should have a dashed arrow pointing forward to the Experiment Runner (outside this diagram).
> *Why this diagram is needed:* The ETL pipeline has both a linear data-transformation flow and a side-output (the test set with ground truth) that feeds the evaluation phase. Without a diagram these two paths are easily conflated.

**Step 1 — Data Loading.** All rows from the `TASK` and `RAWDATA` tables are loaded into Pandas DataFrames and joined on the task identifier.

**Step 2 — Train/Test Split.** Two strategies are supported. The *recent* strategy designates the 200 most recently created tasks as the test set, preserving chronological integrity (the model never sees future information during training). The *modn* strategy samples uniformly across the timeline (every *k*-th task), trading temporal realism for even distribution. The test set and its associated ground-truth file sets are serialised to `test_set.json`.

**Step 3 — Time Window Application.** Within the training set, only tasks from the most recent *w* entries are retained. The `w100` window limits training data to the 100 most recent tasks; `w1000` uses 1,000; `all` uses the full training corpus (~9,599 tasks). This variable directly tests whether temporal recency improves retrieval.

**Step 4 — Text Combination.** For each training task the relevant text fields are concatenated according to the source variant: `title` uses the `TITLE` column only; `desc` prepends the title to the full `DESCRIPTION`; `comments` further appends the `COMMENTS` field. Empty or null fields are silently omitted.

**Step 5 — Batch Encoding.** The combined text strings are encoded in batches (default batch size: 32) using the selected transformer model via the `sentence-transformers` library. The result is a matrix of shape *(N_train_tasks × embedding_dim)*.

**Step 6 — Vector Aggregation.** Each task's embedding is associated with the set of files (or modules) it touched, as recorded in `RAWDATA`. For the `file` target variant, each unique file path receives the mean of all task embeddings associated with it — a **centroid vector** representing the semantic profile of that file. For the `module` target variant, files are grouped by their top-level directory and a single centroid is computed for each module. These centroids constitute the searchable index and are uploaded to the vector database.

### 4.4 Evolution of Vector Storage Architecture

The choice of vector storage technology was not made a priori; it emerged from a sequence of practical failures that progressively revealed the operational requirements of large-scale embedding experiments. This subsection documents that evolution across three stages, illustrated in **Diagram 7**.

> **[Diagram 7 — Three-Generation Evolution of Vector Storage]**
> *What to draw:* A horizontal timeline with three labelled stages connected by arrows. Each stage is a box containing: (1) the generation name (exp2 / exp3-Qdrant / exp3-PostgreSQL), (2) the storage technology icon (SQLite cylinder / Qdrant logo / PostgreSQL elephant), (3) a bullet-point list of key properties, and (4) at the bottom, the reason for transition (a red annotation box). Between stages, the transition arrows should carry short labels: "scalability limit" (exp2 → Qdrant) and "data loss on restart" (Qdrant → PostgreSQL). Above the timeline, show a horizontal axis labelled "Development Timeline". Use green shading for the final (PostgreSQL) stage to indicate the adopted solution.
> *Why this diagram is needed:* The architectural decision to use PostgreSQL + pgvector is not self-evident; without the historical context of the two failed alternatives it appears unnecessarily complex. The diagram makes the reasoning visible and demonstrates that the design was empirically validated rather than arbitrarily chosen.

#### Stage 1 — Embedded Vectors in SQLite (exp2)

In the second experimental generation (*exp2*), the problem of vector storage was solved by the simplest possible means: vectors were serialised as JSON strings and stored directly inside the existing SQLite database in three dedicated tables — `SIMRGL_TASK_VECTOR`, `SIMRGL_MODULE_VECTOR`, and `SIMRGL_FILE_VECTOR`. Each row held a task or file identifier and the corresponding floating-point array as a JSON text column:

```
SIMRGL_TASK_VECTOR(ID, TASK_ID, VECTOR TEXT)
SIMRGL_MODULE_VECTOR(ID, MODULE_ID, VECTOR TEXT, STRATEGY)
SIMRGL_FILE_VECTOR(ID, FILE_ID, VECTOR TEXT, STRATEGY)
```

At query time, all vectors were loaded from the database into NumPy arrays in memory and similarity was computed exhaustively using `sklearn.metrics.pairwise.cosine_similarity`. This approach has clear virtues: it requires no external infrastructure, the entire experimental state is contained in a single portable file, and the implementation is transparent.

*exp2* also supported a richer set of embedding models than the subsequent generations, reflecting the exploratory nature of this stage: a custom Word2Vec model trained on the task corpus itself (Gensim, CBOW/Skip-gram, 100-dimensional), pre-trained FastText vectors from Common Crawl (300-dimensional with subword OOV handling), Stanford GloVe vectors (50–300 dimensions from the 6B token corpus), BERT contextual embeddings via HuggingFace Transformers (mean pooling over the last hidden layer), and sentence-transformers models accessed both locally and via the OpenAI embedding API. An abstract `BaseVectorizer` interface unified all five families, and a factory class (`vectorizer_factory.py`) handled instantiation from configuration.

The SQLite storage approach broke down as the experiment scaled. With 12,532 files and embedding dimensions up to 1,024, loading all vectors into memory for every test query was feasible only marginally. More importantly, the exhaustive O(*n × d*) similarity computation provided no indexing benefit, making the approach fundamentally unscalable to larger datasets or higher-dimensional models. The lack of approximate nearest-neighbour (ANN) indexing was identified as the primary limitation requiring architectural change in the next generation.

#### Stage 2 — Qdrant Vector Database (exp3, initial)

The transition to *exp3* introduced a dedicated vector database for the first time. **Qdrant** was selected as the initial backend due to its native support for cosine-distance collections, efficient batch upsert operations, and straightforward Python client library. Qdrant was deployed as a Docker container at `localhost:6333`. The experiment framework communicated with it over HTTP/REST, uploading centroid vectors as named collections and issuing top-*k* approximate nearest-neighbour queries.

This architecture provided genuine ANN indexing with HNSW graphs, reducing per-query search time to sub-millisecond latency and decoupling the search infrastructure from the Python process memory. The shift also enabled a clean separation of concerns: the ETL pipeline was responsible only for computing and uploading vectors, while the experiment runner issued search requests without needing to manage in-memory arrays.

However, a critical operational problem was encountered during multi-day experiment runs: **Qdrant, in its default containerised configuration, operates as an in-memory store**. When the container was restarted — whether due to system reboots, resource pressure, or routine maintenance — all uploaded collections were lost. For an experiment complex requiring 18 collections per embedding model and 6–10 hours of computation per full run, this represented an unacceptable data-loss risk. Rebuilding all collections from scratch after each restart was not a viable workflow. The need for **durable, restart-safe persistence** became the decisive criterion for the next architectural revision.

> **Note on Qdrant persistence:** Qdrant does support on-disk persistence via a `storage` volume mount in its Docker configuration. However, at the time of the initial *exp3* deployment this option was not configured, and the default behaviour — which stores all data in the container's volatile filesystem — caused repeated data loss. Rather than debugging the container volume configuration, the decision was made to migrate to a storage backend where persistence is a first-class architectural property rather than an optional mount-point setting.

#### Stage 3 — PostgreSQL with pgvector (exp3, final)

The final and currently adopted storage architecture uses **PostgreSQL 15** with the `pgvector` extension. Vectors are stored in typed `vector(d)` columns within regular PostgreSQL tables under the `vectors` schema. An HNSW index (Hierarchical Navigable Small World) is maintained on each vector column, providing the same ANN search capability as Qdrant while inheriting all standard PostgreSQL persistence guarantees — data survives container restarts, process crashes, and power interruptions without any special configuration.

The migration required implementing a second backend class (`PostgresBackend`) behind the same abstract `VectorBackend` interface that had already been defined for Qdrant. This interface design, conceived during the Qdrant phase, proved its value: switching backends required changing a single configuration variable (`VECTOR_BACKEND = 'postgres'`) with no modifications to the ETL pipeline or experiment runner. The PostgreSQL backend stores each collection as a separate table containing columns for the vector, the file or module path, and the target type. Batch upsert operations use parameterised `INSERT` statements to prevent SQL injection.

The operational characteristics of the final architecture are: vector upload throughput of approximately 1,000–5,000 vectors per second (depending on dimensionality), ANN search latency below 10 ms per query, total storage footprint of approximately 112 MB for the full 18-collection experiment set at 384 dimensions, and zero data loss across arbitrary numbers of restart cycles.

The complete three-stage progression is summarised in the following table:

| Property | exp2 — SQLite JSON | exp3 — Qdrant | exp3 — PostgreSQL |
|---|---|---|---|
| Storage location | Embedded SQLite file | Container memory | PostgreSQL on-disk |
| Search algorithm | Brute-force cosine (O*n*) | HNSW ANN | HNSW ANN |
| Persistence across restarts | Yes (file-based) | **No** (in-memory default) | Yes (first-class) |
| External infrastructure | None | Docker container | Docker container |
| Configuration to switch | N/A | `VECTOR_BACKEND = 'qdrant'` | `VECTOR_BACKEND = 'postgres'` |
| Query latency | ~100 ms (10k vectors) | < 10 ms | < 10 ms |
| Max practical scale | ~50k vectors | Millions | Millions |
| Data loss risk | Low | **High** (default config) | Negligible |

### 4.5 Vector Database Layer

The framework supports two interchangeable vector database backends implemented through a common abstract interface (`VectorBackend` ABC in `vector_backends.py`):

- **Qdrant** (Docker container, `localhost:6333`): cosine-distance collections with batch upsert. Retained as an optional alternative when persistence is managed via explicit volume mounts.
- **PostgreSQL + pgvector** (`localhost:5432`): HNSW-indexed tables in the `vectors` schema, supporting sub-millisecond approximate nearest-neighbour queries. This is the recommended and default backend.

Collections are named using the pattern `rag_exp_{source}_{target}_{window}_{split}`, making the configuration of each collection self-documenting. A typical experiment run creates 18 collections per embedding model (3 sources × 2 targets × 3 windows, for one split strategy).

### 4.6 Experiment Execution and Metric Computation

The `ExperimentRunner` class (`run_experiments.py`) iterates over all configured variant combinations. For each combination it encodes the 200 test queries, issues top-10 nearest-neighbour searches against the corresponding collection, and compares the retrieved file or module paths against the ground-truth sets from `test_set.json`.

Four information-retrieval metrics are computed per query and averaged over the test set:

$$\text{AP} = \frac{1}{|R|} \sum_{k=1}^{K} P@k \cdot \text{rel}(k)$$

$$\text{RR} = \frac{1}{\text{rank of first relevant result}}$$

$$P@K = \frac{|\text{retrieved}_K \cap \text{relevant}|}{K}$$

$$R@K = \frac{|\text{retrieved}_K \cap \text{relevant}|}{|\text{relevant}|}$$

where $R$ is the set of relevant files and $\text{rel}(k)$ is a binary relevance indicator at rank $k$. Mean AP (MAP) and Mean RR (MRR) are the primary metrics; Precision and Recall at $K \in \{1, 3, 5, 10\}$ provide a profile of retrieval behaviour across rank positions.

Results from all configurations are appended to a single CSV file (`experiment_results.csv`) containing 18 metric columns plus the four variant identifiers, enabling downstream comparative analysis.

---

## 5. Evaluation Protocol

### 5.1 Dataset Characteristics

All experiments use the Apache SonarQube open-source project as the evaluation corpus. Its characteristics make it a demanding and realistic benchmark:

| Property | Value |
|---|---|
| Total unique Jira issues | 9,799 |
| Total commit–file pairs | 469,836 |
| Distinct source files | 12,532 |
| Top-level modules | 27 |
| Date range | Full project history |
| Test set size | 200 most recent tasks |
| Training set size (max) | ~9,599 tasks |

The dataset is characterised by high **dispersion** (a task on average touches a small subset of a large file space) and significant **vocabulary mismatch** between natural-language task descriptions and technical file-path identifiers, making it a hard retrieval problem.

### 5.2 Baseline Comparison

The results of the *exp3* embedding-based approach are compared against a TF-IDF baseline established in the earlier *exp0* experiment. The baseline applied classical term-frequency weighting to task descriptions and file paths, achieving MAP@10 of 0.5–1.5 % with computation times of 4–48 hours per run. This establishes the lower bound against which the dense retrieval approach is measured.

---

## 6. Key Results

The principal findings across all evaluated configurations are summarised in **Diagram 6** and the table below.

> **[Diagram 6 — Results Overview: Metric Comparison Across Key Configurations]**
> *What to draw:* A grouped bar chart with the x-axis showing six experiment configurations: (1) TF-IDF Baseline, (2) bge-small / title / file / all, (3) bge-small / desc / file / all, (4) bge-small / desc / module / all, (5) bge-small / desc / file / w1000, (6) gte-qwen2 / desc / file / w1000. For each configuration show four grouped bars representing MAP@10, MRR, P@10, and R@10. Use a consistent colour scheme across all bars for each metric. Add a horizontal dashed reference line at the TF-IDF MAP@10 maximum (1.5 %). Annotate the best-performing configuration with a star symbol. The y-axis runs from 0 to 8 % (percentage). Include a legend for metric types.
> *Why this diagram is needed:* The central empirical claim — that dense embeddings substantially outperform TF-IDF — needs a visual summary that makes the magnitude of improvement immediately apparent and allows simultaneous comparison of multiple metrics and configurations without requiring the reader to parse a dense table.

| Configuration | MAP@10 | MRR | P@10 | R@10 |
|---|---|---|---|---|
| TF-IDF baseline (exp0) | 0.5–1.5 % | 0.8–2.0 % | 0.3–1.2 % | 1.0–2.5 % |
| bge-small / title / file / all | ~1.8 % | ~3.0 % | ~0.9 % | ~2.5 % |
| bge-small / desc / file / all | ~2.8 % | ~5.0 % | ~1.4 % | ~3.8 % |
| bge-small / desc / module / all | ~3.0 % | ~5.2 % | ~1.2 % | ~4.2 % |
| bge-small / desc / file / w1000 | ~2.9 % | ~5.1 % | ~1.5 % | ~3.9 % |
| gte-qwen2 / desc / file / w1000 | ~3.5 % | ~6.0 % | ~1.8 % | ~4.5 % |

**Answers to research questions:**

- **RQ1 (Granularity):** Module-level retrieval yields higher recall; file-level yields higher precision. The choice depends on the use-case: initial exploration favours modules; precise editing favours files.
- **RQ2 (Semantic density):** Adding the full description to the title consistently improves MAP by approximately 50 % relative to title-only queries, confirming that richer semantic context improves retrieval.
- **RQ3 (Noise tolerance):** Including user comments degrades MAP in all evaluated configurations, indicating that conversational comment text introduces more noise than signal. The `desc` variant is preferred.
- **RQ4 (Temporal window):** The `w1000` window matches or exceeds the `all` window in most configurations, suggesting that approximately the most recent twelve months of project history provide sufficient and more relevant training signal than the full lifetime.

---

## 7. Infrastructure and Execution Environment

The experimental complex runs on a single workstation. The primary dependencies are: Python 3.10+, PyTorch (CPU or GPU), the `sentence-transformers` library, PostgreSQL 15 with the `pgvector` extension (or Qdrant 1.7+ as an alternative), and Streamlit for the results dashboard. Container orchestration is provided by Podman Compose.

Two execution modes are supported. A **quick-test mode** (≈ 10 minutes) limits the training window to the last 1,000 tasks, allowing rapid verification of system integrity. A **full experiment mode** (60–70 minutes for a single embedding model; 6–10 hours for all seven models) runs the complete factorial design, generating backups of the vector database after each model. A checkpoint mechanism in `run_experiments.py` serialises progress to `checkpoint.json`, enabling interrupted runs to resume from the last completed configuration.

The interactive **Streamlit dashboard** (`experiment_ui.py`) provides three views: a results table with filtering by any variant dimension; a live semantic search interface where the user can enter a free-text task description and receive ranked file or module recommendations in real time; and a research questions panel explaining the experimental design and metric interpretations.

---

## 8. Summary

The experimental programme complex described in this section constitutes a self-contained, reproducible research instrument for the study of automated task-to-code retrieval. Its principal characteristics are:

1. **End-to-end coverage** — from raw version-control data to evaluated retrieval metrics, with no manual annotation required.
2. **Evolutionary design** — the complex developed through three experimental generations, with each architectural revision motivated by a concrete operational failure rather than theoretical preference: exp2 (SQLite JSON vectors) revealed scalability limits; the initial exp3 Qdrant deployment revealed data-loss risk under in-memory default configuration; the final PostgreSQL backend resolved both concerns.
3. **Factorial design** — systematic variation of four independent variables yields 252 evaluated configurations, enabling multi-dimensional comparative analysis.
4. **Dual-backend flexibility** — the abstract vector database interface, introduced during the Qdrant phase and preserved through the PostgreSQL migration, allows transparent switching of storage backends without modifying experiment logic.
5. **Reproducibility** — checkpoint-based execution, deterministic data splits, and serialised test sets ensure that results can be reproduced and extended.
6. **Quantified improvement** — the dense embedding approach delivers a 2–3× improvement in MAP and MRR over the classical TF-IDF baseline, with a 10–100× reduction in computation time.

The complex provides the empirical foundation for the theoretical contributions of the SIMARGL research programme, in particular the SIMARGL evaluation metrics (Novelty@K and Structurality@K) which extend the standard IR metrics used here with domain-specific dimensions of recommendation quality relevant to software architecture and code health.
