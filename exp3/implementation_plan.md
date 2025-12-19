# Implementation Plan - RAG Research Experiment

## Goal
Build a RAG system to answer the 4 Research Questions by testing specific variants of Data Source, Target Granularity, and Time Window.

## User Review Required
> [!IMPORTANT]
> **Vector Database:** Qdrant (via Docker).
> **Embedding Model:** BAAI/bge-small-en-v1.5 (local).
> **Aggregation:** Average Task Vector strategy (Centroid).
> **Evaluation:** Two Test Strategies (Recent vs. ModN).

## Experiment Variants
### 1. Data Source Variants (RQ2 & RQ3)
- **S1: TITLE**
- **S2: TITLE + DESCRIPTION**
- **S3: TITLE + DESCRIPTION + COMMENTS**

### 2. Target Variants (RQ1)
- **T1: FILE**
- **T2: MODULE** (Root folder)

### 3. Time Window Variants (RQ4)
- **W1: NEAREST 100** (Train on last 100 before test)
- **W2: NEAREST 1000** (Train on last 1000 before test)
- **W3: ALL** (Train on all available history)

## Evaluation Protocol
We will support two distinct Test Set strategies. The Test Set size is configurable (default: 200).

### Strategy A: Recent Split (Time-based)
- **Test Set:** The most recent $N$ tasks (e.g., 200).
- **Train Set:** All tasks *older* than the split point.
- **Usage:** Used for W1, W2, and W3.

### Strategy B: ModN Split (Uniform Sampling)
- **Test Set:** Every $k$-th task from the entire history, selected to yield $N$ tasks (e.g., 200).
- **Train Set:** All tasks *except* the selected test tasks.
- **Usage:** Used primarily for W3 (ALL) to test performance across the project's lifetime, not just recent history.

> **Note:** To prevent data leakage, File Embeddings (Centroids) must be re-calculated for each Split Strategy, ensuring Test Task vectors are **excluded** from the average.

## Proposed Changes

### Infrastructure
#### [NEW] [docker-compose.yml](file:///c:/Project/codeXplorer/RAG/docker-compose.yml)
- Service: `qdrant` (Port 6333)

### Data Processing (ETL)
#### [NEW] [etl_pipeline.py](file:///c:/Project/codeXplorer/RAG/etl_pipeline.py)
- **Configurable Arguments:**
    - `--split_strategy`: `recent` or `modn`
    - `--test_size`: Number of tasks in test set (default: 200)
- **Logic:**
    1.  Select Test Set based on strategy.
    2.  Select Train Set (Knowledge Base) = All Tasks - Test Set.
    3.  Vectorize Train Tasks.
    4.  Calculate File Centroids using ONLY Train Tasks.
    5.  Upsert to Qdrant.
    6.  Save `test_set.json`.

### Experiment & UI
#### [NEW] [experiment_ui.py](file:///c:/Project/codeXplorer/RAG/experiment_ui.py)
- **Batch Evaluation:**
    - Load `test_set.json`.
    - Run retrieval for each task.
    - Calculate and display metrics (MAP, MRR, P@K, R@K).

## Verification Plan
- **Automated:** Run ETL with `--split_strategy modn` and verify test set IDs are distributed (not just recent).
- **Manual:** Compare evaluation results of "Recent" vs "ModN" on the "ALL" history variant.
