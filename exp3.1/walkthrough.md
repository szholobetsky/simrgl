# RAG Experiment Walkthrough

The RAG system for linking Task Descriptions to Code Modules is now running.

## System Components
1.  **Vector Database:** Qdrant (running via Podman on port 6333).
2.  **ETL Pipeline:** `etl_pipeline.py` (Processes `sonar.db`, generates embeddings, upserts to Qdrant).
3.  **User Interface:** Streamlit App (running at `http://localhost:8501`).

## How to Use
1.  **Open the UI:** Navigate to [http://localhost:8501](http://localhost:8501) in your browser.
2.  **Interactive Search:**
    - Select "Interactive Search" in the sidebar.
    - Choose your variants (Source: Title/Desc/Comments, Target: File/Module).
    - Enter a Task Title and Description.
    - Click "Search" to see retrieved files.
3.  **Batch Evaluation:**
    - Select "Batch Evaluation" in the sidebar.
    - Click "Run Evaluation" to calculate MAP, MRR, P@K, R@K on the test set (Last 200 tasks).
    - Compare results across different configurations.

## Current Status
- **ETL:** Processing data (Embeddings generation in progress).
- **UI:** Ready.

> **Note:** If search returns empty results, the ETL pipeline might still be populating that specific collection. Please wait a few minutes and try again.

## Experiment Results
Initial evaluation on the last 200 tasks shows:

| Experiment_ID | Source | Target | MAP | MRR | P@10 | R@10 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| title_file | title | file | 0.0098 | 0.0133 | 0.0035 | 0.0164 |
| title_module | title | module | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| desc_file | desc | file | 0.0113 | 0.0183 | 0.0035 | 0.0177 |

*   **Best Performer:** `desc_file` (Description -> File) currently outperforms Title-based retrieval.
*   **Issues:** `title_module` yielded 0 results, indicating a potential issue with module path aggregation. Other variants (`comments`, `desc_module`) are pending ETL completion.

