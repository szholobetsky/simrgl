# Proposed Research Questions for RAG Experiment

Here are the final four Research Questions (RQs). We have retained the Semantic Density and Noise Tolerance questions and restored the Temporal Dynamics question to complete the set.

## RQ1: Granularity Impact
**How does the granularity of the target program unit (File level vs. Module/Directory level) affect the accuracy of the recommendation?**
*   **Hypothesis:** Module-level retrieval will achieve higher Recall@k but lower Precision@k compared to File-level retrieval.
*   **Method:** Compare retrieval performance when the target vector represents a single file vs. a directory of files.

## RQ2: Semantic Density (Title vs. Description)
**Does the high "semantic density" of Task Titles provide a more accurate retrieval signal than the more detailed but less concentrated Task Descriptions?**
*   **Hypothesis:** Titles will outperform Descriptions in Precision@k because they contain the core intent with less noise.
*   **Method:** Direct comparison of `TITLE`-only vs. `DESCRIPTION`-only retrieval.

## RQ3: Noise Tolerance (Impact of Comments)
**Does the inclusion of Task Comments (high noise) degrade retrieval performance compared to using Descriptions alone?**
*   **Hypothesis:** Adding Comments to the embedding source will decrease Precision@k due to the high noise-to-signal ratio.
*   **Method:** Compare retrieval performance of `DESCRIPTION` vs. `DESCRIPTION + " " + COMMENTS`.

## RQ4: Temporal Dynamics (Restored)
**Does limiting the knowledge base to recent tasks (e.g., last 12 months) improve prediction accuracy for new tasks compared to using the entire project history?**
*   **Hypothesis:** A "Recent History" model will outperform a "Full History" model by reducing false positives from obsolete code associations.
*   **Method:** Compare retrieval performance of a model built on *All Tasks* vs. a model built on *Recent Tasks* (e.g., last 1 year).

---
**Implementation Note:**
For all RQs, we will use the **Average Aggregation Strategy** for File representation.
