# Concept: Component-Based Semantic Analysis for Task-to-Code Mapping

## Overview
Moving from a **Centroid-Based** (aggregate) model to a **Component-Based** (multi-vector) model for recommending files and modules based on Jira tasks.

## The Problem: Centroid Decay
In the current implementation, we group Jira tasks by file or module and calculate an `AVG()` or `SUM()` vector to represent that group. This "semantic fingerprint" has several flaws:

1.  **Semantic Dead Zones**: Averaging vectors for a module with multiple responsibilities (e.g., a `User` module handling Auth, Profile, and Billing) results in a "center" vector that might not closely match any specific sub-topic.
2.  **Loss of Specificity**: A new task about "Password Reset" may be highly similar to a historical task "ABC-123: Fix Login Bug," but the module's average vector is diluted by unrelated tasks like "Upload Avatar."
3.  **Outlier Pollution**: A large refactoring task that touches many modules can pull the average vector of those modules toward "Generic Cleanup," reducing the accuracy for functional matches.
4.  **Functional Overload (The "Hub File" Problem)**: Core files like `amortisation.java` often handle multiple intentions (Multicurrency, Future Projections, Client-specific logic). An average vector obfuscates these distinct responsibilities, making it hard to match a new task to a specific sub-function within a large file.

## The Solution: Component-Based Analysis
Instead of representing a module as one vector, we represent it as a **collection of historical task components**. 

### 1. Multi-Vector Representation
A module $M$ or file $F$ is defined as a set of task vectors that modified it:
$F = \{V_{ABC-123}, V_{XYZ-999}, \dots, V_{TaskN}\}$

### 2. Matching Strategy: Max-Similarity / Top-K
When a user provides a new task description $T_{new}$, we calculate the similarity against every individual task vector $V_{task}$ in the database.

*   **Scoring**: The relevance of a file $F$ is determined by its closest component:
    $Score(F) = \max_{V_i \in F} (\text{Similarity}(V_{T_{new}}, V_i))$
*   **Ranking**: Files are ranked by their highest-scoring component.

### 3. Contextual Preservation
If a previous task `INS-444` (Insurance-specific bug) modified both `client/insurance/Handler.java` and `core/amortisation.java`, both files inherit that "Insurance" component. A new insurance-related task will match `INS-444` and correctly identify both the client-specific and core files, without the core file's signal being "drowned out" by general amortisation logic.

### 3. Temporal Decay (Recency Weighting)
To account for code evolution, we apply a weight to each component based on its age:
$$WeightedScore(V_i) = \text{Similarity}(V_{T_{new}}, V_i) \times e^{-\lambda(t_{now} - t_{task})}$$
This ensures that recent architectural changes have more influence than legacy tasks.

## Benefits
*   **Higher Precision**: Matches the specific sub-functionality within a complex module.
*   **Explainability**: We can tell the user exactly *why* a file is recommended: *"File X is recommended because this task is 88% similar to ABC-123 which modified this file."*
*   **Resilience**: New, unrelated tasks in a module don't "corrupt" the matching capability for existing functionality.

## Implementation Plan

### Phase 1: Data Preparation
*   Ensure the vector database (e.g., Qdrant/PostgreSQL) stores the `task_id` and `module/file_path` as metadata for every vector.
*   Stop pre-calculating the aggregate `AVG()` vector for modules during indexing.

### Phase 2: Query Logic Update
*   **Search**: Perform a $K$-Nearest Neighbor (KNN) search for the new task vector against the *entire* collection of historical task vectors.
*   **Aggregation**:
    1.  Retrieve top $K$ matching tasks (e.g., $K=50$).
    2.  Group these tasks by their associated `module` or `file`.
    3.  Calculate a module score based on the top match within that group or a weighted sum of matches in that group.

### Phase 3: Explainability Layer
*   Return the `task_id` and original description of the closest historical match alongside the recommended file path.

### Phase 4: Temporal Tuning
*   Introduce the decay constant ($\lambda$) into the ranking algorithm to prioritize recent code changes.
