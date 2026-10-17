# Experiment 6: Evaluation of RAG Retrieval Modes via SIMARGL CLI

## Objective
Evaluate and compare the retrieval performance of the `simargl` CLI system across three distinct modes: `task`, `file`, and `aggr`. The goal is to determine which mode yields the best precision and recall when predicting code modifications based on natural language task descriptions.

## Research Questions (RQs)

### RQ1: Individual Mode Comparison
**Which individual retrieval mode (`task`, `file`, or `aggr`) provides the best overall performance?**
- **Hypothesis:** One mode will consistently outperform the others across all metrics
- **Decision Rule:** If any mode achieves statistically significant better MAP/MRR across >80% of test tasks, adopt that mode exclusively

### RQ2: Combinatorial Advantage and Combination Strategies
**Can we improve retrieval accuracy by combining results from multiple retrieval modes?**
- **RQ2.1: Combinatorial Advantage**
  - **Hypothesis:** No single mode dominates all cases; each mode excels in different scenarios, and their combination yields superior results
  - **Rationale:** 
    - `file` mode: Direct semantic search may find semantically similar files not directly linked to similar tasks
    - `task` mode: Leverages historical task-to-file mappings, good for recurring patterns
    - `aggr` mode: Averages unit vectors, smooths out noise in individual task vectors
  - **Decision Rule:** If no single mode consistently wins, investigate combination strategies

- **RQ2.2: Combination Strategies**
  - **Which method of combining retrieval results yields the best performance?**
  - **Strategies to evaluate:**
    1. **Intersection (AND):** Files present in ALL three modes' top-K results
    2. **Union (OR):** All unique files from all three modes' top-K results
    3. **Top-K per mode:** Take top-K from each mode, then re-rank by combined score
    4. **Weighted fusion:** Weight results by individual mode performance
    5. **Reciprocal Rank Fusion (RRF):** Combine ranks using RRF formula
    6. **Borda count:** Rank files by positions across modes, sum ranks

### RQ3: Multi-Granularity Prior Knowledge
**Can we improve file-level prediction accuracy by using coarse-grained predictions (modules, file extensions) as prior knowledge?**
- **Hypothesis:** Predicting coarse-grained units (modules, file extensions) first can provide valuable prior information that improves fine-grained file prediction accuracy
- **Rationale:** 
  - Current RAG systems don't incorporate Bayesian prior knowledge directly
  - If module-level prediction is reliable, it can filter/rerank results from fine-grained modes
  - File extensions may serve as semantic proxies (e.g., `.py` = Python logic, `.sql` = database operations)
  - Different modes may work better at different granularity levels
- **Investigation:**
  1. Which mode (`task`/`file`/`aggr`) works best for **module-level** prediction?
  2. Which mode works best for **extension-level** prediction?
  3. Which mode works best for **file-level** prediction?
  4. Can module predictions be used as filtering criteria for file predictions?
  5. Can extension predictions be used as weighting factors?
  6. Does multi-granularity filtering improve final MAP/MRR?
- **Example Workflow:**
  ```
  Query: "Fix authentication bug"
  │
  ├─ Stage 1: Predict MODULES (cheap, coarse-grained)
  │   └─ Top modules: server, sonar-server  → Use as FILTER
  │
  ├─ Stage 2: Predict EXTENSIONS
  │   └─ Top extensions: .js, .ts, .java  → Use as WEIGHT
  │
  ├─ Stage 3: Predict FILES (original task)
  │   └─ Top files: login.js, auth.ts, session.java
  │
  └─ Apply prior:
      - Filter: keep only files from predicted modules
      - Weight: boost files with predicted extensions
      - Result: improved ranking
  ```

## Methodology

### 1. Data Preparation
- **Codebase Repository:** The target project code to be indexed for file-level retrieval.
- **Task Tracker Database:** A SQLite database containing task history (ID, TITLE, DESCRIPTION, etc.).
- **Dataset Splitting:** Split tasks chronologically into Training and Testing sets. Ground truth for Test tasks is determined by the files modified during those specific tasks.

### 2. Indexing (Training Phase)
For the Training Set, build three indices using `simargl`:
- **Files Index:** `simargl index files <repo_path> --project exp6_eval`
- **Units (Tasks) Index:** `simargl index units <db_path> --project exp6_eval`
- **Aggregated Index:** `simargl` handles `aggr` mode automatically during search if the units index (task-to-file mapping) is built.

### 3. Evaluation (Testing Phase)
For each task in the Test Set, query the `simargl` index using the three retrieval modes:

1.  **`file` mode:** Direct semantic search in the Files Index.
2.  **`task` mode:** Semantic search in the Units Index, retrieving tasks and then listing files modified in those tasks.
3.  **`aggr` mode:** Semantic search using the aggregated unit vectors per module/file.

Execute queries using the `simargl` API or CLI:
```bash
# Example command for each mode:
simargl search "query_text" --mode task --project exp6_eval
simargl search "query_text" --mode file --project exp6_eval
simargl search "query_text" --mode aggr --project exp6_eval
```

### 4. Evaluation Protocol

#### Single-Mode Evaluation
For each query, compare retrieved file paths against the ground truth files. Calculate:
- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **P@K** (Precision at K: 5, 10, 20)
- **Recall@K** (Recall at K: 5, 10, 20)

#### Combination Strategy Evaluation
For each combination strategy (RQ2.2), evaluate:
- **Individual mode performance** (baseline)
- **Combined performance** using each strategy
- **Statistical significance** of improvement (Wilcoxon signed-rank test)

#### Winner Detection
- **Deterministic winner:** One mode with significantly better MAP across all tasks
- **Mixed performance:** Multiple modes excel in different scenarios → need combination
- **No clear winner:** All modes similar → investigate combinations

## Execution Plan

### Phase 1: Baseline (Individual Modes)
1. Setup Python environment and install `simargl`
2. Prepare SQLite database (train/test split)
3. Index codebase with `simargl index files`
4. Index tasks with `simargl index units`
5. Run evaluation for each mode (`task`, `file`, `aggr`)
6. Calculate baseline metrics for each mode
7. **Decision:** Does any mode clearly dominate?

### Phase 2: Combination Strategies (if needed)
If no single mode dominates:
1. Implement combination strategies (RQ2.2):
   - Intersection, Union, Top-K per mode, Weighted fusion, RRF, Borda count
2. For each strategy:
   - Query all three modes
   - Apply combination rule
   - Calculate metrics
   - Compare to baseline
3. Select best combination strategy

### Phase 3: Multi-Granularity Prior Knowledge
1. Analyze which mode works best for module-level prediction
2. Analyze which mode works best for extension-level prediction  
3. Implement and evaluate Bayesian-inspired filtering using module/extension priors
4. Compare filtered vs. non-filtered results
5. Generate final report with recommendations

### 5. Output Files
- `results/mode_comparison.csv` - Baseline metrics per mode (RQ1)
- `results/combination_strategies.csv` - Metrics per combination strategy (RQ2.2)
- `results/multi_granularity.csv` - Module/extension/file performance per mode (RQ3)
- `results/prior_filtering.csv` - Evaluation of Bayesian filtering (RQ3)
- `results/README.md` - Complete evaluation report

## Research Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: Baseline                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Task      │  │   File      │  │   Aggr      │           │
│  │   Mode      │  │   Mode      │  │   Mode      │           │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
│         │                 │                 │                 │
│         └─────────────────┴─────────────────┘                 │
│                           │                                   │
│                   ┌───────▼───────┐                           │
│                   │  Baseline     │                           │
│                   │  Comparison   │                           │
│                   └───────┬───────┘                           │
└───────────────────────────┼───────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │ Does one mode dominate?   │
              └─────────────┬─────────────┘
                            │
                ┌───────────┴───────────┐
                │    YES?               │
                │ (Deterministic        │
                │  Winner Identified)   │
                └───────────┬───────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Use best single mode     │
              └───────────────────────────┘

                            │ (Mixed Performance)
                            ▼
                    ┌───────────────────┐
                    │   Phase 2:        │
                    │  Combinations     │
                    └───────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │  Intersection │    Union      │
            │ Top-K fusion  │   RRF         │
            │  Borda count  │  Weighted     │
            └───────────────┴───────────────┘
                            │
                    ┌───────▼───────┐
                    │  Best         │
                    │  Combination  │
                    │  Strategy     │
                    └───────────────┘
```
