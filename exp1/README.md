# CodeXplorer Analysis Scripts

This folder contains Python scripts for advanced analysis of code repository and task tracking data. These scripts process data from the CodeXplorer database and generate additional analytical tables for research and insights.

## Prerequisites

Before running these scripts, you must have:

1. **SQLite database** populated with the following tables:
   - `RAWDATA` - Git commit data (with fields: ID, SHA, AUTHOR_NAME, AUTHOR_EMAIL, CMT_DATE, MESSAGE, PATH, DIFF, TASK_NAME)
   - `TASK` - Jira task data (with fields: ID, NAME, TITLE, DESCRIPTION, COMMENTS)

2. **Python dependencies**:
   ```bash
   pip install tqdm
   ```

## Scripts Overview

### 1. title_term.py
**Purpose**: Extracts and analyzes terms from task titles.

**Input Tables**:
- `TASK` (reads NAME and TITLE fields)

**Output Tables Created**:
- `TITLE_TERM` - Stores unique terms extracted from task titles
  - `ID` (INTEGER PRIMARY KEY): Unique term identifier
  - `TERM` (TEXT UNIQUE): The term itself

- `TITLE_TASK_TERM` - Links tasks to individual term occurrences
  - `ID` (INTEGER PRIMARY KEY): Record identifier
  - `TASK_NAME` (TEXT): Task identifier
  - `TERM_ID` (INTEGER): Reference to TITLE_TERM.ID

- `TITLE_TASK_TERM_AGG` - Aggregated term counts per task
  - `TASK_NAME` (TEXT): Task identifier
  - `TERM_ID` (INTEGER): Reference to TITLE_TERM.ID
  - `CNT` (INTEGER): Number of times term appears in this task title

**What it does**:
- Normalizes task titles (lowercase, removes punctuation)
- Extracts individual words (terms) from titles
- Creates a vocabulary of unique terms
- Links each task to its terms with occurrence counts

---

### 2. module_task.py
**Purpose**: Builds hierarchical file/module structure from commit file paths.

**Input Tables**:
- `RAWDATA` (reads TASK_NAME and PATH fields)

**Output Tables Created**:
- `MODULE` - Hierarchical file/folder structure
  - `ID` (INTEGER PRIMARY KEY): Unique module identifier
  - `PARENT_ID` (INTEGER): Reference to parent module (NULL for root)
  - `NAME` (TEXT): Module/file name
  - `LVL` (INTEGER): Depth level in hierarchy (0 = root)
  - `FILE` (INTEGER): 1 if this is a file, 0 if directory
  - `CNT` (INTEGER): Number of times this module appears in commits

- `MODULE_TASK` - Links tasks to modules/files
  - `TASK_NAME` (TEXT): Task identifier
  - `MODULE_ID` (INTEGER): Reference to MODULE.ID
  - `CNT` (INTEGER): Number of commits for this task-module pair

**What it does**:
- Parses file paths from commits (e.g., `src/main/java/File.java`)
- Creates hierarchical tree structure of directories and files
- Links each task to all modules/files it touched
- Counts how many times each task touched each module

---

### 3. term_rank.py
**Purpose**: Calculates advanced metrics and rankings for terms based on their distribution across files and modules.

**Input Tables**:
- `TITLE_TERM`
- `TITLE_TASK_TERM_AGG`
- `MODULE_TASK`
- `MODULE`

**Output Tables Created**:
- `TERM_RANK` - Term metrics and rankings
  - `TERM_ID` (INTEGER PRIMARY KEY): Reference to TITLE_TERM.ID
  - `RANK` (INTEGER): Hierarchical rank (depth of dispersion)
  - `TASK_CNT` (INTEGER): Number of tasks containing this term
  - `FILE_CNT` (INTEGER): Number of unique files associated with term
  - `ROOT_CNT` (INTEGER): Number of unique root modules
  - `HHI_FILE` (REAL): Herfindahl-Hirschman Index for file concentration
  - `HHI_ROOT` (REAL): Herfindahl-Hirschman Index for root module concentration
  - `COMPOSITE_INDEX` (REAL): Combined concentration metric (HHI_FILE × HHI_ROOT)

**What it does**:
- Analyzes how terms are distributed across the codebase
- Calculates concentration metrics (HHI) to identify focused vs. scattered terms
- Determines hierarchical rank based on common ancestor depth
- Helps identify general vs. specific terms

**Metrics Explained**:
- **RANK**: Higher = term spans wider hierarchy (more general)
- **HHI_FILE**: Lower = term is spread across many files (general), Higher = concentrated in few files (specific)
- **COMPOSITE_INDEX**: Combined measure of specificity
- **TASK_CNT**: Popularity of the term

---

### 4. interlink.py
**Purpose**: Calculates co-occurrence matrices for terms and files.

**Input Tables**:
- `TASK`
- `TITLE_TASK_TERM`
- `MODULE_TASK`
- `MODULE`

**Output Tables Created**:
- `TERM_LINKS` - Term co-occurrence matrix
  - `L_TERM` (INTEGER): First term ID
  - `R_TERM` (INTEGER): Second term ID
  - `CNT` (INTEGER): Number of tasks where both terms appear together

- `FILE_LINKS` - File co-occurrence matrix
  - `L_FILE` (INTEGER): First file/module ID
  - `R_FILE` (INTEGER): Second file/module ID
  - `CNT` (INTEGER): Number of tasks that touched both files

**What it does**:
- Identifies which terms frequently appear together in task titles
- Identifies which files are frequently modified together
- Builds co-occurrence networks for relationship analysis
- Processes in batches to handle large datasets efficiently

---

## Execution Order

**IMPORTANT**: Run scripts in this exact order as they have dependencies:

```bash
# Step 1: Extract terms from task titles
python title_term.py

# Step 2: Build module hierarchy from file paths
python module_task.py

# Step 3: Calculate term rankings and metrics
python term_rank.py

# Step 4: Calculate co-occurrence matrices
python interlink.py
```

## Configuration

Before running, **edit the `DB_FILE` variable** in each script to point to your database:

```python
# In each .py file, update this line:
DB_FILE = "data/sonar.db"  # Change to your database path
```

Alternatively, you can use the provided `config.py` file, but you'll need to modify each script to import from it.

## Usage Example

```bash
# 1. Ensure your database is ready
ls data/sonar.db  # Verify database exists

# 2. Run scripts in order
python title_term.py
# Output: "Дані успішно заповнено." (Data successfully populated)

python module_task.py
# Output: "Дані успішно заповнено."

python term_rank.py
# Output: Shows progress bar and "Розрахунок метрик завершено успішно."

python interlink.py
# Output: Shows progress bar and "Збереження завершено успішно."
```

## Analyzing Results

After running all scripts, you can analyze the data using SQL queries:

### Example Queries

```sql
-- Find most common terms in task titles
SELECT t.TERM, COUNT(DISTINCT tt.TASK_NAME) as task_count
FROM TITLE_TERM t
JOIN TITLE_TASK_TERM tt ON t.ID = tt.TERM_ID
GROUP BY t.TERM
ORDER BY task_count DESC
LIMIT 20;

-- Find most specific terms (high concentration)
SELECT t.TERM, r.COMPOSITE_INDEX, r.TASK_CNT, r.FILE_CNT
FROM TERM_RANK r
JOIN TITLE_TERM t ON r.TERM_ID = t.ID
WHERE r.TASK_CNT >= 5  -- At least 5 tasks
ORDER BY r.COMPOSITE_INDEX DESC
LIMIT 20;

-- Find most general terms (low concentration, high spread)
SELECT t.TERM, r.ROOT_CNT, r.FILE_CNT, r.TASK_CNT, r.HHI_FILE
FROM TERM_RANK r
JOIN TITLE_TERM t ON r.TERM_ID = t.ID
WHERE r.TASK_CNT >= 10
ORDER BY r.HHI_FILE ASC
LIMIT 20;

-- Find terms that frequently co-occur
SELECT
    t1.TERM as term1,
    t2.TERM as term2,
    tl.CNT as co_occurrence_count
FROM TERM_LINKS tl
JOIN TITLE_TERM t1 ON tl.L_TERM = t1.ID
JOIN TITLE_TERM t2 ON tl.R_TERM = t2.ID
ORDER BY tl.CNT DESC
LIMIT 20;

-- Find files that are frequently modified together
SELECT
    m1.NAME as file1,
    m2.NAME as file2,
    fl.CNT as co_modification_count
FROM FILE_LINKS fl
JOIN MODULE m1 ON fl.L_FILE = m1.ID
JOIN MODULE m2 ON fl.R_FILE = m2.ID
WHERE m1.FILE = 1 AND m2.FILE = 1  -- Both are files
ORDER BY fl.CNT DESC
LIMIT 20;

-- Get hierarchical module structure
SELECT
    SUBSTR('    ', 1, LVL * 2) || NAME as hierarchy,
    LVL,
    FILE,
    CNT
FROM MODULE
ORDER BY ID;

-- Find most frequently modified modules
SELECT NAME, LVL, FILE, CNT
FROM MODULE
WHERE FILE = 0  -- Directories only
ORDER BY CNT DESC
LIMIT 20;
```

## Database Schema Summary

After running all scripts, your database will have these tables:

### Input Tables (must exist)
- `RAWDATA` - Git commits
- `TASK` - Jira tasks

### Analysis Tables (created by scripts)
- `TITLE_TERM` - Term vocabulary
- `TITLE_TASK_TERM` - Task-term links
- `TITLE_TASK_TERM_AGG` - Aggregated term counts
- `MODULE` - File hierarchy
- `MODULE_TASK` - Task-module links
- `TERM_RANK` - Term metrics
- `TERM_LINKS` - Term co-occurrence
- `FILE_LINKS` - File co-occurrence

## Performance Notes

- **title_term.py**: Fast, processes in memory
- **module_task.py**: Moderate, builds tree structure incrementally
- **term_rank.py**: Slow for large datasets (analyzes every term-task-file combination)
- **interlink.py**: Moderate, uses batch processing (BATCH_SIZE = 500)

For very large databases (>100,000 tasks):
- Consider increasing `BATCH_SIZE` in interlink.py
- Run term_rank.py overnight if needed
- Monitor disk space (indexes can be large)

## Troubleshooting

**"no such table: TASK"**
→ Ensure your database was created by the CodeXplorer data gathering tool first

**"no such column: TITLE"**
→ Your TASK table needs the TITLE field populated from Jira

**"Out of memory"**
→ Increase BATCH_SIZE in interlink.py or process fewer tasks

**Scripts run but tables are empty**
→ Check that TASK.TITLE and RAWDATA.PATH fields are populated

**Ukrainian text in output**
→ This is normal, scripts use Ukrainian for status messages

## Research Applications

These tables enable various research questions:

1. **Term Analysis**: Which terms are most specific/general?
2. **Module Coupling**: Which files change together frequently?
3. **Task Clustering**: Group similar tasks by term overlap
4. **Impact Analysis**: Predict file changes based on task description
5. **Vocabulary Evolution**: Track term usage over time

## Credits

These scripts are part of the CodeXplorer project for analyzing software repositories and task tracking data.
