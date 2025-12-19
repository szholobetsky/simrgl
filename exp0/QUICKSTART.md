# Quick Start Guide - TF-IDF Experiment (exp0)

## ⚠️ Read This First

**This experiment is NOT recommended for practical use.** It's preserved for:
- Research documentation
- Baseline comparison
- Understanding why modern approaches (exp3) are better

**For actual task-to-code prediction, use exp3 (embedding-based approach).**

---

## Prerequisites

### 1. Database Setup
You need a database created by the CodeXplorer data gathering tool with these tables:
- `TASK` (ID, NAME, TITLE, DESCRIPTION, COMMENTS)
- `RAWDATA` (ID, TASK_NAME, PATH)
- `MODULE` (from exp1 - needed for TFIDF_module_token.py)
- `MODULE_TASK` (from exp1 - needed for TFIDF_module_token.py)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Configuration

Edit `config.py` and update the database path:

```python
DB_FILE = '../data/sonar.db'  # Change to your database
```

---

## Run the Experiment

### Option 1: Simple TF-IDF (Recommended if you must try)

Fastest approach, creates a TF-IDF matrix CSV:

```bash
python tfidfFast.py
```

**Output**: `tfidf_matrix.csv`
**Runtime**: ~10-30 minutes for 10,000 tasks

---

### Option 2: Full Pipeline (Not Recommended - Very Slow)

```bash
# Step 1: Tokenize tasks (5-15 min)
python taskTokenizer.py

# Step 2: Calculate TF-IDF per module (4-48 HOURS!)
# ⚠️ WARNING: This takes extremely long!
python TFIDF_module_token.py
```

**Output**: `TFIDF_MODULE_TOKEN` table in database
**Runtime**: 4-48 hours depending on dataset size

---

### Option 3: Word Groups Analysis

Requires `full_word_group.csv` input file:

```bash
python chainTfidfFast.py
```

**Output**: `full_word_group_with_tfidf.csv`
**Runtime**: ~15-40 minutes for 10,000 tasks

---

## Expected Results

### Performance Metrics (Typical)

After running, you'll see results like:

```
MAP@10:  0.005-0.015 (0.5-1.5%)
MRR:     0.008-0.020 (0.8-2.0%)
P@10:    0.003-0.012 (0.3-1.2%)
R@10:    0.010-0.025 (1.0-2.5%)
```

**Translation**: The system correctly predicts the right files only 0.5-2% of the time.

### Compare with exp3 (Embeddings)

```
exp3 MAP@10:  0.023-0.035 (2.3-3.5%)  - 2-3× BETTER
exp3 Runtime: 20-40 minutes           - 10-100× FASTER
```

---

## Common Issues

### Issue 1: "No module named 'config'"
**Solution**: Make sure `config.py` is in the same directory as the scripts.

### Issue 2: "Table TASK not found"
**Solution**: Run the data gathering tool first to populate the database.

### Issue 3: "Table MODULE not found" (for TFIDF_module_token.py)
**Solution**: Run exp1 scripts first to create MODULE and MODULE_TASK tables.

### Issue 4: NLTK data not found
**Solution**:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Issue 5: Script takes too long
**Solution**: This is expected. Consider:
- Using `tfidfFast.py` instead of `TFIDF_module_token.py`
- Switching to exp3 for practical applications
- Running overnight on a powerful machine

---

## Understanding the Output

### tfidfFast.py Output

Creates `tfidf_matrix.csv`:
- **Rows**: Each row is a task
- **Columns**: Each column is a term
- **Values**: TF-IDF scores (higher = more important)

Example:
```
       fix    bug   login  database  ...
Task1  0.32  0.41  0.89   0.00      ...
Task2  0.00  0.33  0.00   0.76      ...
```

### TFIDF_MODULE_TOKEN Table

| MODULE_ID | TOKEN_ID | TFIDF | TF   | IDF  |
|-----------|----------|-------|------|------|
| 1         | 45       | 0.234 | 0.12 | 1.95 |
| 1         | 89       | 0.156 | 0.08 | 1.95 |

- **MODULE_ID**: Which code module
- **TOKEN_ID**: Which term
- **TFIDF**: Combined score
- **TF**: How often term appears in this module
- **IDF**: How rare the term is globally

---

## Why This Doesn't Work Well

1. **Semantic Gap**: Can't understand "bug" = "defect" = "issue"
2. **Sparsity**: Most combinations have zero score
3. **No Context**: "test" could mean unit test or QA test
4. **Too Slow**: Hours of computation for poor results

---

## Next Steps

After seeing these results:

1. **Don't try to optimize this** - It's a fundamental limitation
2. **Read README.md** for detailed analysis
3. **Move to exp3** for 2-3× better accuracy and 100× faster speed

---

## Files Created

After running:
- `tfidf_matrix.csv` - Full TF-IDF matrix
- `full_word_group_with_tfidf.csv` - Word groups with TF-IDF
- Database tables: `TOKEN`, `TASK_TOKEN_INDEX`, `TFIDF_MODULE_TOKEN`

---

## Cleanup

To remove generated files:

```bash
rm tfidf_matrix.csv
rm full_word_group_with_tfidf.csv
```

To remove database tables:

```sql
DROP TABLE IF EXISTS TOKEN;
DROP TABLE IF EXISTS TASK_TOKEN_INDEX;
DROP TABLE IF EXISTS ORDER_TOKEN_INDEX;
DROP TABLE IF EXISTS TFIDF_MODULE_TOKEN;
```

---

## Help & Support

This experiment is historical. For support on modern approaches:
- See exp3 documentation
- Check main README in refactor folder
- Review research_questions.md in exp3

---

**Remember**: This experiment demonstrates why traditional IR methods don't work for code prediction. Use exp3 instead!
