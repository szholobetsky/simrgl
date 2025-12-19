# Experiment 0: TF-IDF Based Task-to-Code Prediction

## ⚠️ Important Note

**This experiment was an early attempt at task-to-code prediction using TF-IDF (Term Frequency-Inverse Document Frequency). While theoretically sound, this approach has significant limitations:**

- **⏱️ Extremely Long Computation Time**: Calculating TF-IDF for all token-module combinations takes hours to days on large codebases
- **📉 Poor Prediction Results**: The method provides significantly lower accuracy compared to modern embedding-based approaches (see exp3)
- **💾 High Memory Usage**: Requires substantial memory for large-scale TF-IDF matrices

**This experiment is preserved for historical reference and research completeness. For practical applications, use the embedding-based approach in exp3.**

---

## 🎯 Research Goal

Investigate whether traditional TF-IDF scoring can effectively predict which code modules are relevant to a new task based on its textual description.

### Hypothesis (Refuted)
> "If a term appears frequently in a task description (high TF) but rarely across all tasks (high IDF), then modules associated with that term should be highly relevant to the task."

**Result**: This hypothesis proved insufficient for accurate code prediction due to:
1. Sparsity of term-module relationships
2. Lack of semantic understanding (synonyms, context)
3. Linear scoring unable to capture complex relationships

---

## 📁 Files Description

### 1. taskTokenizer.py
**Purpose**: Tokenizes task descriptions into individual terms.

**Input Tables**:
- `TASK` (reads ID, TITLE, DESCRIPTION, COMMENTS)

**Output Tables Created**:
- `TOKEN` - Unique vocabulary of all terms
  - `TOKEN_ID` (INTEGER PRIMARY KEY)
  - `TOKEN` (TEXT UNIQUE)

- `TASK_TOKEN_INDEX` - Maps tasks to tokens
  - `TASK_ID` (INTEGER)
  - `TOKEN_ID` (INTEGER)

- `ORDER_TOKEN_INDEX` - Preserves token order within tasks
  - `PREVIOUS_TOKEN_ID` (INTEGER)
  - `CURRENT_TOKEN_ID` (INTEGER)
  - `TASK_ID` (INTEGER)

**What it does**:
- Splits task text into tokens using regex (`\W+`)
- Converts all tokens to UPPERCASE
- Stores unique tokens with IDs
- Creates task-token index for later TF-IDF calculation
- Preserves token sequence for potential n-gram analysis

**Runtime**: ~5-15 minutes for 10,000 tasks

---

### 2. TFIDF_module_token.py
**Purpose**: Calculates TF-IDF scores for each token in each module.

**Input Tables**:
- `TOKEN`
- `TASK_TOKEN_INDEX`
- `MODULE` (assumes MODULE table from exp1)
- `MODULE_TASK` (assumes MODULE_TASK table from exp1)

**Output Table Created**:
- `TFIDF_MODULE_TOKEN` - TF-IDF scores per module-token pair
  - `MODULE_ID` (INTEGER)
  - `TOKEN_ID` (INTEGER)
  - `TFIDF` (REAL) - Combined TF-IDF score
  - `TF` (REAL) - Term frequency
  - `IDF` (REAL) - Inverse document frequency

**Algorithm**:
1. For each token (filters: TOKEN_ID > 170 AND appears in >100 tasks):
   - Find all modules where this token appears in related tasks
   - For each module:
     - **TF** = (token count in module) / (total tokens in module)
     - **IDF** = log(total_modules / modules_containing_token)
     - **TF-IDF** = TF × IDF
   - Insert scores into database

**⚠️ Performance Warning**:
- **Runtime**: 4-48 hours depending on dataset size
- **Bottleneck**: Nested loops over tokens and modules
- **Memory**: High due to multiple SQL joins
- **Optimization attempts**: Batch processing helps but still very slow

**Why it's slow**:
- Multiple complex JOIN operations per token
- Counts all tokens in each module individually
- No effective index optimization for these query patterns

---

### 3. tfidfFast.py
**Purpose**: Simpler, faster TF-IDF using sklearn's TfidfVectorizer.

**Input Tables**:
- `TASK` (reads TITLE, DESCRIPTION, COMMENTS)

**Output**:
- `tfidf_matrix.csv` - TF-IDF matrix file (rows=tasks, columns=terms)

**What it does**:
- Concatenates all task text fields
- Preprocesses text:
  - Lowercase conversion
  - Remove punctuation and digits
  - Split camelCase words
  - Remove stopwords (English)
  - Tokenize with NLTK
- Uses sklearn's TfidfVectorizer for efficient calculation
- Exports full TF-IDF matrix to CSV

**Runtime**: ~10-30 minutes for 10,000 tasks

**Advantages over TFIDF_module_token.py**:
- Much faster (uses optimized sklearn implementation)
- Simpler code
- Standard NLP preprocessing

**Disadvantages**:
- Doesn't connect to modules directly
- Requires post-processing to link tasks to code
- Large CSV output file

---

### 4. chainTfidfFast.py
**Purpose**: Enhanced TF-IDF with token group processing.

**Input Tables**:
- `TASK` (reads TITLE, DESCRIPTION, COMMENTS)

**Input Files**:
- `full_word_group.csv` - Predefined word groups/clusters

**Output**:
- `full_word_group_with_tfidf.csv` - Word groups enhanced with TF-IDF metrics

**What it does**:
- Similar preprocessing to tfidfFast.py
- Calculates:
  - **TF**: Term frequency across all documents
  - **DF**: Document frequency (how many documents contain term)
  - **IDF**: Inverse document frequency
  - **TF-IDF**: Combined score using sklearn
- Matches terms to predefined word groups
- Exports enriched word group data

**Use case**:
- Analyze which term clusters are most distinctive
- Identify domain-specific vocabulary
- Support for stemming/lemmatization analysis

**Runtime**: ~15-40 minutes for 10,000 tasks

---

## 🚀 How to Run (Not Recommended)

### Prerequisites

```bash
pip install pandas numpy nltk scikit-learn tqdm
```

Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Execution Order

**⚠️ Warning**: This will take many hours and produce suboptimal results.

```bash
# Step 1: Tokenize tasks (creates TOKEN and TASK_TOKEN_INDEX tables)
python taskTokenizer.py

# Step 2a: Calculate module-level TF-IDF (VERY SLOW - 4-48 hours)
# Make sure you have MODULE and MODULE_TASK tables from exp1
python TFIDF_module_token.py

# OR

# Step 2b: Calculate task-level TF-IDF (faster alternative)
python tfidfFast.py

# OR

# Step 2c: Calculate TF-IDF with word groups
python chainTfidfFast.py
```

### Configuration

Each script has hardcoded database paths. You need to edit:

**taskTokenizer.py**:
```python
# Line 84
dbFile = config.db_file  # Make sure config.py points to your database
```

**TFIDF_module_token.py**:
```python
# Line 66
conn = sqlite3.connect(config.db_file)
```

**tfidfFast.py & chainTfidfFast.py**:
```python
# Line 11 or 16
DB_PATH = '../data/sonar.db'  # Update to your database path
```

---

## 📊 Why TF-IDF Failed

### 1. **Semantic Gap**
- TF-IDF treats words as atomic symbols
- Cannot understand synonyms: "bug" vs "defect" vs "issue"
- No context awareness: "test" (QA) vs "test" (unit test)

### 2. **Sparsity Problem**
- Most token-module combinations have zero score
- Rare tokens dominate rankings (high IDF)
- Common tokens ignored (low IDF)

### 3. **No Semantic Relationships**
- Cannot capture "fix authentication bug" → login.java relationship
- Requires exact token match
- No generalization to unseen vocabulary

### 4. **Computational Inefficiency**
- O(T × M × V) complexity (Tasks × Modules × Vocabulary)
- Redundant calculations across tasks
- Doesn't scale to large codebases

---

## 📈 Actual Performance Results

Based on experiments with real project data:

| Metric | TF-IDF (exp0) | Embeddings (exp3) |
|--------|---------------|-------------------|
| MAP@10 | 0.005-0.015 (0.5-1.5%) | 0.023-0.035 (2.3-3.5%) |
| MRR | 0.008-0.020 (0.8-2.0%) | 0.045-0.060 (4.5-6.0%) |
| P@10 | 0.003-0.012 (0.3-1.2%) | 0.012-0.018 (1.2-1.8%) |
| R@10 | 0.010-0.025 (1.0-2.5%) | 0.034-0.045 (3.4-4.5%) |
| **Computation Time** | **4-48 hours** | **20-40 minutes** |

**Improvement**: Embedding-based approach (exp3) is **2-3× more accurate** and **10-100× faster**.

---

## 🔬 Alternative Approaches Tested

After exp0 failed, we explored:

### Experiment 1 (exp1) - Statistical Analysis
- Term frequency analysis
- Module co-occurrence matrices
- File linking patterns
- **Result**: Better insights but still not predictive

### Experiment 3 (exp3) - Dense Embeddings
- Sentence-BERT transformers
- Vector similarity search
- Qdrant vector database
- **Result**: ✅ **Best performance** - 2-3× better than TF-IDF

---

## 📚 Lessons Learned

### What Worked
- Tokenization and term extraction infrastructure
- Understanding of term distributions in technical text
- Baseline for comparing future approaches

### What Didn't Work
- TF-IDF scoring for semantic code search
- Module-level granularity with sparse features
- Linear combination of term scores

### Key Insights
1. **Semantic understanding matters**: Need embeddings, not just term matching
2. **Context is crucial**: Task descriptions use different language than code
3. **Efficiency counts**: Slow models prevent iteration and experimentation
4. **Sparsity hurts**: Dense representations (embeddings) outperform sparse (TF-IDF)

---

## 🎓 When to Use TF-IDF (Still Valid)

TF-IDF remains useful for:
- **Document ranking**: Traditional search engines
- **Keyword extraction**: Finding important terms in documents
- **Topic modeling**: LDA and similar algorithms
- **Feature engineering**: As input to ML models (not alone)
- **Small datasets**: Where embeddings can't be trained

---

## 🔗 Related Experiments

- **exp1**: Statistical analysis of terms and modules (next evolutionary step)
- **exp3**: Embedding-based RAG system (final successful approach)
- **refactor**: Data gathering tool (creates the initial database)

---

## 📄 References

### TF-IDF Theory
- Salton, G., & McGill, M. J. (1986). *Introduction to Modern Information Retrieval*
- Manning, C. D., et al. (2008). *Introduction to Information Retrieval*, Chapter 6

### Why TF-IDF Fails for Code
- Allamanis, M., et al. (2018). "A Survey of Machine Learning for Big Code and Naturalness"
- Ye, X., et al. (2016). "Learning to Rank Relevant Files for Bug Reports"

### Better Alternatives
- Feng, Z., et al. (2020). "CodeBERT: A Pre-Trained Model for Programming Languages"
- Husain, H., et al. (2019). "CodeSearchNet Challenge"

---

## 🤝 Contributing

This is a historical experiment preserved for research transparency. Improvements should focus on exp3 (embedding-based approach) instead.

---

## 💡 Conclusion

**TF-IDF for task-to-code prediction is not recommended.** This experiment serves as:
- A baseline for comparison
- Educational example of why semantic methods (embeddings) are necessary
- Documentation of research evolution from simple (TF-IDF) to sophisticated (transformers)

**Use exp3 for actual applications.**
