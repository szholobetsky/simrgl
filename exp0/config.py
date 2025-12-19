"""
Configuration file for TF-IDF Experiment (exp0)
"""

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Path to your SQLite database file
# This database should be populated by the CodeXplorer data gathering tool
# with RAWDATA and TASK tables
DB_FILE = '../data/sonar.db'

# Alternative database path (uncomment if needed)
# DB_FILE = 'C:/Projects/data/your_project.db'


# =============================================================================
# TFIDF MODULE TOKEN SETTINGS
# =============================================================================

# Minimum token ID to start processing (to skip very common tokens)
# Lower IDs typically represent very frequent terms that add little value
MIN_TOKEN_ID = 170

# Minimum number of tasks a token must appear in to be included
# Tokens appearing in fewer tasks are likely too rare to be meaningful
MIN_TOKEN_FREQUENCY = 100

# Module level to consider (0 = root level)
# Set to 0 for top-level modules/directories
MODULE_LEVEL = 0

# Batch size for processing (not used in current implementation)
# Reserved for future optimization
BATCH_SIZE = 1000


# =============================================================================
# TEXT PREPROCESSING SETTINGS
# =============================================================================

# Remove English stopwords during preprocessing
REMOVE_STOPWORDS = True

# Convert tokens to uppercase (used in taskTokenizer.py)
# Set to False if you want case-sensitive tokens
UPPERCASE_TOKENS = True

# Minimum token length to consider
# Tokens shorter than this will be filtered out
MIN_TOKEN_LENGTH = 2


# =============================================================================
# OUTPUT FILE PATHS
# =============================================================================

# Output file for tfidfFast.py
TFIDF_MATRIX_OUTPUT = 'tfidf_matrix.csv'

# Input file for chainTfidfFast.py (word groups)
WORD_GROUPS_INPUT = 'full_word_group.csv'

# Output file for chainTfidfFast.py
WORD_GROUPS_WITH_TFIDF_OUTPUT = 'full_word_group_with_tfidf.csv'


# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Maximum number of tasks to process in test mode
# Set to None to process all tasks
TEST_LIMIT = None

# Enable progress bars using tqdm
SHOW_PROGRESS = True

# Log level for detailed execution information
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = 'INFO'
