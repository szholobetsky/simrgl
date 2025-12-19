"""
Configuration file for CodeXplorer Analysis Scripts
Modify the database file path according to your project.
"""

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Path to your SQLite database file containing RAWDATA and TASK tables
# This database should already be populated by the data gathering tool
# Examples:
#   - Relative path: "data/sonar.db"
#   - Absolute path: "C:/Projects/data/flink.db"
DB_FILE = "data/sonar.db"


# =============================================================================
# TABLE NAMES (DO NOT MODIFY UNLESS YOU HAVE CUSTOM SCHEMA)
# =============================================================================

# Source tables (should already exist in your database)
RAWDATA_TABLE = "RAWDATA"          # Contains commit data with PATH field
TASK_TABLE = "TASK"                # Contains task data with NAME and TITLE fields
TASK_NAME_COLUMN = "NAME"          # Task identifier column
TASK_TITLE_COLUMN = "TITLE"        # Task title column

# Output tables (will be created by the scripts)
# Terms extraction tables
TITLE_TERM_TABLE = "TITLE_TERM"                    # Unique terms from task titles
TITLE_TASK_TERM_TABLE = "TITLE_TASK_TERM"          # Task-term relationships
TITLE_TASK_TERM_AGG_TABLE = "TITLE_TASK_TERM_AGG"  # Aggregated term counts

# Module/file structure tables
MODULE_TABLE = "MODULE"                # Hierarchical file/module structure
MODULE_TASK_TABLE = "MODULE_TASK"      # Task-module relationships

# Analysis tables
TERM_RANK_TABLE = "TERM_RANK"          # Term ranking metrics
TERM_LINKS_TABLE = "TERM_LINKS"        # Term co-occurrence matrix
FILE_LINKS_TABLE = "FILE_LINKS"        # File co-occurrence matrix


# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Batch size for processing tasks in interlink.py
# Higher values = more memory usage but potentially faster processing
# Recommended: 500-1000 for large databases
BATCH_SIZE = 500
