"""
Configuration file for CodeXplorer Data Gathering Tool.
Modify these settings according to your project requirements.
"""

# ============================================================================
# GIT REPOSITORY SETTINGS
# ============================================================================

# Path to the local Git repository to analyze
# Example: '/home/user/projects/my-repo' or 'C:/Projects/my-repo'
REPO_PATH = '/home/stzh/Projects/data/repos/Flink/flink'

# Git branch to extract commits from
# Common values: 'master', 'main', 'develop'
BRANCH = 'master'


# ============================================================================
# DATABASE SETTINGS
# ============================================================================

# Path to the SQLite database file where data will be stored
# The file will be created if it doesn't exist
# Example: './data.db' or '../data/project.db'
DB_FILE = '../data/flink.db'


# ============================================================================
# JIRA SETTINGS
# ============================================================================

# Base URL of your Jira instance (without trailing slash)
# Examples:
#   - Public Jira: 'https://issues.apache.org/jira'
#   - Private Jira: 'https://your-company.atlassian.net'
JIRA_URL = 'https://issues.apache.org/jira'

# Jira connector type - determines how task details are fetched
# Options:
#   'api'      - Use Jira REST API (fastest, requires API access)
#   'html'     - Parse HTML pages (works for public Jira, no API needed)
#   'selenium' - Use browser automation (slowest, works when JavaScript is required)
JIRA_CONNECTOR_TYPE = 'selenium'


# ============================================================================
# TASK EXTRACTION SETTINGS
# ============================================================================

# Regex pattern to extract task identifiers from commit messages
# Common patterns:
#   r'^[A-Z]+-\d+'           - Matches: JIRA-123 at the start of message
#   r'\[([A-Z]+-\d+)\]'      - Matches: [JIRA-123] in brackets
#   r'[A-Z]+-\d+'            - Matches: JIRA-123 anywhere in message

# Pattern for task numbers in simple format (e.g., "JIRA-123 Fix bug")
SIMPLE_NUMBER_MASK = r'^[A-Z]+-\d+'

# Pattern for task numbers in bracketed format (e.g., "[JIRA-123] Fix bug")
BRACKETED_MASK = r'\[([A-Z]+-\d+)\]'

# Active pattern to use for task extraction
CURRENT_MASK = BRACKETED_MASK


# ============================================================================
# RATE LIMITING SETTINGS
# ============================================================================

# Maximum number of Jira API requests per minute
# Adjust based on your Jira instance's rate limits
# Typical values: 50-200
REQUEST_RATE_LIMIT = 50

# Number of database updates to batch before committing
# Higher values = better performance but more memory usage
# Typical values: 10-100
DB_BATCH_SIZE = 10


# ============================================================================
# EXECUTION SETTINGS
# ============================================================================

# Test mode: limits data extraction for testing purposes
# If True, stops after processing 100,000 commits
# Set to False for production use
TEST_MODE = False

# Print commit details to console during extraction
# Useful for debugging, but slows down execution
# Set to False for production use
PRINT_CONTENT = False


# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Log file path for error logging
LOG_FILE = 'error.log'

# Logging level
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = 'INFO'
