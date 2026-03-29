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
# TRACKER SETTINGS
# ============================================================================

# Which issue tracker to use: 'jira', 'github', or 'youtrack'
TRACKER_TYPE = 'jira'


# ============================================================================
# GITHUB SETTINGS (used when TRACKER_TYPE = 'github')
# ============================================================================

# Repository owner (username or organization)
GITHUB_OWNER = 'django'

# Repository name
GITHUB_REPO = 'django'

# Personal access token — optional for public repos, but strongly recommended.
# Without a token: 60 requests/hour (not enough for large projects).
# With a token:  5000 requests/hour.
# Create one at: https://github.com/settings/tokens (no scopes needed for public repos)
GITHUB_TOKEN = ''


# ============================================================================
# BULK FETCH SETTINGS (used by bulk_task_fetcher.py)
# ============================================================================

# For Jira bulk fetch — project key to fetch all issues for
# Example: 'KAFKA', 'SPARK', 'FLINK'
JIRA_PROJECT_KEY = 'KAFKA'

# For YouTrack bulk fetch — project key (prefix before the dash)
# Example: 'KT' for Kotlin, 'IDEA' for IntelliJ IDEA, 'IJPL' for Platform
YOUTRACK_PROJECT_KEY = 'KT'


# ============================================================================
# YOUTRACK SETTINGS (used when TRACKER_TYPE = 'youtrack')
# ============================================================================

# Base URL of the YouTrack instance
# JetBrains public: 'https://youtrack.jetbrains.com'
# Self-hosted:      'https://your-company.youtrack.cloud'
YOUTRACK_URL = 'https://youtrack.jetbrains.com'

# Permanent token — optional for public instances like youtrack.jetbrains.com.
# Generate at: YouTrack → Profile → Account Security → New token
# Verified 2026-03: jetbrains YouTrack works without token (~0.48s/req, ~7500 issues/hr)
YOUTRACK_TOKEN = ''


# ============================================================================
# GITLAB SETTINGS (used when TRACKER_TYPE = 'gitlab')
# ============================================================================

# Base URL of the GitLab instance
# Options:
#   'https://gitlab.com'               — public projects (Inkscape, etc.)
#   'https://gitlab.gnome.org'         — GNOME projects
#   'https://gitlab.freedesktop.org'   — Mesa, Wayland, D-Bus
GITLAB_URL = 'https://gitlab.com'

# Project path as 'namespace/repo', e.g. 'inkscape/inkscape'
GITLAB_PROJECT = 'inkscape/inkscape'

# Personal access token.
# REQUIRED for fetching comments (notes) — gitlab.com returns 401 without token.
# Issues and commits are public without token.
# Generate at: GitLab → Settings → Access Tokens (scope: 'read_api')
# Rate limit: 500 req/10min without token, 2000 req/min with token
GITLAB_TOKEN = ''

# WARNING: GitLab projects typically have very low commit-to-issue link rates
# (Inkscape ~2%, GNOME Shell ~1%, Mesa ~6%) because teams use Merge Requests,
# not commit messages, to link issues. Consider MR-based collection instead.
# See GitLabApiConnector.fetch_mr_issue_links() for the MR-based approach.


# ============================================================================
# JIRA SETTINGS (used when TRACKER_TYPE = 'jira')
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

# --- Jira patterns ---

# Pattern for task numbers in simple format (e.g., "JIRA-123 Fix bug")
SIMPLE_NUMBER_MASK = r'^[A-Z]+-\d+'

# Pattern for task numbers in bracketed format (e.g., "[JIRA-123] Fix bug")
BRACKETED_MASK = r'\[([A-Z]+-\d+)\]'

# --- GitHub patterns ---
# Note: all GitHub patterns use a capture group — extract_task_name() returns group(1).

# Django: "Fixed #12345 -- description" (~60% coverage)
DJANGO_MASK = r'[Ff]ixed\s+#(\d+)'

# Generic GitHub: "Fixes #123", "Closes #123", "Resolves #123" (covers most projects)
GITHUB_GENERIC_MASK = r'(?:fix(?:e[sd])?|clos(?:e[sd]?)|resolv(?:e[sd]?))\s*[:#]?\s*#(\d+)'

# Broad fallback — matches any #NNN (more noise, fewer misses; use as last resort)
GITHUB_BROAD_MASK = r'#(\d+)'

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
