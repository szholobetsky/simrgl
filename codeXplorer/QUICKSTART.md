# Quick Start Guide

Get started with CodeXplorer in 5 minutes!

## Step 1: Install Dependencies

### On Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### On Windows:
```cmd
setup.bat
venv\Scripts\activate.bat
```

## Step 2: Configure the Tool

Edit `config.py` and set these required parameters:

```python
# Path to your Git repository
REPO_PATH = '/path/to/your/repo'

# Path where database will be created
DB_FILE = './data.db'

# Your Jira instance URL
JIRA_URL = 'https://your-jira-instance.com'

# How to connect to Jira (try 'selenium' first)
JIRA_CONNECTOR_TYPE = 'selenium'

# Pattern to extract task IDs from commits
CURRENT_MASK = r'\[([A-Z]+-\d+)\]'  # For [JIRA-123] format
# OR
CURRENT_MASK = r'^[A-Z]+-\d+'       # For JIRA-123 format
```

## Step 3: Run the Tool

```bash
python main.py
```

The tool will:
1. Create database tables
2. Extract all commits from your Git repo
3. Find task IDs in commit messages
4. Fetch task details from Jira
5. Save everything to SQLite database

## Step 4: Explore Your Data

Use any SQLite browser or command line:

```bash
sqlite3 data.db
```

```sql
-- See all tables
.tables

-- Count commits
SELECT COUNT(*) FROM RAWDATA;

-- Count tasks
SELECT COUNT(*) FROM TASK;

-- View sample data
SELECT * FROM TASK LIMIT 5;
```

## Common Issues

**"Repository not found"**
→ Check that REPO_PATH points to a valid Git repository

**"Jira connection failed"**
→ Try changing JIRA_CONNECTOR_TYPE from 'api' to 'html' to 'selenium'

**"No tasks found"**
→ Check that CURRENT_MASK matches your commit message format

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Analyze your data using SQL queries
- Export data for further analysis

## Example Configuration

### Apache Flink Project
```python
REPO_PATH = '/home/user/repos/flink'
JIRA_URL = 'https://issues.apache.org/jira'
JIRA_CONNECTOR_TYPE = 'selenium'
CURRENT_MASK = r'\[([A-Z]+-\d+)\]'
```

### Company Private Project
```python
REPO_PATH = 'C:/Projects/my-project'
JIRA_URL = 'https://company.atlassian.net'
JIRA_CONNECTOR_TYPE = 'api'
CURRENT_MASK = r'^[A-Z]+-\d+'
```

That's it! You're ready to explore your codebase.
