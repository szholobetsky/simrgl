# CodeXplorer Data Gathering Tool

A Python tool for extracting commit data from Git repositories and enriching it with task tracking information from Jira. This tool helps analyze the relationship between code changes and project tasks.

## Features

- **Git Commit Extraction**: Extracts detailed commit information including SHA, author, date, message, file paths, and diffs
- **Task Identifier Extraction**: Automatically extracts task identifiers (e.g., JIRA-123) from commit messages using configurable regex patterns
- **Jira Integration**: Fetches task details (title, description, comments) from Jira using multiple connection methods
- **SQLite Storage**: Stores all data in a lightweight SQLite database with two main tables
- **Multiple Jira Connectors**: Supports API, HTML parsing, and Selenium-based fetching methods
- **Rate Limiting**: Built-in rate limiting to respect Jira API limits
- **Progress Tracking**: Visual progress bars for long-running operations

## Installation

### Prerequisites

- Python 3.7 or higher
- Git installed on your system
- A local Git repository to analyze
- Access to a Jira instance (optional, for task details)

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the tool by editing `config.py`:
```python
# Set your Git repository path
REPO_PATH = '/path/to/your/git/repo'

# Set your Jira URL
JIRA_URL = 'https://your-jira-instance.com'

# Configure other settings as needed
```

## Configuration

All configuration is done in `config.py`. Key settings:

### Git Repository Settings
- `REPO_PATH`: Path to the local Git repository
- `BRANCH`: Git branch to analyze (default: 'master')

### Database Settings
- `DB_FILE`: Path to the SQLite database file

### Jira Settings
- `JIRA_URL`: Base URL of your Jira instance
- `JIRA_CONNECTOR_TYPE`: Choose from:
  - `'api'`: Use Jira REST API (fastest, requires API access)
  - `'html'`: Parse HTML pages (works for public Jira)
  - `'selenium'`: Use browser automation (for JavaScript-heavy pages)

### Task Extraction Settings
- `CURRENT_MASK`: Regex pattern to extract task IDs from commit messages
  - `SIMPLE_NUMBER_MASK`: Matches "JIRA-123 Fix bug"
  - `BRACKETED_MASK`: Matches "[JIRA-123] Fix bug"

### Performance Settings
- `REQUEST_RATE_LIMIT`: Max Jira requests per minute (default: 50)
- `DB_BATCH_SIZE`: Database batch size (default: 10)
- `TEST_MODE`: Limit to 100,000 commits for testing (default: False)

## Usage

Run the tool:

```bash
python main.py
```

The tool will execute the following steps:

1. **Initialize Database**: Creates SQLite database and tables
2. **Extract Commits**: Reads all commits from the Git repository
3. **Extract Task IDs**: Identifies task identifiers in commit messages
4. **Fetch Task Details**: Retrieves task information from Jira
5. **Complete**: Data is saved to the SQLite database

## Database Schema

### RAWDATA Table
Stores Git commit information:

| Column        | Type    | Description                    |
|---------------|---------|--------------------------------|
| ID            | INTEGER | Primary key                    |
| SHA           | TEXT    | Git commit SHA                 |
| AUTHOR_NAME   | TEXT    | Commit author name             |
| AUTHOR_EMAIL  | TEXT    | Commit author email            |
| CMT_DATE      | TEXT    | Commit date (ISO format)       |
| MESSAGE       | BLOB    | Commit message                 |
| PATH          | BLOB    | File path                      |
| DIFF          | BLOB    | Git diff content               |
| TASK_NAME     | TEXT    | Extracted task identifier      |

### TASK Table
Stores Jira task information:

| Column       | Type    | Description                    |
|--------------|---------|--------------------------------|
| ID           | INTEGER | Primary key (auto-increment)   |
| NAME         | TEXT    | Task identifier (unique)       |
| TITLE        | TEXT    | Task title/summary             |
| DESCRIPTION  | TEXT    | Task description               |
| COMMENTS     | TEXT    | All task comments concatenated |

Indexes are created on TASK.ID and TASK.NAME for efficient querying.

## Jira Connector Types

### API Connector
- **Best for**: Jira instances with API access
- **Pros**: Fastest, most reliable
- **Cons**: Requires API permissions
- **Use when**: You have API access to Jira

### HTML Connector
- **Best for**: Public Jira instances
- **Pros**: No API required, simple
- **Cons**: May break if Jira HTML structure changes
- **Use when**: Jira API is not available but pages are accessible

### Selenium Connector
- **Best for**: JavaScript-heavy Jira pages
- **Pros**: Works with dynamic content
- **Cons**: Slower, requires Chrome/ChromeDriver
- **Use when**: HTML connector doesn't work due to JavaScript rendering

## Examples

### Example 1: Analyzing Apache Flink
```python
# In config.py
REPO_PATH = '/home/user/repos/flink'
JIRA_URL = 'https://issues.apache.org/jira'
JIRA_CONNECTOR_TYPE = 'selenium'
CURRENT_MASK = r'\[([A-Z]+-\d+)\]'  # Matches [FLINK-123]
```

### Example 2: Private Company Repository
```python
# In config.py
REPO_PATH = 'C:/Projects/my-company-repo'
JIRA_URL = 'https://my-company.atlassian.net'
JIRA_CONNECTOR_TYPE = 'api'
CURRENT_MASK = r'^[A-Z]+-\d+'  # Matches PROJ-456 at start
```

## Troubleshooting

### "Git repository not found"
- Verify `REPO_PATH` points to a valid Git repository
- Ensure the repository has commits on the specified branch

### "Jira connection failed"
- Check that `JIRA_URL` is correct and accessible
- Try different connector types (api → html → selenium)
- Check firewall/network settings

### "No tasks extracted"
- Verify `CURRENT_MASK` matches your commit message format
- Check sample commit messages in your repository
- Use `PRINT_CONTENT = True` to debug

### Selenium errors
- Install ChromeDriver: `pip install webdriver-manager`
- Update Chrome browser to the latest version
- Check for antivirus/firewall blocking Chrome

## Performance Tips

1. **Use API connector** when possible (fastest)
2. **Increase batch size** for better database performance
3. **Enable test mode** first to verify configuration
4. **Disable print content** in production for speed
5. **Adjust rate limiting** based on your Jira instance

## Data Analysis

After running the tool, you can analyze the data using SQL queries:

```sql
-- Find most active contributors
SELECT AUTHOR_NAME, COUNT(*) as commits
FROM RAWDATA
GROUP BY AUTHOR_NAME
ORDER BY commits DESC;

-- Find tasks with most commits
SELECT TASK_NAME, COUNT(*) as commits
FROM RAWDATA
WHERE TASK_NAME IS NOT NULL
GROUP BY TASK_NAME
ORDER BY commits DESC;

-- Join commits with task details
SELECT r.SHA, r.MESSAGE, t.TITLE, t.DESCRIPTION
FROM RAWDATA r
LEFT JOIN TASK t ON r.TASK_NAME = t.NAME
WHERE t.TITLE IS NOT NULL;
```

## Project Structure

```
refactor/
├── main.py                      # Main entry point
├── config.py                    # Configuration file
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── database/
│   ├── __init__.py
│   └── db_manager.py           # Database operations
├── connectors/
│   ├── __init__.py
│   ├── git/
│   │   ├── __init__.py
│   │   └── git_connector.py   # Git repository connector
│   └── jira/
│       ├── __init__.py
│       ├── jira_api_connector.py      # Jira API connector
│       ├── jira_html_connector.py     # HTML parser connector
│       └── jira_selenium_connector.py # Selenium connector
├── task_extractor.py           # Task ID extraction
└── task_fetcher.py             # Task details fetching
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- All modules have proper documentation
- Configuration parameters are well-commented

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review `error.log` for detailed error messages
3. Verify your configuration in `config.py`
