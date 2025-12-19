"""
CodeXplorer Data Gathering Tool
Main entry point for extracting commit data from Git repositories
and fetching task details from Jira.
"""

import logging
import sys

import config
from database import DatabaseManager
from connectors.git import GitConnector
from task_extractor import TaskExtractor
from task_fetcher import TaskFetcher


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        filename=config.LOG_FILE,
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def print_config():
    """Print current configuration settings."""
    print("\n" + "=" * 80)
    print("CodeXplorer Configuration")
    print("=" * 80)
    print(f"Repository Path:     {config.REPO_PATH}")
    print(f"Branch:              {config.BRANCH}")
    print(f"Database File:       {config.DB_FILE}")
    print(f"Jira URL:            {config.JIRA_URL}")
    print(f"Jira Connector:      {config.JIRA_CONNECTOR_TYPE}")
    print(f"Task Pattern:        {config.CURRENT_MASK}")
    print(f"Test Mode:           {config.TEST_MODE}")
    print(f"Print Content:       {config.PRINT_CONTENT}")
    print("=" * 80 + "\n")


def main():
    """Main execution flow."""
    setup_logging()
    logger = logging.getLogger(__name__)

    print_config()

    try:
        # Initialize database
        print("\n[1/5] Initializing database...")
        db_manager = DatabaseManager(config.DB_FILE)
        db_manager.create_tables()
        print("✓ Database initialized")

        # Extract commits from Git repository
        print("\n[2/5] Extracting commits from Git repository...")
        git_connector = GitConnector(config.REPO_PATH)
        git_connector.extract_commits(
            db_manager,
            branch=config.BRANCH,
            test_mode=config.TEST_MODE,
            print_content=config.PRINT_CONTENT
        )
        print("✓ Commits extracted")

        # Extract task identifiers from commit messages
        print("\n[3/5] Extracting task identifiers from commit messages...")
        task_extractor = TaskExtractor(config.CURRENT_MASK)
        task_extractor.process_all_commits(db_manager)
        print("✓ Task identifiers extracted")

        # Fetch task details from Jira
        print("\n[4/5] Fetching task details from Jira...")
        task_fetcher = TaskFetcher(
            config.JIRA_URL,
            connector_type=config.JIRA_CONNECTOR_TYPE
        )
        task_fetcher.fetch_all_tasks(
            db_manager,
            rate_limit=config.REQUEST_RATE_LIMIT,
            batch_size=config.DB_BATCH_SIZE
        )
        print("✓ Task details fetched")

        print("\n[5/5] Data gathering complete!")
        print(f"\nData has been saved to: {config.DB_FILE}")
        print("\nDatabase contains two tables:")
        print("  - RAWDATA: Git commit information")
        print("  - TASK: Jira task details")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        print(f"Check {config.LOG_FILE} for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
