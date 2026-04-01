"""
CodeXplorer - Commits-Only Script
Collects only Git commits for projects that have no issue tracker.
Executes steps 1 and 2 only:
  [1/2] Initialize database
  [2/2] Extract commits from Git repository
"""

import logging
import sys

import config
from database import DatabaseManager
from connectors.git import GitConnector


def setup_logging():
    logging.basicConfig(
        filename=config.LOG_FILE,
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)


def print_config():
    print("\n" + "=" * 80)
    print("CodeXplorer - Commits Only Mode (no tracker)")
    print("=" * 80)
    print(f"Repository Path:     {config.REPO_PATH}")
    print(f"Branch:              {config.BRANCH}")
    print(f"Database File:       {config.DB_FILE}")
    print(f"Test Mode:           {config.TEST_MODE}")
    print(f"Print Content:       {config.PRINT_CONTENT}")
    print("=" * 80 + "\n")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    print_config()

    try:
        print("[1/2] Initializing database...")
        db_manager = DatabaseManager(config.DB_FILE)
        db_manager.create_tables()
        print("      Database initialized\n")

        print("[2/2] Extracting commits from Git repository...")
        git_connector = GitConnector(config.REPO_PATH)
        git_connector.extract_commits(
            db_manager,
            branch=config.BRANCH,
            test_mode=config.TEST_MODE,
            print_content=config.PRINT_CONTENT
        )
        print("      Commits extracted\n")

        print("Done.")
        print(f"\nData saved to: {config.DB_FILE}")
        print("  Table: RAWDATA — Git commit information")

    except Exception as e:
        logger.error(f"Error during commit extraction: {e}", exc_info=True)
        print(f"\nError: {e}")
        print(f"Check {config.LOG_FILE} for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
