"""
CodeXplorer - Task Resume Script
Resumes the pipeline from the TASK step, assuming commits are already
extracted and stored in the RAWDATA table. Executes steps 3 and 4 only:
  [3/4] Extract task identifiers from commit messages
  [4/4] Fetch task details from tracker
"""

import logging
import sys

import config
from database import DatabaseManager
from task_extractor import TaskExtractor
from task_fetcher import TaskFetcher


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
    print("CodeXplorer - Task Resume Mode")
    print("=" * 80)
    print(f"Database File:       {config.DB_FILE}")
    print(f"Task Pattern:        {config.CURRENT_MASK}")
    print(f"Tracker Type:        {config.TRACKER_TYPE}")
    print(f"Jira URL:            {config.JIRA_URL}")
    print(f"Jira Connector:      {config.JIRA_CONNECTOR_TYPE}")
    print(f"Rate Limit:          {config.REQUEST_RATE_LIMIT} req/min")
    print(f"Batch Size:          {config.DB_BATCH_SIZE}")
    print("=" * 80 + "\n")


def verify_rawdata(db_manager: DatabaseManager) -> int:
    """Check RAWDATA table exists and has rows. Returns row count."""
    import sqlite3
    conn = sqlite3.connect(config.DB_FILE)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM RAWDATA")
        count = cursor.fetchone()[0]
        return count
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            f"RAWDATA table not found or inaccessible: {e}\n"
            "Make sure commit extraction (step 2) has completed first."
        ) from e
    finally:
        conn.close()


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    print_config()

    try:
        # Connect to existing database — do NOT call create_tables() to avoid
        # overwriting existing data; tables must already exist.
        print("[1/4] Connecting to existing database...")
        db_manager = DatabaseManager(config.DB_FILE)

        row_count = verify_rawdata(db_manager)
        print(f"      RAWDATA: {row_count:,} commit rows found")
        if row_count == 0:
            print("      WARNING: RAWDATA is empty — no commits to process.")
            sys.exit(1)
        print("      Database connection OK\n")

        # Extract task identifiers from commit messages
        print("[2/4] Extracting task identifiers from commit messages...")
        task_extractor = TaskExtractor(config.CURRENT_MASK)
        task_extractor.process_all_commits(db_manager)
        print("      Task identifiers extracted\n")

        # Fetch task details from tracker
        print("[3/4] Fetching task details from tracker...")
        task_fetcher = TaskFetcher(
            config.JIRA_URL,
            connector_type=config.JIRA_CONNECTOR_TYPE
        )
        task_fetcher.fetch_all_tasks(
            db_manager,
            rate_limit=config.REQUEST_RATE_LIMIT,
            batch_size=config.DB_BATCH_SIZE
        )
        print("      Task details fetched\n")

        print("[4/4] Done.")
        print(f"\nData saved to: {config.DB_FILE}")

    except Exception as e:
        logger.error(f"Error during task extraction: {e}", exc_info=True)
        print(f"\nError: {e}")
        print(f"Check {config.LOG_FILE} for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
