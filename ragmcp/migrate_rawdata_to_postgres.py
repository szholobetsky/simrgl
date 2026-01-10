#!/usr/bin/env python3
"""
Migration Script: SQLite RAWDATA -> PostgreSQL
Migrates TASK_NAME, PATH, MESSAGE, and DIFF from SQLite to PostgreSQL
"""

import sqlite3
import psycopg2
from psycopg2.extras import execute_batch
import logging
import sys
from tqdm import tqdm
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_postgres_table(pg_conn):
    """Create RAWDATA table in PostgreSQL"""
    logger.info("Creating RAWDATA table in PostgreSQL...")

    with pg_conn.cursor() as cursor:
        # Create schema if not exists
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {config.POSTGRES_SCHEMA}")

        # Drop existing table if exists
        cursor.execute(f"""
            DROP TABLE IF EXISTS {config.POSTGRES_SCHEMA}.rawdata CASCADE
        """)

        # Create new table
        cursor.execute(f"""
            CREATE TABLE {config.POSTGRES_SCHEMA}.rawdata (
                id SERIAL PRIMARY KEY,
                task_name TEXT NOT NULL,
                path TEXT NOT NULL,
                message TEXT,
                diff TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for faster lookups
        cursor.execute(f"""
            CREATE INDEX idx_rawdata_task_name
            ON {config.POSTGRES_SCHEMA}.rawdata(task_name)
        """)

        cursor.execute(f"""
            CREATE INDEX idx_rawdata_path
            ON {config.POSTGRES_SCHEMA}.rawdata(path)
        """)

        cursor.execute(f"""
            CREATE INDEX idx_rawdata_task_path
            ON {config.POSTGRES_SCHEMA}.rawdata(task_name, path)
        """)

        pg_conn.commit()
        logger.info("Table created successfully")


def migrate_data(sqlite_path, pg_conn, batch_size=1000):
    """Migrate data from SQLite to PostgreSQL"""
    logger.info(f"Connecting to SQLite: {sqlite_path}")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_cursor = sqlite_conn.cursor()

    # Get total count (only valid records with both TASK_NAME and PATH)
    sqlite_cursor.execute("""
        SELECT COUNT(*) FROM RAWDATA
        WHERE TASK_NAME IS NOT NULL AND PATH IS NOT NULL
    """)
    total_count = sqlite_cursor.fetchone()[0]

    # Get count of invalid records (will be skipped)
    sqlite_cursor.execute("""
        SELECT COUNT(*) FROM RAWDATA
        WHERE TASK_NAME IS NULL OR PATH IS NULL
    """)
    invalid_count = sqlite_cursor.fetchone()[0]

    logger.info(f"Total records to migrate: {total_count:,}")
    if invalid_count > 0:
        logger.warning(f"Skipping {invalid_count:,} records with NULL TASK_NAME or PATH")

    # Fetch data in batches (skip rows with NULL TASK_NAME or PATH)
    sqlite_cursor.execute("""
        SELECT TASK_NAME, PATH, MESSAGE, DIFF
        FROM RAWDATA
        WHERE TASK_NAME IS NOT NULL AND PATH IS NOT NULL
    """)

    pg_cursor = pg_conn.cursor()
    batch = []
    migrated = 0

    with tqdm(total=total_count, desc="Migrating records") as pbar:
        while True:
            rows = sqlite_cursor.fetchmany(batch_size)
            if not rows:
                break

            # Prepare batch data
            for row in rows:
                task_name, path, message, diff = row

                # Convert BLOB to text if needed
                if isinstance(path, bytes):
                    path = path.decode('utf-8', errors='ignore')
                if isinstance(message, bytes):
                    message = message.decode('utf-8', errors='ignore')
                if isinstance(diff, bytes):
                    diff = diff.decode('utf-8', errors='ignore')

                batch.append((task_name, path, message, diff))

            # Insert batch into PostgreSQL
            execute_batch(
                pg_cursor,
                f"""
                INSERT INTO {config.POSTGRES_SCHEMA}.rawdata
                (task_name, path, message, diff)
                VALUES (%s, %s, %s, %s)
                """,
                batch,
                page_size=batch_size
            )

            migrated += len(batch)
            pbar.update(len(batch))
            batch = []

            # Commit periodically
            pg_conn.commit()

    sqlite_conn.close()
    logger.info(f"Migration completed: {migrated:,} records migrated")

    # Verify count
    pg_cursor.execute(f"SELECT COUNT(*) FROM {config.POSTGRES_SCHEMA}.rawdata")
    pg_count = pg_cursor.fetchone()[0]
    logger.info(f"PostgreSQL record count: {pg_count:,}")

    if pg_count == total_count:
        logger.info("✓ Verification successful: counts match")
    else:
        logger.warning(f"⚠ Count mismatch: SQLite={total_count}, PostgreSQL={pg_count}")


def get_migration_stats(pg_conn):
    """Get statistics about migrated data"""
    logger.info("Gathering migration statistics...")

    with pg_conn.cursor() as cursor:
        # Total records
        cursor.execute(f"SELECT COUNT(*) FROM {config.POSTGRES_SCHEMA}.rawdata")
        total = cursor.fetchone()[0]

        # Unique tasks
        cursor.execute(f"SELECT COUNT(DISTINCT task_name) FROM {config.POSTGRES_SCHEMA}.rawdata")
        unique_tasks = cursor.fetchone()[0]

        # Unique files
        cursor.execute(f"SELECT COUNT(DISTINCT path) FROM {config.POSTGRES_SCHEMA}.rawdata")
        unique_files = cursor.fetchone()[0]

        # Top 10 tasks by file count
        cursor.execute(f"""
            SELECT task_name, COUNT(*) as file_count
            FROM {config.POSTGRES_SCHEMA}.rawdata
            GROUP BY task_name
            ORDER BY file_count DESC
            LIMIT 10
        """)
        top_tasks = cursor.fetchall()

        print("\n" + "="*60)
        print("MIGRATION STATISTICS")
        print("="*60)
        print(f"Total records:    {total:,}")
        print(f"Unique tasks:     {unique_tasks:,}")
        print(f"Unique files:     {unique_files:,}")
        print(f"Avg files/task:   {total/unique_tasks:.1f}")
        print("\nTop 10 tasks by file count:")
        for i, (task, count) in enumerate(top_tasks, 1):
            print(f"  {i:2d}. {task:15s} - {count:4d} files")
        print("="*60 + "\n")


def main():
    """Main migration function"""
    logger.info("Starting RAWDATA migration: SQLite → PostgreSQL")
    logger.info(f"SQLite DB: {config.DB_PATH}")
    logger.info(f"PostgreSQL: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}")

    try:
        # Connect to PostgreSQL
        pg_conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        logger.info("✓ Connected to PostgreSQL")

        # Create table
        create_postgres_table(pg_conn)

        # Migrate data
        migrate_data(config.DB_PATH, pg_conn)

        # Get statistics
        get_migration_stats(pg_conn)

        pg_conn.close()
        logger.info("✓ Migration completed successfully")

    except Exception as e:
        logger.error(f"✗ Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
