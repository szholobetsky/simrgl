#!/usr/bin/env python3
"""
Clear all vector collections from PostgreSQL
Useful for starting fresh with new indexing strategy
"""

import psycopg2
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import config


def clear_all_vectors():
    """Drop all vector tables and recreate schema"""
    logger.info("Connecting to PostgreSQL...")

    try:
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )

        cursor = conn.cursor()

        # Get list of all tables in vectors schema
        cursor.execute(f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = '{config.POSTGRES_SCHEMA}'
            AND table_type = 'BASE TABLE'
        """)

        tables = cursor.fetchall()

        if not tables:
            logger.info(f"No tables found in schema '{config.POSTGRES_SCHEMA}'")
            conn.close()
            return

        logger.info(f"Found {len(tables)} tables to drop:")
        for table in tables:
            logger.info(f"  - {table[0]}")

        # Ask for confirmation
        response = input("\nAre you sure you want to DROP ALL these tables? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborted by user")
            conn.close()
            return

        # Drop all tables
        logger.info("Dropping all tables...")
        for table in tables:
            table_name = table[0]
            # Quote table name to handle special characters like hyphens
            cursor.execute(f'DROP TABLE IF EXISTS {config.POSTGRES_SCHEMA}."{table_name}" CASCADE')
            logger.info(f"  ✓ Dropped {table_name}")

        conn.commit()

        # Also drop rawdata table if it exists
        cursor.execute(f"DROP TABLE IF EXISTS {config.POSTGRES_SCHEMA}.rawdata CASCADE")
        logger.info(f"  ✓ Dropped rawdata table")

        conn.commit()

        logger.info("✓ All vector tables cleared successfully")
        logger.info(f"Schema '{config.POSTGRES_SCHEMA}' is now empty")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error clearing vectors: {e}")
        sys.exit(1)


def main():
    """Main function"""
    print("="*60)
    print("CLEAR POSTGRESQL VECTOR COLLECTIONS")
    print("="*60)
    print()
    print(f"Database: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}")
    print(f"Schema: {config.POSTGRES_SCHEMA}")
    print()
    print("WARNING: This will delete ALL vector collections and rawdata!")
    print("="*60)
    print()

    clear_all_vectors()


if __name__ == "__main__":
    main()
