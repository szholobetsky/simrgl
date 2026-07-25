#!/usr/bin/env python3
"""
Test PostgreSQL + pgvector setup
"""

import sys

def test_postgres_connection():
    """Test basic PostgreSQL connection"""
    print("=" * 60)
    print("Testing PostgreSQL Connection")
    print("=" * 60)
    print()

    try:
        import psycopg2
        print("✓ psycopg2 package installed")
    except ImportError:
        print("✗ psycopg2 not installed!")
        print("  Install: pip install psycopg2-binary")
        return False

    import config

    try:
        print(f"\nConnecting to {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}...")
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        cursor = conn.cursor()
        print("✓ Connected to PostgreSQL")

        # Test pgvector extension
        print("\nTesting pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("✓ pgvector extension available")

        # Test vector operations
        print("\nTesting vector operations...")
        cursor.execute("SELECT '[1,2,3]'::vector;")
        result = cursor.fetchone()
        print(f"✓ Vector type working: {result[0]}")

        # Test cosine distance
        print("\nTesting cosine similarity...")
        cursor.execute("""
        SELECT 1 - ('[1,2,3]'::vector <=> '[1,2,3]'::vector) as similarity;
        """)
        result = cursor.fetchone()
        print(f"✓ Cosine similarity working: {result[0]}")

        # Create test schema
        print(f"\nCreating schema: {config.POSTGRES_SCHEMA}...")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {config.POSTGRES_SCHEMA};")
        conn.commit()
        print(f"✓ Schema '{config.POSTGRES_SCHEMA}' ready")

        # Test vector dimensions
        print("\nTesting 384-dim vector (bge-small)...")
        test_vector = "[" + ",".join(["0.1"] * 384) + "]"
        cursor.execute(f"SELECT '{test_vector}'::vector(384);")
        result = cursor.fetchone()
        print(f"✓ 384-dim vector working")

        cursor.close()
        conn.close()

        print("\n" + "=" * 60)
        print("All PostgreSQL tests passed!")
        print("=" * 60)
        print("\nYou can now use PostgreSQL backend:")
        print("  1. Edit config.py: VECTOR_BACKEND = 'postgres'")
        print("  2. Run ETL: python etl_pipeline.py --backend postgres ...")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  Windows: start_postgres.bat")
        print("  Linux: ./start_postgres.sh")
        return False


if __name__ == "__main__":
    success = test_postgres_connection()
    sys.exit(0 if success else 1)
