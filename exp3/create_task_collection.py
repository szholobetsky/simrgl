#!/usr/bin/env python3
"""
Create a collection with individual TASK embeddings
Useful for task-to-task similarity search and module recreation
Supports both Qdrant and PostgreSQL backends
"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config
from vector_backends import get_vector_backend
from utils import logger

def create_task_collection(backend_type: str = None, model_key: str = None, window: str = 'all'):
    """
    Create collection with individual task embeddings

    Args:
        backend_type: 'qdrant' or 'postgres' (None = use config.VECTOR_BACKEND)
        model_key: Key from EMBEDDING_MODELS (None = use default)
        window: 'w100' for last 100 tasks, 'all' for complete history
    """

    backend_type = backend_type or config.VECTOR_BACKEND
    model_config = config.get_model_config(model_key)
    model_name = model_config['name']
    model_suffix = f"_{model_key}" if model_key else "_bge-small"

    logger.info("="*60)
    logger.info("Creating Task Embeddings Collection")
    logger.info(f"Backend: {backend_type}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Window: {window}")
    logger.info("="*60)

    # Connect to database
    logger.info(f"\n1. Loading tasks from {config.DB_PATH}...")
    conn = sqlite3.connect(config.DB_PATH)

    # Filter tasks based on window
    if window == 'w100':
        # Get last 100 tasks
        query = """
        SELECT ID, NAME, TITLE, DESCRIPTION, COMMENTS
        FROM TASK
        ORDER BY ID DESC
        LIMIT 100
        """
        logger.info(f"   Filtering: Last 100 tasks (window={window})")
    else:
        # Get all tasks
        query = """
        SELECT ID, NAME, TITLE, DESCRIPTION, COMMENTS
        FROM TASK
        ORDER BY ID
        """
        logger.info(f"   Filtering: All tasks (window={window})")

    tasks_df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"   Loaded {len(tasks_df)} tasks")

    # Initialize model
    logger.info(f"\n2. Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=model_config.get('trust_remote_code', False))
    vector_size = model.get_sentence_embedding_dimension()

    # Combine title + description
    logger.info("\n3. Combining text fields...")
    texts = []
    for _, row in tasks_df.iterrows():
        title = str(row['TITLE']) if pd.notna(row['TITLE']) else ''
        desc = str(row['DESCRIPTION']) if pd.notna(row['DESCRIPTION']) else ''
        text = f"{title}\n{desc}".strip()
        texts.append(text)

    # Generate embeddings
    logger.info("\n4. Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=config.BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Initialize backend
    logger.info(f"\n5. Connecting to {backend_type} backend...")
    backend = get_vector_backend(backend_type)
    backend.connect()

    # Create collection with window suffix
    if window == 'w100':
        collection_name = f"task_embeddings_w100{model_suffix}"
    else:
        collection_name = f"task_embeddings_all{model_suffix}"

    logger.info(f"\n6. Creating collection: {collection_name}...")
    backend.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        recreate=True
    )

    logger.info(f"   Created collection with dimension {vector_size}")

    # Prepare task vectors as dictionary
    logger.info("\n7. Preparing task embeddings...")
    task_vectors = {}
    task_metadata = {}

    for idx, row in tqdm(tasks_df.iterrows(), total=len(tasks_df), desc="Preparing"):
        task_id = str(row['ID'])
        task_vectors[task_id] = embeddings[idx]
        task_metadata[task_id] = {
            'task_name': row['NAME'],
            'title': str(row['TITLE'])[:500] if pd.notna(row['TITLE']) else '',
            'description': str(row['DESCRIPTION'])[:500] if pd.notna(row['DESCRIPTION']) else '',
            'has_comments': pd.notna(row['COMMENTS']) and len(str(row['COMMENTS'])) > 0
        }

    # Upload embeddings using backend-specific logic
    logger.info("\n8. Uploading task embeddings...")
    if backend_type == 'qdrant':
        _upload_to_qdrant(backend, collection_name, task_vectors, task_metadata)
    elif backend_type == 'postgres':
        _upload_to_postgres(backend, collection_name, task_vectors, task_metadata)

    logger.info(f"\n   Uploaded {len(tasks_df)} task embeddings")

    # Verify
    collection_info = backend.get_collection_info(collection_name)
    logger.info(f"\n✓ Collection created successfully!")
    logger.info(f"  Name: {collection_name}")
    logger.info(f"  Vectors: {collection_info.get('count', 0):,}")
    logger.info(f"  Vector size: {collection_info.get('vector_size', 0)}")

    logger.info("\n" + "="*60)
    logger.info("Task collection ready!")
    logger.info("="*60)
    logger.info("\nYou can now:")
    logger.info("  - Search for similar historical tasks")
    logger.info("  - Find tasks by description")
    logger.info("  - See what similar problems were solved before")
    logger.info("  - Recreate module embeddings from task history")
    logger.info("="*60)


def _upload_to_qdrant(backend, collection_name, task_vectors, task_metadata):
    """Upload task embeddings to Qdrant (with proper ID handling)"""
    from qdrant_client.http import models
    from tqdm import tqdm

    points = []
    for task_id, vector in task_vectors.items():
        metadata = task_metadata[task_id]
        point = models.PointStruct(
            id=int(task_id),  # Use task ID as point ID
            vector=vector.tolist(),
            payload={
                'task_id': task_id,
                'task_name': metadata['task_name'],
                'title': metadata['title'],
                'description': metadata['description'],
                'has_comments': metadata['has_comments']
            }
        )
        points.append(point)

        # Upload in batches
        if len(points) >= config.UPSERT_BATCH_SIZE:
            backend.client.upsert(collection_name=collection_name, points=points, wait=True)
            points = []

    # Upload remaining
    if points:
        backend.client.upsert(collection_name=collection_name, points=points, wait=True)


def _upload_to_postgres(backend, collection_name, task_vectors, task_metadata):
    """Upload task embeddings to PostgreSQL (with metadata in payload)"""
    from psycopg2.extras import execute_values
    from tqdm import tqdm

    table_name = f'{backend.schema}."{collection_name}"'  # Quote table name to handle hyphens

    # Prepare data with metadata as JSONB
    data = []
    for task_id, vector in task_vectors.items():
        metadata = task_metadata[task_id]
        # Store metadata as JSON string
        import json
        metadata_json = json.dumps(metadata)

        data.append((
            task_id,
            'task',
            vector.tolist(),
            metadata_json
        ))

    # Batch insert
    batch_size = config.UPSERT_BATCH_SIZE
    for i in tqdm(range(0, len(data), batch_size), desc="Upserting batches"):
        batch = data[i:i + batch_size]

        # First, modify table to add metadata column if it doesn't exist
        try:
            backend.cursor.execute(f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS metadata JSONB;
            """)
            backend.conn.commit()
        except:
            pass

        insert_sql = f"""
        INSERT INTO {table_name} (path, type, vector, metadata)
        VALUES %s
        """
        execute_values(backend.cursor, insert_sql, batch)
        backend.conn.commit()


def main():
    parser = argparse.ArgumentParser(description='Create Task Embeddings Collection')
    parser.add_argument(
        '--backend',
        type=str,
        default=None,
        choices=['qdrant', 'postgres'],
        help=f'Vector backend to use (default: {config.VECTOR_BACKEND})'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=list(config.EMBEDDING_MODELS.keys()),
        help='Embedding model to use (default: bge-small)'
    )
    parser.add_argument(
        '--window',
        type=str,
        default='all',
        choices=['w100', 'all'],
        help='Window size: w100 for last 100 tasks, all for complete history (default: all)'
    )

    args = parser.parse_args()

    create_task_collection(
        backend_type=args.backend,
        model_key=args.model,
        window=args.window
    )


if __name__ == "__main__":
    main()
