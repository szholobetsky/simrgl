"""
Vector Backend Abstraction Layer
Supports both Qdrant and PostgreSQL+pgvector
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
from utils import logger


class VectorBackend(ABC):
    """Abstract base class for vector storage backends"""

    @abstractmethod
    def connect(self):
        """Initialize connection to vector database"""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, recreate: bool = True):
        """
        Create or recreate a collection/table

        Args:
            collection_name: Name of the collection/table
            vector_size: Dimension of vectors
            recreate: If True, delete existing collection
        """
        pass

    @abstractmethod
    def upsert_vectors(
        self,
        collection_name: str,
        vectors_dict: Dict[str, np.ndarray],
        target_variant: str
    ):
        """
        Insert/update vectors in the collection

        Args:
            collection_name: Name of the collection/table
            vectors_dict: Dictionary mapping path to vector
            target_variant: Type of target (file or module)
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search for similar vectors

        Args:
            collection_name: Name of the collection/table
            query_vector: Query vector
            top_k: Number of results to return

        Returns:
            List of results with path, score, and metadata
        """
        pass

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict:
        """Get collection information (size, count, etc.)"""
        pass

    @abstractmethod
    def close(self):
        """Close connection"""
        pass


class QdrantBackend(VectorBackend):
    """Qdrant vector database backend"""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = None

    def connect(self):
        """Initialize Qdrant client"""
        if self.client is None:
            from qdrant_client import QdrantClient
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}...")
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info("Connected to Qdrant")

    def create_collection(self, collection_name: str, vector_size: int, recreate: bool = True):
        """Create Qdrant collection"""
        from qdrant_client.http import models

        self.connect()

        if recreate:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Created Qdrant collection: {collection_name} (dim={vector_size})")

    def upsert_vectors(
        self,
        collection_name: str,
        vectors_dict: Dict[str, np.ndarray],
        target_variant: str
    ):
        """Upsert vectors to Qdrant"""
        from qdrant_client.http import models
        from tqdm import tqdm
        import config

        logger.info(f"Upserting {len(vectors_dict)} vectors to Qdrant collection: {collection_name}")

        points = []
        for idx, (path, vector) in enumerate(vectors_dict.items()):
            points.append(models.PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    "path": path,
                    "type": target_variant
                }
            ))

        # Batch upsert
        batch_size = config.UPSERT_BATCH_SIZE
        for i in tqdm(range(0, len(points), batch_size), desc="Upserting batches"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True
            )

        logger.info(f"Upserted {len(points)} points to {collection_name}")

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """Search Qdrant collection"""
        self.connect()

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True
        )

        return [
            {
                'path': result.payload.get('path', result.payload.get('task_id', 'unknown')),
                'score': result.score,
                'type': result.payload.get('type', 'unknown'),
                **result.payload  # Include all payload fields
            }
            for result in results
        ]

    def get_collection_info(self, collection_name: str) -> Dict:
        """Get Qdrant collection info"""
        self.connect()
        info = self.client.get_collection(collection_name)
        return {
            'name': collection_name,
            'count': info.points_count,
            'vector_size': info.config.params.vectors.size
        }

    def close(self):
        """Close Qdrant connection (no-op for Qdrant client)"""
        pass


class PostgresBackend(VectorBackend):
    """PostgreSQL+pgvector backend"""

    def __init__(self, host: str, port: int, database: str, user: str, password: str, schema: str = 'vectors'):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema
        self.conn = None
        self.cursor = None

    def connect(self):
        """Initialize PostgreSQL connection"""
        if self.conn is None:
            import psycopg2
            from psycopg2.extras import execute_values

            logger.info(f"Connecting to PostgreSQL at {self.host}:{self.port}/{self.database}...")

            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.conn.autocommit = False
            self.cursor = self.conn.cursor()

            # Create pgvector extension if not exists
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create schema if not exists
            self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema};")
            self.conn.commit()

            logger.info(f"Connected to PostgreSQL (schema: {self.schema})")

    def create_collection(self, collection_name: str, vector_size: int, recreate: bool = True):
        """Create PostgreSQL table for vectors"""
        self.connect()

        try:
            # Properly quote table and index names (handles hyphens in collection names)
            table_name = f'{self.schema}."{collection_name}"'
            index_name_vector = f'"{collection_name}_vector_idx"'
            index_name_path = f'"{collection_name}_path_idx"'

            if recreate:
                self.cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                logger.info(f"Deleted existing table: {table_name}")

            # Create table with vector column
            create_table_sql = f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                path TEXT NOT NULL,
                type TEXT NOT NULL,
                vector vector({vector_size}) NOT NULL
            );
            """
            self.cursor.execute(create_table_sql)

            # Create index for vector similarity search (HNSW for cosine similarity)
            index_sql = f"""
            CREATE INDEX {index_name_vector}
            ON {table_name}
            USING hnsw (vector vector_cosine_ops);
            """
            self.cursor.execute(index_sql)

            # Create index on path for faster lookups
            path_index_sql = f"""
            CREATE INDEX {index_name_path}
            ON {table_name} (path);
            """
            self.cursor.execute(path_index_sql)

            self.conn.commit()
            logger.info(f"Created PostgreSQL table: {table_name} (dim={vector_size}) with HNSW index")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def upsert_vectors(
        self,
        collection_name: str,
        vectors_dict: Dict[str, np.ndarray],
        target_variant: str
    ):
        """Upsert vectors to PostgreSQL"""
        from psycopg2.extras import execute_values
        from tqdm import tqdm
        import config

        self.connect()

        try:
            # Properly quote table name
            table_name = f'{self.schema}."{collection_name}"'
            logger.info(f"Upserting {len(vectors_dict)} vectors to PostgreSQL table: {table_name}")

            # Prepare data for batch insert
            data = [
                (path, target_variant, vector.tolist())
                for path, vector in vectors_dict.items()
            ]

            # Batch insert
            batch_size = config.UPSERT_BATCH_SIZE
            for i in tqdm(range(0, len(data), batch_size), desc="Upserting batches"):
                batch = data[i:i + batch_size]

                insert_sql = f"""
                INSERT INTO {table_name} (path, type, vector)
                VALUES %s
                """
                execute_values(self.cursor, insert_sql, batch)
                self.conn.commit()

            logger.info(f"Upserted {len(data)} vectors to {table_name}")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error upserting vectors to {collection_name}: {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """Search PostgreSQL table using pgvector"""
        self.connect()

        try:
            # Properly quote table name
            table_name = f'{self.schema}."{collection_name}"'

            # Check if metadata column exists
            self.cursor.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = 'metadata';
            """, (self.schema, collection_name))
            has_metadata = self.cursor.fetchone() is not None

            # Use cosine distance (1 - cosine similarity)
            # pgvector returns distance, we convert to similarity
            if has_metadata:
                search_sql = f"""
                SELECT path, type, 1 - (vector <=> %s::vector) as score, metadata
                FROM {table_name}
                ORDER BY vector <=> %s::vector
                LIMIT %s;
                """
            else:
                search_sql = f"""
                SELECT path, type, 1 - (vector <=> %s::vector) as score
                FROM {table_name}
                ORDER BY vector <=> %s::vector
                LIMIT %s;
                """

            vector_str = f"[{','.join(map(str, query_vector.tolist()))}]"
            self.cursor.execute(search_sql, (vector_str, vector_str, top_k))

            results = []
            for row in self.cursor.fetchall():
                result = {
                    'path': row[0],
                    'type': row[1],
                    'score': float(row[2])
                }

                # If metadata exists, parse and merge it
                if has_metadata and len(row) > 3 and row[3]:
                    import json
                    try:
                        metadata = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                        result.update(metadata)
                    except:
                        pass

                results.append(result)

            return results
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error searching collection {collection_name}: {e}")
            raise

    def get_collection_info(self, collection_name: str) -> Dict:
        """Get PostgreSQL table info"""
        self.connect()

        # Properly quote table name
        table_name = f'{self.schema}."{collection_name}"'

        # Get row count
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = self.cursor.fetchone()[0]

        # Get vector dimension
        self.cursor.execute(f"""
        SELECT vector_dims(vector)
        FROM {table_name}
        LIMIT 1;
        """)
        result = self.cursor.fetchone()
        vector_size = result[0] if result else 0

        return {
            'name': collection_name,
            'count': count,
            'vector_size': vector_size
        }

    def close(self):
        """Close PostgreSQL connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Closed PostgreSQL connection")


def get_vector_backend(backend_type: str = None) -> VectorBackend:
    """
    Factory function to get the configured vector backend

    Args:
        backend_type: 'qdrant' or 'postgres' (None = use config.VECTOR_BACKEND)

    Returns:
        VectorBackend instance
    """
    import config

    backend_type = backend_type or config.VECTOR_BACKEND

    if backend_type == 'qdrant':
        return QdrantBackend(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
    elif backend_type == 'postgres':
        return PostgresBackend(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            schema=config.POSTGRES_SCHEMA
        )
    else:
        raise ValueError(f"Unknown vector backend: {backend_type}. Use 'qdrant' or 'postgres'")
