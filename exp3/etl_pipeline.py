"""
ETL Pipeline for RAG Research Experiment
Loads data, creates train/test splits, generates embeddings, and populates Qdrant
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import argparse
from typing import Tuple, Dict, List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from utils import (
    combine_text_fields,
    extract_file_path,
    extract_module_path,
    logger
)
from vector_backends import get_vector_backend


class ETLPipeline:
    """ETL Pipeline for RAG Experiment"""

    def __init__(self, split_strategy: str = 'recent', test_size: int = None, model_key: str = None, backend_type: str = None):
        """
        Initialize ETL Pipeline

        Args:
            split_strategy: 'recent' or 'modn'
            test_size: Number of test tasks (default from config)
            model_key: Key from EMBEDDING_MODELS (None = use default)
            backend_type: Vector backend type - 'qdrant' or 'postgres' (None = use config)
        """
        self.split_strategy = split_strategy
        self.test_size = test_size or config.TEST_SIZE
        self.model_key = model_key
        self.model_config = config.get_model_config(model_key)
        self.backend_type = backend_type or config.VECTOR_BACKEND
        self.model = None
        self.backend = None
        self.vector_size = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from SQLite database"""
        logger.info(f"Loading data from {config.DB_PATH}...")

        conn = sqlite3.connect(config.DB_PATH)

        # Load tasks
        tasks_df = pd.read_sql_query("SELECT * FROM TASK", conn)

        # Load commits/files
        rawdata_df = pd.read_sql_query("SELECT * FROM RAWDATA", conn)

        conn.close()

        logger.info(f"Loaded {len(tasks_df)} tasks and {len(rawdata_df)} commits")
        return tasks_df, rawdata_df

    def create_split(
        self,
        tasks_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split tasks into train and test sets

        Args:
            tasks_df: DataFrame with all tasks

        Returns:
            train_tasks, test_tasks
        """
        # Sort by ID (chronological order)
        tasks_df = tasks_df.sort_values('ID').reset_index(drop=True)
        total_tasks = len(tasks_df)
        indices = np.arange(total_tasks)

        logger.info(f"Creating split using strategy: {self.split_strategy}")
        logger.info(f"Total tasks: {total_tasks}, Test size: {self.test_size}")

        if self.split_strategy == 'recent':
            # Test on most recent tasks
            test_indices = indices[-self.test_size:]
            train_indices = indices[:-self.test_size]

        elif self.split_strategy == 'modn':
            # Test on uniformly sampled tasks
            step = total_tasks // self.test_size
            test_indices = indices[::step][:self.test_size]
            train_indices = np.setdiff1d(indices, test_indices)

        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

        train_tasks = tasks_df.iloc[train_indices].copy()
        test_tasks = tasks_df.iloc[test_indices].copy()

        logger.info(f"Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")

        return train_tasks, test_tasks

    def apply_time_window(
        self,
        train_tasks: pd.DataFrame,
        test_tasks: pd.DataFrame,
        window_size: int = None
    ) -> pd.DataFrame:
        """
        Apply time window filter to training data

        Args:
            train_tasks: All training tasks
            test_tasks: Test tasks (to determine cutoff)
            window_size: Number of recent tasks to use (None = all)

        Returns:
            Filtered training tasks
        """
        if window_size is None:
            logger.info("Using ALL training tasks (no window limit)")
            return train_tasks

        # For 'recent' split, take last N tasks before test set
        if self.split_strategy == 'recent':
            windowed_tasks = train_tasks.iloc[-window_size:]
            logger.info(f"Applied window: using last {window_size} tasks from training set")

        # For 'modn' split, take tasks closest to each test task
        else:
            # For modn, we'll just take the most recent window_size tasks overall
            # This is a simplification but reasonable for the experiment
            windowed_tasks = train_tasks.iloc[-window_size:]
            logger.info(f"Applied window: using {window_size} most recent training tasks")

        logger.info(f"Window filtered: {len(windowed_tasks)} tasks (from {len(train_tasks)})")
        return windowed_tasks

    def prepare_test_set(
        self,
        test_tasks: pd.DataFrame,
        merged_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Prepare test set with ground truth files

        Args:
            test_tasks: Test task DataFrame
            merged_df: Merged tasks+commits DataFrame

        Returns:
            List of test task dictionaries with ground truth
        """
        test_set = []

        for _, task in test_tasks.iterrows():
            relevant_files = merged_df[
                merged_df['NAME'] == task['NAME']
            ]['PATH'].unique().tolist()

            # Clean and normalize file paths
            relevant_files = [extract_file_path(f) for f in relevant_files]
            relevant_files = [f for f in relevant_files if f != "unknown"]

            test_set.append({
                'NAME': task['NAME'],
                'TITLE': task.get('TITLE', ''),
                'DESCRIPTION': task.get('DESCRIPTION', ''),
                'COMMENTS': task.get('COMMENTS', ''),
                'relevant_files': relevant_files
            })

        logger.info(f"Prepared test set with {len(test_set)} tasks")
        return test_set

    def initialize_model(self):
        """Initialize embedding model"""
        if self.model is None:
            model_name = self.model_config['name']
            trust_remote = self.model_config.get('trust_remote_code', False)
            logger.info(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Vector size: {self.vector_size}")

    def initialize_backend(self):
        """Initialize vector backend (Qdrant or PostgreSQL)"""
        if self.backend is None:
            logger.info(f"Initializing vector backend: {self.backend_type}")
            self.backend = get_vector_backend(self.backend_type)
            self.backend.connect()
            logger.info(f"Vector backend ready: {self.backend_type}")

    def generate_embeddings(
        self,
        tasks_df: pd.DataFrame,
        source_variant: str
    ) -> pd.DataFrame:
        """
        Generate embeddings for tasks using specified source variant

        Args:
            tasks_df: DataFrame with task data
            source_variant: Key from config.SOURCE_VARIANTS

        Returns:
            DataFrame with 'text' and 'vector' columns added
        """
        self.initialize_model()

        variant_config = config.SOURCE_VARIANTS[source_variant]
        fields = variant_config['fields']

        logger.info(f"Generating embeddings for source variant: {source_variant}")
        logger.info(f"Using fields: {fields}")

        # Combine text fields
        texts = []
        for _, row in tasks_df.iterrows():
            text = combine_text_fields(row, fields)
            texts.append(text)

        tasks_df = tasks_df.copy()
        tasks_df['text'] = texts

        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True
        )

        tasks_df['vector'] = list(embeddings)

        return tasks_df

    def aggregate_by_target(
        self,
        merged_df: pd.DataFrame,
        target_variant: str
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate task vectors by target unit (file or module)

        Args:
            merged_df: DataFrame with task-commit pairs and vectors
            target_variant: Key from config.TARGET_VARIANTS

        Returns:
            Dictionary mapping target path to centroid vector
        """
        logger.info(f"Aggregating by target variant: {target_variant}")

        # Extract target paths
        if target_variant == 'file':
            merged_df['target_path'] = merged_df['PATH'].apply(extract_file_path)
        elif target_variant == 'module':
            merged_df['target_path'] = merged_df['PATH'].apply(extract_module_path)
        else:
            raise ValueError(f"Unknown target variant: {target_variant}")

        # Remove unknown paths
        merged_df = merged_df[merged_df['target_path'] != 'unknown']

        # Group by target and compute centroids
        target_vectors = {}
        grouped = merged_df.groupby('target_path')

        for target_path, group in tqdm(grouped, desc="Computing centroids"):
            vectors = np.stack(group['vector'].values)
            centroid = np.mean(vectors, axis=0)
            target_vectors[target_path] = centroid

        logger.info(f"Aggregated {len(target_vectors)} {target_variant} units")
        return target_vectors

    def create_collection(
        self,
        collection_name: str,
        recreate: bool = True
    ):
        """
        Create or recreate vector collection

        Args:
            collection_name: Name of the collection
            recreate: If True, delete existing collection
        """
        self.initialize_backend()
        self.initialize_model()

        self.backend.create_collection(
            collection_name=collection_name,
            vector_size=self.vector_size,
            recreate=recreate
        )
        logger.info(f"Created collection: {collection_name} (backend: {self.backend_type})")

    def upsert_vectors(
        self,
        collection_name: str,
        target_vectors: Dict[str, np.ndarray],
        target_variant: str
    ):
        """
        Upsert target vectors to vector backend

        Args:
            collection_name: Name of the collection
            target_vectors: Dictionary mapping paths to vectors
            target_variant: Type of target (file or module)
        """
        self.backend.upsert_vectors(
            collection_name=collection_name,
            vectors_dict=target_vectors,
            target_variant=target_variant
        )

    def run(
        self,
        source_variants: List[str] = None,
        target_variants: List[str] = None,
        window_variants: List[str] = None
    ):
        """
        Run the complete ETL pipeline

        Args:
            source_variants: List of source variant keys (default: all)
            target_variants: List of target variant keys (default: all)
            window_variants: List of window variant keys (default: all)
        """
        # Default to all variants
        source_variants = source_variants or list(config.SOURCE_VARIANTS.keys())
        target_variants = target_variants or list(config.TARGET_VARIANTS.keys())
        window_variants = window_variants or list(config.WINDOW_VARIANTS.keys())

        logger.info("=" * 80)
        logger.info("Starting ETL Pipeline")
        logger.info(f"Vector Backend: {self.backend_type}")
        logger.info(f"Embedding Model: {self.model_config['name']}")
        logger.info(f"Model Key: {self.model_key or 'default'}")
        logger.info(f"Split Strategy: {self.split_strategy}")
        logger.info(f"Test Size: {self.test_size}")
        logger.info(f"Source Variants: {source_variants}")
        logger.info(f"Target Variants: {target_variants}")
        logger.info(f"Window Variants: {window_variants}")
        logger.info("=" * 80)

        # 1. Load data
        tasks_df, rawdata_df = self.load_data()

        # 2. Create train/test split
        train_tasks_all, test_tasks = self.create_split(tasks_df)

        # 3. Merge tasks with commits
        merged_df = pd.merge(
            rawdata_df,
            tasks_df[['NAME', 'TITLE', 'DESCRIPTION', 'COMMENTS']],
            left_on='TASK_NAME',
            right_on='NAME',
            how='inner'
        )
        logger.info(f"Merged dataset has {len(merged_df)} task-commit pairs")

        # 4. Prepare and save test set
        test_set = self.prepare_test_set(test_tasks, merged_df)
        with open(config.TEST_SET_FILE, 'w') as f:
            json.dump(test_set, f, indent=2)
        logger.info(f"Saved test set to {config.TEST_SET_FILE}")

        # 5. Process each combination of variants
        for window_key in window_variants:
            window_config = config.WINDOW_VARIANTS[window_key]
            window_size = window_config['size']

            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Window: {window_config['name']} ({window_key})")
            logger.info(f"{'='*80}")

            # Apply time window to training data
            train_tasks = self.apply_time_window(
                train_tasks_all,
                test_tasks,
                window_size
            )

            # Filter merged data to only include training tasks
            train_task_names = set(train_tasks['NAME'])
            train_merged = merged_df[merged_df['NAME'].isin(train_task_names)].copy()

            for source_key in source_variants:
                logger.info(f"\n{'-'*80}")
                logger.info(f"Processing Source: {source_key}")
                logger.info(f"{'-'*80}")

                # Generate embeddings for training tasks
                train_tasks_embedded = self.generate_embeddings(train_tasks, source_key)

                # Map vectors to merged data
                task_vector_map = dict(zip(
                    train_tasks_embedded['NAME'],
                    train_tasks_embedded['vector']
                ))
                train_merged['vector'] = train_merged['NAME'].map(task_vector_map)
                train_merged = train_merged.dropna(subset=['vector'])

                for target_key in target_variants:
                    # Include model in collection name if specified
                    model_suffix = f"_{self.model_key}" if self.model_key else ""
                    collection_name = f"{config.COLLECTION_PREFIX}_{source_key}_{target_key}_{window_key}_{self.split_strategy}{model_suffix}"

                    logger.info(f"Processing Target: {target_key}")
                    logger.info(f"Collection: {collection_name}")

                    # Aggregate vectors by target
                    target_vectors = self.aggregate_by_target(
                        train_merged,
                        target_key
                    )

                    if not target_vectors:
                        logger.warning(f"No vectors for {collection_name}, skipping")
                        continue

                    # Create collection and upsert
                    self.create_collection(collection_name)
                    self.upsert_vectors(
                        collection_name,
                        target_vectors,
                        target_key
                    )

        logger.info("\n" + "=" * 80)
        logger.info("ETL Pipeline Complete!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='RAG ETL Pipeline')
    parser.add_argument(
        '--split_strategy',
        type=str,
        default='recent',
        choices=['recent', 'modn'],
        help='Test/train split strategy'
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=config.TEST_SIZE,
        help='Number of test tasks'
    )
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
        help='Embedding model to use (default: bge-small). Options: ' +
             ', '.join(config.EMBEDDING_MODELS.keys())
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=list(config.SOURCE_VARIANTS.keys()),
        default=None,
        help='Source variants to process (default: all)'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=list(config.TARGET_VARIANTS.keys()),
        default=None,
        help='Target variants to process (default: all)'
    )
    parser.add_argument(
        '--windows',
        nargs='+',
        choices=list(config.WINDOW_VARIANTS.keys()),
        default=None,
        help='Window variants to process (default: all)'
    )

    args = parser.parse_args()

    pipeline = ETLPipeline(
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        model_key=args.model,
        backend_type=args.backend
    )

    pipeline.run(
        source_variants=args.sources,
        target_variants=args.targets,
        window_variants=args.windows
    )


if __name__ == '__main__':
    main()
