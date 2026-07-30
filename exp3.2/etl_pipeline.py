"""
ETL Pipeline for RAG Research Experiment
Loads data, creates train/test splits, generates embeddings, and populates Qdrant
"""

import os
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
    get_commits_table_name,
    logger
)
from vector_backends import get_vector_backend


class ETLPipeline:
    """ETL Pipeline for RAG Experiment"""

    # TASK_UNIT='commit' noise filtering thresholds (see _build_commit_tasks)
    MAX_FILES_PER_COMMIT = 50
    MIN_SUBJECT_LEN = 8
    MAX_DIFF_CHARS = 4000

    def __init__(self, split_strategy: str = 'recent', test_size: int = None, model_key: str = None,
                 backend_type: str = None, task_unit: str = None):
        """
        Initialize ETL Pipeline

        Args:
            split_strategy: 'recent' or 'modn'
            test_size: Number of test tasks (default from config)
            model_key: Key from EMBEDDING_MODELS (None = use default)
            backend_type: Vector backend type - 'qdrant' or 'postgres' (None = use config)
            task_unit: 'ticket' or 'commit' (None = use config.TASK_UNIT)
        """
        self.split_strategy = split_strategy
        self.test_size = test_size or config.TEST_SIZE
        self.model_key = model_key
        self.model_config = config.get_model_config(model_key)
        self.backend_type = backend_type or config.VECTOR_BACKEND
        self.task_unit = task_unit or config.TASK_UNIT
        self.model = None
        self.backend = None
        self.vector_size = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from SQLite database"""
        logger.info(f"Loading data from {config.DB_PATH} (task_unit={self.task_unit})...")

        conn = sqlite3.connect(config.DB_PATH)

        # Load commits/files (table is named RAWDATA or COMMITS depending on
        # which CodeXplorer generation produced this database).
        # Explicit column list - skips AUTHOR_NAME/AUTHOR_EMAIL/ID (nothing
        # here reads them) and DIFF too, EXCEPT in commit-mode, where the
        # 'diff' source variant needs it - DIFF is the dominant share of row
        # bytes (~82% of DIFF+MESSAGE+PATH on rubocop.db), so it stays out
        # of the default (ticket-mode) query entirely.
        commits_table = get_commits_table_name(conn)
        diff_column = ", DIFF" if self.task_unit == 'commit' else ""
        rawdata_df = pd.read_sql_query(
            f"SELECT SHA, PATH, MESSAGE, CMT_DATE, TASK_NAME{diff_column} FROM {commits_table}",
            conn
        )

        if self.task_unit == 'commit':
            conn.close()
            tasks_df, rawdata_df = self._build_commit_tasks(rawdata_df)
        else:
            tasks_df = pd.read_sql_query("SELECT * FROM TASK", conn)
            conn.close()

        logger.info(f"Loaded {len(tasks_df)} tasks and {len(rawdata_df)} commits")
        return tasks_df, rawdata_df

    def _build_commit_tasks(self, rawdata_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build a synthetic per-commit "task" table for TASK_UNIT='commit'.

        Each surviving commit (SHA) becomes one task: TITLE is the git
        subject line (first line of MESSAGE), DESCRIPTION is the full
        message. There's no COMMENTS equivalent (no comments concept for a
        raw commit) - instead, DIFF (all changed files' diffs for that SHA,
        concatenated and capped at MAX_DIFF_CHARS) fills the analogous
        "extra noisy context" role for the 'diff' source variant, mirroring
        ticket-mode's title/desc/comments density progression with
        title/desc/diff instead. Ground truth files are all PATHs sharing
        that SHA, mirroring how ticket-mode groups commits by TASK_NAME -
        achieved here by setting TASK_NAME=SHA on survivors, so every
        downstream method (create_split, apply_time_window,
        prepare_test_set, generate_embeddings, aggregate_by_target) works
        unchanged regardless of which mode produced its input.

        Noise filtering happens here, before anything downstream sees the
        data - a commit that fails these checks never becomes a "task":
          - merge commits (subject starts with "Merge ")
          - mass-diff commits (over MAX_FILES_PER_COMMIT distinct paths -
            a repo-wide reformat/rename, not a real single task)
          - trivial commits (subject line under MIN_SUBJECT_LEN chars)
        """
        df = rawdata_df.copy()
        df['MESSAGE'] = df['MESSAGE'].fillna('')
        df['_subject'] = df['MESSAGE'].str.split('\n').str[0].str.strip()

        is_merge = df['_subject'].str.lower().str.startswith('merge ')
        files_per_sha = df.groupby('SHA')['PATH'].transform('nunique')
        is_mass_diff = files_per_sha > self.MAX_FILES_PER_COMMIT
        is_trivial = df['_subject'].str.len() < self.MIN_SUBJECT_LEN

        df = df[~(is_merge | is_mass_diff | is_trivial)].copy()

        if df.empty:
            raise ValueError(
                "No commits survived noise filtering for TASK_UNIT='commit' - "
                "check commit message quality for this project"
            )

        df['TASK_NAME'] = df['SHA']
        df['_cmt_dt'] = pd.to_datetime(df['CMT_DATE'], utc=True, errors='coerce')

        one_per_sha = df.sort_values('_cmt_dt').groupby('SHA', as_index=False).first()
        one_per_sha = one_per_sha.sort_values('_cmt_dt').reset_index(drop=True)
        one_per_sha['ID'] = one_per_sha.index + 1
        one_per_sha['NAME'] = one_per_sha['SHA']
        one_per_sha['TITLE'] = one_per_sha['_subject']
        one_per_sha['DESCRIPTION'] = one_per_sha['MESSAGE']
        one_per_sha['COMMENTS'] = ''

        keep_cols = ['ID', 'NAME', 'TITLE', 'DESCRIPTION', 'COMMENTS']

        if 'DIFF' in df.columns:
            # One SHA can span many rows (one per changed file) - join all
            # of that commit's diffs into one text blob per task. Capped:
            # embedding models truncate at a few hundred tokens anyway, so
            # anything past MAX_DIFF_CHARS would never be read regardless -
            # capping keeps memory/compute bounded for large commits without
            # losing anything a real embedding call would have used.
            diff_per_sha = (
                df.groupby('SHA')['DIFF']
                .apply(lambda diffs: '\n'.join(diffs.fillna('').astype(str))[:self.MAX_DIFF_CHARS])
                .rename('DIFF')
            )
            # one_per_sha already carries a (meaningless - single arbitrary
            # row's) 'DIFF' column from the earlier groupby().first(); drop
            # it first so the merge below doesn't collide and suffix into
            # DIFF_x/DIFF_y instead of a plain 'DIFF' column.
            one_per_sha = one_per_sha.drop(columns=['DIFF']).merge(
                diff_per_sha, left_on='SHA', right_index=True, how='left'
            )
            keep_cols.append('DIFF')

        tasks_df = one_per_sha[keep_cols]
        drop_cols = [c for c in ['_subject', '_cmt_dt', 'DIFF'] if c in df.columns]
        rawdata_df = df.drop(columns=drop_cols)

        logger.info(
            f"Commit-mode: {len(tasks_df)} tasks survived noise filtering "
            f"(from {rawdata_df['SHA'].nunique()} distinct commits after filtering, "
            f"{len(rawdata_df)} file rows)"
        )
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
                merged_df['TASK_NAME'] == task['NAME']
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

    def save_test_set(
        self,
        test_tasks: pd.DataFrame,
        rawdata_df: pd.DataFrame,
        output_file: str
    ):
        """
        Save test set to JSON file

        Args:
            test_tasks: Test task DataFrame
            rawdata_df: Raw commit data
            output_file: Path to output JSON file
        """
        import json

        # Prepare test set
        merged_df = rawdata_df[rawdata_df['TASK_NAME'].isin(test_tasks['NAME'])]
        test_set = self.prepare_test_set(test_tasks, merged_df)

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(test_set, f, indent=2)

        logger.info(f"Test set saved to {output_file}")

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
            source_variant: Key from config.TRAIN_SOURCE_VARIANTS

        Returns:
            DataFrame with 'text' and 'vector' columns added
        """
        self.initialize_model()

        variant_config = config.TRAIN_SOURCE_VARIANTS[source_variant]
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
        source_variants = source_variants or list(config.TRAIN_SOURCE_VARIANTS.keys())
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
                    collection_name = config.collection_name(
                        source_key, target_key, window_key, self.split_strategy,
                        model_key=self.model_key
                    )

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
        choices=list(config.TRAIN_SOURCE_VARIANTS.keys()),
        default=None,
        help='Source variants to process (default: all) - index/train side (title/desc/diff)'
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
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help=f'Project name, maps to {config.PROJECTS_DIR}/<project>.db '
             f'(default: {config.PROJECT})'
    )
    parser.add_argument(
        '--task-unit',
        type=str,
        default=None,
        choices=config.TASK_UNITS,
        help=f'Task unit criterion (default: {config.TASK_UNIT})'
    )

    args = parser.parse_args()

    if args.project:
        config.PROJECT = args.project
        config.DB_PATH = os.path.join(config.PROJECTS_DIR, f'{args.project}.db')
    if args.task_unit:
        config.TASK_UNIT = args.task_unit

    pipeline = ETLPipeline(
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        model_key=args.model,
        backend_type=args.backend,
        task_unit=args.task_unit
    )

    pipeline.run(
        source_variants=args.sources,
        target_variants=args.targets,
        window_variants=args.windows
    )


if __name__ == '__main__':
    main()
