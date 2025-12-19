"""
Run RAG Research Experiments
Evaluates all experiment combinations and saves results
"""

import json
import os
import pandas as pd
import argparse
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import config
from utils import (
    combine_text_fields,
    calculate_metrics_for_query,
    aggregate_metrics,
    format_metrics_row,
    validate_test_set,
    log_experiment_start,
    log_experiment_complete,
    extract_module_path,
    logger
)


class ExperimentRunner:
    """Run and evaluate RAG experiments"""

    def __init__(self, split_strategy: str = 'recent', model_key: str = None):
        """
        Initialize Experiment Runner

        Args:
            split_strategy: Split strategy used in ETL ('recent' or 'modn')
            model_key: Key from EMBEDDING_MODELS (None = use default)
        """
        self.split_strategy = split_strategy
        self.model_key = model_key
        self.model_config = config.get_model_config(model_key)
        self.model = None
        self.client = None
        self.test_set = None

    def load_test_set(self) -> List[Dict[str, Any]]:
        """Load test set from JSON file"""
        if not os.path.exists(config.TEST_SET_FILE):
            raise FileNotFoundError(
                f"Test set file not found: {config.TEST_SET_FILE}. "
                "Please run ETL pipeline first."
            )

        with open(config.TEST_SET_FILE, 'r') as f:
            test_set = json.load(f)

        if not validate_test_set(test_set):
            raise ValueError("Test set validation failed")

        self.test_set = test_set
        return test_set

    def initialize_model(self):
        """Initialize embedding model"""
        if self.model is None:
            model_name = self.model_config['name']
            trust_remote = self.model_config.get('trust_remote_code', False)
            logger.info(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote)
            logger.info("Model loaded")

    def initialize_client(self):
        """Initialize Qdrant client"""
        if self.client is None:
            logger.info(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}...")
            self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            logger.info("Connected to Qdrant")

    def check_collection_exists(self, collection_name: str) -> bool:
        """Check if a Qdrant collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"Error checking collection {collection_name}: {e}")
            return False

    def encode_queries(
        self,
        test_set: List[Dict[str, Any]],
        source_variant: str
    ) -> List[List[float]]:
        """
        Encode test queries using the specified source variant

        Args:
            test_set: Test tasks
            source_variant: Source variant key

        Returns:
            List of query vectors
        """
        variant_config = config.SOURCE_VARIANTS[source_variant]
        fields = variant_config['fields']

        queries = []
        for task in test_set:
            query_text = combine_text_fields(task, fields)
            queries.append(query_text)

        logger.info(f"Encoding {len(queries)} queries...")
        query_vectors = self.model.encode(
            queries,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True
        )

        return query_vectors.tolist()

    def query_single_task(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int
    ) -> List[str]:
        """
        Query Qdrant for a single task

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding
            top_k: Number of results to retrieve

        Returns:
            List of retrieved file/module paths
        """
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True
            ).points

            retrieved_paths = [res.payload['path'] for res in results]
            return retrieved_paths

        except Exception as e:
            logger.error(f"Error querying {collection_name}: {e}")
            return []

    def evaluate_experiment(
        self,
        source_variant: str,
        target_variant: str,
        window_variant: str,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single experiment configuration

        Args:
            source_variant: Source variant key
            target_variant: Target variant key
            window_variant: Window variant key
            top_k: Number of results to retrieve (default from config)

        Returns:
            Dictionary with experiment results
        """
        top_k = top_k or config.DEFAULT_TOP_K

        # Build collection name (include model suffix if specified)
        model_suffix = f"_{self.model_key}" if self.model_key else ""
        collection_name = (
            f"{config.COLLECTION_PREFIX}_{source_variant}_{target_variant}_"
            f"{window_variant}_{self.split_strategy}{model_suffix}"
        )

        experiment_id = f"{source_variant}_{target_variant}_{window_variant}_{self.split_strategy}{model_suffix}"

        log_experiment_start(experiment_id, {
            'source': source_variant,
            'target': target_variant,
            'window': window_variant,
            'split': self.split_strategy,
            'collection': collection_name
        })

        # Check if collection exists
        if not self.check_collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist. Skipping.")
            return None

        # Encode queries
        query_vectors = self.encode_queries(self.test_set, source_variant)

        # Query in parallel
        all_metrics = []

        def query_and_evaluate(idx: int) -> Dict[str, float]:
            query_vector = query_vectors[idx]
            task = self.test_set[idx]

            retrieved_paths = self.query_single_task(
                collection_name,
                query_vector,
                top_k
            )

            relevant_files = set(task['relevant_files'])

            # For module-level evaluation, convert file paths to module paths
            if target_variant == 'module':
                relevant_files = set(extract_module_path(f) for f in task['relevant_files'])

            metrics = calculate_metrics_for_query(
                retrieved_paths,
                relevant_files,
                config.TOP_K_VALUES
            )

            return metrics

        logger.info(f"Querying {len(self.test_set)} tasks in parallel...")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(query_and_evaluate, i): i
                for i in range(len(self.test_set))
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Evaluating {experiment_id}"
            ):
                try:
                    metrics = future.result()
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    continue

        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics, config.TOP_K_VALUES)

        log_experiment_complete(experiment_id, aggregated)

        # Format result row
        result = format_metrics_row(
            experiment_id,
            source_variant,
            target_variant,
            window_variant,
            self.split_strategy,
            aggregated,
            config.TOP_K_VALUES
        )

        return result

    def run_all_experiments(
        self,
        source_variants: List[str] = None,
        target_variants: List[str] = None,
        window_variants: List[str] = None
    ) -> pd.DataFrame:
        """
        Run all experiment combinations

        Args:
            source_variants: List of source variant keys (default: all)
            target_variants: List of target variant keys (default: all)
            window_variants: List of window variant keys (default: all)

        Returns:
            DataFrame with all experiment results
        """
        # Default to all variants
        source_variants = source_variants or list(config.SOURCE_VARIANTS.keys())
        target_variants = target_variants or list(config.TARGET_VARIANTS.keys())
        window_variants = window_variants or list(config.WINDOW_VARIANTS.keys())

        logger.info("=" * 80)
        logger.info("Running All Experiments")
        logger.info(f"Embedding Model: {self.model_config['name']}")
        logger.info(f"Model Key: {self.model_key or 'default'}")
        logger.info(f"Source Variants: {source_variants}")
        logger.info(f"Target Variants: {target_variants}")
        logger.info(f"Window Variants: {window_variants}")
        logger.info(f"Split Strategy: {self.split_strategy}")
        logger.info("=" * 80)

        # Initialize resources
        self.load_test_set()
        self.initialize_model()
        self.initialize_client()

        # Run all combinations
        results = []

        for source in source_variants:
            for target in target_variants:
                for window in window_variants:
                    result = self.evaluate_experiment(source, target, window)
                    if result is not None:
                        results.append(result)

        # Create DataFrame
        if results:
            df = pd.DataFrame(results)
            logger.info(f"\nCompleted {len(results)} experiments")
            return df
        else:
            logger.warning("No experiment results generated")
            return pd.DataFrame()

    def save_results(self, results_df: pd.DataFrame, output_file: str = None):
        """Save results to CSV file"""
        output_file = output_file or config.EXPERIMENT_RESULTS_FILE

        if results_df.empty:
            logger.warning("No results to save")
            return

        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        logger.info(f"\nResults Summary:\n{results_df.to_string()}")


def main():
    parser = argparse.ArgumentParser(description='Run RAG Experiments')
    parser.add_argument(
        '--split_strategy',
        type=str,
        default='recent',
        choices=['recent', 'modn'],
        help='Test/train split strategy (must match ETL)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=list(config.EMBEDDING_MODELS.keys()),
        help='Embedding model (must match ETL). Options: ' +
             ', '.join(config.EMBEDDING_MODELS.keys())
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=list(config.SOURCE_VARIANTS.keys()),
        default=None,
        help='Source variants to evaluate (default: all)'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=list(config.TARGET_VARIANTS.keys()),
        default=None,
        help='Target variants to evaluate (default: all)'
    )
    parser.add_argument(
        '--windows',
        nargs='+',
        choices=list(config.WINDOW_VARIANTS.keys()),
        default=None,
        help='Window variants to evaluate (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file (default: experiment_results.csv)'
    )

    args = parser.parse_args()

    runner = ExperimentRunner(split_strategy=args.split_strategy, model_key=args.model)

    results_df = runner.run_all_experiments(
        source_variants=args.sources,
        target_variants=args.targets,
        window_variants=args.windows
    )

    runner.save_results(results_df, args.output)


if __name__ == '__main__':
    main()
