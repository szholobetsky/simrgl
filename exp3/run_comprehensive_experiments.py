"""
Comprehensive RAG Experiment Runner
Runs experiments for multiple models, both split strategies, with resume capability
"""

import argparse
import sys
import os
import pandas as pd
from typing import List, Tuple
from datetime import datetime
import traceback

import config
from checkpoint_manager import CheckpointManager
from gpu_utils import clear_gpu_memory, log_gpu_memory, cleanup_model
from etl_pipeline import ETLPipeline
from run_experiments import ExperimentRunner
from utils import logger


class ComprehensiveExperimentRunner:
    """Runs comprehensive experiments with resume capability"""

    def __init__(
        self,
        models: List[str],
        strategies: List[str] = None,
        sources: List[str] = None,
        targets: List[str] = None,
        windows: List[str] = None,
        resume: bool = True,
        backend: str = None
    ):
        """
        Initialize comprehensive experiment runner

        Args:
            models: List of model keys (e.g., ['bge-small', 'bge-large'])
            strategies: List of split strategies (default: ['recent', 'modn'])
            sources: List of source variants (default: all)
            targets: List of target variants (default: all)
            windows: List of window variants (default: all)
            resume: Whether to resume from checkpoint
            backend: Vector backend type (default: from config)
        """
        self.models = models
        self.strategies = strategies or ['recent', 'modn']
        self.sources = sources or list(config.SOURCE_VARIANTS.keys())
        self.targets = targets or list(config.TARGET_VARIANTS.keys())
        self.windows = windows or list(config.WINDOW_VARIANTS.keys())
        self.backend = backend or config.VECTOR_BACKEND

        self.checkpoint = CheckpointManager()
        self.results_dir = "experiment_results"
        self.all_results = []

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Handle resume
        if resume and self.checkpoint.should_resume():
            print(self.checkpoint.get_summary())
            print()
            response = input("Resume from checkpoint? (y/n): ").strip().lower()
            if response != 'y':
                print("Starting fresh...")
                self.checkpoint.clear_checkpoint()
            else:
                # Load existing results from CSV when resuming
                self._load_existing_results()
        else:
            if not resume:
                self.checkpoint.clear_checkpoint()

    def _load_existing_results(self):
        """Load existing results from CSV when resuming"""
        output_file = f"{self.results_dir}/comprehensive_results.csv"

        if os.path.exists(output_file):
            try:
                df = pd.read_csv(output_file)
                self.all_results = df.to_dict('records')
                logger.info(f"Loaded {len(self.all_results)} existing results from {output_file}")
                print(f"Loaded {len(self.all_results)} existing results from previous run")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}. Starting with empty results.")
                self.all_results = []
        else:
            logger.info("No existing results file found - starting fresh")
            self.all_results = []

    def get_total_variants(self) -> int:
        """Calculate total number of experiment variants"""
        return (
            len(self.models) *
            len(self.strategies) *
            len(self.sources) *
            len(self.targets) *
            len(self.windows)
        )

    def run_etl_variant(
        self,
        model_key: str,
        strategy: str,
        source: str,
        target: str,
        window: str
    ) -> bool:
        """
        Run ETL for a specific variant

        Returns:
            True if successful, False if failed
        """
        # Check if already completed
        if self.checkpoint.is_etl_completed(model_key, strategy, source, target, window):
            logger.info(f"[SKIP] ETL already completed: {model_key}_{strategy}_{source}_{target}_{window}")
            return True

        try:
            logger.info(f"[ETL] Starting: {model_key}_{strategy}_{source}_{target}_{window}")

            # Create ETL pipeline
            pipeline = ETLPipeline(
                split_strategy=strategy,
                test_size=config.TEST_SIZE,
                model_key=model_key,
                backend_type=self.backend
            )

            # Load data
            tasks_df, rawdata_df = pipeline.load_data()

            # Create split
            train_tasks, test_tasks = pipeline.create_split(tasks_df)

            # Save test set (only once per strategy)
            test_set_file = f"{self.results_dir}/test_set_{strategy}_{model_key}.json"
            pipeline.save_test_set(test_tasks, rawdata_df, test_set_file)

            # Generate embeddings
            train_tasks_with_vectors = pipeline.generate_embeddings(train_tasks, source)

            # Merge with file data (only for train set)
            train_task_ids = set(train_tasks['ID'].values)
            train_rawdata = rawdata_df[rawdata_df['TASK_NAME'].isin(train_tasks['NAME'])].copy()
            merged_df = train_rawdata.merge(
                train_tasks_with_vectors[['NAME', 'vector']],
                left_on='TASK_NAME',
                right_on='NAME',
                how='inner'
            )

            # Filter by window
            if window != 'all':
                window_size = config.WINDOW_VARIANTS[window]['size']
                # Get last N tasks before test set
                train_task_ids_list = sorted(train_task_ids)[-window_size:]
                train_task_names = train_tasks[train_tasks['ID'].isin(train_task_ids_list)]['NAME'].values
                merged_df = merged_df[merged_df['TASK_NAME'].isin(train_task_names)]

            # Aggregate by target
            target_vectors = pipeline.aggregate_by_target(merged_df, target)

            # Create collection name
            model_suffix = model_key.replace('-', '_')
            collection_name = f"{config.COLLECTION_PREFIX}_{source}_{target}_{window}_{strategy}_{model_suffix}"

            # Create collection
            pipeline.create_collection(collection_name, recreate=True)

            # Upsert vectors
            pipeline.upsert_vectors(collection_name, target_vectors, target)

            # Cleanup
            cleanup_model(pipeline.model)
            pipeline.backend.close()

            # Mark as completed
            self.checkpoint.mark_etl_completed(model_key, strategy, source, target, window)

            logger.info(f"[ETL] ✓ Completed: {model_key}_{strategy}_{source}_{target}_{window}")
            return True

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[ETL] ✗ Failed: {model_key}_{strategy}_{source}_{target}_{window}")
            logger.error(f"Error: {error_msg}")
            logger.error(traceback.format_exc())

            self.checkpoint.mark_etl_failed(model_key, strategy, source, target, window, error_msg)

            # Cleanup on failure
            try:
                clear_gpu_memory()
            except:
                pass

            return False

    def run_experiment_variant(
        self,
        model_key: str,
        strategy: str,
        source: str,
        target: str,
        window: str
    ) -> bool:
        """
        Run experiment evaluation for a specific variant

        Returns:
            True if successful, False if failed
        """
        # Check if already completed
        if self.checkpoint.is_experiment_completed(model_key, strategy, source, target, window):
            logger.info(f"[SKIP] Experiment already completed: {model_key}_{strategy}_{source}_{target}_{window}")
            return True

        try:
            logger.info(f"[EVAL] Starting: {model_key}_{strategy}_{source}_{target}_{window}")

            # Create experiment runner
            runner = ExperimentRunner(
                split_strategy=strategy,
                model_key=model_key
            )

            # Load test set
            test_set_file = f"{self.results_dir}/test_set_{strategy}_{model_key}.json"
            if not os.path.exists(test_set_file):
                logger.error(f"Test set not found: {test_set_file}")
                return False

            runner.load_test_set(test_set_file)

            # Initialize model and backend
            runner.initialize_model()
            runner.initialize_backend()

            # Build collection name
            model_suffix = model_key.replace('-', '_')
            collection_name = f"{config.COLLECTION_PREFIX}_{source}_{target}_{window}_{strategy}_{model_suffix}"

            # Check if collection exists
            if not runner.check_collection_exists(collection_name):
                logger.error(f"Collection not found: {collection_name}")
                return False

            # Run evaluation
            results = runner.run_single_experiment(
                collection_name=collection_name,
                source_variant=source,
                target_variant=target,
                window_variant=window,
                experiment_id=f"{source}_{target}_{window}_{strategy}"
            )

            # Add model and strategy to results
            results['model'] = model_key
            results['split_strategy'] = strategy

            # Store results
            self.all_results.append(results)

            # Cleanup
            cleanup_model(runner.model)
            runner.backend.close()

            # Mark as completed
            self.checkpoint.mark_experiment_completed(model_key, strategy, source, target, window)

            # Save results incrementally after each experiment
            self.save_results()

            logger.info(f"[EVAL] ✓ Completed: {model_key}_{strategy}_{source}_{target}_{window}")
            return True

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[EVAL] ✗ Failed: {model_key}_{strategy}_{source}_{target}_{window}")
            logger.error(f"Error: {error_msg}")
            logger.error(traceback.format_exc())

            self.checkpoint.mark_experiment_failed(model_key, strategy, source, target, window, error_msg)

            # Cleanup on failure
            try:
                clear_gpu_memory()
            except:
                pass

            return False

    def _cleanup_collection(self, model_key: str, strategy: str, source: str, target: str, window: str):
        """
        Delete collection to free memory after experiment completes

        Args:
            model_key: Model identifier
            strategy: Split strategy
            source: Source variant
            target: Target variant
            window: Window variant
        """
        from vector_backends import get_vector_backend

        # Build collection name (same as in run_experiment_variant)
        model_suffix = model_key.replace('-', '_')
        collection_name = f"{config.COLLECTION_PREFIX}_{source}_{target}_{window}_{strategy}_{model_suffix}"

        try:
            # Get backend instance
            backend = get_vector_backend(self.backend)
            backend.connect()

            # Delete collection
            success = backend.delete_collection(collection_name)

            if success:
                logger.info(f"✓ Cleaned up collection: {collection_name}")
            else:
                logger.warning(f"⚠ Failed to cleanup collection: {collection_name}")

            # Close backend connection
            backend.close()

        except Exception as e:
            logger.warning(f"⚠ Error during collection cleanup for {collection_name}: {e}")

    def run_all(self):
        """Run all experiments with memory-efficient single-collection approach"""
        total_variants = self.get_total_variants()

        print("=" * 80)
        print("COMPREHENSIVE RAG EXPERIMENT (Memory-Efficient Mode)")
        print("=" * 80)
        print(f"Models: {', '.join(self.models)}")
        print(f"Strategies: {', '.join(self.strategies)}")
        print(f"Sources: {', '.join(self.sources)}")
        print(f"Targets: {', '.join(self.targets)}")
        print(f"Windows: {', '.join(self.windows)}")
        print(f"Backend: {self.backend}")
        print(f"Total variants: {total_variants}")
        print("=" * 80)
        print(f"Note: Running ETL → Experiment → Cleanup for each variant")
        print(f"      (Only 1 collection in memory at a time)")
        print("=" * 80)
        print()

        completed_count = 0
        failed_count = 0
        start_time = datetime.now()

        # Process each variant: ETL → Experiment → Cleanup
        for model_key in self.models:
            for strategy in self.strategies:
                self.checkpoint.set_current_progress(model_key, strategy)

                for source in self.sources:
                    for target in self.targets:
                        for window in self.windows:
                            variant_id = f"{model_key}_{strategy}_{source}_{target}_{window}"

                            print("\n" + "-" * 80)
                            print(f"Processing variant {completed_count + failed_count + 1}/{total_variants}: {variant_id}")
                            print("-" * 80)

                            log_gpu_memory()

                            # Step 1: ETL (create and populate collection)
                            print(f"[1/3] Running ETL...")
                            etl_success = self.run_etl_variant(
                                model_key, strategy, source, target, window
                            )

                            if not etl_success:
                                failed_count += 1
                                print(f"✗ ETL failed for {variant_id}, skipping experiment")
                                print(f"Progress: {completed_count}/{total_variants} completed, {failed_count} failed")
                                continue

                            # Step 2: Run experiment
                            print(f"[2/3] Running experiment...")
                            exp_success = self.run_experiment_variant(
                                model_key, strategy, source, target, window
                            )

                            if exp_success:
                                completed_count += 1
                                print(f"✓ Experiment completed: {variant_id}")
                            else:
                                failed_count += 1
                                print(f"✗ Experiment failed for {variant_id}")

                            # Step 3: Cleanup collection to free memory
                            print(f"[3/3] Cleaning up collection...")
                            self._cleanup_collection(model_key, strategy, source, target, window)

                            print(f"Progress: {completed_count}/{total_variants} completed, {failed_count} failed")
                            log_gpu_memory()

        # Save combined results
        self.save_results()

        # Print final summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"Total time: {duration}")
        print(f"ETL completed: {completed_count}/{total_variants}")
        print(f"ETL failed: {failed_count}")
        print(f"Experiments completed: {experiment_count}/{total_variants}")
        print(f"Experiments failed: {experiment_failed}")
        print(f"Results saved to: {self.results_dir}/")
        print("=" * 80)

    def save_results(self):
        """Save all results to CSV"""
        if not self.all_results:
            logger.warning("No results to save")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)

        # Reorder columns
        column_order = [
            'model', 'split_strategy', 'experiment_id',
            'source', 'target', 'window',
            'MAP', 'MRR',
            'P@1', 'R@1', 'P@3', 'R@3', 'P@5', 'R@5', 'P@10', 'R@10'
        ]

        # Only include columns that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]

        # Save to CSV
        output_file = f"{self.results_dir}/comprehensive_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        # Also save per-model results
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            model_file = f"{self.results_dir}/results_{model}.csv"
            model_df.to_csv(model_file, index=False)
            logger.info(f"Model results saved to {model_file}")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive RAG experiments')

    parser.add_argument(
        '--models',
        nargs='+',
        default=['bge-small'],
        help='Model keys to test (e.g., bge-small bge-large gte-large)'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['recent', 'modn'],
        default=['recent', 'modn'],
        help='Split strategies to test'
    )
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=list(config.SOURCE_VARIANTS.keys()),
        default=list(config.SOURCE_VARIANTS.keys()),
        help='Source variants to test'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=list(config.TARGET_VARIANTS.keys()),
        default=list(config.TARGET_VARIANTS.keys()),
        help='Target variants to test'
    )
    parser.add_argument(
        '--windows',
        nargs='+',
        choices=list(config.WINDOW_VARIANTS.keys()),
        default=list(config.WINDOW_VARIANTS.keys()),
        help='Window variants to test'
    )
    parser.add_argument(
        '--backend',
        choices=['qdrant', 'postgres'],
        default=config.VECTOR_BACKEND,
        help='Vector backend to use'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh (ignore checkpoint)'
    )

    args = parser.parse_args()

    # Create and run experiment
    runner = ComprehensiveExperimentRunner(
        models=args.models,
        strategies=args.strategies,
        sources=args.sources,
        targets=args.targets,
        windows=args.windows,
        resume=not args.no_resume,
        backend=args.backend
    )

    runner.run_all()


if __name__ == '__main__':
    main()
