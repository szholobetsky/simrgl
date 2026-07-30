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
        train_sources: List[str] = None,
        query_sources: List[str] = None,
        targets: List[str] = None,
        windows: List[str] = None,
        resume: bool = True,
        backend: str = None,
        project: str = None,
        task_unit: str = None,
        auto_resume: bool = False,
        test_size: int = None
    ):
        """
        Initialize comprehensive experiment runner (exp3.2 cross-vocabulary
        grid: index built from commit messages, queried with ticket text)

        Args:
            models: List of model keys (e.g., ['bge-small', 'bge-large'])
            strategies: List of split strategies (default: ['recent', 'modn'])
            train_sources: Index-side source variants, commit vocabulary
                (config.TRAIN_SOURCE_VARIANTS, default: all)
            query_sources: Query-side source variants, ticket vocabulary
                (config.QUERY_SOURCE_VARIANTS, default: all)
            targets: List of target variants (default: all)
            windows: List of window variants (default: all)
            resume: Whether to resume from checkpoint
            backend: Vector backend type (default: from config)
            project: Project name (default: config.PROJECT) - scopes results_dir/checkpoint
            task_unit: expected to be 'cross' for exp3.2 (default: config.TASK_UNIT) -
                only used to scope results_dir/checkpoint path and collection
                naming; run_all() always builds the index from commit-mode
                data and queries from ticket-mode data regardless of this value
            auto_resume: Skip the interactive y/n prompt and resume automatically
                when a checkpoint exists. Required for unattended/background runs -
                `input()` would hang (or raise EOFError) with no attached TTY.
            test_size: Held-out test task count (default: config.TEST_SIZE=200).
                Override for tiny projects (e.g. agilebill: 120 commits total -
                200 would leave nothing to train on).
        """
        self.models = models
        self.strategies = strategies or ['recent', 'modn']
        self.train_sources = train_sources or list(config.TRAIN_SOURCE_VARIANTS.keys())
        self.query_sources = query_sources or list(config.QUERY_SOURCE_VARIANTS.keys())
        self.targets = targets or list(config.TARGET_VARIANTS.keys())
        self.windows = windows or list(config.WINDOW_VARIANTS.keys())
        self.backend = backend or config.VECTOR_BACKEND
        self.test_size = test_size or config.TEST_SIZE
        self.project = project or config.PROJECT
        self.task_unit = task_unit or config.TASK_UNIT

        # Per (project, task_unit) isolation: separate checkpoint/results dir
        # so one project's crash/rerun can't touch another's state, and so
        # resuming after an interruption just means re-invoking with the
        # same --project/--task-unit.
        self.results_dir = f"experiment_results/{self.project}/{self.task_unit}"
        self.checkpoint = CheckpointManager(f"{self.results_dir}/checkpoint.json")
        self.all_results = []

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Handle resume
        if resume and self.checkpoint.should_resume():
            print(self.checkpoint.get_summary())
            print()
            if auto_resume:
                print("--yes given: resuming automatically.")
                response = 'y'
            else:
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
        """Calculate total number of eval variants (ETL builds x query fan-out)"""
        return (
            len(self.models) *
            len(self.strategies) *
            len(self.train_sources) *
            len(self.query_sources) *
            len(self.targets) *
            len(self.windows)
        )

    def _run_etl_for_target(
        self,
        model_key: str,
        strategy: str,
        train_source: str,
        target: str,
        window: str,
        pipeline: ETLPipeline,
        train_merged
    ) -> bool:
        """
        Aggregate/upsert a single (train_source,target,window) variant using
        embeddings already computed by the caller (see
        `_process_source_window`) - embeddings only depend on `train_source`,
        not `target`/`window`/`query_source`, so computing them once per
        (train_source,window) and sharing them across targets AND across
        every query_source avoids redundant recompute.

        Returns:
            True if successful (or already completed), False if failed
        """
        if self.checkpoint.is_etl_completed(model_key, strategy, train_source, target, window):
            logger.info(f"[SKIP] ETL already completed: {model_key}_{strategy}_{train_source}_{target}_{window}")
            return True

        try:
            logger.info(f"[ETL] Starting: {model_key}_{strategy}_{train_source}_{target}_{window}")

            target_vectors = pipeline.aggregate_by_target(train_merged, target)

            if not target_vectors:
                logger.warning(f"No vectors for {model_key}_{strategy}_{train_source}_{target}_{window}, skipping")
                return False

            collection_name = config.collection_name(train_source, target, window, strategy, model_key=model_key)
            pipeline.create_collection(collection_name, recreate=True)
            pipeline.upsert_vectors(collection_name, target_vectors, target)

            self.checkpoint.mark_etl_completed(model_key, strategy, train_source, target, window)

            logger.info(f"[ETL] ✓ Completed: {model_key}_{strategy}_{train_source}_{target}_{window}")
            return True

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[ETL] ✗ Failed: {model_key}_{strategy}_{train_source}_{target}_{window}")
            logger.error(f"Error: {error_msg}")
            logger.error(traceback.format_exc())

            self.checkpoint.mark_etl_failed(model_key, strategy, train_source, target, window, error_msg)

            try:
                clear_gpu_memory()
            except Exception:
                pass

            return False

    def _process_source_window(
        self,
        model_key: str,
        strategy: str,
        train_source: str,
        window: str,
        pipeline: ETLPipeline,
        train_tasks_windowed,
        merged_df_base,
        counters: dict
    ):
        """
        Handle every target for one (train_source, window) pair, computing
        embeddings at most once (skipped entirely if every target already
        has its ETL step checkpointed). For each target, once the index is
        built, fan out over every `query_source` before cleaning up the
        collection - cleanup happens once per target, after all
        `query_source` evals finish, not after each one, so the same
        collection isn't rebuilt 3x just to answer 3 different queries
        against it.
        """
        needs_embedding = any(
            not self.checkpoint.is_etl_completed(model_key, strategy, train_source, t, window)
            for t in self.targets
        )

        train_merged = None
        if needs_embedding:
            try:
                train_tasks_embedded = pipeline.generate_embeddings(train_tasks_windowed, train_source)
                task_vector_map = dict(zip(train_tasks_embedded['NAME'], train_tasks_embedded['vector']))
                train_task_names = set(train_tasks_windowed['NAME'])
                train_merged = merged_df_base[merged_df_base['NAME'].isin(train_task_names)].copy()
                train_merged['vector'] = train_merged['NAME'].map(task_vector_map)
                train_merged = train_merged.dropna(subset=['vector'])
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"[ETL] ✗ Embedding failed for {model_key}_{strategy}_{train_source}_*_{window}: {error_msg}")
                logger.error(traceback.format_exc())
                for target in self.targets:
                    if not self.checkpoint.is_etl_completed(model_key, strategy, train_source, target, window):
                        self.checkpoint.mark_etl_failed(model_key, strategy, train_source, target, window, error_msg)
                        counters['failed'] += len(self.query_sources)
                try:
                    clear_gpu_memory()
                except Exception:
                    pass
                return

        for target in self.targets:
            variant_id = f"{model_key}_{strategy}_{train_source}_{target}_{window}"
            print("\n" + "-" * 80)
            print(f"Processing variant: {variant_id}")
            print("-" * 80)

            etl_success = self._run_etl_for_target(
                model_key, strategy, train_source, target, window, pipeline, train_merged
            )
            if not etl_success:
                counters['failed'] += len(self.query_sources)
                print(f"✗ ETL failed for {variant_id}, skipping {len(self.query_sources)} query evals")
                continue

            for query_source in self.query_sources:
                eval_id = f"{variant_id}_q{query_source}"
                exp_success = self.run_experiment_variant(
                    model_key, strategy, train_source, query_source, target, window
                )
                if exp_success:
                    counters['completed'] += 1
                    print(f"✓ Experiment completed: {eval_id}")
                else:
                    counters['failed'] += 1
                    print(f"✗ Experiment failed for {eval_id}")

            self._cleanup_collection(model_key, strategy, train_source, target, window)

        if needs_embedding and pipeline.model is not None:
            cleanup_model(pipeline.model)
            pipeline.model = None

    def run_experiment_variant(
        self,
        model_key: str,
        strategy: str,
        train_source: str,
        query_source: str,
        target: str,
        window: str
    ) -> bool:
        """
        Run experiment evaluation for a specific (train_source, query_source)
        variant - index built from train_source (commit vocabulary), queried
        with query_source (ticket vocabulary).

        Returns:
            True if successful, False if failed
        """
        # Composite checkpoint key - CheckpointManager's `source` slot is
        # opaque (just interpolated into a string id), so this needs no
        # CheckpointManager changes. ETL completion (train_source alone,
        # see _run_etl_for_target) is intentionally a separate, coarser key
        # than this one.
        variant_key = f"{train_source}_q{query_source}"

        # Check if already completed
        if self.checkpoint.is_experiment_completed(model_key, strategy, variant_key, target, window):
            logger.info(f"[SKIP] Experiment already completed: {model_key}_{strategy}_{variant_key}_{target}_{window}")
            return True

        try:
            logger.info(f"[EVAL] Starting: {model_key}_{strategy}_{variant_key}_{target}_{window}")

            # Create experiment runner
            runner = ExperimentRunner(
                split_strategy=strategy,
                model_key=model_key
            )

            # Load test set - always the ticket-mode one (query side is
            # always Jira ticket text, regardless of train_source/target/
            # window - see run_all()).
            test_set_file = f"{self.results_dir}/test_set_ticket_{strategy}_{model_key}.json"
            if not os.path.exists(test_set_file):
                logger.error(f"Test set not found: {test_set_file}")
                return False

            runner.load_test_set(test_set_file)

            # Initialize model and backend
            runner.initialize_model()
            runner.initialize_backend()

            # Build collection name - depends only on train_source (index side)
            collection_name = config.collection_name(train_source, target, window, strategy, model_key=model_key)

            # Check if collection exists
            if not runner.check_collection_exists(collection_name):
                logger.error(f"Collection not found: {collection_name}")
                return False

            # Run evaluation
            results = runner.run_single_experiment(
                collection_name=collection_name,
                train_source=train_source,
                query_source=query_source,
                target_variant=target,
                window_variant=window,
                experiment_id=f"{train_source}_q{query_source}_{target}_{window}_{strategy}"
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
            self.checkpoint.mark_experiment_completed(model_key, strategy, variant_key, target, window)

            # Save results incrementally after each experiment
            self.save_results()

            logger.info(f"[EVAL] ✓ Completed: {model_key}_{strategy}_{variant_key}_{target}_{window}")
            return True

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[EVAL] ✗ Failed: {model_key}_{strategy}_{variant_key}_{target}_{window}")
            logger.error(f"Error: {error_msg}")
            logger.error(traceback.format_exc())

            self.checkpoint.mark_experiment_failed(model_key, strategy, variant_key, target, window, error_msg)

            # Cleanup on failure
            try:
                clear_gpu_memory()
            except:
                pass

            return False

    def _cleanup_collection(self, model_key: str, strategy: str, train_source: str, target: str, window: str):
        """
        Delete collection to free memory - called once per (train_source,
        target, window) after ALL query_source evals against it have
        finished (see _process_source_window), not after each individual
        eval.

        Args:
            model_key: Model identifier
            strategy: Split strategy
            train_source: Index-side source variant
            target: Target variant
            window: Window variant
        """
        from vector_backends import get_vector_backend

        # Build collection name (same as in run_experiment_variant)
        collection_name = config.collection_name(train_source, target, window, strategy, model_key=model_key)

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
        print("COMPREHENSIVE RAG EXPERIMENT - CROSS-VOCABULARY MODE (exp3.2)")
        print("=" * 80)
        print(f"Models: {', '.join(self.models)}")
        print(f"Strategies: {', '.join(self.strategies)}")
        print(f"Train sources (index, commit vocabulary): {', '.join(self.train_sources)}")
        print(f"Query sources (query, ticket vocabulary): {', '.join(self.query_sources)}")
        print(f"Targets: {', '.join(self.targets)}")
        print(f"Windows: {', '.join(self.windows)}")
        print(f"Backend: {self.backend}")
        print(f"Total eval variants: {total_variants}")
        print("=" * 80)
        print(f"Note: index built from COMMIT messages, queried with TICKET text.")
        print(f"      Collection built once per (train_source,target,window),")
        print(f"      reused across all {len(self.query_sources)} query_sources.")
        print("=" * 80)
        print()

        counters = {'completed': 0, 'failed': 0}
        start_time = datetime.now()

        # Loop order is window -> train_source -> target (not the more
        # obvious train_source -> target -> window): embeddings depend only
        # on `train_source`, and windowing narrows the train set *before*
        # embedding, so this order lets `_process_source_window` compute
        # embeddings once per (train_source,window) and share them across
        # both `target`s AND every `query_source`, instead of recomputing
        # per (train_source,target,window,query_source) combination.
        for model_key in self.models:
            for strategy in self.strategies:
                self.checkpoint.set_current_progress(model_key, strategy)

                # Index side: synthetic per-commit tasks, same construction
                # as exp3.1's commit mode - this is what
                # generate_embeddings()/aggregate_by_target() build the
                # collections from, independent of self.task_unit (which
                # only scopes the results_dir/checkpoint/collection path,
                # expected to be 'cross').
                index_pipeline = ETLPipeline(
                    split_strategy=strategy,
                    test_size=self.test_size,
                    model_key=model_key,
                    backend_type=self.backend,
                    task_unit='commit'
                )
                tasks_df, rawdata_df = index_pipeline.load_data()
                train_tasks_all, index_test_tasks = index_pipeline.create_split(tasks_df)

                merged_df_base = pd.merge(
                    rawdata_df,
                    tasks_df[['NAME', 'TITLE', 'DESCRIPTION', 'COMMENTS']],
                    left_on='TASK_NAME', right_on='NAME', how='inner'
                )

                # Query side: real Jira tickets. Regenerated locally
                # (deterministic split -> byte-identical to exp3.1's own
                # ticket-mode test set for the same project/strategy/
                # test_size, see exp3.2/README.md "Dependency on exp3.1")
                # rather than read across the exp3.1/exp3.2 folder boundary.
                # No embedding happens on this pipeline - cheap (DB read +
                # pandas split only).
                query_pipeline = ETLPipeline(
                    split_strategy=strategy,
                    test_size=self.test_size,
                    model_key=model_key,
                    backend_type=self.backend,
                    task_unit='ticket'
                )
                query_tasks_df, query_rawdata_df = query_pipeline.load_data()
                _query_train_tasks, query_test_tasks = query_pipeline.create_split(query_tasks_df)

                test_set_file = f"{self.results_dir}/test_set_ticket_{strategy}_{model_key}.json"
                query_pipeline.save_test_set(query_test_tasks, query_rawdata_df, test_set_file)

                for window in self.windows:
                    window_size = config.WINDOW_VARIANTS[window]['size']
                    # Windowing is a property of the INDEX side (how much
                    # commit history feeds the centroids) - the query side
                    # (200 held-out tickets) has no window concept.
                    train_tasks_windowed = index_pipeline.apply_time_window(
                        train_tasks_all, index_test_tasks, window_size
                    )

                    log_gpu_memory()

                    for train_source in self.train_sources:
                        self._process_source_window(
                            model_key, strategy, train_source, window,
                            index_pipeline, train_tasks_windowed, merged_df_base, counters
                        )

                    print(f"Progress: {counters['completed']}/{total_variants} completed, {counters['failed']} failed")
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
        print(f"Completed: {counters['completed']}/{total_variants}")
        print(f"Failed: {counters['failed']}")
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
            'train_source', 'query_source', 'target', 'window',
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
        '--train-sources',
        nargs='+',
        choices=list(config.TRAIN_SOURCE_VARIANTS.keys()),
        default=list(config.TRAIN_SOURCE_VARIANTS.keys()),
        help='Index-side (commit-message) source variants to test'
    )
    parser.add_argument(
        '--query-sources',
        nargs='+',
        choices=list(config.QUERY_SOURCE_VARIANTS.keys()),
        default=list(config.QUERY_SOURCE_VARIANTS.keys()),
        help='Query-side (ticket) source variants to test'
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
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help=f'Project name, maps to {config.PROJECTS_DIR}/<project>.db (default: {config.PROJECT})'
    )
    parser.add_argument(
        '--task-unit',
        type=str,
        default=None,
        choices=config.TASK_UNITS,
        help=f'Task unit criterion (default: {config.TASK_UNIT})'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Auto-resume without the interactive y/n prompt (required for unattended/background runs)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help=f'Held-out test task count (default: {config.TEST_SIZE}). '
             f'Override for tiny projects (e.g. agilebill: 20).'
    )

    args = parser.parse_args()

    if args.project:
        config.PROJECT = args.project
        config.DB_PATH = os.path.join(config.PROJECTS_DIR, f'{args.project}.db')
    if args.task_unit:
        config.TASK_UNIT = args.task_unit

    # Create and run experiment
    runner = ComprehensiveExperimentRunner(
        models=args.models,
        strategies=args.strategies,
        train_sources=args.train_sources,
        query_sources=args.query_sources,
        targets=args.targets,
        windows=args.windows,
        resume=not args.no_resume,
        backend=args.backend,
        project=config.PROJECT,
        task_unit=config.TASK_UNIT,
        auto_resume=args.yes,
        test_size=args.test_size
    )

    runner.run_all()


if __name__ == '__main__':
    main()
