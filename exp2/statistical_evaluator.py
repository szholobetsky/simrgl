#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Evaluator for Semantic Similarity
Evaluates semantic similarity performance using MAP, MRR, and Recall@K metrics
on test tasks that were excluded from model training (proper train/test split)
Updated to use the new modular vectorizer system
"""

import sqlite3
import json
import numpy as np
import random
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import statistics
import properties as props
from task_selector import TaskSelector
from sci_log import sci_log

log_enable = True

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import euclidean, cityblock
    from tqdm import tqdm
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Missing required libraries. Please install: pip install scikit-learn scipy tqdm")
    print(f"Import error: {e}")
    SCIPY_AVAILABLE = False
    # Try fallback without scipy
    try:
        from tqdm import tqdm
        print("[INFO] Continuing with limited functionality (no scipy)")
    except ImportError:
        print("[ERROR] tqdm also not available. Please install: pip install tqdm")
        exit(1)

class StatisticalEvaluator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.filtered_terms = set()
        self._setup_database_optimizations()
        
        # Initialize vectorizer-related attributes
        self._pipeline_checked = False
        self._vectorizer = None
        self._module_vectors_cache = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
    
    def _setup_database_optimizations(self):
        """Setup database optimizations for better performance"""
        try:
            # Create index on RAWDATA.TASK_NAME if it doesn't exist
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rawdata_task_name ON RAWDATA(TASK_NAME)")
            # Create index on TASK.NAME if it doesn't exist
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_task_name ON TASK(NAME)")
            self.conn.commit()
        except sqlite3.Error as e:
            self.log(f"Warning: Could not create indexes: {e}")
    
    def log(self, message: str):
        """Print log message if verbose mode is enabled"""
        if props.VERBOSE:
            print(f"[INFO] {message}")
    
    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector to unit length (magnitude = 1)
        
        Args:
            vector: Input vector to normalize
            
        Returns:
            Unit vector with same direction but magnitude = 1
        """
        # Calculate the L2 norm (Euclidean length)
        norm = np.linalg.norm(vector)
        
        # Handle zero vectors to avoid division by zero
        if norm == 0:
            return vector  # Return zero vector as is
        
        # Return normalized vector
        return vector / norm
    
    def load_filtered_terms(self):
        """Load terms that pass HHI and count filters for preprocessing"""
        if not props.PREPROCESS_TEST_TASK:
            return
            
        self.log(f"Loading filtered terms for preprocessing...")
        
        cursor = self.conn.execute("""
        SELECT ct.TERM 
        FROM SIMRGL_TERM_RANK ctr 
        JOIN SIMRGL_TERMS ct ON ctr.TERM_ID = ct.ID
        WHERE ctr.CNT > ? AND ctr.HHI_ROOT > ?
        ORDER BY ctr.HHI_ROOT DESC, ctr.CNT DESC
        """, (props.MIN_TERM_COUNT, props.MIN_HHI_ROOT))
        
        self.filtered_terms = set(term[0] for term in cursor.fetchall())
        self.log(f"Loaded {len(self.filtered_terms)} filtered terms for preprocessing")
    
    def tokenize_and_filter_text(self, text: str) -> List[str]:
        """Tokenize text and optionally filter using HHI terms"""
        if not text:
            return []
        
        tokens = re.findall(r'\b[a-zA-Z]+[a-zA-Z0-9#_]*\b|\b[a-zA-Z0-9#_]*[a-zA-Z]+\b', text.lower())
        
        filtered_tokens = []
        for token in tokens:
            if len(token) >= props.MIN_WORD_LENGTH:
                if not props.IGNORE_PURE_NUMBERS or not token.isdigit():
                    if not props.IGNORE_PURE_SYMBOLS or any(c.isalnum() for c in token):
                        if props.PREPROCESS_TEST_TASK:
                            if token in self.filtered_terms:
                                filtered_tokens.append(token)
                        else:
                            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def _load_vectorizer_model(self):
        """Load the configured vectorizer model (same approach as vectoriser.py)"""
        try:
            from vectorizer_factory import create_vectorizer
            
            print(f"    Loading {props.VECTORISER_MODEL} vectorizer for evaluation...")
            self._vectorizer = create_vectorizer(self.db_path)
            
            # Load filtered terms (needed for tokenization)
            self._vectorizer.load_filtered_terms()
            
            if not self._vectorizer.filtered_terms:
                raise ValueError("No terms passed the filtering criteria")
            
            # Load the model
            self._vectorizer.load_or_train_model()
            
            print(f"    Loaded vectorizer: {self._vectorizer.get_model_info()}")
            
        except ImportError as e:
            print(f"    Error: Vectorizer factory not available: {e}")
            raise ValueError("Vectorizer factory not available. Please ensure all vectorizer files are present.")
        except Exception as e:
            print(f"    Error loading vectorizer model: {e}")
            raise ValueError(f"Could not load vectorizer model: {e}")
    
    def vectorize_task_text(self, text: str) -> np.ndarray:
        """Vectorize task text using the configured embedding model"""
        # Check if we have the pipeline tables
        if not self._pipeline_checked:
            print("    Checking pipeline status...")
            
            # Check if terms exist
            cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_TERMS")
            terms_count = cursor.fetchone()[0]
            
            # Check if vectors exist
            cursor = self.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='SIMRGL_TASK_VECTOR'")
            vectors_table_exists = cursor.fetchone()[0] > 0
            
            if terms_count == 0:
                raise ValueError("No terms found in database. Please run main.py first to extract terms.")
            
            if not vectors_table_exists:
                raise ValueError("No task vectors found in database. Please run vectoriser.py first to create vectors.")
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_TASK_VECTOR")
            vectors_count = cursor.fetchone()[0]
            
            if vectors_count == 0:
                raise ValueError("No task vectors found in database. Please run vectoriser.py first to create vectors.")
            
            print(f"    Pipeline check: {terms_count} terms, {vectors_count} task vectors")
            self._pipeline_checked = True
        
        # Load vectorizer if not already loaded
        if not hasattr(self, '_vectorizer') or self._vectorizer is None:
            self._load_vectorizer_model()
        
        # Vectorize the text using the same approach as the training pipeline
        if hasattr(self._vectorizer, 'vectorize_raw_text'):
            vector = self._vectorizer.vectorize_raw_text(text)
        else:
            tokens = self.tokenize_and_filter_text(text)
            vector = self._vectorizer.vectorize_text(tokens)
        
        # Normalize the vector if enabled (to match the stored vectors)
        if getattr(props, 'NORMALIZE_VECTORS', True):
            normalized_vector = self.normalize_vector(vector)
            return normalized_vector
        
        return vector
    
    def get_task_ground_truth_modules(self, task_id: int) -> Set[int]:
        """Get the ground truth modules (modules that were actually changed) for a task"""
        cursor = self.conn.execute("""
        SELECT DISTINCT m.ID
        FROM SIMRGL_MODULES m
        JOIN SIMRGL_FILES f ON m.ID = f.MODULE_ID
        JOIN RAWDATA r ON f.FILE_PATH = r.PATH
        JOIN TASK t ON r.TASK_NAME = t.NAME
        WHERE t.ID = ?
        """, (task_id,))
        
        return set(row[0] for row in cursor.fetchall())
    
    def calculate_distances(self, test_vector: np.ndarray, target_vector: np.ndarray) -> dict[str, float]:
        """Calculate various distance metrics between two vectors"""
        distances = {}
        
        test_vec_2d = test_vector.reshape(1, -1)
        target_vec_2d = target_vector.reshape(1, -1)
        
        for metric in props.DISTANCE_METRICS:
            try:
                if metric == 'cosine':
                    # For normalized vectors, cosine similarity = dot product (more efficient)
                    if getattr(props, 'NORMALIZE_VECTORS', True):
                        similarity = np.dot(test_vector, target_vector)
                    else:
                        if SCIPY_AVAILABLE:
                            similarity = cosine_similarity(test_vec_2d, target_vec_2d)[0][0]
                        else:
                            # Fallback cosine calculation without sklearn
                            dot_product = np.dot(test_vector, target_vector)
                            norm_a = np.linalg.norm(test_vector)
                            norm_b = np.linalg.norm(target_vector)
                            similarity = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
                    distances[metric] = 1.0 - similarity
                elif metric == 'euclidean':
                    if SCIPY_AVAILABLE:
                        distances[metric] = euclidean(test_vector, target_vector)
                    else:
                        distances[metric] = np.linalg.norm(test_vector - target_vector)
                elif metric == 'manhattan':
                    if SCIPY_AVAILABLE:
                        distances[metric] = cityblock(test_vector, target_vector)
                    else:
                        distances[metric] = np.sum(np.abs(test_vector - target_vector))
            except Exception as e:
                self.log(f"Error calculating {metric} distance: {e}")
                distances[metric] = float('inf')
        
        return distances
    
    def get_ranked_modules(self, test_vector: np.ndarray, sort_metric: str = 'cosine') -> List[Tuple[int, float]]:
        """Get all modules ranked by similarity to test vector"""
        # Initialize module vectors cache for efficiency
        if not hasattr(self, '_module_vectors_cache') or not self._module_vectors_cache:
            print("    Loading module vectors...")
            cursor = self.conn.execute("""
            SELECT mv.MODULE_ID, mv.VECTOR
            FROM SIMRGL_MODULE_VECTOR mv
            """)
            
            self._module_vectors_cache = []
            for module_id, vector_json in cursor.fetchall():
                try:
                    module_vector = np.array(json.loads(vector_json))
                    # Verify the vector is normalized (should already be from vectoriser.py)
                    norm = np.linalg.norm(module_vector)
                    if norm > 0 and abs(norm - 1.0) > 1e-6:
                        # Re-normalize if needed (shouldn't happen with updated vectoriser.py)
                        module_vector = module_vector / norm
                    self._module_vectors_cache.append((module_id, module_vector))
                except Exception as e:
                    self.log(f"Error loading vector for module {module_id}: {e}")
            
            print(f"    Loaded {len(self._module_vectors_cache)} normalized module vectors")
        
        results = []
        
        for module_id, module_vector in self._module_vectors_cache:
            try:
                distances = self.calculate_distances(test_vector, module_vector)
                distance = distances.get(sort_metric, float('inf'))
                results.append((module_id, distance))
            except Exception as e:
                self.log(f"Error processing module {module_id}: {e}")
                continue
        
        # Sort by distance (smaller is better)
        results.sort(key=lambda x: x[1])
        return results
    
    def calculate_average_precision(self, ranked_modules: List[Tuple[int, float]], 
                                  ground_truth: Set[int]) -> float:
        """Calculate Average Precision for a single query"""
        if not ground_truth:
            return 0.0
        
        relevant_found = 0
        precision_sum = 0.0
        
        for i, (module_id, _) in enumerate(ranked_modules):
            if module_id in ground_truth:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        if relevant_found == 0:
            return 0.0
        
        return precision_sum / len(ground_truth)
    
    def calculate_reciprocal_rank(self, ranked_modules: List[Tuple[int, float]], 
                                ground_truth: Set[int]) -> float:
        """Calculate Reciprocal Rank for a single query"""
        if not ground_truth:
            return 0.0
        
        for i, (module_id, _) in enumerate(ranked_modules):
            if module_id in ground_truth:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def calculate_recall_at_k(self, ranked_modules: List[Tuple[int, float]], 
                            ground_truth: Set[int], k: int) -> float:
        """Calculate Recall@K for a single query"""
        if not ground_truth:
            return 0.0
        
        top_k_modules = set(module_id for module_id, _ in ranked_modules[:k])
        relevant_retrieved = len(top_k_modules.intersection(ground_truth))
        
        return relevant_retrieved / len(ground_truth)
    
    def get_test_tasks(self) -> List[Tuple[int, str, str]]:
        """Get test tasks based on train/test split configuration"""
        if not props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            # If no split is enabled, fall back to random sampling
            self.log("Train/test split disabled - using random sampling for evaluation")
            return self.get_random_test_tasks_fallback(props.SUMMARY_TEST_TASK_COUNT)
        
        # Use proper test set
        self.log("Using proper test set from train/test split...")
        
        with TaskSelector(self.db_path) as selector:
            _, test_task_ids = selector.get_train_test_split()
        
        if not test_task_ids:
            raise ValueError("No test tasks available from split")
        
        # Build text source query
        text_fields = ["t.TITLE"]
        if props.USE_DESCRIPTION:
            text_fields.append("t.DESCRIPTION")
        if props.USE_COMMENTS:
            text_fields.append("t.COMMENTS")
        
        text_concat = " || ' ' || ".join([f"COALESCE({field}, '')" for field in text_fields])
        
        # Get test tasks
        test_ids_str = ",".join(map(str, test_task_ids))
        cursor = self.conn.execute(f"""
        SELECT t.ID, t.NAME, {text_concat} as combined_text
        FROM TASK t
        WHERE t.ID IN ({test_ids_str})
        ORDER BY t.ID
        """)
        
        test_tasks = cursor.fetchall()
        
        self.log(f"Retrieved {len(test_tasks)} test tasks from proper test set")
        return test_tasks
    
    def get_random_test_tasks_fallback(self, count: int) -> List[Tuple[int, str, str]]:
        """Fallback method: Simple random sampling without commit data check"""
        self.log(f"Using fallback method: simple random sampling...")
        
        # Build text source query
        text_fields = ["t.TITLE"]
        if props.USE_DESCRIPTION:
            text_fields.append("t.DESCRIPTION")
        if props.USE_COMMENTS:
            text_fields.append("t.COMMENTS")
        
        text_concat = " || ' ' || ".join([f"COALESCE({field}, '')" for field in text_fields])
        
        # Get random tasks using ORDER BY RANDOM() with LIMIT
        print(f"    Selecting {count} random tasks...")
        cursor = self.conn.execute(f"""
        SELECT t.ID, t.NAME, {text_concat} as combined_text
        FROM TASK t
        WHERE t.NAME IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """, (count,))
        
        selected_tasks = cursor.fetchall()
        
        if not selected_tasks:
            raise ValueError("No tasks available")
        
        self.log(f"Selected {len(selected_tasks)} tasks using fallback method")
        return selected_tasks
    
    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a list of values"""
        if not values:
            return {
                'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0,
                'q1': 0.0, 'q3': 0.0, 'std': 0.0
            }
        
        values_sorted = sorted(values)
        n = len(values_sorted)
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'q1': values_sorted[n // 4] if n >= 4 else values_sorted[0],
            'q3': values_sorted[3 * n // 4] if n >= 4 else values_sorted[-1],
            'std': statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    def print_statistics_table(self, metric_name: str, stats: Dict[str, float]):
        """Print statistics in a formatted table"""
        print(f"\n{metric_name} Statistics:")
        print("-" * 50)
        print(f"{'Metric':<12} {'Value':<10}")
        print("-" * 50)
        print(f"{'Min':<12} {stats['min']:<10.4f}")
        print(f"{'Max':<12} {stats['max']:<10.4f}")
        print(f"{'Mean':<12} {stats['mean']:<10.4f}")
        print(f"{'Median':<12} {stats['median']:<10.4f}")
        print(f"{'Q1':<12} {stats['q1']:<10.4f}")
        print(f"{'Q3':<12} {stats['q3']:<10.4f}")
        print(f"{'Std Dev':<12} {stats['std']:<10.4f}")
        if log_enable:
            sci_log.key(f"{metric_name} - min", f"{stats['min']:<10.4f}")
            sci_log.key(f"{metric_name} - q1", f"{stats['q1']:<10.4f}")
            sci_log.key(f"{metric_name} - mean", f"{stats['mean']:<10.4f}")
            sci_log.key(f"{metric_name} - median", f"{stats['median']:<10.4f}")
            sci_log.key(f"{metric_name} - q3", f"{stats['q3']:<10.4f}")
            sci_log.key(f"{metric_name} - max", f"{stats['max']:<10.4f}")
        
    def run_evaluation(self, test_count: int = None, sort_metric: str = 'cosine', 
                      recall_k_values: List[int] = None):
        """Run statistical evaluation with proper train/test split"""
        if test_count is None:
            test_count = props.SUMMARY_TEST_TASK_COUNT
        
        if recall_k_values is None:
            recall_k_values = [5, 10, 20, 50]
        
        print(f"\n{'='*80}")
        print(f"STARTING STATISTICAL EVALUATION")
        print(f"Using current configuration from properties.py")
        print(f"{'='*80}")
        print(f"Vectorizer model: {props.VECTORISER_MODEL}")
        print(f"Train/test split enabled: {props.EXCLUDE_TEST_TASKS_FROM_MODEL}")
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print(f"Test task selection strategy: {props.EXCLUDE_TEST_TASKS_STRATEGY}")
            print(f"Target test tasks: {props.SUMMARY_TEST_TASK_COUNT}")
        else:
            print(f"Fallback: Random sampling of {test_count} tasks")
        print(f"Distance metric: {sort_metric}")
        print(f"Recall@K values: {recall_k_values}")
        print(f"Preprocessing enabled: {props.PREPROCESS_TEST_TASK}")
        print(f"Vector normalization: {props.NORMALIZE_VECTORS}")
        print(f"Module vector strategy: {props.MODULE_VECTOR_STRATEGY}")
        print()
        
        # Step 1: Load preprocessing terms if needed
        print("[STEP 1/6] Loading preprocessing configuration...")
        self.load_filtered_terms()
        print("> Preprocessing configuration loaded")
        
        # Step 2: Get test tasks (proper split or fallback)
        print(f"\n[STEP 2/6] Getting test tasks...")
        test_tasks = self.get_test_tasks()
        
        if not test_tasks:
            raise ValueError("No test tasks available")
        
        print(f"> Retrieved {len(test_tasks)} tasks for evaluation")
        
        # Show train/test split info if enabled
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            try:
                with TaskSelector(self.db_path) as selector:
                    train_tasks, test_task_ids = selector.get_train_test_split()
                    print(f"  Training tasks: {len(train_tasks)}")
                    print(f"  Test tasks: {len(test_task_ids)}")
                    print(f"  Tasks being evaluated: {len(test_tasks)}")
            except Exception as e:
                self.log(f"Error showing split info: {e}")
        
        # Step 3: Initialize metrics storage
        print(f"\n[STEP 3/6] Initializing metrics storage...")
        map_scores = []
        mrr_scores = []
        recall_scores = {k: [] for k in recall_k_values}
        
        print("> Metrics storage initialized")
        
        # Step 4: Process each test task
        print(f"\n[STEP 4/6] Processing test tasks...")
        print("This may take a while depending on the number of tasks and modules...")
        
        successful_evaluations = 0
        zero_vector_count = 0
        no_ground_truth_count = 0
        
        with tqdm(test_tasks, desc="Evaluating test tasks", unit="tasks") as pbar:
            for i, (task_id, task_name, text) in enumerate(pbar):
                pbar.set_postfix({
                    "task": task_name[:15], 
                    "success": successful_evaluations,
                    "zero_vec": zero_vector_count,
                    "no_gt": no_ground_truth_count
                })
                
                try:
                    # Show detailed progress for first few tasks
                    if i < 3 and props.VERBOSE:
                        print(f"\n  Processing test task {i+1}: {task_name}")
                    
                    # Vectorize task text (returns normalized vector)
                    if i < 3 and props.VERBOSE:
                        print("    Vectorizing task text...")
                    task_vector = self.vectorize_task_text(text)
                    
                    if np.all(task_vector == 0):
                        zero_vector_count += 1
                        if props.VERBOSE:
                            self.log(f"Zero vector for task {task_name}")
                        continue
                    
                    # Get ground truth modules
                    if i < 3 and props.VERBOSE:
                        print("    Getting ground truth modules...")
                    ground_truth_modules = self.get_task_ground_truth_modules(task_id)
                    
                    if not ground_truth_modules:
                        no_ground_truth_count += 1
                        if props.VERBOSE:
                            self.log(f"No ground truth modules for task {task_name}")
                        continue
                    
                    if i < 3 and props.VERBOSE:
                        print(f"    Found {len(ground_truth_modules)} ground truth modules")
                    
                    # Get ranked modules
                    if i < 3 and props.VERBOSE:
                        print("    Ranking modules by similarity...")
                    ranked_modules = self.get_ranked_modules(task_vector, sort_metric)
                    
                    if not ranked_modules:
                        self.log(f"No module vectors available")
                        continue
                    
                    # Calculate metrics
                    if i < 3 and props.VERBOSE:
                        print("    Calculating metrics...")
                    map_score = self.calculate_average_precision(ranked_modules, ground_truth_modules)
                    mrr_score = self.calculate_reciprocal_rank(ranked_modules, ground_truth_modules)
                    
                    map_scores.append(map_score)
                    mrr_scores.append(mrr_score)
                    
                    # Calculate Recall@K for different K values
                    for k in recall_k_values:
                        recall_k = self.calculate_recall_at_k(ranked_modules, ground_truth_modules, k)
                        recall_scores[k].append(recall_k)
                    
                    successful_evaluations += 1
                    
                    if i < 3 and props.VERBOSE:
                        print(f"    > Task evaluated: MAP={map_score:.3f}, MRR={mrr_score:.3f}")
                    
                except Exception as e:
                    self.log(f"Error processing task {task_name}: {e}")
                    continue
        
        print(f"\n> Task processing completed!")
        print(f"  Successful evaluations: {successful_evaluations}/{len(test_tasks)}")
        if zero_vector_count > 0:
            print(f"  Tasks with zero vectors: {zero_vector_count}")
        if no_ground_truth_count > 0:
            print(f"  Tasks without ground truth: {no_ground_truth_count}")
        
        # Step 5: Calculate and print statistics
        print(f"\n[STEP 5/6] Calculating statistics...")
        if not map_scores:
            print("[!]  No valid evaluations completed!")
            return
        
        print(f"> Statistics calculated for {len(map_scores)} evaluations")
        
        # Step 6: Print results
        print(f"\n[STEP 6/6] Generating results...")
        self.print_evaluation_results(map_scores, mrr_scores, recall_scores, 
                                    len(test_tasks), sort_metric)
        print("> Evaluation completed successfully!")
    
    def print_evaluation_results(self, map_scores: List[float], mrr_scores: List[float], 
                               recall_scores: Dict[int, List[float]], 
                               total_tasks: int, sort_metric: str):
        """Print comprehensive evaluation results"""
        print("\n" + "="*80)
        print("STATISTICAL EVALUATION RESULTS")
        print("="*80)
        print(f"Configuration used:")
        print(f"  Vectorizer model: {props.VECTORISER_MODEL}")
        print(f"  Distance metric: {sort_metric}")
        print(f"  Module strategy: {props.MODULE_VECTOR_STRATEGY}")
        print(f"  Vector normalization: {props.NORMALIZE_VECTORS}")
        print(f"  Preprocessing: {props.PREPROCESS_TEST_TASK}")
        print(f"  Train/test split: {props.EXCLUDE_TEST_TASKS_FROM_MODEL}")
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print(f"  Test selection strategy: {props.EXCLUDE_TEST_TASKS_STRATEGY}")
        
        print(f"\nEvaluation summary:")
        print(f"  Total test tasks processed: {total_tasks}")
        print(f"  Valid evaluations: {len(map_scores)}")
        
        if not map_scores:
            print("No valid evaluations completed!")
            return
        
        # Calculate statistics for each metric
        map_stats = self.calculate_statistics(map_scores)
        mrr_stats = self.calculate_statistics(mrr_scores)
        
        # Print MAP statistics
        self.print_statistics_table("Mean Average Precision (MAP)", map_stats)
        
        # Print MRR statistics
        self.print_statistics_table("Mean Reciprocal Rank (MRR)", mrr_stats)
        
        # Print Recall@K statistics
        for k in sorted(recall_scores.keys()):
            recall_stats = self.calculate_statistics(recall_scores[k])
            self.print_statistics_table(f"Recall@{k}", recall_stats)
        
        # Print summary
        print(f"\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print("Results based on PROPER TRAIN/TEST SPLIT:")
            print("- Model trained on training tasks only")
            print("- Evaluation performed on unseen test tasks")
            print("- This provides unbiased performance estimates")
        else:
            print("Results based on RANDOM SAMPLING (no train/test split):")
            print("- Model may have seen test tasks during training")
            print("- Results may be optimistically biased")
            print("- Enable EXCLUDE_TEST_TASKS_FROM_MODEL for proper evaluation")
        
        print(f"\nOverall system performance (higher is better):")
        print(f"  MAP: {map_stats['mean']:.4f} +/- {map_stats['std']:.4f}")
        print(f"  MRR: {mrr_stats['mean']:.4f} +/- {mrr_stats['std']:.4f}")
        
        for k in sorted(recall_scores.keys()):
            recall_stats = self.calculate_statistics(recall_scores[k])
            print(f"  Recall@{k}: {recall_stats['mean']:.4f} +/- {recall_stats['std']:.4f}")
        
        print("="*80)

def main():
    """Main entry point"""
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Statistical evaluation using current properties configuration')
    parser.add_argument('--count', type=int, default=props.SUMMARY_TEST_TASK_COUNT if hasattr(props, 'SUMMARY_TEST_TASK_COUNT') else 100,
                       help='Number of tasks to test (ignored if proper split is enabled, default: from config)')
    parser.add_argument('--sort', type=int, default=1, choices=[1, 2, 3],
                       help='Sort by distance metric: 1=cosine (default), 2=euclidean, 3=manhattan')
    parser.add_argument('--recall-k', nargs='+', type=int, default=[5, 10, 20, 50],
                       help='K values for Recall@K calculation (default: 5 10 20 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Map sort parameter to distance metric - use configured metrics if available
    if hasattr(props, 'DISTANCE_METRICS') and props.DISTANCE_METRICS:
        # Use the configured distance metrics
        if args.sort == 1:
            sort_metric = 'cosine' if 'cosine' in props.DISTANCE_METRICS else props.DISTANCE_METRICS[0]
        elif args.sort == 2:
            sort_metric = 'euclidean' if 'euclidean' in props.DISTANCE_METRICS else props.DISTANCE_METRICS[0]
        elif args.sort == 3:
            sort_metric = 'manhattan' if 'manhattan' in props.DISTANCE_METRICS else props.DISTANCE_METRICS[0]
        else:
            sort_metric = props.DISTANCE_METRICS[0]
    else:
        # Fallback to old mapping
        sort_metric_map = {1: 'cosine', 2: 'euclidean', 3: 'manhattan'}
        sort_metric = sort_metric_map.get(args.sort, 'cosine')
    
    print(f"Using configuration from properties.py:")
    print(f"  Vectorizer model: {props.VECTORISER_MODEL}")
    print(f"  Distance metrics configured: {getattr(props, 'DISTANCE_METRICS', ['cosine'])}")
    print(f"  Evaluation will use: {sort_metric}")
    
    try:
        with StatisticalEvaluator(props.DATABASE_PATH) as evaluator:
            evaluator.run_evaluation(args.count, sort_metric, args.recall_k)
            
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    if log_enable:
        sci_log.start('properties.py')
    main()
    if log_enable:
        sci_log.stop('csv')