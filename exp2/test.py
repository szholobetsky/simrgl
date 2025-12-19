#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Compass Tester
Tests semantic similarity between task text and modules using current properties configuration
Runs a single test with parameters from properties.py
"""

import sqlite3
import json
import numpy as np
import re
from typing import List, Tuple
import properties as props
from task_selector import TaskSelector

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import euclidean, cityblock
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Visualization libraries not available: {e}")
    print("Visualization features will be disabled")
    VISUALIZATION_AVAILABLE = False

class SemanticTester:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.filtered_terms = set()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
    
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
        
        # Same tokenization as in main.py and vectoriser.py
        tokens = re.findall(r'\b[a-zA-Z]+[a-zA-Z0-9#_]*\b|\b[a-zA-Z0-9#_]*[a-zA-Z]+\b', text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if len(token) >= props.MIN_WORD_LENGTH:
                if not props.IGNORE_PURE_NUMBERS or not token.isdigit():
                    if not props.IGNORE_PURE_SYMBOLS or any(c.isalnum() for c in token):
                        # Apply HHI filtering if enabled
                        if props.PREPROCESS_TEST_TASK:
                            if token in self.filtered_terms:
                                filtered_tokens.append(token)
                        else:
                            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def vectorize_test_text(self, text: str) -> np.ndarray:
        """Vectorize task text using the configured embedding model"""
        # Check if we have the pipeline tables
        if not hasattr(self, '_pipeline_checked'):
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
            
            # Check train/test split status
            if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
                print(f"    Train/test split: ENABLED ({props.EXCLUDE_TEST_TASKS_STRATEGY})")
                with TaskSelector(self.db_path) as selector:
                    train_tasks, test_tasks = selector.get_train_test_split()
                    print(f"    Training tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")
            else:
                print(f"    Train/test split: DISABLED")
            
            self._pipeline_checked = True
        
        # Load the vectorizer to use the same model as configured
        try:
            from vectorizer_factory import create_vectorizer
            
            if not hasattr(self, '_vectorizer') or self._vectorizer is None:
                print(f"    Loading {props.VECTORISER_MODEL} vectorizer for test text processing...")
                self._vectorizer = create_vectorizer(self.db_path)
                self._vectorizer.load_filtered_terms()
                
                # Load the model (this might take time for some models)
                self._vectorizer.load_or_train_model()
                print(f"    Vectorizer ready: {self._vectorizer.get_model_info()}")
        
        except ImportError:
            raise ValueError("Vectorizer factory not available. Please ensure all vectorizer files are present.")
        
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
    
    def calculate_distances(self, test_vector: np.ndarray, target_vector: np.ndarray) -> dict[str, float]:
        """Calculate configured distance metrics between two vectors"""
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
                        if VISUALIZATION_AVAILABLE:
                            similarity = cosine_similarity(test_vec_2d, target_vec_2d)[0][0]
                        else:
                            # Fallback cosine calculation without sklearn
                            dot_product = np.dot(test_vector, target_vector)
                            norm_a = np.linalg.norm(test_vector)
                            norm_b = np.linalg.norm(target_vector)
                            similarity = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0
                    distances[metric] = 1.0 - similarity
                elif metric == 'euclidean':
                    if VISUALIZATION_AVAILABLE:
                        distances[metric] = euclidean(test_vector, target_vector)
                    else:
                        distances[metric] = np.linalg.norm(test_vector - target_vector)
                elif metric == 'manhattan':
                    if VISUALIZATION_AVAILABLE:
                        distances[metric] = cityblock(test_vector, target_vector)
                    else:
                        distances[metric] = np.sum(np.abs(test_vector - target_vector))
            except Exception as e:
                self.log(f"Error calculating {metric} distance: {e}")
                distances[metric] = float('inf')
        
        return distances
    
    def find_similar_modules(self, test_vector: np.ndarray, top_k: int = 50, sort_metric: str = None) -> List[Tuple[int, str, dict[str, float]]]:
        """Find modules most similar to test vector using configured distance metric"""
        if sort_metric is None:
            sort_metric = props.DISTANCE_METRICS[0]  # Use first configured metric
        
        self.log(f"Finding top {top_k} similar modules, sorted by {sort_metric}...")
        
        # Get all module vectors
        cursor = self.conn.execute("""
        SELECT mv.MODULE_ID, m.MODULE_NAME, mv.VECTOR
        FROM SIMRGL_MODULE_VECTOR mv
        JOIN SIMRGL_MODULES m ON mv.MODULE_ID = m.ID
        """)
        
        results = []
        
        for module_id, module_name, vector_json in cursor.fetchall():
            try:
                module_vector = np.array(json.loads(vector_json))
                
                # Verify the module vector is normalized (should already be from updated vectoriser.py)
                norm = np.linalg.norm(module_vector)
                if norm > 0 and abs(norm - 1.0) > 1e-6:
                    # Re-normalize if needed (shouldn't happen with updated vectoriser.py)
                    self.log(f"Re-normalizing vector for module {module_name} (norm was {norm:.6f})")
                    module_vector = module_vector / norm
                
                distances = self.calculate_distances(test_vector, module_vector)
                results.append((module_id, module_name, distances))
            except Exception as e:
                self.log(f"Error processing module {module_name}: {e}")
                continue
        
        # Sort by specified distance metric
        if sort_metric in props.DISTANCE_METRICS and results:
            results.sort(key=lambda x: x[2].get(sort_metric, float('inf')))
        else:
            self.log(f"Warning: Sort metric {sort_metric} not in configured metrics, using first available")
            available_metric = props.DISTANCE_METRICS[0]
            results.sort(key=lambda x: x[2].get(available_metric, float('inf')))
        
        return results[:top_k]
    
    def load_test_task(self, task_file: str) -> str:
        """Load test task from file"""
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            self.log(f"Loaded test task from {task_file}: {len(text)} characters")
            return text
        except Exception as e:
            raise ValueError(f"Could not load test task from {task_file}: {e}")
    
    def print_configuration(self):
        """Print current configuration from properties"""
        print(f"\n{'='*80}")
        print("CURRENT CONFIGURATION FROM PROPERTIES.PY")
        print(f"{'='*80}")
        print(f"Vectorizer model: {props.VECTORISER_MODEL}")
        print(f"Distance metrics: {', '.join(props.DISTANCE_METRICS)}")
        print(f"Module vector strategy: {props.MODULE_VECTOR_STRATEGY}")
        print(f"Preprocess test task: {props.PREPROCESS_TEST_TASK}")
        print(f"Exclude test tasks from model: {props.EXCLUDE_TEST_TASKS_FROM_MODEL}")
        print(f"Normalize vectors: {props.NORMALIZE_VECTORS}")
        print(f"Clear embeddings: {props.CLEAR_EMBEDDINGS}")
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print(f"Test task selection strategy: {props.EXCLUDE_TEST_TASKS_STRATEGY}")
            print(f"Target test tasks: {props.SUMMARY_TEST_TASK_COUNT}")
        
        print(f"Vector dimension: {props.VECTOR_DIMENSION}")
        print(f"{'='*80}")
    
    def print_results(self, results: List[Tuple[int, str, dict[str, float]]], test_text: str, sort_metric: str):
        """Print similarity results in a formatted way"""
        print(f"\n{'='*80}")
        print("SEMANTIC SIMILARITY RESULTS")
        print(f"{'='*80}")
        print(f"Test task text: {test_text[:200]}{'...' if len(test_text) > 200 else ''}")
        print(f"Model: {props.VECTORISER_MODEL}")
        print(f"Distance metrics configured: {', '.join(props.DISTANCE_METRICS)}")
        print(f"Results sorted by: {sort_metric}")
        print(f"Preprocessing enabled: {props.PREPROCESS_TEST_TASK}")
        print(f"Vector normalization: {props.NORMALIZE_VECTORS}")
        print(f"Module aggregation strategy: {props.MODULE_VECTOR_STRATEGY}")
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print(f"Train/test split: ENABLED - model trained on training tasks only")
        else:
            print(f"Train/test split: DISABLED - model may have seen similar tasks during training")
        
        print(f"\n{'-'*80}")
        print(f"{'Rank':<4} {'Module ID':<10} {'Module Name':<30} {'Distances'}")
        print(f"{'-'*80}")
        
        for rank, (module_id, module_name, distances) in enumerate(results, 1):
            distance_str = []
            for metric in props.DISTANCE_METRICS:
                if metric in distances:
                    distance_str.append(f"{metric}:{distances[metric]:.4f}")
            
            print(f"{rank:<4} {module_id:<10} {module_name:<30} {' | '.join(distance_str)}")
        
        print(f"{'='*80}")
        
        # Additional statistics
        if results:
            sort_distances = [r[2].get(sort_metric, float('inf')) for r in results if sort_metric in r[2]]
            if sort_distances:
                print(f"\n{sort_metric.capitalize()} distance statistics:")
                print(f"  Minimum: {min(sort_distances):.4f}")
                print(f"  Maximum: {max(sort_distances):.4f}")
                print(f"  Average: {np.mean(sort_distances):.4f}")
                print(f"  Median: {np.median(sort_distances):.4f}")
                
                if sort_metric == 'cosine':
                    print(f"\nCosine similarity statistics (1 - cosine_distance):")
                    similarities = [1.0 - d for d in sort_distances]
                    print(f"  Minimum similarity: {min(similarities):.4f}")
                    print(f"  Maximum similarity: {max(similarities):.4f}")
                    print(f"  Average similarity: {np.mean(similarities):.4f}")
    
    def visualize_semantic_space(self, task_file: str, output_file: str = None):
        """Create 2D visualization of module vectors and test task"""
        if not VISUALIZATION_AVAILABLE:
            print("Visualization libraries not available. Skipping visualization.")
            return
        
        if output_file is None:
            output_file = getattr(props, 'VIZ_DEFAULT_OUTPUT', 'semantic_map.png')
        
        self.log("Creating 2D semantic space visualization...")
        
        try:
            # Load test task and vectorize it
            test_text = self.load_test_task(task_file)
            test_vector = self.vectorize_test_text(test_text)
            
            # Get all module vectors and names
            cursor = self.conn.execute("""
            SELECT mv.MODULE_ID, m.MODULE_NAME, mv.VECTOR
            FROM SIMRGL_MODULE_VECTOR mv
            JOIN SIMRGL_MODULES m ON mv.MODULE_ID = m.ID
            ORDER BY m.MODULE_NAME
            """)
            
            module_vectors = []
            module_names = []
            module_ids = []
            
            for module_id, module_name, vector_json in cursor.fetchall():
                try:
                    vector = np.array(json.loads(vector_json))
                    
                    # Verify vector is normalized
                    norm = np.linalg.norm(vector)
                    if norm > 0 and abs(norm - 1.0) > 1e-6:
                        self.log(f"Re-normalizing vector for module {module_name} (norm was {norm:.6f})")
                        vector = vector / norm
                    
                    module_vectors.append(vector)
                    module_names.append(module_name)
                    module_ids.append(module_id)
                except Exception as e:
                    self.log(f"Error loading vector for module {module_name}: {e}")
                    continue
            
            if len(module_vectors) < 2:
                self.log("Need at least 2 module vectors for visualization")
                return
            
            # Combine all vectors
            all_vectors = np.array(module_vectors + [test_vector])
            all_labels = module_names + ['TEST_TASK']
            
            self.log(f"Visualizing {len(module_vectors)} modules + 1 test task")
            
            # Reduce dimensionality to 2D
            try:
                self.log("Applying t-SNE dimensionality reduction...")
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_vectors)-1))
                vectors_2d = tsne.fit_transform(all_vectors)
                reduction_method = "t-SNE"
            except Exception as e:
                self.log(f"t-SNE failed ({e}), falling back to PCA...")
                pca = PCA(n_components=2, random_state=42)
                vectors_2d = pca.fit_transform(all_vectors)
                reduction_method = "PCA"
            
            # Separate module points from test task point
            module_points = vectors_2d[:-1]
            test_point = vectors_2d[-1]
            
            # Create the plot
            plt.figure(figsize=getattr(props, 'VIZ_FIGURE_SIZE', (12, 8)))
            
            # Plot modules as scattered points
            plt.scatter(module_points[:, 0], module_points[:, 1], 
                       c='lightblue', s=100, alpha=0.7, 
                       edgecolors='navy', linewidth=1, label='Modules')
            
            # Plot test task as a red star
            plt.scatter(test_point[0], test_point[1], 
                       c='red', s=200, marker='*', 
                       edgecolors='darkred', linewidth=2, label='Test Task')
            
            # Add module names as annotations
            for i, (name, point) in enumerate(zip(module_names, module_points)):
                display_name = name if len(name) <= 15 else name[:12] + '...'
                plt.annotate(display_name, (point[0], point[1]), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # Find and highlight top similar modules
            top_k = getattr(props, 'VIZ_TOP_MODULES_HIGHLIGHT', 5)
            test_results = self.find_similar_modules(test_vector, top_k=top_k)
            top_module_ids = [r[0] for r in test_results]
            
            # Highlight top modules
            for i, (module_id, point) in enumerate(zip(module_ids, module_points)):
                if module_id in top_module_ids:
                    rank = top_module_ids.index(module_id) + 1
                    plt.scatter(point[0], point[1], 
                               c='orange', s=150, alpha=0.8,
                               edgecolors='red', linewidth=2)
                    plt.annotate(f'#{rank}', (point[0], point[1]), 
                                xytext=(-10, -10), textcoords='offset points',
                                fontsize=10, ha='center', va='center',
                                bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', alpha=0.8),
                                color='white', weight='bold')
            
            # Customize the plot
            title_parts = [f'Semantic Space Visualization ({reduction_method})']
            title_parts.append(f'Model: {props.VECTORISER_MODEL}')
            
            plt.title(f'{" - ".join(title_parts)}\nTest Task: {test_text[:50]}...', 
                     fontsize=14, pad=20)
            plt.xlabel(f'{reduction_method} Component 1', fontsize=12)
            plt.ylabel(f'{reduction_method} Component 2', fontsize=12)
            plt.legend(loc='upper right', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add configuration info
            config_info = f'Distance: {props.DISTANCE_METRICS[0]}, Strategy: {props.MODULE_VECTOR_STRATEGY}, Normalized: {props.NORMALIZE_VECTORS}'
            plt.figtext(0.02, 0.02, config_info, fontsize=9, style='italic')
            
            # Save the plot
            plt.tight_layout()
            dpi = getattr(props, 'VIZ_DPI', 300)
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
            self.log(f"Visualization saved to {output_file}")
            
            # Display if possible
            try:
                plt.show()
            except:
                self.log("Could not display plot (no GUI available)")
            
            plt.close()
            
        except Exception as e:
            self.log(f"Visualization failed: {e}")
            raise
    
    def run_test(self, task_file: str = "task.txt", top_k: int = 50, visualize: bool = False):
        """Run semantic similarity test using current properties configuration"""
        self.log(f"Starting semantic similarity test with current configuration...")
        
        try:
            # Print current configuration
            self.print_configuration()
            
            # Step 1: Load preprocessing terms if needed
            self.load_filtered_terms()
            
            # Step 2: Load test task
            test_text = self.load_test_task(task_file)
            
            if not test_text:
                raise ValueError("Test task is empty")
            
            # Step 3: Vectorize test text
            self.log("Vectorizing test task...")
            test_vector = self.vectorize_test_text(test_text)
            
            # Verify test vector is normalized
            test_norm = np.linalg.norm(test_vector)
            self.log(f"Test vector norm: {test_norm:.6f}")
            
            if np.all(test_vector == 0):
                self.log("Warning: Test vector is all zeros. This might indicate:")
                self.log("  - No terms from test task passed the filtering criteria")
                self.log("  - Test task contains only stop words or rare terms")
                self.log("  - Consider adjusting MIN_TERM_COUNT or MIN_HHI_ROOT parameters")
            
            # Step 4: Find similar modules using configured distance metric
            sort_metric = props.DISTANCE_METRICS[0]  # Use first configured metric for sorting
            results = self.find_similar_modules(test_vector, top_k, sort_metric)
            
            if not results:
                print("No module vectors found. Please run vectoriser.py first.")
                return
            
            # Step 5: Print results
            self.print_results(results, test_text, sort_metric)
            
            # Step 6: Create visualization if requested
            if visualize:
                self.visualize_semantic_space(task_file)
            
            self.log(f"Semantic similarity test completed successfully!")
            
        except Exception as e:
            self.log(f"Test failed: {e}")
            raise

def main():
    """Main entry point"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test semantic similarity using current properties configuration')
    parser.add_argument('task_file', nargs='?', default='task.txt', 
                       help='File containing task text (default: task.txt)')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Number of top results to show (default: 50)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create 2D visualization of modules and test task')
    parser.add_argument('--viz-file', default=None,
                       help='Output file for visualization (default: from properties.VIZ_DEFAULT_OUTPUT)')
    
    args = parser.parse_args()
    
    print("Running semantic similarity test with current properties configuration...")
    print(f"Task file: {args.task_file}")
    print(f"Top results: {args.top_k}")
    print(f"Visualization: {args.visualize}")
    
    try:
        with SemanticTester(props.DATABASE_PATH) as tester:
            tester.run_test(args.task_file, args.top_k, args.visualize)
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        raise

if __name__ == "__main__":
    main()