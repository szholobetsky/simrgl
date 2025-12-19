#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Compass Vectoriser
Enhanced with modular embedding support for multiple models
Supports Word2Vec, FastText, GloVe, BERT, and LLM embeddings
"""

import sqlite3
import json
import numpy as np
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import properties as props
from task_selector import TaskSelector
from vectorizer_factory import create_vectorizer, validate_model_config

try:
    from sklearn.cluster import KMeans
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing required libraries. Please install: pip install scikit-learn tqdm")
    print(f"Import error: {e}")
    exit(1)

class SemanticVectoriser:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.vectorizer = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
    
    def log(self, message: str):
        """Print log message if verbose mode is enabled"""
        if props.VERBOSE:
            print(f"[VECTORISER] {message}")
    
    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize vector to unit length (magnitude = 1)
        
        Args:
            vector: Input vector to normalize
            
        Returns:
            Unit vector with same direction but magnitude = 1
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector  # Return zero vector as is
        return vector / norm

    def should_rebuild_module_vectors(self) -> bool:
        """Check if module vectors need rebuilding due to strategy change"""
        if props.CLEAR_EMBEDDINGS:
            return True
        
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_MODULE_VECTOR")
            if cursor.fetchone()[0] == 0:
                return True
        except sqlite3.Error:
            return True
        
        # Check if strategy has changed
        metadata = self.get_embedding_metadata()
        expected_strategy = props.MODULE_VECTOR_STRATEGY
        stored_strategy = metadata.get('module_strategy', '')
        
        return stored_strategy != expected_strategy

    def should_clear_embeddings(self) -> bool:
        """Check if embeddings should be cleared based on configuration and model changes"""
        if props.CLEAR_EMBEDDINGS:
            self.log("CLEAR_EMBEDDINGS is True, will recreate all embeddings")
            return True
        
        # Check if model has changed
        metadata = self.get_embedding_metadata()
        current_model = props.VECTORISER_MODEL
        stored_model = metadata.get('model_type', '')
        
        if stored_model != current_model:
            self.log(f"Model changed from '{stored_model}' to '{current_model}', will recreate embeddings")
            return True
        
        # Check if embeddings exist
        if not self.check_existing_embeddings():
            self.log("No existing embeddings found, will create new ones")
            return True
        
        self.log(f"Reusing existing embeddings for model: {current_model}")
        return False
    
    def check_existing_embeddings(self) -> bool:
        """Check if embeddings already exist in the database"""
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_TASK_VECTOR")
            count = cursor.fetchone()[0]
            return count > 0
        except sqlite3.Error:
            return False
    
    def get_embedding_metadata(self) -> Dict[str, str]:
        """Get metadata about current embeddings"""
        try:
            cursor = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='SIMRGL_EMBEDDING_METADATA'
            """)
            if not cursor.fetchone():
                return {}
            
            cursor = self.conn.execute("SELECT key, value FROM SIMRGL_EMBEDDING_METADATA")
            return {key: value for key, value in cursor.fetchall()}
        except sqlite3.Error:
            return {}
    
    def save_embedding_metadata(self, metadata: Dict[str, str]):
        """Save metadata about current embeddings"""
        try:
            # Create metadata table if it doesn't exist
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS SIMRGL_EMBEDDING_METADATA (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """)
            
            # Clear existing metadata
            self.conn.execute("DELETE FROM SIMRGL_EMBEDDING_METADATA")
            
            # Insert new metadata
            for key, value in metadata.items():
                self.conn.execute(
                    "INSERT INTO SIMRGL_EMBEDDING_METADATA (key, value) VALUES (?, ?)",
                    (key, value)
                )
            
            self.conn.commit()
        except sqlite3.Error as e:
            self.log(f"Error saving metadata: {e}")
    
    def drop_vector_tables(self):
        """Drop all vector tables if they exist"""
        self.log("Dropping existing vector tables...")
        
        tables_to_drop = [
            'SIMRGL_FILE_VECTOR',
            'SIMRGL_MODULE_VECTOR', 
            'SIMRGL_TASK_VECTOR'
        ]
        
        for table in tables_to_drop:
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
                self.log(f"Dropped table {table}")
            except sqlite3.Error as e:
                self.log(f"Error dropping table {table}: {e}")
        
        self.conn.commit()
    
    def create_vector_tables(self):
        """Create vector tables"""
        self.log("Creating vector tables...")
        
        # Task vectors
        self.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS SIMRGL_TASK_VECTOR (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            TASK_ID INTEGER NOT NULL,
            VECTOR TEXT NOT NULL,
            FOREIGN KEY (TASK_ID) REFERENCES TASK(ID)
        )
        """)
        
        # Module vectors
        self.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS SIMRGL_MODULE_VECTOR (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            MODULE_ID INTEGER NOT NULL,
            VECTOR TEXT NOT NULL,
            STRATEGY TEXT NOT NULL,
            FOREIGN KEY (MODULE_ID) REFERENCES SIMRGL_MODULES(ID)
        )
        """)
        
        # File vectors
        self.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS SIMRGL_FILE_VECTOR (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            FILE_ID INTEGER NOT NULL,
            VECTOR TEXT NOT NULL,
            STRATEGY TEXT NOT NULL,
            FOREIGN KEY (FILE_ID) REFERENCES SIMRGL_FILES(ID)
        )
        """)
        
        self.conn.commit()
        self.log("Vector tables created successfully")
    
    def vectorize_tasks(self):
        """Create vectors for all tasks (training tasks only if split enabled)"""
        split_info = ""
        train_filter = ""
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            with TaskSelector(self.db_path) as selector:
                train_filter, _ = selector.create_train_test_filter_clause("t")
                split_info = " (training tasks only)"
        
        norm_text = "with normalization" if getattr(props, 'NORMALIZE_VECTORS', True) else ""
        self.log(f"Vectorizing tasks {norm_text}{split_info} using {props.VECTORISER_MODEL}")
        
        # Build text source query
        text_fields = ["t.TITLE"]
        if props.USE_DESCRIPTION:
            text_fields.append("t.DESCRIPTION")
        if props.USE_COMMENTS:
            text_fields.append("t.COMMENTS")
        
        text_concat = " || ' ' || ".join([f"COALESCE({field}, '')" for field in text_fields])
        
        query = f"""
        SELECT t.ID, {text_concat} as combined_text
        FROM TASK t
        WHERE t.NAME IS NOT NULL
        {train_filter}
        """
        
        cursor = self.conn.execute(query)
        tasks = cursor.fetchall()
        vectorized_count = 0
        
        # Add progress bar for task vectorization
        desc_parts = ["Vectorizing tasks"]
        if getattr(props, 'NORMALIZE_VECTORS', True):
            desc_parts.append("(normalized)")
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            desc_parts.append("(training only)")
        desc = " ".join(desc_parts)
        
        with tqdm(tasks, desc=desc, unit="tasks") as pbar:
            for task_id, text in pbar:
                # Use the vectorizer's method for text processing
                if hasattr(self.vectorizer, 'vectorize_raw_text'):
                    vector = self.vectorizer.vectorize_raw_text(text)
                else:
                    tokens = self.vectorizer.tokenize_and_filter_text(text)
                    vector = self.vectorizer.vectorize_text(tokens)
                
                # Normalize the vector if enabled
                if getattr(props, 'NORMALIZE_VECTORS', True):
                    vector = self.normalize_vector(vector)
                
                # Store vector as JSON
                vector_json = json.dumps(vector.tolist())
                
                self.conn.execute(
                    "INSERT INTO SIMRGL_TASK_VECTOR (TASK_ID, VECTOR) VALUES (?, ?)",
                    (task_id, vector_json)
                )
                
                vectorized_count += 1
                pbar.set_postfix({"vectorized": vectorized_count})
        
        self.conn.commit()
        self.log(f"Vectorized {vectorized_count} tasks {norm_text}{split_info}")
    
    def get_task_module_relationships(self) -> Dict[int, Dict[int, int]]:
        """Get mapping of modules to tasks with file change counts (only training tasks if split enabled)"""
        split_info = ""
        train_filter = ""
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            with TaskSelector(self.db_path) as selector:
                train_filter, _ = selector.create_train_test_filter_clause("t")
                split_info = " (training tasks only)"
        
        self.log(f"Building task-module relationships{split_info}...")
        
        query = f"""
        SELECT m.ID as module_id, t.ID as task_id, COUNT(f.ID) as file_count
        FROM SIMRGL_MODULES m
        JOIN SIMRGL_FILES f ON m.ID = f.MODULE_ID
        JOIN RAWDATA r ON f.FILE_PATH = r.PATH
        JOIN TASK t ON r.TASK_NAME = t.NAME
        WHERE 1=1
        {train_filter}
        GROUP BY m.ID, t.ID
        """
        
        cursor = self.conn.execute(query)
        
        # module_id -> {task_id: file_count}
        module_tasks = defaultdict(dict)
        
        for module_id, task_id, file_count in cursor.fetchall():
            module_tasks[module_id][task_id] = file_count
        
        self.log(f"Built relationships for {len(module_tasks)} modules{split_info}")
        return dict(module_tasks)
    
    def aggregate_vectors_weighted_avg(self, task_vectors: List[np.ndarray], weights: List[int]) -> np.ndarray:
        """Calculate weighted average of vectors"""
        if not task_vectors or not weights:
            return np.zeros(props.VECTOR_DIMENSION)
        
        vectors = np.array(task_vectors)
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()  # normalize weights
        
        return np.average(vectors, axis=0, weights=weights)
    
    def aggregate_vectors_cluster(self, task_vectors: List[np.ndarray], progress_info: str = "") -> np.ndarray:
        """Use clustering to find representative vector with progress indication"""
        if not task_vectors:
            return np.zeros(props.VECTOR_DIMENSION)
        
        if len(task_vectors) == 1:
            return task_vectors[0]
        
        # Use K-means with k=1 to find centroid
        vectors = np.array(task_vectors)
        
        # Create a simple progress indicator for clustering
        if props.VERBOSE and progress_info:
            print(f"    Clustering {len(task_vectors)} vectors for {progress_info}...", end="", flush=True)
        
        start_time = time.time()
        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans.fit(vectors)
        elapsed = time.time() - start_time
        
        if props.VERBOSE and progress_info:
            print(f" done ({elapsed:.2f}s)")
        
        return kmeans.cluster_centers_[0]
    
    def vectorize_modules(self):
        """Create vectors for modules using specified strategy (with optional normalization)"""
        split_info = " (from training tasks only)" if props.EXCLUDE_TEST_TASKS_FROM_MODEL else ""
        norm_text = "with normalization" if getattr(props, 'NORMALIZE_VECTORS', True) else ""
        self.log(f"Vectorizing modules {norm_text} using strategy: {props.MODULE_VECTOR_STRATEGY}{split_info}")
        
        # Get task vectors (may be normalized depending on configuration)
        cursor = self.conn.execute("SELECT TASK_ID, VECTOR FROM SIMRGL_TASK_VECTOR")
        task_vectors = {}
        for task_id, vector_json in cursor.fetchall():
            vector = np.array(json.loads(vector_json))
            task_vectors[task_id] = vector
        
        # Get task-module relationships (filtered for training tasks if enabled)
        module_tasks = self.get_task_module_relationships()
        
        # Get module names for better progress display
        cursor = self.conn.execute("SELECT ID, MODULE_NAME FROM SIMRGL_MODULES")
        module_names = {module_id: name for module_id, name in cursor.fetchall()}
        
        vectorized_count = 0
        
        # Create progress bar for modules
        module_items = list(module_tasks.items())
        desc_parts = ["Vectorizing modules"]
        if getattr(props, 'NORMALIZE_VECTORS', True):
            desc_parts.append("(normalized)")
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            desc_parts.append("(from training)")
        desc = " ".join(desc_parts)
        
        with tqdm(module_items, desc=desc, unit="modules") as pbar:
            for module_id, task_weights in pbar:
                module_name = module_names.get(module_id, f"Module_{module_id}")
                pbar.set_postfix({"current": module_name[:20]})
                
                # Collect vectors and weights for this module
                vectors = []
                weights = []
                
                for task_id, weight in task_weights.items():
                    if task_id in task_vectors:
                        vectors.append(task_vectors[task_id])
                        weights.append(weight)
                
                if not vectors:
                    continue
                
                # Apply aggregation strategy
                if props.MODULE_VECTOR_STRATEGY == 'avg':
                    module_vector = np.mean(vectors, axis=0)
                elif props.MODULE_VECTOR_STRATEGY == 'sum':
                    module_vector = np.sum(vectors, axis=0)
                elif props.MODULE_VECTOR_STRATEGY == 'median':
                    module_vector = np.median(vectors, axis=0)
                elif props.MODULE_VECTOR_STRATEGY == 'weighted_avg':
                    module_vector = self.aggregate_vectors_weighted_avg(vectors, weights)
                elif props.MODULE_VECTOR_STRATEGY == 'cluster':
                    module_vector = self.aggregate_vectors_cluster(vectors, module_name)
                else:
                    self.log(f"Unknown strategy {props.MODULE_VECTOR_STRATEGY}, using avg")
                    module_vector = np.mean(vectors, axis=0)
                
                # Normalize the vector if enabled
                if getattr(props, 'NORMALIZE_VECTORS', True):
                    module_vector = self.normalize_vector(module_vector)
                
                # Store vector
                vector_json = json.dumps(module_vector.tolist())
                
                self.conn.execute(
                    "INSERT INTO SIMRGL_MODULE_VECTOR (MODULE_ID, VECTOR, STRATEGY) VALUES (?, ?, ?)",
                    (module_id, vector_json, props.MODULE_VECTOR_STRATEGY)
                )
                
                vectorized_count += 1
        
        self.conn.commit()
        self.log(f"Vectorized {vectorized_count} modules {norm_text}{split_info}")
    
    def verify_vector_normalization(self, table_name: str, sample_size: int = 10):
        """Verify that stored vectors are properly normalized"""
        cursor = self.conn.execute(f"SELECT VECTOR FROM {table_name} LIMIT ?", (sample_size,))
        
        print(f"\nVerifying normalization for {table_name}:")
        print("Vector ID | Magnitude")
        print("-" * 25)
        
        for i, (vector_json,) in enumerate(cursor.fetchall()):
            vector = np.array(json.loads(vector_json))
            magnitude = np.linalg.norm(vector)
            print(f"Vector {i:2d}  | {magnitude:.6f}")
        
        print()
    
    def print_summary_statistics(self):
        """Print vectorization summary"""
        print("\n" + "="*60)
        print("VECTORIZATION SUMMARY")
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print("(Training data only - test tasks excluded from model)")
        print("="*60)
        
        # Task vectors
        cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_TASK_VECTOR")
        task_count = cursor.fetchone()[0]
        print(f"Task vectors created: {task_count}")
        
        # Module vectors
        cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_MODULE_VECTOR")
        module_count = cursor.fetchone()[0]
        print(f"Module vectors created: {module_count}")
        
        # File vectors
        cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_FILE_VECTOR")
        file_count = cursor.fetchone()[0]
        print(f"File vectors created: {file_count}")
        
        print(f"Vectorization model: {props.VECTORISER_MODEL}")
        print(f"Vector dimension: {props.VECTOR_DIMENSION}")
        print(f"Module vectorization strategy: {props.MODULE_VECTOR_STRATEGY}")
        print(f"Vector normalization: {'ENABLED' if props.NORMALIZE_VECTORS else 'DISABLED'}")
        print(f"Clear embeddings: {props.CLEAR_EMBEDDINGS}")
        
        if self.vectorizer:
            print(f"Model info: {self.vectorizer.get_model_info()}")
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print(f"Train/test split: ENABLED")
            print(f"Test task selection strategy: {props.EXCLUDE_TEST_TASKS_STRATEGY}")
            print(f"Target test tasks: {props.SUMMARY_TEST_TASK_COUNT}")
        else:
            print(f"Train/test split: DISABLED")
        
        # Verify normalization for each table
        if task_count > 0:
            self.verify_vector_normalization('SIMRGL_TASK_VECTOR', 5)
        if module_count > 0:
            self.verify_vector_normalization('SIMRGL_MODULE_VECTOR', 5)
        if file_count > 0:
            self.verify_vector_normalization('SIMRGL_FILE_VECTOR', 5)
        
        # Show train/test split info
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            try:
                with TaskSelector(self.db_path) as selector:
                    selector.print_split_summary()
            except Exception as e:
                self.log(f"Error showing split summary: {e}")
        
        print("="*60)
    
    def run_vectorization(self):
        """Run the complete vectorization process"""
        self.log("Starting enhanced vectorization process...")
        
        try:
            # Step 1: Validate configuration
            validate_model_config()
            
            # Step 2: Check what needs to be rebuilt
            need_full_rebuild = self.should_clear_embeddings()
            need_module_rebuild = self.should_rebuild_module_vectors() if not need_full_rebuild else False
            
            if need_full_rebuild:
                self.log("Full rebuild required - clearing all embeddings")
                self.drop_vector_tables()
                self.create_vector_tables()
            elif need_module_rebuild:
                self.log("Module rebuild required - keeping task vectors")
                # Only clear module vectors
                self.conn.execute("DELETE FROM SIMRGL_MODULE_VECTOR")
                self.conn.commit()
            else:
                # Ensure ALL tables exist even if not clearing
                required_tables = ['SIMRGL_TASK_VECTOR', 'SIMRGL_MODULE_VECTOR', 'SIMRGL_FILE_VECTOR']
                missing_tables = []
                
                for table in required_tables:
                    try:
                        self.conn.execute(f"SELECT COUNT(*) FROM {table}")
                    except sqlite3.Error:
                        missing_tables.append(table)
                
                if missing_tables:
                    self.log(f"Missing vector tables: {missing_tables}, creating them...")
                    self.create_vector_tables()
                    need_full_rebuild = True  # Force full rebuild if tables were missing
            
            # Step 3: Create vectorizer instance (ALWAYS needed)
            self.log(f"Creating {props.VECTORISER_MODEL} vectorizer...")
            self.vectorizer = create_vectorizer(self.db_path)
            
            # Step 4: Load filtered terms (needed for tokenization)
            self.vectorizer.load_filtered_terms()
            
            if not self.vectorizer.filtered_terms:
                raise ValueError("No terms passed the filtering criteria")
            
            # Step 5: Load/train model (ALWAYS needed for vectorization)
            self.vectorizer.load_or_train_model()
            
            # Step 6: Perform vectorization based on what's needed
            if need_full_rebuild:
                # Full vectorization: tasks + modules
                self.vectorize_tasks()
                self.vectorize_modules()
                rebuild_type = "full"
            elif need_module_rebuild:
                # Only rebuild module vectors
                self.vectorize_modules()
                rebuild_type = "modules_only"
            else:
                # No vectorization needed
                self.log("Reusing existing embeddings")
                rebuild_type = "none"
            
            # Step 7: Save metadata (always update to reflect current config)
            metadata = {
                'model_type': props.VECTORISER_MODEL,
                'vector_dimension': str(props.VECTOR_DIMENSION),
                'normalize_vectors': str(props.NORMALIZE_VECTORS),
                'module_strategy': props.MODULE_VECTOR_STRATEGY,
                'model_info': self.vectorizer.get_model_info(),
                'last_rebuild': rebuild_type
            }
            self.save_embedding_metadata(metadata)
            
            if rebuild_type != "none":
                self.log(f"Embeddings updated ({rebuild_type})")
            
            # Step 8: Print summary
            self.print_summary_statistics()
            
            self.log("Vectorization completed successfully!")
            
        except Exception as e:
            self.log(f"Vectorization failed: {e}")
            raise

def main():
    """Main entry point"""
    try:
        with SemanticVectoriser(props.DATABASE_PATH) as vectoriser:
            vectoriser.run_vectorization()
    except Exception as e:
        print(f"[ERROR] Vectorization failed: {e}")
        raise

if __name__ == "__main__":
    main()