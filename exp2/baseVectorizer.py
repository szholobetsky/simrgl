#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Vectorizer Class
Abstract base class for all vectorization implementations
"""

import sqlite3
import json
import numpy as np
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import properties as props
from task_selector import TaskSelector

class BaseVectorizer(ABC):
    """Abstract base class for all vectorizers"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.filtered_terms = set()
        self.model = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
    
    def log(self, message: str):
        """Print log message if verbose mode is enabled"""
        if props.VERBOSE:
            print(f"[{self.__class__.__name__}] {message}")
    
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
    
    def tokenize_and_filter_text(self, text: str) -> List[str]:
        """Tokenize text and filter using HHI terms"""
        if not text:
            return []
        
        # Same tokenization as in main.py
        tokens = re.findall(r'\b[a-zA-Z]+[a-zA-Z0-9#_]*\b|\b[a-zA-Z0-9#_]*[a-zA-Z]+\b', text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if len(token) >= props.MIN_WORD_LENGTH:
                if not props.IGNORE_PURE_NUMBERS or not token.isdigit():
                    if not props.IGNORE_PURE_SYMBOLS or any(c.isalnum() for c in token):
                        if token in self.filtered_terms:
                            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def load_filtered_terms(self):
        """Load terms that pass HHI and count filters"""
        self.log(f"Loading filtered terms (CNT > {props.MIN_TERM_COUNT}, HHI_ROOT > {props.MIN_HHI_ROOT})...")
        
        cursor = self.conn.execute("""
        SELECT ct.TERM 
        FROM SIMRGL_TERM_RANK ctr 
        JOIN SIMRGL_TERMS ct ON ctr.TERM_ID = ct.ID
        WHERE ctr.CNT > ? AND ctr.HHI_ROOT > ?
        ORDER BY ctr.HHI_ROOT DESC, ctr.CNT DESC
        """, (props.MIN_TERM_COUNT, props.MIN_HHI_ROOT))
        
        self.filtered_terms = set(term[0] for term in cursor.fetchall())
        self.log(f"Loaded {len(self.filtered_terms)} filtered terms")
    
    def prepare_training_corpus(self) -> List[List[str]]:
        """Prepare corpus for training (only training tasks if split is enabled)"""
        split_info = ""
        train_filter = ""
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            with TaskSelector(self.db_path) as selector:
                train_filter, _ = selector.create_train_test_filter_clause("t")
                split_info = " (training tasks only)"
        
        self.log(f"Preparing training corpus{split_info}...")
        
        # Build text source query based on configuration
        text_fields = ["t.TITLE"]
        if props.USE_DESCRIPTION:
            text_fields.append("t.DESCRIPTION")
        if props.USE_COMMENTS:
            text_fields.append("t.COMMENTS")
        
        text_concat = " || ' ' || ".join([f"COALESCE({field}, '')" for field in text_fields])
        
        query = f"""
        SELECT {text_concat} as combined_text
        FROM TASK t
        WHERE t.NAME IS NOT NULL
        {train_filter}
        """
        
        cursor = self.conn.execute(query)
        tasks = cursor.fetchall()
        corpus = []
        
        for (text,) in tasks:
            tokens = self.tokenize_and_filter_text(text)
            if tokens:  # Only add non-empty documents
                corpus.append(tokens)
        
        self.log(f"Prepared corpus with {len(corpus)} documents{split_info}")
        return corpus
    
    def prepare_text_corpus(self) -> List[str]:
        """Prepare text corpus for models that work with raw text"""
        split_info = ""
        train_filter = ""
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            with TaskSelector(self.db_path) as selector:
                train_filter, _ = selector.create_train_test_filter_clause("t")
                split_info = " (training tasks only)"
        
        self.log(f"Preparing text corpus{split_info}...")
        
        # Build text source query based on configuration
        text_fields = ["t.TITLE"]
        if props.USE_DESCRIPTION:
            text_fields.append("t.DESCRIPTION")
        if props.USE_COMMENTS:
            text_fields.append("t.COMMENTS")
        
        text_concat = " || ' ' || ".join([f"COALESCE({field}, '')" for field in text_fields])
        
        query = f"""
        SELECT {text_concat} as combined_text
        FROM TASK t
        WHERE t.NAME IS NOT NULL
        {train_filter}
        """
        
        cursor = self.conn.execute(query)
        tasks = cursor.fetchall()
        corpus = []
        
        for (text,) in tasks:
            if text and text.strip():
                corpus.append(text.strip())
        
        self.log(f"Prepared text corpus with {len(corpus)} documents{split_info}")
        return corpus
    
    @abstractmethod
    def load_or_train_model(self):
        """Load pretrained or train the embedding model"""
        pass
    
    @abstractmethod
    def vectorize_text(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to vector using the specific model"""
        pass
    
    def vectorize_raw_text(self, text: str) -> np.ndarray:
        """Convert raw text to vector (for models that don't need tokenization)"""
        # Default implementation: tokenize and use vectorize_text
        tokens = self.tokenize_and_filter_text(text)
        return self.vectorize_text(tokens)
    
    @abstractmethod
    def get_vector_dimension(self) -> int:
        """Get the dimension of vectors produced by this model"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> str:
        """Get information about the loaded model"""
        pass
    
    def should_clear_embeddings(self) -> bool:
        """Determine if embeddings should be cleared based on configuration"""
        return props.CLEAR_EMBEDDINGS
    
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
    
    def needs_recomputation(self) -> bool:
        """Check if embeddings need to be recomputed"""
        if self.should_clear_embeddings():
            return True
        
        if not self.check_existing_embeddings():
            return True
        
        # Check if model configuration has changed
        current_metadata = self.get_embedding_metadata()
        expected_model = props.VECTORISER_MODEL
        stored_model = current_metadata.get('model_type', '')
        
        if stored_model != expected_model:
            self.log(f"Model changed from {stored_model} to {expected_model}, recomputing embeddings")
            return True
        
        return False