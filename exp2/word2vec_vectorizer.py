#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word2Vec Vectorizer
Handles training and using custom Word2Vec models
"""

import numpy as np
from typing import List
import properties as props
from baseVectorizer import BaseVectorizer

try:
    from gensim.models import Word2Vec
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Gensim not available. Please install: pip install gensim")
    raise e

class Word2VecVectorizer(BaseVectorizer):
    """Word2Vec vectorizer implementation for training custom models"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.word2vec_model = None
    
    def load_or_train_model(self):
        """Train custom Word2Vec model"""
        self.log("Training custom Word2Vec model...")
        
        corpus = self.prepare_training_corpus()
        
        if not corpus:
            raise ValueError("Empty corpus for Word2Vec training")
        
        try:
            self.log(f"Training Word2Vec with parameters:")
            self.log(f"  Vector size: {props.VECTOR_DIMENSION}")
            self.log(f"  Window: {props.W2V_WINDOW}")
            self.log(f"  Min count: {props.W2V_MIN_COUNT}")
            self.log(f"  Epochs: {props.W2V_EPOCHS}")
            self.log(f"  Algorithm: {'Skip-gram' if props.W2V_SG else 'CBOW'}")
            
            self.word2vec_model = Word2Vec(
                sentences=corpus,
                vector_size=props.VECTOR_DIMENSION,
                window=props.W2V_WINDOW,
                min_count=props.W2V_MIN_COUNT,
                epochs=props.W2V_EPOCHS,
                sg=props.W2V_SG,
                workers=4  # Use multiple cores for training
            )
            
            self.model = self.word2vec_model
            self.log(f"Word2Vec model trained successfully with vocabulary size: {len(self.word2vec_model.wv)}")
            
        except Exception as e:
            self.log(f"Failed to train Word2Vec model: {e}")
            raise ValueError(f"Could not train Word2Vec model: {e}")
    
    def vectorize_text(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to vector using Word2Vec embeddings"""
        if not tokens or not self.word2vec_model:
            return np.zeros(self.get_vector_dimension())
        
        # Get vectors for tokens that exist in vocabulary
        vectors = []
        found_tokens = 0
        
        for token in tokens:
            if token in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[token])
                found_tokens += 1
        
        if not vectors:
            return np.zeros(self.get_vector_dimension())
        
        # Use average aggregation
        vectors = np.array(vectors)
        return np.mean(vectors, axis=0)
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of vectors produced by Word2Vec"""
        return props.VECTOR_DIMENSION
    
    def get_model_info(self) -> str:
        """Get information about the trained Word2Vec model"""
        if self.word2vec_model:
            return f"Custom Word2Vec model, vocab size: {len(self.word2vec_model.wv)}, dimension: {props.VECTOR_DIMENSION}"
        return "Word2Vec model not trained"