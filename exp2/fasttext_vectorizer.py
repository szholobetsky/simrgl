#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastText Vectorizer
Handles FastText embedding model loading and vectorization
"""

import numpy as np
from typing import List
import properties as props
from baseVectorizer import BaseVectorizer

try:
    import fasttext
    import fasttext.util
except ImportError as e:
    print(f"[ERROR] FastText not available. Please install: pip install fasttext")
    raise e

class FastTextVectorizer(BaseVectorizer):
    """FastText vectorizer implementation"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.fasttext_model = None
    
    def load_or_train_model(self):
        """Load pretrained FastText model"""
        self.log("Loading pretrained FastText model...")
        
        try:
            # Download FastText model if needed
            self.log(f"Downloading FastText model for language: {props.FASTTEXT_MODEL_LANG}")
            fasttext.util.download_model(props.FASTTEXT_MODEL_LANG, if_exists='ignore')
            
            # Load the model
            model_file = f'cc.{props.FASTTEXT_MODEL_LANG}.{props.FASTTEXT_MODEL_DIM}.bin'
            self.log(f"Loading FastText model: {model_file}")
            self.fasttext_model = fasttext.load_model(model_file)
            
            self.log(f"FastText model loaded successfully")
            self.model = self.fasttext_model
            
        except Exception as e:
            self.log(f"Failed to load FastText model: {e}")
            raise ValueError(f"Could not load FastText model: {e}")
    
    def vectorize_text(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to vector using FastText embeddings"""
        if not tokens or not self.fasttext_model:
            return np.zeros(self.get_vector_dimension())
        
        # Get vectors for all tokens (FastText can handle any word)
        vectors = []
        for token in tokens:
            vector = self.fasttext_model.get_word_vector(token)
            # Resize vector to target dimension if needed
            if len(vector) != props.VECTOR_DIMENSION:
                if len(vector) > props.VECTOR_DIMENSION:
                    vector = vector[:props.VECTOR_DIMENSION]
                else:
                    vector = np.pad(vector, (0, props.VECTOR_DIMENSION - len(vector)))
            vectors.append(vector)
        
        if not vectors:
            return np.zeros(self.get_vector_dimension())
        
        # Use average aggregation by default
        vectors = np.array(vectors)
        return np.mean(vectors, axis=0)
    
    def vectorize_raw_text(self, text: str) -> np.ndarray:
        """Convert raw text to vector using FastText sentence vector"""
        if not text or not self.fasttext_model:
            return np.zeros(self.get_vector_dimension())
        
        try:
            # FastText can compute sentence vectors directly
            vector = self.fasttext_model.get_sentence_vector(text)
            
            # Resize vector to target dimension if needed
            if len(vector) != props.VECTOR_DIMENSION:
                if len(vector) > props.VECTOR_DIMENSION:
                    vector = vector[:props.VECTOR_DIMENSION]
                else:
                    vector = np.pad(vector, (0, props.VECTOR_DIMENSION - len(vector)))
            
            return vector
        except Exception as e:
            self.log(f"Error computing sentence vector: {e}")
            # Fallback to token-based approach
            return super().vectorize_raw_text(text)
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of vectors produced by FastText"""
        return props.VECTOR_DIMENSION
    
    def get_model_info(self) -> str:
        """Get information about the loaded FastText model"""
        if self.fasttext_model:
            return f"FastText {props.FASTTEXT_MODEL_LANG} model, target dimension: {props.VECTOR_DIMENSION}"
        return "FastText model not loaded"