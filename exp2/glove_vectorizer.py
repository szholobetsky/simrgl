#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GloVe Vectorizer
Handles GloVe embedding model loading and vectorization
"""

import os
import zipfile
import requests
import numpy as np
from typing import List, Dict
import properties as props
from baseVectorizer import BaseVectorizer

class GloVeVectorizer(BaseVectorizer):
    """GloVe vectorizer implementation"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.glove_embeddings = {}
        self.embedding_dim = 0
        
        # GloVe model URLs and info
        self.glove_models = {
            'glove.6B.50d': {
                'url': 'http://nlp.stanford.edu/data/glove.6B.zip',
                'file': 'glove.6B.50d.txt',
                'dim': 50
            },
            'glove.6B.100d': {
                'url': 'http://nlp.stanford.edu/data/glove.6B.zip',
                'file': 'glove.6B.100d.txt',
                'dim': 100
            },
            'glove.6B.200d': {
                'url': 'http://nlp.stanford.edu/data/glove.6B.zip',
                'file': 'glove.6B.200d.txt',
                'dim': 200
            },
            'glove.6B.300d': {
                'url': 'http://nlp.stanford.edu/data/glove.6B.zip',
                'file': 'glove.6B.300d.txt',
                'dim': 300
            }
        }
    
    def download_glove_model(self, model_name: str):
        """Download and extract GloVe model if not already available"""
        if model_name not in self.glove_models:
            raise ValueError(f"Unknown GloVe model: {model_name}")
        
        model_info = self.glove_models[model_name]
        cache_dir = props.GLOVE_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        
        model_file = os.path.join(cache_dir, model_info['file'])
        
        if os.path.exists(model_file):
            self.log(f"GloVe model already exists: {model_file}")
            return model_file
        
        # Download the zip file
        zip_file = os.path.join(cache_dir, 'glove.6B.zip')
        
        if not os.path.exists(zip_file):
            self.log(f"Downloading GloVe model from {model_info['url']}")
            self.log("This may take several minutes...")
            
            response = requests.get(model_info['url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_file, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end="", flush=True)
            
            print()  # New line after progress
            self.log("Download completed")
        
        # Extract the specific model file
        if not os.path.exists(model_file):
            self.log(f"Extracting {model_info['file']} from zip archive")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extract(model_info['file'], cache_dir)
        
        return model_file
    
    def load_glove_embeddings(self, model_file: str) -> Dict[str, np.ndarray]:
        """Load GloVe embeddings from file"""
        self.log(f"Loading GloVe embeddings from {model_file}")
        embeddings = {}
        
        with open(model_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 50000 == 0:
                    self.log(f"Loaded {line_num} embeddings...")
                
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                word = parts[0]
                try:
                    vector = np.array([float(x) for x in parts[1:]])
                    if self.embedding_dim == 0:
                        self.embedding_dim = len(vector)
                    embeddings[word] = vector
                except ValueError:
                    continue
        
        self.log(f"Loaded {len(embeddings)} GloVe embeddings, dimension: {self.embedding_dim}")
        return embeddings
    
    def load_or_train_model(self):
        """Load GloVe embeddings"""
        self.log(f"Loading GloVe model: {props.GLOVE_MODEL_NAME}")
        
        try:
            model_file = self.download_glove_model(props.GLOVE_MODEL_NAME)
            self.glove_embeddings = self.load_glove_embeddings(model_file)
            
            if not self.glove_embeddings:
                raise ValueError("No embeddings loaded")
            
            self.model = self.glove_embeddings
            self.log("GloVe model loaded successfully")
            
        except Exception as e:
            self.log(f"Failed to load GloVe model: {e}")
            raise ValueError(f"Could not load GloVe model: {e}")
    
    def vectorize_text(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to vector using GloVe embeddings"""
        if not tokens or not self.glove_embeddings:
            return np.zeros(self.get_vector_dimension())
        
        # Get vectors for tokens that exist in vocabulary
        vectors = []
        found_tokens = 0
        
        for token in tokens:
            if token in self.glove_embeddings:
                vector = self.glove_embeddings[token]
                # Resize vector to target dimension if needed
                if len(vector) != props.VECTOR_DIMENSION:
                    if len(vector) > props.VECTOR_DIMENSION:
                        vector = vector[:props.VECTOR_DIMENSION]
                    else:
                        vector = np.pad(vector, (0, props.VECTOR_DIMENSION - len(vector)))
                vectors.append(vector)
                found_tokens += 1
        
        if not vectors:
            return np.zeros(self.get_vector_dimension())
        
        # Use average aggregation
        vectors = np.array(vectors)
        return np.mean(vectors, axis=0)
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of vectors produced by GloVe"""
        return props.VECTOR_DIMENSION
    
    def get_model_info(self) -> str:
        """Get information about the loaded GloVe model"""
        if self.glove_embeddings:
            return f"GloVe model: {props.GLOVE_MODEL_NAME}, vocab size: {len(self.glove_embeddings)}, target dimension: {props.VECTOR_DIMENSION}"
        return "GloVe model not loaded"