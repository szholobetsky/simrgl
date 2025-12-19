#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Vectorizer
Handles various LLM embedding models (sentence-transformers, OpenAI, etc.)
"""

import numpy as np
from typing import List, Optional
import properties as props
from baseVectorizer import BaseVectorizer

class LLMVectorizer(BaseVectorizer):
    """LLM vectorizer implementation supporting multiple providers"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.embedding_model = None
        self.api_client = None
        self.provider = props.LLM_API_PROVIDER
    
    def load_or_train_model(self):
        """Load LLM model (local or API-based)"""
        if self.provider:
            self._load_api_model()
        else:
            self._load_local_model()
    
    def _load_local_model(self):
        """Load local sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.log(f"Loading sentence-transformers model: {props.LLM_MODEL_NAME}")
            
            # Determine device
            if props.LLM_DEVICE == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = props.LLM_DEVICE
            
            self.log(f"Using device: {device}")
            
            self.embedding_model = SentenceTransformer(props.LLM_MODEL_NAME, device=device)
            self.model = self.embedding_model
            
            self.log("Sentence-transformers model loaded successfully")
            
        except ImportError as e:
            self.log(f"sentence-transformers not available: {e}")
            raise ValueError("Please install sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            self.log(f"Failed to load sentence-transformers model: {e}")
            raise ValueError(f"Could not load model: {e}")
    
    def _load_api_model(self):
        """Load API-based model client"""
        if self.provider == 'openai':
            self._load_openai_client()
        elif self.provider == 'anthropic':
            self._load_anthropic_client()
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")
    
    def _load_openai_client(self):
        """Load OpenAI API client"""
        try:
            import openai
            
            if not props.LLM_API_KEY:
                raise ValueError("LLM_API_KEY must be set for OpenAI")
            
            self.log("Initializing OpenAI client")
            self.api_client = openai.OpenAI(api_key=props.LLM_API_KEY)
            self.model = self.api_client
            
            self.log("OpenAI client initialized successfully")
            
        except ImportError:
            raise ValueError("Please install OpenAI: pip install openai")
        except Exception as e:
            self.log(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"Could not initialize OpenAI: {e}")
    
    def _load_anthropic_client(self):
        """Load Anthropic API client"""
        try:
            import anthropic
            
            if not props.LLM_API_KEY:
                raise ValueError("LLM_API_KEY must be set for Anthropic")
            
            self.log("Initializing Anthropic client")
            self.api_client = anthropic.Anthropic(api_key=props.LLM_API_KEY)
            self.model = self.api_client
            
            self.log("Anthropic client initialized successfully")
            
        except ImportError:
            raise ValueError("Please install Anthropic: pip install anthropic")
        except Exception as e:
            self.log(f"Failed to initialize Anthropic client: {e}")
            raise ValueError(f"Could not initialize Anthropic: {e}")
    
    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts using the selected LLM"""
        if not texts:
            return np.zeros((0, self.get_vector_dimension()))
        
        if self.provider:
            return self._encode_texts_api(texts)
        else:
            return self._encode_texts_local(texts)
    
    def _encode_texts_local(self, texts: List[str]) -> np.ndarray:
        """Encode texts using local sentence-transformers model"""
        try:
            from tqdm import tqdm
            
            embeddings = []
            
            # Process in batches
            for i in tqdm(range(0, len(texts), props.LLM_BATCH_SIZE), 
                         desc="Encoding texts", unit="batch"):
                batch_texts = texts[i:i + props.LLM_BATCH_SIZE]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Resize embeddings to target dimension if needed
                if batch_embeddings.shape[1] != props.VECTOR_DIMENSION:
                    resized_embeddings = []
                    for emb in batch_embeddings:
                        if len(emb) > props.VECTOR_DIMENSION:
                            resized_emb = emb[:props.VECTOR_DIMENSION]
                        else:
                            resized_emb = np.pad(emb, (0, props.VECTOR_DIMENSION - len(emb)))
                        resized_embeddings.append(resized_emb)
                    batch_embeddings = np.array(resized_embeddings)
                
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings)
            
        except Exception as e:
            self.log(f"Error encoding texts with local model: {e}")
            return np.zeros((len(texts), self.get_vector_dimension()))
    
    def _encode_texts_api(self, texts: List[str]) -> np.ndarray:
        """Encode texts using API-based model"""
        if self.provider == 'openai':
            return self._encode_texts_openai(texts)
        elif self.provider == 'anthropic':
            return self._encode_texts_anthropic(texts)
        else:
            return np.zeros((len(texts), self.get_vector_dimension()))
    
    def _encode_texts_openai(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenAI API"""
        try:
            from tqdm import tqdm
            import time
            
            embeddings = []
            
            # Process in batches with rate limiting
            for i in tqdm(range(0, len(texts), props.LLM_BATCH_SIZE), 
                         desc="Encoding with OpenAI", unit="batch"):
                batch_texts = texts[i:i + props.LLM_BATCH_SIZE]
                
                try:
                    response = self.api_client.embeddings.create(
                        input=batch_texts,
                        model=props.LLM_API_MODEL
                    )
                    
                    batch_embeddings = []
                    for embedding_obj in response.data:
                        emb = np.array(embedding_obj.embedding)
                        
                        # Resize to target dimension if needed
                        if len(emb) != props.VECTOR_DIMENSION:
                            if len(emb) > props.VECTOR_DIMENSION:
                                emb = emb[:props.VECTOR_DIMENSION]
                            else:
                                emb = np.pad(emb, (0, props.VECTOR_DIMENSION - len(emb)))
                        
                        batch_embeddings.append(emb)
                    
                    embeddings.extend(batch_embeddings)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.log(f"Error in OpenAI API call: {e}")
                    # Add zero vectors for failed batch
                    embeddings.extend([np.zeros(props.VECTOR_DIMENSION)] * len(batch_texts))
            
            return np.array(embeddings)
            
        except Exception as e:
            self.log(f"Error encoding texts with OpenAI: {e}")
            return np.zeros((len(texts), self.get_vector_dimension()))
    
    def _encode_texts_anthropic(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Anthropic API (placeholder - Anthropic doesn't provide embeddings API)"""
        self.log("Warning: Anthropic doesn't provide direct embedding API")
        # This is a placeholder - you would need to implement text-to-embedding
        # via Claude's text generation capabilities, which is not standard
        return np.zeros((len(texts), self.get_vector_dimension()))
    
    def vectorize_text(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to vector using LLM embeddings"""
        if not tokens:
            return np.zeros(self.get_vector_dimension())
        
        # Join tokens back to text for LLM processing
        text = ' '.join(tokens)
        embeddings = self.encode_texts_batch([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.get_vector_dimension())
    
    def vectorize_raw_text(self, text: str) -> np.ndarray:
        """Convert raw text to vector using LLM"""
        if not text:
            return np.zeros(self.get_vector_dimension())
        
        embeddings = self.encode_texts_batch([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.get_vector_dimension())
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of vectors produced by LLM"""
        return props.VECTOR_DIMENSION
    
    def get_model_info(self) -> str:
        """Get information about the loaded LLM model"""
        if self.provider:
            return f"LLM API: {self.provider}, model: {props.LLM_API_MODEL}, target dimension: {props.VECTOR_DIMENSION}"
        elif self.embedding_model:
            return f"LLM local: {props.LLM_MODEL_NAME}, target dimension: {props.VECTOR_DIMENSION}"
        return "LLM model not loaded"