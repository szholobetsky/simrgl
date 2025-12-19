#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Vectorizer
Handles BERT embedding model loading and vectorization
"""

import numpy as np
import torch
from typing import List
import properties as props
from baseVectorizer import BaseVectorizer

try:
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Transformers not available. Please install: pip install transformers torch tqdm")
    raise e

class BERTVectorizer(BaseVectorizer):
    """BERT vectorizer implementation"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self.tokenizer = None
        self.bert_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_or_train_model(self):
        """Load pretrained BERT model"""
        self.log(f"Loading BERT model: {props.BERT_MODEL_NAME}")
        self.log(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(props.BERT_MODEL_NAME)
            self.bert_model = AutoModel.from_pretrained(props.BERT_MODEL_NAME)
            
            # Move model to device
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            self.model = self.bert_model
            self.log(f"BERT model loaded successfully")
            
        except Exception as e:
            self.log(f"Failed to load BERT model: {e}")
            raise ValueError(f"Could not load BERT model: {e}")
    
    def encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts using BERT"""
        if not texts or not self.bert_model:
            return np.zeros((len(texts), self.get_vector_dimension()))
        
        embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), props.BERT_BATCH_SIZE):
            batch_texts = texts[i:i + props.BERT_BATCH_SIZE]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=props.BERT_MAX_LENGTH
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
                # Extract embeddings from specified layer
                if props.BERT_LAYER == -1:
                    # Last layer
                    hidden_states = outputs.last_hidden_state
                else:
                    # Specific layer (need all hidden states)
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        hidden_states = outputs.hidden_states[props.BERT_LAYER]
                    else:
                        # Fallback to last layer if hidden states not available
                        hidden_states = outputs.last_hidden_state
                
                # Pool the embeddings (mean pooling over sequence length)
                attention_mask = inputs['attention_mask']
                masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
                pooled_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                # Resize to target dimension if needed
                if pooled_embeddings.shape[1] != props.VECTOR_DIMENSION:
                    if pooled_embeddings.shape[1] > props.VECTOR_DIMENSION:
                        pooled_embeddings = pooled_embeddings[:, :props.VECTOR_DIMENSION]
                    else:
                        padding = torch.zeros(pooled_embeddings.shape[0], 
                                            props.VECTOR_DIMENSION - pooled_embeddings.shape[1],
                                            device=self.device)
                        pooled_embeddings = torch.cat([pooled_embeddings, padding], dim=1)
                
                embeddings.extend(pooled_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def vectorize_text(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to vector using BERT embeddings"""
        if not tokens or not self.bert_model:
            return np.zeros(self.get_vector_dimension())
        
        # Join tokens back to text for BERT processing
        text = ' '.join(tokens)
        embeddings = self.encode_text_batch([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.get_vector_dimension())
    
    def vectorize_raw_text(self, text: str) -> np.ndarray:
        """Convert raw text to vector using BERT"""
        if not text or not self.bert_model:
            return np.zeros(self.get_vector_dimension())
        
        embeddings = self.encode_text_batch([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(self.get_vector_dimension())
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of vectors produced by BERT"""
        return props.VECTOR_DIMENSION
    
    def get_model_info(self) -> str:
        """Get information about the loaded BERT model"""
        if self.bert_model:
            return f"BERT model: {props.BERT_MODEL_NAME}, layer: {props.BERT_LAYER}, target dimension: {props.VECTOR_DIMENSION}, device: {self.device}"
        return "BERT model not loaded"