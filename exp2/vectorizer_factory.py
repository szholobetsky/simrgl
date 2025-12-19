#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vectorizer Factory
Creates appropriate vectorizer instances based on configuration
"""

import properties as props
from baseVectorizer import BaseVectorizer

def create_vectorizer(db_path: str) -> BaseVectorizer:
    """
    Factory function to create appropriate vectorizer based on configuration
    
    Args:
        db_path: Path to the database
        
    Returns:
        Appropriate vectorizer instance
        
    Raises:
        ValueError: If unknown vectorizer model specified
    """
    model_type = props.VECTORISER_MODEL.lower()
    
    if model_type == 'own':
        from word2vec_vectorizer import Word2VecVectorizer
        return Word2VecVectorizer(db_path)
    
    elif model_type == 'fast_text':
        from fasttext_vectorizer import FastTextVectorizer
        return FastTextVectorizer(db_path)
    
    elif model_type == 'glove':
        from glove_vectorizer import GloVeVectorizer
        return GloVeVectorizer(db_path)
    
    elif model_type == 'bert':
        from bert_vectorizer import BERTVectorizer
        return BERTVectorizer(db_path)
    
    elif model_type == 'llm':
        from llm_vectorizer import LLMVectorizer
        return LLMVectorizer(db_path)
    
    else:
        raise ValueError(f"Unknown vectorizer model: {props.VECTORISER_MODEL}. "
                        f"Supported models: 'own', 'fast_text', 'glove', 'bert', 'llm'")

def get_available_models():
    """Get list of available vectorizer models"""
    return ['own', 'fast_text', 'glove', 'bert', 'llm']

def validate_model_config():
    """Validate the current model configuration"""
    model_type = props.VECTORISER_MODEL.lower()
    
    if model_type not in get_available_models():
        raise ValueError(f"Invalid VECTORISER_MODEL: {props.VECTORISER_MODEL}")
    
    # Model-specific validation
    if model_type == 'bert':
        if not hasattr(props, 'BERT_MODEL_NAME') or not props.BERT_MODEL_NAME:
            raise ValueError("BERT_MODEL_NAME must be specified for BERT vectorizer")
    
    elif model_type == 'llm':
        if props.LLM_API_PROVIDER and not props.LLM_API_KEY:
            raise ValueError("LLM_API_KEY must be specified when using API provider")
        
        if not hasattr(props, 'LLM_MODEL_NAME') or not props.LLM_MODEL_NAME:
            raise ValueError("LLM_MODEL_NAME must be specified for LLM vectorizer")
    
    elif model_type == 'glove':
        if not hasattr(props, 'GLOVE_MODEL_NAME') or not props.GLOVE_MODEL_NAME:
            raise ValueError("GLOVE_MODEL_NAME must be specified for GloVe vectorizer")
    
    elif model_type == 'fast_text':
        if not hasattr(props, 'FASTTEXT_MODEL_LANG') or not props.FASTTEXT_MODEL_LANG:
            raise ValueError("FASTTEXT_MODEL_LANG must be specified for FastText vectorizer")
    
    return True