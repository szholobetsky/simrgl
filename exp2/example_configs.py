#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Model Configurations
Shows how to configure different embedding models for experiments

To use a specific configuration:
1. Copy the desired configuration to your properties.py
2. Set CLEAR_EMBEDDINGS = True for the first run with a new model
3. Set CLEAR_EMBEDDINGS = False for subsequent runs with the same model
"""

# Configuration 1: Custom Word2Vec model
WORD2VEC_CONFIG = {
    'VECTORISER_MODEL': 'own',
    'CLEAR_EMBEDDINGS': True,  # Set to False after first run
    'VECTOR_DIMENSION': 100,
    'NORMALIZE_VECTORS': True,
    
    # Word2Vec specific settings
    'W2V_WINDOW': 5,
    'W2V_MIN_COUNT': 1,
    'W2V_EPOCHS': 10,
    'W2V_SG': 0,  # 0=CBOW, 1=Skip-gram
}

# Configuration 2: FastText pretrained model
FASTTEXT_CONFIG = {
    'VECTORISER_MODEL': 'fast_text',
    'CLEAR_EMBEDDINGS': True,  # Set to False after first run
    'VECTOR_DIMENSION': 300,  # Will be resized to this dimension
    'NORMALIZE_VECTORS': True,
    
    # FastText specific settings
    'FASTTEXT_MODEL_LANG': 'en',
    'FASTTEXT_MODEL_DIM': 300,
}

# Configuration 3: GloVe pretrained model
GLOVE_CONFIG = {
    'VECTORISER_MODEL': 'glove',
    'CLEAR_EMBEDDINGS': True,  # Set to False after first run
    'VECTOR_DIMENSION': 100,  # Will be resized to this dimension
    'NORMALIZE_VECTORS': True,
    
    # GloVe specific settings
    'GLOVE_MODEL_NAME': 'glove.6B.100d',  # Options: 50d, 100d, 200d, 300d
    'GLOVE_CACHE_DIR': './glove_cache',
}

# Configuration 4: BERT model
BERT_CONFIG = {
    'VECTORISER_MODEL': 'bert',
    'CLEAR_EMBEDDINGS': True,  # Set to False after first run
    'VECTOR_DIMENSION': 768,  # Will be resized if needed
    'NORMALIZE_VECTORS': True,
    
    # BERT specific settings
    'BERT_MODEL_NAME': 'bert-base-uncased',
    'BERT_MAX_LENGTH': 512,
    'BERT_BATCH_SIZE': 16,
    'BERT_LAYER': -2,  # -1 = last layer, -2 = second to last
}

# Configuration 5: Local sentence-transformers model
SENTENCE_TRANSFORMER_CONFIG = {
    'VECTORISER_MODEL': 'llm',
    'CLEAR_EMBEDDINGS': True,  # Set to False after first run
    'VECTOR_DIMENSION': 384,  # Will be resized if needed
    'NORMALIZE_VECTORS': True,
    
    # LLM specific settings (local model)
    'LLM_MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',
    'LLM_MAX_LENGTH': 384,
    'LLM_BATCH_SIZE': 32,
    'LLM_DEVICE': 'auto',  # 'auto', 'cpu', 'cuda'
    'LLM_API_PROVIDER': None,  # None for local models
}

# Configuration 6: OpenAI API embeddings
OPENAI_CONFIG = {
    'VECTORISER_MODEL': 'llm',
    'CLEAR_EMBEDDINGS': True,  # Set to False after first run
    'VECTOR_DIMENSION': 1536,  # text-embedding-ada-002 dimension
    'NORMALIZE_VECTORS': True,
    
    # LLM specific settings (OpenAI API)
    'LLM_API_PROVIDER': 'openai',
    'LLM_API_KEY': 'your-openai-api-key-here',
    'LLM_API_MODEL': 'text-embedding-ada-002',
    'LLM_BATCH_SIZE': 100,  # OpenAI allows larger batches
}

# Configuration 7: Large BERT model for better quality
BERT_LARGE_CONFIG = {
    'VECTORISER_MODEL': 'bert',
    'CLEAR_EMBEDDINGS': True,  # Set to False after first run
    'VECTOR_DIMENSION': 1024,  # BERT-large dimension
    'NORMALIZE_VECTORS': True,
    
    # BERT specific settings
    'BERT_MODEL_NAME': 'bert-large-uncased',
    'BERT_MAX_LENGTH': 512,
    'BERT_BATCH_SIZE': 8,  # Smaller batch for larger model
    'BERT_LAYER': -1,  # Last layer for best representations
}

# Configuration 8: Multilingual model
MULTILINGUAL_CONFIG = {
    'VECTORISER_MODEL': 'llm',
    'CLEAR_EMBEDDINGS': True,  # Set to False after first run
    'VECTOR_DIMENSION': 768,
    'NORMALIZE_VECTORS': True,
    
    # Multilingual sentence transformer
    'LLM_MODEL_NAME': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'LLM_MAX_LENGTH': 384,
    'LLM_BATCH_SIZE': 32,
    'LLM_DEVICE': 'auto',
    'LLM_API_PROVIDER': None,
}

def apply_config(config_dict, properties_module):
    """
    Apply a configuration to the properties module
    
    Args:
        config_dict: Dictionary with configuration values
        properties_module: The properties module to update
    """
    for key, value in config_dict.items():
        setattr(properties_module, key, value)
    
    print(f"Applied configuration: {config_dict['VECTORISER_MODEL']}")
    print(f"Vector dimension: {config_dict['VECTOR_DIMENSION']}")
    print(f"Clear embeddings: {config_dict['CLEAR_EMBEDDINGS']}")

# Example usage:
"""
import properties as props
from example_model_configs import BERT_CONFIG, apply_config

# Apply BERT configuration
apply_config(BERT_CONFIG, props)

# Run vectorization (first time with new model)
# python vectoriser.py

# For subsequent experiments with the same model, change:
props.CLEAR_EMBEDDINGS = False

# Run vectorization again (will reuse embeddings)
# python vectoriser.py
"""