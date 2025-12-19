# Configuration templates for the new models in your focused experiment

# =============================================================================
# BERT MODEL CONFIGURATIONS
# =============================================================================

# Standard BERT (baseline)
BERT_BASE_CONFIG = {
    'VECTORISER_MODEL': 'bert',
    'BERT_MODEL_NAME': 'bert-base-uncased',
    'VECTOR_DIMENSION': 768,
    'BERT_MAX_LENGTH': 512,
    'BERT_BATCH_SIZE': 16,
    'BERT_LAYER': -2,
}

# RoBERTa (often better than BERT)
ROBERTA_CONFIG = {
    'VECTORISER_MODEL': 'bert',
    'BERT_MODEL_NAME': 'roberta-base',
    'VECTOR_DIMENSION': 768,
    'BERT_MAX_LENGTH': 512,
    'BERT_BATCH_SIZE': 16,
    'BERT_LAYER': -2,
}

# CodeBERT (specialized for code)
CODEBERT_CONFIG = {
    'VECTORISER_MODEL': 'bert',
    'BERT_MODEL_NAME': 'microsoft/codebert-base',
    'VECTOR_DIMENSION': 768,
    'BERT_MAX_LENGTH': 256,  # Shorter for code snippets
    'BERT_BATCH_SIZE': 16,
    'BERT_LAYER': -2,
}

# =============================================================================
# LLM MODEL CONFIGURATIONS  
# =============================================================================

# MPNet (current default, good baseline)
MPNET_CONFIG = {
    'VECTORISER_MODEL': 'llm',
    'LLM_MODEL_NAME': 'sentence-transformers/all-mpnet-base-v2',
    'VECTOR_DIMENSION': 768,
    'LLM_MAX_LENGTH': 384,
    'LLM_BATCH_SIZE': 32,
    'LLM_DEVICE': 'auto',
    'LLM_API_PROVIDER': None,
}

# BGE Large (state-of-the-art)
BGE_LARGE_CONFIG = {
    'VECTORISER_MODEL': 'llm',
    'LLM_MODEL_NAME': 'BAAI/bge-large-en-v1.5',
    'VECTOR_DIMENSION': 1024,
    'LLM_MAX_LENGTH': 512,
    'LLM_BATCH_SIZE': 16,  # Smaller batch for larger model
    'LLM_DEVICE': 'auto',
    'LLM_API_PROVIDER': None,
}

# MS MARCO DistilBERT (optimized for search/retrieval)
MSMARCO_CONFIG = {
    'VECTORISER_MODEL': 'llm',
    'LLM_MODEL_NAME': 'sentence-transformers/msmarco-distilbert-base-v4',
    'VECTOR_DIMENSION': 768,
    'LLM_MAX_LENGTH': 384,
    'LLM_BATCH_SIZE': 32,
    'LLM_DEVICE': 'auto',
    'LLM_API_PROVIDER': None,
}

# =============================================================================
# INSTALLATION COMMANDS
# =============================================================================

"""
To install the new models, run these commands:

# For BERT models (including RoBERTa and CodeBERT)
pip install torch transformers --index-url https://download.pytorch.org/whl/cpu

# For LLM models
pip install sentence-transformers

# Check if specific models are available:
python -c "from transformers import AutoTokenizer; print('RoBERTa available:', AutoTokenizer.from_pretrained('roberta-base'))"
python -c "from transformers import AutoTokenizer; print('CodeBERT available:', AutoTokenizer.from_pretrained('microsoft/codebert-base'))"
python -c "from sentence_transformers import SentenceTransformer; print('BGE available:', SentenceTransformer('BAAI/bge-large-en-v1.5'))"
"""

# =============================================================================
# EXPECTED PERFORMANCE CHARACTERISTICS
# =============================================================================

MODEL_CHARACTERISTICS = {
    'bert-base-uncased': {
        'quality': 'Good',
        'speed': 'Slow', 
        'memory': 'High',
        'best_for': 'General text understanding'
    },
    
    'roberta-base': {
        'quality': 'Very Good',
        'speed': 'Slow',
        'memory': 'High', 
        'best_for': 'Better than BERT for most tasks'
    },
    
    'microsoft/codebert-base': {
        'quality': 'Very Good',
        'speed': 'Slow',
        'memory': 'High',
        'best_for': 'Code analysis and software projects'
    },
    
    'sentence-transformers/all-mpnet-base-v2': {
        'quality': 'Excellent',
        'speed': 'Medium',
        'memory': 'Medium',
        'best_for': 'General sentence similarity'
    },
    
    'BAAI/bge-large-en-v1.5': {
        'quality': 'Excellent',
        'speed': 'Medium',
        'memory': 'High',
        'best_for': 'State-of-the-art retrieval'
    },
    
    'sentence-transformers/msmarco-distilbert-base-v4': {
        'quality': 'Very Good',
        'speed': 'Fast',
        'memory': 'Medium',
        'best_for': 'Search and retrieval tasks'
    }
}

def apply_model_config(config_dict):
    """Apply a model configuration to properties"""
    import properties as props
    
    for key, value in config_dict.items():
        setattr(props, key, value)
    
    print(f"Applied configuration for {config_dict['VECTORISER_MODEL']}")
    if 'BERT_MODEL_NAME' in config_dict:
        print(f"BERT model: {config_dict['BERT_MODEL_NAME']}")
    elif 'LLM_MODEL_NAME' in config_dict:
        print(f"LLM model: {config_dict['LLM_MODEL_NAME']}")

# Example usage:
"""
import properties as props
from model_configurations import BGE_LARGE_CONFIG, apply_model_config

# Apply BGE configuration
apply_model_config(BGE_LARGE_CONFIG)

# Run experiment
# python vectoriser.py
"""