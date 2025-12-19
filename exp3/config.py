"""
Configuration file for RAG Research Experiment
Defines all experiment variants and parameters
"""

# Database Configuration
DB_PATH = 'sonar.db'

# Qdrant Configuration
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
COLLECTION_PREFIX = 'rag_exp'

# Embedding Configuration
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'  # Default model (used when --model not specified)

# Available Embedding Models for multi-model experiments
# Usage: python etl_pipeline.py --model bge-large
EMBEDDING_MODELS = {
    'bge-small': {
        'name': 'BAAI/bge-small-en-v1.5',
        'dim': 384,
        'description': 'BGE Small - Fast, lightweight',
        'trust_remote_code': False
    },
    'bge-large': {
        'name': 'BAAI/bge-large-en-v1.5',
        'dim': 1024,
        'description': 'BGE Large - Better quality',
        'trust_remote_code': False
    },
    'bge-m3': {
        'name': 'BAAI/bge-m3',
        'dim': 1024,
        'description': 'BGE M3 - Multilingual, long context',
        'trust_remote_code': True
    },
    'gte-qwen2': {
        'name': 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
        'dim': 1536,
        'description': 'GTE Qwen2 1.5B - High quality, Qwen-based',
        'trust_remote_code': True
    },
    'nomic-embed': {
        'name': 'nomic-ai/nomic-embed-text-v1.5',
        'dim': 768,
        'description': 'Nomic Embed - Good quality, efficient',
        'trust_remote_code': True
    },
    'gte-large': {
        'name': 'thenlper/gte-large',
        'dim': 1024,
        'description': 'GTE Large - Strong on technical text',
        'trust_remote_code': False
    },
    'e5-large': {
        'name': 'intfloat/e5-large-v2',
        'dim': 1024,
        'description': 'E5 Large - Microsoft, strong general',
        'trust_remote_code': False
    }
}

def get_model_config(model_key: str = None) -> dict:
    """Get model configuration by key. Returns default if key is None."""
    if model_key is None:
        return {'name': EMBEDDING_MODEL, 'dim': 384, 'trust_remote_code': False}
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(EMBEDDING_MODELS.keys())}")
    return EMBEDDING_MODELS[model_key]

# Experiment Variants
# RQ2 & RQ3: Data Source Variants
SOURCE_VARIANTS = {
    'title': {
        'name': 'TITLE',
        'description': 'Task title only',
        'fields': ['TITLE']
    },
    'desc': {
        'name': 'TITLE + DESCRIPTION',
        'description': 'Task title and description',
        'fields': ['TITLE', 'DESCRIPTION']
    },
    'comments': {
        'name': 'TITLE + DESCRIPTION + COMMENTS',
        'description': 'Task title, description, and comments',
        'fields': ['TITLE', 'DESCRIPTION', 'COMMENTS']
    }
}

# RQ1: Target Granularity Variants
TARGET_VARIANTS = {
    'file': {
        'name': 'FILE',
        'description': 'Individual file level',
        'extractor': 'file'
    },
    'module': {
        'name': 'MODULE',
        'description': 'Root folder level',
        'extractor': 'module'
    }
}

# RQ4: Time Window Variants
WINDOW_VARIANTS = {
    'w100': {
        'name': 'NEAREST 100',
        'description': 'Train on last 100 tasks before test',
        'size': 100
    },
    'w1000': {
        'name': 'NEAREST 1000',
        'description': 'Train on last 1000 tasks before test',
        'size': 1000
    },
    'all': {
        'name': 'ALL',
        'description': 'Train on all available history',
        'size': None
    }
}

# Test Strategy Variants
SPLIT_STRATEGIES = {
    'recent': {
        'name': 'Recent Split',
        'description': 'Test on most recent N tasks'
    },
    'modn': {
        'name': 'ModN Split',
        'description': 'Test on every k-th task (uniform sampling)'
    }
}

# Evaluation Parameters
TEST_SIZE = 200
TOP_K_VALUES = [1, 3, 5, 10]
DEFAULT_TOP_K = 10

# Batch Processing
BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 100

# Output Files
TEST_SET_FILE = 'test_set.json'
EXPERIMENT_RESULTS_FILE = 'experiment_results.csv'
LOG_FILE = 'experiment.log'
