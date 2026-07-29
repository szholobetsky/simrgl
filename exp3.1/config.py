"""
Configuration file for RAG Research Experiment
Defines all experiment variants and parameters
"""

import hashlib

# Directory containing one SQLite database per project (built by
# codeXplorer). See db/project_stats.csv (repo root) for an inventory of
# what's in each one, including which commits-table name it uses.
PROJECTS_DIR = '/home/stzh/Projects/db'

# Active project - must match <PROJECTS_DIR>/<PROJECT>.db
# Override per run with --project on etl_pipeline.py / create_task_collection.py
PROJECT = 'sonar'

# Task-unit criterion: 'ticket' (TASK table, current default) or 'commit'
# (synthetic per-commit task built from RAWDATA/COMMITS.MESSAGE - see
# ETLPipeline.load_data_commit_mode()). Override with --task-unit.
TASK_UNIT = 'ticket'
TASK_UNITS = ['ticket', 'commit']

# Database Configuration
# The commits table is named RAWDATA in older databases and COMMITS in
# newer ones (see utils.get_commits_table_name) - same columns either way,
# so DB_PATH just needs to point at the right file.
DB_PATH = f'{PROJECTS_DIR}/{PROJECT}.db'

# Vector Backend Selection
# Options: 'qdrant' or 'postgres'
VECTOR_BACKEND = 'postgres'  # Options: 'postgres' (PostgreSQL+pgvector) or 'qdrant' (Qdrant vector DB)

# Qdrant Configuration
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
COLLECTION_PREFIX = 'rag_exp'


def collection_name(source, target, window, split, model_key=None, project=None, task_unit=None):
    """
    Build a project+task_unit-namespaced collection name.

    Centralizing this avoids the naming pattern drifting out of sync
    across etl_pipeline.py / run_comprehensive_experiments.py /
    run_experiments.py - all of which previously built this string
    independently with no project dimension, so two projects could
    silently overwrite each other's Postgres/Qdrant collection.
    """
    project = project or PROJECT
    task_unit = task_unit or TASK_UNIT
    model_suffix = f"_{model_key}" if model_key else ""
    base = f"{COLLECTION_PREFIX}_{project}_{task_unit}_{source}_{target}_{window}_{split}{model_suffix}"

    # PostgreSQL silently truncates identifiers at 63 bytes (NAMEDATALEN),
    # and vector_backends.py appends "_vector_idx" (11 chars) to this name
    # for the HNSW index - so budget for that, not just the 63-byte table
    # name limit, or two different variants (e.g. differing only in window
    # or model, which sort near the end of the string) can truncate to the
    # identical relation name and collide with "already exists" (hit for
    # real on kubernetes/ticket/comments - the longest project+source combo).
    max_safe_len = 63 - len("_vector_idx")
    if len(base) > max_safe_len:
        digest = hashlib.md5(base.encode()).hexdigest()[:8]
        base = f"{base[:max_safe_len - len(digest) - 1]}_{digest}"
    return base


def task_collection_name(window, model_key=None, project=None):
    """Namespaced name for the standalone task-to-task collection (create_task_collection.py)."""
    project = project or PROJECT
    model_suffix = f"_{model_key}" if model_key else "_bge-small"
    return f"task_embeddings_{project}_{window}{model_suffix}"


# Task Embeddings Collection Name
TASK_COLLECTION = 'task_embeddings_all_bge-small'

# PostgreSQL Configuration (for pgvector backend)
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = 5432
POSTGRES_DB = 'semantic_vectors'
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'postgres'  # Change in production!
POSTGRES_SCHEMA = 'vectors'  # Schema for vector tables

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
    },
    # commit-mode only (TASK_UNIT='commit'): COMMENTS is always '' there
    # (no comments concept for a raw commit), so 'comments' above is a
    # byte-identical duplicate of 'desc' in that mode. 'diff' fills the
    # same "extra noisy context" role that COMMENTS plays for tickets,
    # using each commit's actual diff content instead (see
    # ETLPipeline._build_commit_tasks).
    'diff': {
        'name': 'TITLE + DESCRIPTION + DIFF',
        'description': 'Commit subject+message plus diff content (commit-mode only)',
        'fields': ['TITLE', 'DESCRIPTION', 'DIFF']
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
