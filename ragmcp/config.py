"""
Configuration for RAG MCP Server
"""

# Vector Backend Selection
# Options: 'qdrant' or 'postgres'
VECTOR_BACKEND = 'postgres'  # Using PostgreSQL+pgvector for stability

# Qdrant Configuration
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333

# PostgreSQL Configuration (for pgvector backend)
POSTGRES_HOST = 'localhost'
POSTGRES_PORT = 5432
POSTGRES_DB = 'semantic_vectors'
POSTGRES_USER = 'postgres'
POSTGRES_PASSWORD = 'postgres'  # Change in production!
POSTGRES_SCHEMA = 'vectors'  # Schema for vector tables

# Collection names (should match exp3 ETL output)
# Format: rag_exp_{source}_{target}_{window}_{split}_{model}

# DUAL COLLECTION MODE: Use both RECENT and ALL collections for optimal results
# RECENT collections: High precision for current development work (last 100 tasks)
# ALL collections: Comprehensive coverage for finding older/rare functionality

# RECENT collections (last 100 tasks) - for current work
COLLECTION_MODULE_RECENT = 'rag_exp_desc_module_w100_modn_bge-small'
COLLECTION_FILE_RECENT = 'rag_exp_desc_file_w100_modn_bge-small'
COLLECTION_TASK_RECENT = 'task_embeddings_w100_bge-small'

# ALL collections (complete history) - for comprehensive coverage
COLLECTION_MODULE_ALL = 'rag_exp_desc_module_all_modn_bge-small'
COLLECTION_FILE_ALL = 'rag_exp_desc_file_all_modn_bge-small'
COLLECTION_TASK_ALL = 'task_embeddings_all_bge-small'

# Legacy single collection names (kept for backward compatibility)
COLLECTION_MODULE = COLLECTION_MODULE_RECENT  # Default to recent
COLLECTION_FILE = COLLECTION_FILE_RECENT      # Default to recent
COLLECTION_TASK = COLLECTION_TASK_ALL         # Default to all for task search

# Embedding Model (should match exp3 model)
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'  # Must match ETL model
VECTOR_SIZE = 384  # bge-small dimension

# LLM Configuration (Ollama)
OLLAMA_URL = 'http://localhost:11434'
OLLAMA_MODEL = 'qwen2.5-coder:1.5b'  # Default model for all agents

# Alternative models (change OLLAMA_MODEL to any of these):
# - 'gemma3:1b' (815 MB) - Fastest, best for testing
# - 'qwen2.5-coder:1.5b' (986 MB) - Fast, better for code
# - 'qwen2.5-coder:latest' (4.7 GB) - Best quality, slower
# - 'qwen2.5-coder:14b' - Excellent quality, needs 16GB RAM
# - 'codellama:7b' - Alternative coding model

# Search Configuration
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# Project configuration
PROJECTS = ['sonar', 'flink']  # Add more as needed

# Database and Code Retrieval
DB_PATH = '../data/sonar.db'  # Path to SQLite database with RAWDATA
CODE_ROOT = r'C:\Project\codeXplorer\capestone\repository\SONAR\sonarqube'  # Root directory for actual source code retrieval

# Security Configuration for Gradio UI
# Options: 'localhost', 'network', 'public'
GRADIO_ACCESS_MODE = 'localhost'  # Default: localhost only (most secure)

# Authentication (optional, uncomment to enable)
# GRADIO_USERNAME = 'admin'
# GRADIO_PASSWORD = 'your_secure_password_here'

# Server Configuration
GRADIO_PORT = 7860
GRADIO_SHARE = False  # Never set to True for sensitive data!
