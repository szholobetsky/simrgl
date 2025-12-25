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
COLLECTION_MODULE = 'rag_exp_desc_module_w1000_modn_bge-small'  # TEST MODE - Last 1000 tasks
COLLECTION_FILE = 'rag_exp_desc_file_w1000_modn_bge-small'      # TEST MODE - Last 1000 tasks
COLLECTION_TASK = 'task_embeddings_all_bge-small'              # Individual task embeddings for task search

# Embedding Model (should match exp3 model)
EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'  # Must match ETL model
VECTOR_SIZE = 384  # bge-small dimension

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
