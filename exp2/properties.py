# Configuration file for commit analysis

# Database settings
DATABASE_PATH = "data/flink.db"

# Text processing settings
USE_DESCRIPTION = False  # Include TASK.DESCRIPTION in analysis
USE_COMMENTS = False     # Include TASK.COMMENTS in analysis

# Word filtering settings
MIN_WORD_LENGTH = 1      # Minimum word length (set to 1 to include all words)
IGNORE_PURE_NUMBERS = True  # Ignore tokens that are pure numbers
IGNORE_PURE_SYMBOLS = True  # Ignore tokens that are pure symbols

# Module filtering settings
IGNORE_FILES_WITHOUT_ROOT_MODULE = True  # Skip files that don't have a root module

# Bradford ranking settings
BRADFORD_ZONES = 4  # Number of Bradford zones (typically 4)

# Debug settings
VERBOSE = True  # Print progress information

# Vectorization settings
NORMALIZE_VECTORS = True# Normalize all vectors to unit length
VECTOR_DIMENSION = 100      # Dimension of vectors (will be adjusted per model)

# NEW: Vectorization model selection
VECTORISER_MODEL = 'fast_text'# Options: 'own', 'fast_text', 'glove', 'bert', 'llm'
CLEAR_EMBEDDINGS = False# Set to False to reuse existing embeddings when model hasn't changed

# Word2Vec settings (for VECTORISER_MODEL = 'own')
W2V_WINDOW = 50          # Word2Vec window size
W2V_MIN_COUNT = 1       # Word2Vec minimum word count
W2V_EPOCHS = 10         # Word2Vec training epochs
W2V_SG = 0              # Word2Vec algorithm (0=CBOW, 1=Skip-gram)

# FastText settings (for VECTORISER_MODEL = 'fast_text')
FASTTEXT_MODEL_LANG = 'en'  # Language for pretrained FastText model
FASTTEXT_MODEL_DIM = 300    # Dimension of pretrained FastText model

# GloVe settings (for VECTORISER_MODEL = 'glove')
GLOVE_MODEL_NAME = 'glove.6B.100d'  # Options: glove.6B.50d, glove.6B.100d, glove.6B.200d, glove.6B.300d
GLOVE_CACHE_DIR = './glove_cache'   # Directory to cache GloVe models

# BERT settings (for VECTORISER_MODEL = 'bert')
BERT_MODEL_NAME = 'bert-base-uncased'  # BERT model name from HuggingFace
BERT_MAX_LENGTH = 512                  # Maximum sequence length for BERT
BERT_BATCH_SIZE = 16                   # Batch size for BERT processing
BERT_LAYER = -2                        # Which BERT layer to use (-1 = last, -2 = second to last)

# LLM settings (for VECTORISER_MODEL = 'llm')
LLM_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # LLM model for embeddings
LLM_MAX_LENGTH = 384                   # Maximum sequence length for LLM
LLM_BATCH_SIZE = 32                    # Batch size for LLM processing
LLM_DEVICE = 'auto'                    # Device: 'auto', 'cpu', 'cuda'
LLM_API_PROVIDER = None                # For API-based LLMs: 'openai', 'anthropic', etc.
LLM_API_KEY = None                     # API key if using API-based LLM
LLM_API_MODEL = 'text-embedding-ada-002'  # API model name

# Term filtering for vectorization
MIN_TERM_COUNT = 0      # Minimum term count for inclusion
MIN_HHI_ROOT = 0.0     # Minimum HHI_ROOT for inclusion

# Module vectorization strategy
# Options: 'avg', 'sum', 'median', 'weighted_avg', 'cluster'
MODULE_VECTOR_STRATEGY = 'sum'

# Distance metrics for similarity calculation
# Options: 'cosine', 'euclidean', 'manhattan'
DISTANCE_METRICS = ['cosine']

# Test settings
PREPROCESS_TEST_TASK = False# Filter test task text using HHI and count thresholds

# Statistical evaluation settings
SUMMARY_TEST_TASK_COUNT = 200  # Number of tasks to use for statistical evaluation

# Train/Test split settings
EXCLUDE_TEST_TASKS_FROM_MODEL = True# Exclude test tasks from training/analysis
EXCLUDE_TEST_TASKS_STRATEGY = 'modN'  # Strategy for selecting test tasks
# Possible values: 'lastN', 'firstN', 'middleN', 'modN'
# lastN - exclude N tasks with highest IDs
# firstN - exclude N tasks with lowest IDs
# middleN - exclude N tasks from the middle of ID range
# modN - exclude tasks where ID % target_count ≈ 0

# Visualization settings
VIZ_DEFAULT_OUTPUT = 'semantic_map.png'  # Default output file for visualization
VIZ_FIGURE_SIZE = (12, 8)                # Figure size in inches
VIZ_DPI = 300                            # Resolution for saved images
VIZ_TOP_MODULES_HIGHLIGHT = 5            # Number of top similar modules to highlight