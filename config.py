import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
UPLOADS_DIR = DATA_DIR / "uploads"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ─── SQLite Persistent Database ───────────────────────────────────────────────
DB_PATH = DATA_DIR / "rag_app.db"
MEMORY_DIR = DATA_DIR / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ─── Google Gemini — last-resort fallback ───────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
EMBEDDING_GEMINI_MODEL = "text-embedding-004"

# ─── Local Embedding Model ───────────────────────────────────────────────────
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # MiniLM-L6-v2 output dimension — must match Pinecone index

# ─── Pinecone — Primary Vector Database ─────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-ntpc")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ─── OpenRouter — Primary LLM Provider (free :free models) ──────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_SITE_URL = "https://github.com/WaqassKhn/Vector_RAG"
OPENROUTER_APP_NAME = "RAG-NTPC"

# Task → model priority lists (index 0 = highest priority).
# The OpenRouterLLM class verifies these against the live free model list at startup
# and skips any that are currently unavailable. Update this list as OpenRouter's
# free model catalog changes (check https://openrouter.ai/models?q=:free).
OPENROUTER_MODELS: dict[str, list[str]] = {
    "answer": [                              # Main answer generation — needs a capable model
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen3-8b:free",
    ],
    "decompose": [                           # Query planning — speed > raw quality
        "meta-llama/llama-3.1-8b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "qwen/qwen3-8b:free",
        "mistralai/mistral-7b-instruct:free",
    ],
    "judge": [                               # LLM-as-judge grounding evaluation
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen3-8b:free",
    ],
    "compress": [                            # Memory summarisation — small + fast
        "meta-llama/llama-3.1-8b-instruct:free",
        "qwen/qwen3-8b:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "mistralai/mistral-7b-instruct:free",
    ],
    "triage": [                              # Document scope selection
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen3-8b:free",
        "microsoft/phi-3-mini-128k-instruct:free",
    ],
}

# ─── Chunking Parameters ─────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120

# ─── Hybrid Retrieval Parameters ─────────────────────────────────────────────
INITIAL_TOP_K = 15
RERANKED_TOP_K = 5
RRF_K = 60   # Reciprocal Rank Fusion constant

# ─── Grounding Evaluation ────────────────────────────────────────────────────
GROUNDING_PASS_THRESHOLD = 0.70
NUMERICAL_TOLERANCE = 1e-4

# ─── Conversation Memory ─────────────────────────────────────────────────────
MEMORY_MAX_TURNS_BEFORE_COMPRESS = 8   # compress when total stored turns exceed this
MEMORY_RECENT_TURNS_IN_PROMPT = 5      # always inject this many recent turns verbatim

# ─── Semantic Answer Cache ───────────────────────────────────────────────────
CACHE_SIMILARITY_THRESHOLD = 0.95      # cosine similarity threshold for cache hit
CACHE_MAX_ENTRIES = 100                # FIFO eviction after this many entries

# ─── Daily Token Budget & Quota Controls (Conservative Estimates) ────────────
# Based on Gemini / Google Pro / Free Tier limits
GEMINI_PRO_DAILY_REQUEST_CAP = 50       # Conservative cap for Gemini Pro free tier (50 RPD)
GEMINI_FLASH_DAILY_REQUEST_CAP = 1500   # Cap for Gemini Flash free tier (1,500 RPD)
MAX_ESTIMATED_TOKENS_PER_QUERY = 3500   # Upper bound budget per simple grounded query
MAX_ESTIMATED_TOKENS_COMPLEX_QUERY = 8000 # Upper bound budget per decomposed query

# ─── Cognitive Memory Engine ─────────────────────────────────────────────────
EPISODIC_TIME_DECAY_LAMBDA = 0.01       # Exponential decay constant per hour (~24h halflife)
EPISODIC_SIMILARITY_THRESHOLD = 0.65    # Minimum cosine similarity for episodic recall
MAX_EPISODIC_RECORDS = 500              # Maximum episodes stored
MAX_SEMANTIC_FACTS = 1000               # Maximum domain facts stored

