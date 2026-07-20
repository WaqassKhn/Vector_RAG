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

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
EMBEDDING_GEMINI_MODEL = "text-embedding-004"

# Open-Source Local Embedding Model
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for MiniLM-L6-v2

# Chunking Parameters
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120

# Hybrid Retrieval Parameters
INITIAL_TOP_K = 15
RERANKED_TOP_K = 5
RRF_K = 60  # Reciprocal Rank Fusion constant

# Grounding Evaluation Parameters
GROUNDING_PASS_THRESHOLD = 0.70
NUMERICAL_TOLERANCE = 1e-4
