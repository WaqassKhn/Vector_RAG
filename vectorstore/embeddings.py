import numpy as np
from typing import List, Union
from config import DEFAULT_LOCAL_EMBEDDING_MODEL, EMBEDDING_GEMINI_MODEL, GEMINI_API_KEY

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False


class EmbeddingManager:
    """
    Embedding manager providing local HuggingFace sentence-transformers (free)
    and optional Google Gemini embedding integration.
    """

    def __init__(self, use_gemini: bool = False, model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL, api_key: str = None):
        self.use_gemini = use_gemini
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.local_model = None
        self.gemini_client = None

        if self.use_gemini:
            if not HAS_GEMINI_SDK:
                raise ImportError("google-genai package is required for Gemini Embeddings.")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is required to use Gemini Embeddings.")
            self.gemini_client = genai.Client(api_key=self.api_key)
        else:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers package is required for local embeddings.")
            self.local_model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of text strings into normalized floating-point vectors.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        if self.use_gemini:
            embeddings = []
            # Batch embedding requests
            for text in texts:
                res = self.gemini_client.models.embed_content(
                    model=EMBEDDING_GEMINI_MODEL,
                    contents=text
                )
                embeddings.append(res.embedding.values)
            vecs = np.array(embeddings, dtype=np.float32)
        else:
            vecs = self.local_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            vecs = vecs.astype(np.float32)

        # L2 Normalize vectors for Cosine Similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_vecs = vecs / norms
        return normalized_vecs

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single search query string."""
        res = self.embed_texts([query])
        return res[0]
