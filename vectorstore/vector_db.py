import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from config import VECTOR_DB_DIR

class VectorDatabase:
    """
    FAISS-backed Vector Database managing document chunk embeddings and metadata storage.
    Supports cosine similarity search and index persistence.
    """

    def __init__(self, vector_dir: Path = VECTOR_DB_DIR):
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.vector_dir / "faiss.index"
        self.metadata_path = self.vector_dir / "metadata.json"
        
        self.index: faiss.Index = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.dimension: int = 0

    def initialize_new_index(self, dimension: int):
        """Initializes a new FAISS Inner Product index (Cosine similarity on normalized vectors)."""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks_metadata = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adds chunk metadata and embeddings to the FAISS index."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if self.index is None:
            self.initialize_new_index(dimension=embeddings.shape[1])

        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 15) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes semantic similarity search.
        Returns list of (chunk_metadata, similarity_score).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.chunks_metadata):
                results.append((self.chunks_metadata[idx], float(score)))

        return results

    def save_index(self):
        """Saves FAISS index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks_metadata, f, indent=2, ensure_ascii=False)

    def load_index(self) -> bool:
        """Loads FAISS index and metadata from disk if present."""
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)
            self.dimension = self.index.d
            return True
        return False

    def clear(self):
        """Resets vector index and metadata."""
        if self.index is not None:
            self.index.reset()
        self.chunks_metadata = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
