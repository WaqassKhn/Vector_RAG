"""
rag/cache.py
────────────
In-memory semantic answer cache for the RAG system.

Skips the full retrieval + generation pipeline when the incoming query is
semantically very similar (cosine similarity >= threshold) to a previously
answered query. Session-scoped — does not persist across server restarts.

Usage:
    cache = SemanticAnswerCache()
    hit = cache.lookup(query_embedding)
    if hit:
        return hit["answer"]
    # ... run pipeline ...
    cache.store(query_embedding, answer, citations)
"""

import logging
from typing import Optional

import numpy as np

from config import CACHE_SIMILARITY_THRESHOLD, CACHE_MAX_ENTRIES

logger = logging.getLogger(__name__)


class SemanticAnswerCache:
    """
    FIFO in-memory cache keyed by query embeddings.
    Returns cached answers for semantically identical queries.
    """

    def __init__(
        self,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
        max_entries: int = CACHE_MAX_ENTRIES,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._cache: list[dict] = []   # [{query_embedding, answer, citations, query_text}]
        self._hit_count: int = 0
        self._miss_count: int = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D float32 vectors."""
        if a.shape != b.shape:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def lookup(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
    ) -> Optional[dict]:
        """
        Returns cached result if a stored query has cosine similarity >= threshold.
        Returns None on cache miss.
        """
        if not self._cache:
            self._miss_count += 1
            return None

        qe = query_embedding.flatten().astype(np.float32)

        best_sim = 0.0
        best_entry = None
        for entry in self._cache:
            sim = self._cosine_similarity(qe, entry["query_embedding"])
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self.similarity_threshold:
            self._hit_count += 1
            logger.debug(f"[Cache] HIT (similarity={best_sim:.4f}): '{best_entry['query_text'][:60]}...'")
            return {
                "answer": best_entry["answer"],
                "citations": best_entry["citations"],
                "cache_hit": True,
                "similarity": round(best_sim, 4),
                "cache_similarity": round(best_sim, 4),
                "matched_query": best_entry.get("query_text", ""),
            }

        self._miss_count += 1
        return None

    def store(
        self,
        query_embedding: np.ndarray,
        answer: str,
        citations: list[str],
        query_text: str = "",
    ) -> None:
        """Stores a query-answer pair. Evicts oldest entry if at capacity."""
        if len(self._cache) >= self.max_entries:
            self._cache.pop(0)   # FIFO eviction

        self._cache.append({
            "query_embedding": query_embedding.flatten().astype(np.float32),
            "answer": answer,
            "citations": citations,
            "query_text": query_text,
        })

    def get_hit_count(self) -> int:
        return self._hit_count

    def get_miss_count(self) -> int:
        return self._miss_count

    def get_stats(self) -> dict:
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0
        return {
            "entries": len(self._cache),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate_pct": round(hit_rate * 100, 1),
        }

    def clear(self) -> None:
        self._cache = []
        self._hit_count = 0
        self._miss_count = 0
