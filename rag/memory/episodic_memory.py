"""
rag/memory/episodic_memory.py
──────────────────────────────
Episodic Memory Engine with Time-Decay Vector Retrieval.

Stores user interaction episodes across sessions with timestamps, citations,
and query embeddings in the persistent SQLite database.

Retrieval uses exponential time-decayed cosine similarity:
    Score(q, e) = cosine_similarity(q_vec, e_vec) * exp(-lambda * delta_t_hours)

Ensures recent, relevant past interactions are prioritized while older interactions
gracefully fade unless strongly semantically aligned.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np

from config import (
    EPISODIC_TIME_DECAY_LAMBDA,
    EPISODIC_SIMILARITY_THRESHOLD,
    MAX_EPISODIC_RECORDS,
)
from database.db_manager import DatabaseManager, get_db

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D float arrays."""
    if vec_a.shape != vec_b.shape:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class EpisodicMemory:
    """
    Persistent episodic memory store with time-decayed similarity retrieval.
    Backed by SQLite via DatabaseManager.
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        time_decay_lambda: float = EPISODIC_TIME_DECAY_LAMBDA,
        similarity_threshold: float = EPISODIC_SIMILARITY_THRESHOLD,
        max_records: int = MAX_EPISODIC_RECORDS,
    ):
        self.db = db or get_db()
        self.time_decay_lambda = time_decay_lambda
        self.similarity_threshold = similarity_threshold
        self.max_records = max_records

    def record_episode(
        self,
        query: str,
        answer: str,
        citations: Optional[List[str]] = None,
        query_embedding: Optional[Union[np.ndarray, List[float]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Records an interaction episode to persistent SQLite storage.
        """
        emb_list = []
        if query_embedding is not None:
            if isinstance(query_embedding, np.ndarray):
                emb_list = query_embedding.flatten().tolist()
            else:
                emb_list = list(query_embedding)

        # Truncate answer if very long to preserve clean episodic memory
        clean_answer = answer[:1200] + ("..." if len(answer) > 1200 else "")

        episode_id = self.db.save_episode(
            query=query.strip(),
            answer=clean_answer.strip(),
            citations=citations or [],
            embedding=emb_list,
            session_id=session_id,
        )
        logger.info(f"[EpisodicMemory] Recorded episode '{episode_id[:8]}' (query: {query[:40]}...)")
        return episode_id

    def search_episodes(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        top_k: int = 3,
        threshold: Optional[float] = None,
        current_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant past interaction episodes using
        time-decayed cosine similarity.

        Returns list of dicts: {
            "episode": episode_dict,
            "raw_similarity": float,
            "decay_factor": float,
            "time_decayed_score": float,
            "age_hours": float,
        }
        """
        if isinstance(query_embedding, list):
            q_vec = np.array(query_embedding, dtype=np.float32)
        else:
            q_vec = query_embedding.flatten().astype(np.float32)

        min_score = threshold if threshold is not None else self.similarity_threshold
        episodes = self.db.get_all_episodes(limit=self.max_records)
        if not episodes:
            return []

        now_utc = datetime.now(timezone.utc)
        scored_episodes = []

        for ep in episodes:
            # Skip current session's immediate questions if needed (handled by working memory)
            if current_session_id and ep.get("session_id") == current_session_id:
                continue

            emb_raw = ep.get("embedding")
            if not emb_raw or len(emb_raw) == 0:
                continue

            e_vec = np.array(emb_raw, dtype=np.float32)
            raw_sim = _cosine_similarity(q_vec, e_vec)

            # Calculate time difference in hours
            try:
                ep_time = datetime.fromisoformat(ep["timestamp"])
                if ep_time.tzinfo is None:
                    ep_time = ep_time.replace(tzinfo=timezone.utc)
                age_hours = max(0.0, (now_utc - ep_time).total_seconds() / 3600.0)
            except Exception:
                age_hours = 0.0

            decay = math.exp(-self.time_decay_lambda * age_hours)
            decayed_score = raw_sim * decay

            if decayed_score >= min_score:
                scored_episodes.append({
                    "episode": ep,
                    "raw_similarity": round(raw_sim, 4),
                    "decay_factor": round(decay, 4),
                    "time_decayed_score": round(decayed_score, 4),
                    "age_hours": round(age_hours, 1),
                })

        # Sort descending by time-decayed score
        scored_episodes.sort(key=lambda x: x["time_decayed_score"], reverse=True)
        return scored_episodes[:top_k]

    def get_context_string(
        self,
        query_embedding: Optional[Union[np.ndarray, List[float]]] = None,
        top_k: int = 2,
        current_session_id: Optional[str] = None,
    ) -> str:
        """
        Formats top relevant past episodes into a prompt context section.
        """
        if query_embedding is None:
            return ""

        matches = self.search_episodes(
            query_embedding=query_embedding,
            top_k=top_k,
            current_session_id=current_session_id,
        )
        if not matches:
            return ""

        blocks = ["[RELEVANT PAST INTERACTIONS (Episodic Memory)]:"]
        for match in matches:
            ep = match["episode"]
            age_str = f"{match['age_hours']:.0f}h ago" if match["age_hours"] < 48 else f"{match['age_hours']/24:.1f}d ago"
            blocks.append(
                f"- Past Query ({age_str}, relevance {match['time_decayed_score']:.2f}): \"{ep['query']}\"\n"
                f"  Key Answer Summary: {ep['answer'][:250]}..."
            )
        return "\n".join(blocks)

    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent episodes for UI inspection."""
        return self.db.get_all_episodes(limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        """Returns episodic memory statistics."""
        episodes = self.db.get_all_episodes(limit=self.max_records)
        return {
            "total_episodes": len(episodes),
            "decay_lambda": self.time_decay_lambda,
            "similarity_threshold": self.similarity_threshold,
        }

    def clear(self) -> None:
        """Clears all episodic memory records."""
        self.db.clear_episodes()
        logger.info("[EpisodicMemory] Cleared all episodes.")
