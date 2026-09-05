"""
rag/memory/cognitive_hub.py
───────────────────────────
CognitiveMemoryHub — Unified Coordinator for Multi-Tier Cognitive Memory.

Synchronizes and orchestrates:
  1. Working Memory: Short-term verbatim turn buffer + periodic LLM compression (ConversationMemory)
  2. Episodic Memory: Cross-session interaction recall with time-decayed vector search (EpisodicMemory)
  3. Semantic Memory: Persistent user preferences and domain entity fact graphs (SemanticMemory)
  4. Procedural Memory: Domain execution workflows and task-specific reasoning recipes (ProceduralMemory)

Provides a unified interface for prompt context formulation and post-interaction learning.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from database.db_manager import DatabaseManager, get_db
from rag.memory.conversation_memory import ConversationMemory
from rag.memory.episodic_memory import EpisodicMemory
from rag.memory.procedural_memory import ProceduralMemory
from rag.memory.semantic_memory import SemanticMemory

logger = logging.getLogger(__name__)


class CognitiveMemoryHub:
    """
    Unified cognitive memory coordinator integrating Working, Episodic,
    Semantic, and Procedural memory tiers.
    """

    def __init__(
        self,
        llm=None,
        db: Optional[DatabaseManager] = None,
    ):
        self.db = db or get_db()
        self.llm = llm

        # Tier 1: Working Memory (Short-Term & Compressed Conversation)
        self.working_memory = ConversationMemory(llm=self.llm)

        # Tier 2: Episodic Memory (Time-decayed cross-session recall)
        self.episodic_memory = EpisodicMemory(db=self.db)

        # Tier 3: Semantic Memory (User preferences & domain facts)
        self.semantic_memory = SemanticMemory(db=self.db)

        # Tier 4: Procedural Memory (Task execution recipes)
        self.procedural_memory = ProceduralMemory(db=self.db)

    # ─── Unified Prompt Context Assembly ──────────────────────────────────────

    def build_cognitive_context(
        self,
        query: str,
        query_embedding: Optional[Union[np.ndarray, List[float]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Builds a compact, high-value cognitive memory context block
        for injection into the QueryPlanner and RAGChain generation prompts.
        """
        sections = []

        # 1. Procedural Memory: Matched Task Recipe
        recipe_context = self.procedural_memory.get_context_string(query)
        if recipe_context:
            sections.append(recipe_context)

        # 2. Semantic Memory: User Preferences & Domain Facts
        semantic_context = self.semantic_memory.get_context_string(query)
        if semantic_context:
            sections.append(semantic_context)

        # 3. Episodic Memory: Relevant Past Session Episodes (Time-decayed)
        if query_embedding is not None:
            episodic_context = self.episodic_memory.get_context_string(
                query_embedding=query_embedding,
                top_k=2,
                current_session_id=session_id,
            )
            if episodic_context:
                sections.append(episodic_context)

        # 4. Working Memory: Recent Dialogue & Summary
        working_context = self.working_memory.get_context_string()
        if working_context:
            sections.append(working_context)

        if not sections:
            return ""

        return "\n\n".join(sections)

    def build_augmented_prompt_context(
        self,
        query: str,
        query_embedding: Optional[Union[np.ndarray, List[float]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Alias for build_cognitive_context."""
        return self.build_cognitive_context(
            query=query,
            query_embedding=query_embedding,
            session_id=session_id,
        )

    # ─── Post-Interaction Lifecycle ───────────────────────────────────────────

    def post_interaction_update(
        self,
        query: str,
        answer: str,
        citations: Optional[List[str]] = None,
        query_embedding: Optional[Union[np.ndarray, List[float]]] = None,
        session_id: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
        task: str = "answer",
    ) -> None:
        """
        Executes post-interaction learning across all cognitive memory tiers:
          - Updates Working Memory with new turn.
          - Records interaction Episode into Episodic Memory.
          - Extracts potential preferences / domain facts into Semantic Memory.
          - Logs token usage into persistent SQLite DB.
        """
        # 1. Working Memory turn update
        self.working_memory.add_turn(user_msg=query, assistant_msg=answer)

        # 2. Episodic Memory episode record
        try:
            self.episodic_memory.record_episode(
                query=query,
                answer=answer,
                citations=citations or [],
                query_embedding=query_embedding,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning(f"[CognitiveHub] Failed to record episode: {exc}")

        # 3. Semantic Memory extraction
        try:
            self.semantic_memory.extract_and_update(user_msg=query, assistant_msg=answer)
        except Exception as exc:
            logger.warning(f"[CognitiveHub] Semantic extraction warning: {exc}")

        # 4. Token logging
        if prompt_tokens > 0 or completion_tokens > 0:
            try:
                self.db.log_token_usage(
                    session_id=session_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=model,
                    task=task,
                )
            except Exception as exc:
                logger.warning(f"[CognitiveHub] Token logging error: {exc}")

    # ─── Metrics & Dashboard Summary ──────────────────────────────────────────

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Returns consolidated metrics across all 4 cognitive memory tiers."""
        return {
            "working_memory": {
                "recent_turns": self.working_memory.get_recent_turn_count(),
                "total_turns": self.working_memory.get_turn_count(),
                "has_compressed_summary": bool(self.working_memory.get_summary()),
            },
            "episodic_memory": self.episodic_memory.get_stats(),
            "semantic_memory": self.semantic_memory.get_stats(),
            "procedural_memory": self.procedural_memory.get_stats(),
            "token_metrics": self.db.get_token_usage_summary(),
        }

    def clear_all(self) -> None:
        """Clears working, episodic, and semantic memory (resets procedural to defaults)."""
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.procedural_memory.clear()
        logger.info("[CognitiveHub] Cleared all memory tiers.")
