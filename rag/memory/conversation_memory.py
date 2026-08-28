"""
rag/memory/conversation_memory.py
──────────────────────────────────
Per-session working conversation memory with LLM-based compression.

Keeps the last N turns verbatim and compresses older turns into a summary
paragraph when the total turn count exceeds a configurable threshold.
"""

import logging
from typing import Optional, TYPE_CHECKING

from config import MEMORY_MAX_TURNS_BEFORE_COMPRESS, MEMORY_RECENT_TURNS_IN_PROMPT

if TYPE_CHECKING:
    from rag.openrouter_llm import OpenRouterLLM

logger = logging.getLogger(__name__)

_COMPRESS_SYSTEM_PROMPT = (
    "You are a concise summarizer. Summarize the following conversation history "
    "into a single dense paragraph preserving all key facts, figures, and topics. "
    "Do not add commentary or opinions."
)


class ConversationMemory:
    """
    Conversation history manager with automatic LLM-based compression.

    Keeps `recent_turns` pairs verbatim in _recent_turns.
    When total turn count exceeds `max_turns_before_compress`, the excess
    older turns are compressed into _summary using the LLM.
    """

    def __init__(
        self,
        llm: Optional["OpenRouterLLM"] = None,
        max_turns_before_compress: int = MEMORY_MAX_TURNS_BEFORE_COMPRESS,
        recent_turns: int = MEMORY_RECENT_TURNS_IN_PROMPT,
    ):
        self.llm = llm
        self.max_turns_before_compress = max_turns_before_compress
        self.recent_turns = recent_turns

        # Internal state
        self._recent_turns: list[dict] = []    # [{"role": "user"|"assistant", "content": str}]
        self._summary: str = ""                # compressed summary of older turns
        self._total_turn_count: int = 0        # total turns ever added (including compressed)

    # ─── Public API ───────────────────────────────────────────────────────────

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Appends a user+assistant turn. Triggers compression when threshold exceeded."""
        self._recent_turns.append({"role": "user", "content": user_msg})
        self._recent_turns.append({"role": "assistant", "content": assistant_msg})
        self._total_turn_count += 1

        # Each "turn" = 2 entries (user + assistant)
        max_entries = self.max_turns_before_compress * 2
        if len(self._recent_turns) > max_entries:
            self._compress_old_turns()

    def get_context_string(self) -> str:
        """
        Returns a formatted string to inject into the RAG prompt before the
        context chunks. Includes the compressed summary (if any) and recent turns.
        """
        parts = []

        if self._summary:
            parts.append(
                "[CONVERSATION SUMMARY — earlier turns compressed]:\n"
                + self._summary
            )

        if self._recent_turns:
            recent_lines = []
            for entry in self._recent_turns[-(self.recent_turns * 2):]:
                role = "User" if entry["role"] == "user" else "Assistant"
                recent_lines.append(f"{role}: {entry['content']}")
            parts.append("[RECENT CONVERSATION]:\n" + "\n".join(recent_lines))

        if not parts:
            return ""

        return "\n\n".join(parts)

    def get_summary(self) -> str:
        """Returns the compressed summary of older turns (shown in UI sidebar)."""
        return self._summary

    def get_turn_count(self) -> int:
        """Total number of turns ever processed (including compressed ones)."""
        return self._total_turn_count

    def get_recent_turn_count(self) -> int:
        """Number of turns currently in the verbatim recent buffer."""
        return len(self._recent_turns) // 2

    def clear(self) -> None:
        """Resets all memory state."""
        self._recent_turns = []
        self._summary = ""
        self._total_turn_count = 0
        logger.info("[Memory] Conversation memory cleared.")

    # ─── Internal: compression ────────────────────────────────────────────────

    def _compress_old_turns(self) -> None:
        """
        Compresses turns older than the recent_turns window into _summary.
        Keeps the most recent `recent_turns` pairs verbatim.
        """
        keep_entries = self.recent_turns * 2
        entries_to_compress = self._recent_turns[:-keep_entries]
        self._recent_turns = self._recent_turns[-keep_entries:]

        if not entries_to_compress:
            return

        dialogue_lines = []
        for entry in entries_to_compress:
            role = "User" if entry["role"] == "user" else "Assistant"
            dialogue_lines.append(f"{role}: {entry['content']}")
        dialogue_text = "\n".join(dialogue_lines)

        if self._summary:
            dialogue_text = f"[Previous summary]: {self._summary}\n\n[New turns to incorporate]:\n{dialogue_text}"

        if self.llm and self.llm.is_available():
            try:
                new_summary = self.llm.generate(
                    prompt=f"Conversation history to summarize:\n\n{dialogue_text}",
                    task="compress",
                    system_instruction=_COMPRESS_SYSTEM_PROMPT,
                    temperature=0.0,
                )
                self._summary = new_summary.strip()
                logger.info(
                    f"[Memory] Compressed {len(entries_to_compress)//2} turns into summary "
                    f"({len(self._summary)} chars)."
                )
            except Exception as exc:
                logger.warning(f"[Memory] Compression failed: {exc}. Keeping raw older turns as text summary.")
                self._summary = dialogue_text[:1500]
        else:
            self._summary = dialogue_text[:1500]
            logger.warning("[Memory] LLM unavailable for compression. Using truncated raw text.")
