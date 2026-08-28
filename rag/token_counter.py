"""
rag/token_counter.py
────────────────────
Token counter and conservative daily quota estimator for the RAG system.

Features:
- Fast, dependency-free token estimation (~4 characters per token heuristic or exact word splitting).
- Per-query token accounting (Input tokens, Context tokens, Output tokens, Total tokens).
- Session-wide and daily quota tracking against Gemini Pro (50 RPD) & Gemini Flash (1500 RPD) limits.
- Tracks tokens saved via Local Embeddings (MiniLM) and Semantic Cache Hits.
"""

from typing import Dict, Any, Optional
from config import (
    GEMINI_PRO_DAILY_REQUEST_CAP,
    GEMINI_FLASH_DAILY_REQUEST_CAP,
    MAX_ESTIMATED_TOKENS_PER_QUERY,
)


class TokenTracker:
    """
    Tracks token consumption, estimated costs ($0.00 on free tier),
    and progress against daily rate limits and quotas.
    """

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_queries = 0
        self.total_planner_calls = 0
        self.total_eval_calls = 0
        self.total_compress_calls = 0
        self.saved_cache_tokens = 0
        self.saved_local_embed_tokens = 0

    @staticmethod
    def estimate_tokens(text: Optional[str]) -> int:
        """
        Conservative token count estimation.
        Standard heuristic: ~4 characters per token for English text,
        or max(len(words) * 1.3, len(chars) / 4).
        """
        if not text:
            return 0
        char_count = len(text)
        word_count = len(text.split())
        # Conservative estimate (takes upper bound)
        return max(int(word_count * 1.33), int(char_count / 3.8))

    def record_query(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        task: str = "answer",
    ) -> Dict[str, int]:
        """
        Records token usage for a single operation and returns the token delta.
        """
        input_tokens = self.estimate_tokens(prompt) + self.estimate_tokens(context)
        output_tokens = self.estimate_tokens(response)
        total = input_tokens + output_tokens

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if task == "answer":
            self.total_queries += 1
        elif task == "decompose":
            self.total_planner_calls += 1
        elif task == "judge":
            self.total_eval_calls += 1
        elif task == "compress":
            self.total_compress_calls += 1

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
        }

    def record_cache_hit(self, saved_query: str, saved_answer: str, saved_context: Optional[str] = None) -> int:
        """Records tokens saved by a semantic cache hit."""
        saved = self.estimate_tokens(saved_query) + self.estimate_tokens(saved_answer) + self.estimate_tokens(saved_context)
        self.saved_cache_tokens += saved
        return saved

    def record_local_embedding(self, texts: list[str]) -> int:
        """Records tokens processed locally (0 API cost)."""
        tokens = sum(self.estimate_tokens(t) for t in texts)
        self.saved_local_embed_tokens += tokens
        return tokens

    def get_summary(self) -> Dict[str, Any]:
        """Returns cumulative token stats and quota usage percentages."""
        total_tokens = self.total_input_tokens + self.total_output_tokens
        total_requests = (
            self.total_queries
            + self.total_planner_calls
            + self.total_eval_calls
            + self.total_compress_calls
        )

        pro_pct = round((total_requests / max(1, GEMINI_PRO_DAILY_REQUEST_CAP)) * 100, 1)
        flash_pct = round((total_requests / max(1, GEMINI_FLASH_DAILY_REQUEST_CAP)) * 100, 1)

        return {
            "total_tokens": total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": total_requests,
            "queries_count": self.total_queries,
            "pro_quota_used_pct": min(100.0, pro_pct),
            "flash_quota_used_pct": min(100.0, flash_pct),
            "pro_remaining_requests": max(0, GEMINI_PRO_DAILY_REQUEST_CAP - total_requests),
            "flash_remaining_requests": max(0, GEMINI_FLASH_DAILY_REQUEST_CAP - total_requests),
            "saved_cache_tokens": self.saved_cache_tokens,
            "saved_local_embed_tokens": self.saved_local_embed_tokens,
            "estimated_cost_usd": 0.0,
        }
