"""
rag/agents/merge_agent.py
─────────────────────────
MergeAgent — synthesizes answers from multiple RAGChain.run() results.

Used when the QueryPlannerAgent classifies a query as "complex" and the system
runs N separate retrieval passes (one per sub-query). The MergeAgent:
  1. Deduplicates retrieved chunks by chunk_id across all sub-results
  2. Calls the LLM once with the merged context and the original query
  3. Returns a unified answer with cross-sub-query citations

Usage:
    merger = MergeAgent(llm=openrouter_llm)
    result = merger.merge(
        original_query="Compare Q1 and Q4 revenue",
        sub_results=[rag_result_1, rag_result_2],
    )
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.openrouter_llm import OpenRouterLLM

logger = logging.getLogger(__name__)

_MERGE_SYSTEM_PROMPT = """You are a strict, grounded AI assistant specializing in corporate and financial document analysis.

You have been given context retrieved from multiple targeted retrieval passes, each answering a different aspect of the user's question.

Rules:
1. Accuracy & Grounding: Every statement MUST be supported by the provided context chunks.
2. Numerical Precision: Quote exact figures as they appear. Do NOT round or extrapolate.
3. Citations: Cite sources inline as [Source: <Filename>, Page <N>] for every key claim.
4. Synthesis: Connect the answers from different retrieval passes into a single coherent response.
5. Missing Information: If any aspect cannot be answered from the context, state this explicitly.
"""


class MergeAgent:
    """
    Synthesizes answers from multiple RAGChain.run() outputs into a single
    coherent, grounded response with deduplicated context.
    """

    def __init__(self, llm: Optional["OpenRouterLLM"] = None):
        self.llm = llm

    def merge(
        self,
        original_query: str,
        sub_results: list[dict],
    ) -> dict:
        """
        Merges N RAGChain.run() output dicts into a single answer.

        Args:
            original_query: The original user question (not the sub-queries).
            sub_results: List of dicts from RAGChain.run(), each containing
                         "query", "reranked_chunks", "citations", "formatted_context".

        Returns:
            Dict with keys: query, answer, reranked_chunks, citations, sub_queries.
        """
        if not sub_results:
            return {
                "query": original_query,
                "answer": "No retrieval results to merge.",
                "reranked_chunks": [],
                "citations": [],
                "sub_queries": [],
            }

        # Single result — no merging needed
        if len(sub_results) == 1:
            result = sub_results[0]
            result["sub_queries"] = [result.get("query", original_query)]
            result["query"] = original_query
            return result

        # Deduplicate chunks by chunk_id
        seen_ids: set[str] = set()
        all_unique_chunks: list[dict] = []
        for sub in sub_results:
            for chunk in sub.get("reranked_chunks", []):
                cid = chunk.get("chunk_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    all_unique_chunks.append(chunk)

        # Collect all citations (deduped, preserving order)
        all_citations: list[str] = []
        seen_citations: set[str] = set()
        for sub in sub_results:
            for cit in sub.get("citations", []):
                if cit not in seen_citations:
                    seen_citations.add(cit)
                    all_citations.append(cit)

        # Build merged context string
        context_blocks = []
        for idx, chunk in enumerate(all_unique_chunks):
            citation_tag = f"Source: {chunk['filename']}, Page {chunk['page_number']}"
            block = f"--- CONTEXT CHUNK #{idx+1} [{citation_tag}] ---\n{chunk['text']}\n"
            context_blocks.append(block)

        merged_context = "\n\n".join(context_blocks)

        sub_query_labels = "\n".join(
            f"  {i+1}. {sub.get('query', '')}"
            for i, sub in enumerate(sub_results)
        )

        user_prompt = f"""CONTEXT CHUNKS (merged from multiple retrieval passes):
{merged_context}

This context was retrieved by breaking your question into these sub-queries:
{sub_query_labels}

ORIGINAL USER QUESTION:
{original_query}

GROUNDED SYNTHESIZED ANSWER WITH INLINE CITATIONS:"""

        # Generate merged answer
        if self.llm and self.llm.is_available():
            answer = self.llm.generate(
                prompt=user_prompt,
                task="answer",
                system_instruction=_MERGE_SYSTEM_PROMPT,
                temperature=0.1,
            )
        else:
            # Fallback: concatenate individual answers
            individual_answers = [
                f"[Sub-query: {sub.get('query', '')}]\n{sub.get('answer', '')}"
                for sub in sub_results
            ]
            answer = (
                "[Note: LLM unavailable for synthesis. Individual sub-answers below:]\n\n"
                + "\n\n---\n\n".join(individual_answers)
            )

        logger.info(
            f"[MergeAgent] Merged {len(sub_results)} sub-results, "
            f"{len(all_unique_chunks)} unique chunks → answer generated."
        )

        return {
            "query": original_query,
            "answer": answer,
            "reranked_chunks": all_unique_chunks,
            "citations": all_citations,
            "sub_queries": [sub.get("query", "") for sub in sub_results],
            "formatted_context": merged_context,
        }
