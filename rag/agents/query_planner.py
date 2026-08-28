"""
rag/agents/query_planner.py
───────────────────────────
QueryPlannerAgent — classifies query complexity and produces a retrieval plan.

Uses a small, fast LLM (llama-3.1-8b or equivalent) to decide:
  - "simple"  → single-pass retrieval is sufficient
  - "complex" → the query needs 2–4 targeted sub-queries

Also identifies which specific documents to scope retrieval to (doc_scope)
when the query clearly references a specific file or topic.

Output schema:
    {
        "complexity": "simple" | "complex",
        "sub_queries": ["..."],        # 1 item if simple, 2-4 if complex
        "doc_scope": ["filename.pdf"] | null
    }

Usage:
    planner = QueryPlannerAgent(llm=openrouter_llm, known_documents=["report.pdf"])
    plan = planner.plan("Compare Q1 and Q4 revenue figures across all reports")
    # plan = {"complexity": "complex", "sub_queries": ["Q1 revenue", "Q4 revenue"], "doc_scope": null}
"""

import json
import logging
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.openrouter_llm import OpenRouterLLM

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """You are a query planning assistant for a document RAG system.
Given a user query and a list of available indexed documents, output a retrieval plan as JSON.

Rules:
- "simple": single factual question answerable in one retrieval pass.
  Examples: "What is NTPC's revenue?", "Who is the CEO?", "What is the date of the report?"
- "complex": requires comparison, aggregation across time periods, or multiple distinct facts.
  Examples: "Compare Q1 vs Q4 revenue", "Summarize financial highlights AND operational metrics",
            "What changed between FY23 and FY24?"
- sub_queries: for simple, return [original_query]. For complex, return 2-4 focused sub-queries
  that each target a specific, narrow fact.
- doc_scope: ONLY set if the query explicitly names a document or mentions a topic that clearly
  maps to ONE of the listed documents. Otherwise null.

Output ONLY valid JSON. No explanation. No markdown fences."""

_PLANNER_USER_TEMPLATE = """Available documents: {doc_list}

User query: {query}

JSON plan:"""


class QueryPlannerAgent:
    """
    Classifies query complexity and decomposes complex queries into sub-queries.
    Falls back to treating the query as simple if the LLM is unavailable or
    returns malformed JSON.
    """

    def __init__(
        self,
        llm: Optional["OpenRouterLLM"] = None,
        known_documents: Optional[list[str]] = None,
    ):
        self.llm = llm
        self.known_documents = known_documents or []

    def plan(self, query: str, procedural_hint: Optional[str] = None) -> dict:
        """
        Produces a retrieval plan for the given query.

        Args:
            query: User question to plan.
            procedural_hint: Optional procedural recipe guidelines to steer decomposition.

        Returns:
            {
                "complexity": "simple" | "complex",
                "sub_queries": list[str],
                "doc_scope": list[str] | None,
            }
        """
        fallback = {
            "complexity": "simple",
            "sub_queries": [query],
            "doc_scope": None,
        }

        if not self.llm or not self.llm.is_available():
            logger.debug("[QueryPlanner] LLM unavailable — treating query as simple.")
            return fallback

        doc_list = ", ".join(self.known_documents) if self.known_documents else "No documents listed"
        prompt = _PLANNER_USER_TEMPLATE.format(doc_list=doc_list, query=query)
        if procedural_hint:
            prompt += f"\n\nWorkflow Guidance:\n{procedural_hint}"

        try:
            raw = self.llm.generate(
                prompt=prompt,
                task="decompose",
                system_instruction=_PLANNER_SYSTEM_PROMPT,
                temperature=0.0,
            )
            plan = self._parse_plan(raw, query)
            logger.info(
                f"[QueryPlanner] complexity={plan['complexity']}, "
                f"sub_queries={len(plan['sub_queries'])}, "
                f"doc_scope={plan['doc_scope']}"
            )
            return plan

        except Exception as exc:
            logger.warning(f"[QueryPlanner] Failed to plan query: {exc}. Using simple fallback.")
            return fallback

    def _parse_plan(self, raw: str, original_query: str) -> dict:
        """
        Parses LLM output into a validated plan dict.
        Falls back to simple plan on any parse error.
        """
        # Extract JSON block (handles markdown fences, leading text, etc.)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in planner output.")

        data = json.loads(json_match.group(0))

        complexity = data.get("complexity", "simple")
        if complexity not in ("simple", "complex"):
            complexity = "simple"

        sub_queries = data.get("sub_queries", [original_query])
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [original_query]
        # Enforce 1 item for simple, 2-4 for complex
        if complexity == "simple":
            sub_queries = sub_queries[:1] or [original_query]
        else:
            sub_queries = sub_queries[:4] or [original_query]

        doc_scope = data.get("doc_scope")
        if doc_scope is not None:
            if not isinstance(doc_scope, list):
                doc_scope = None
            else:
                # Filter to only known documents
                doc_scope = [d for d in doc_scope if d in self.known_documents] or None

        return {
            "complexity": complexity,
            "sub_queries": sub_queries,
            "doc_scope": doc_scope,
        }
