"""
rag/memory/procedural_memory.py
───────────────────────────────
Procedural Memory Engine for Task Execution Recipes and Reasoning Workflows.

Stores pre-compiled and learned workflows for handling complex financial,
tabular, regulatory, and multi-document query patterns.

When a query arrives, ProceduralMemory matches the query intent to the best recipe,
providing step-by-step reasoning instructions to the QueryPlanner and RAGChain.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from database.db_manager import DatabaseManager, get_db

logger = logging.getLogger(__name__)

# Standard Built-in Corporate RAG Task Recipes
_DEFAULT_RECIPES = [
    {
        "name": "financial_comparison",
        "trigger_patterns": [
            "compare", "versus", "vs", "growth", "difference between",
            "q1 vs", "fy23 vs", "fy24 vs", "year on year", "yoy", "quarter on quarter", "qoq"
        ],
        "steps": [
            "1. Extract baseline period figures and comparison period figures with exact source citations.",
            "2. Ensure monetary units (₹ Crore, Lakh, Millions) and periods are strictly aligned before comparing.",
            "3. Calculate the absolute delta and percentage change (do not extrapolate beyond source data).",
            "4. Format the final output in a structured markdown comparison table.",
        ],
        "few_shot_examples": [
            {
                "query": "Compare FY23 vs FY24 revenue and EBITDA",
                "approach": "Extract FY23 revenue, FY24 revenue, compute delta (%), extract FY23 EBITDA, FY24 EBITDA, tabulate.",
            }
        ],
    },
    {
        "name": "executive_summary",
        "trigger_patterns": [
            "executive summary", "overview", "highlights", "key takeaways",
            "annual summary", "briefing", "snapshot"
        ],
        "steps": [
            "1. Extract top-line capacity metrics (Total Installed Capacity, Commercial Capacity in GW/MW).",
            "2. Extract operational highlights (Generation in BUs, PLF percentage).",
            "3. Extract financial highlights (Total Income, EBITDA, PAT in ₹ Crore).",
            "4. Summarize strategic initiatives (Renewable Energy transition, Capex roadmap).",
            "5. Structure with clear bold headings and bullet points.",
        ],
        "few_shot_examples": [],
    },
    {
        "name": "tabular_analysis",
        "trigger_patterns": [
            "table", "breakdown", "schedule", "balance sheet", "p&l",
            "profit and loss", "cash flow", "statement", "segment"
        ],
        "steps": [
            "1. Cross-reference row labels, column headers, and currency unit designations (₹ in Crore).",
            "2. Quote exact numeric cell values and note any parenthesized negative figures.",
            "3. Reconstruct a clean, aligned Markdown table representing the source data.",
            "4. Add an explanatory footnote citing the exact table title and page number.",
        ],
        "few_shot_examples": [],
    },
    {
        "name": "regulatory_compliance",
        "trigger_patterns": [
            "compliance", "regulation", "policy", "norm", "mandate",
            "statutory", "emission", "cerc", "esg", "environmental"
        ],
        "steps": [
            "1. Identify the specific regulation, act, or statutory body referenced.",
            "2. Extract exact compliance thresholds, emission standards, or deadline dates.",
            "3. Verify reported status (Compliant / Under Review / Target Year).",
            "4. Provide unambiguous citations to the specific clause, section, or annexure.",
        ],
        "few_shot_examples": [],
    },
]


class ProceduralMemory:
    """
    Persistent procedural memory store for domain execution recipes.
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or get_db()
        self._ensure_default_recipes()

    def _ensure_default_recipes(self) -> None:
        """Seeds built-in procedural recipes into SQLite if not present."""
        for r in _DEFAULT_RECIPES:
            existing = self.db.get_recipe(r["name"])
            if not existing:
                self.db.save_recipe(
                    name=r["name"],
                    trigger_patterns=r["trigger_patterns"],
                    steps=r["steps"],
                    few_shot_examples=r.get("few_shot_examples", []),
                )

    def register_recipe(
        self,
        name: str,
        trigger_patterns: List[str],
        steps: List[str],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Registers or updates a task execution recipe."""
        return self.db.save_recipe(
            name=name.strip().lower().replace(" ", "_"),
            trigger_patterns=[p.strip().lower() for p in trigger_patterns],
            steps=[s.strip() for s in steps],
            few_shot_examples=few_shot_examples or [],
        )

    def get_recipe(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a recipe by its name."""
        return self.db.get_recipe(name.strip().lower().replace(" ", "_"))

    def get_all_recipes(self) -> List[Dict[str, Any]]:
        """Returns all registered recipes."""
        return self.db.get_all_recipes()

    def match_recipe(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Matches a user query to the best procedural execution recipe.
        Returns the best matching recipe dict with a match score, or None.
        """
        query_clean = query.lower()
        recipes = self.get_all_recipes()
        best_match = None
        highest_score = 0

        for r in recipes:
            score = 0
            for pattern in r.get("trigger_patterns", []):
                # Pattern match check
                if pattern in query_clean:
                    # Longer pattern matches get higher weight
                    score += len(pattern.split()) * 2 + 1

            if score > highest_score:
                highest_score = score
                best_match = r

        if best_match and highest_score >= 2:
            return best_match
        return None

    def get_context_string(self, query: str) -> str:
        """
        Returns a formatted prompt block if a recipe matches the user's query intent.
        """
        recipe = self.match_recipe(query)
        if not recipe:
            return ""

        steps_formatted = "\n".join(f"  {step}" for step in recipe.get("steps", []))
        return (
            f"[PROCEDURAL REASONING RECIPE: {recipe['name'].replace('_', ' ').upper()}]\n"
            f"Apply the following domain workflow for this question:\n{steps_formatted}"
        )

    def delete_recipe(self, name: str) -> bool:
        """Deletes a custom recipe."""
        return self.db.delete_recipe(name.strip().lower().replace(" ", "_"))

    def get_stats(self) -> Dict[str, Any]:
        """Returns procedural memory statistics."""
        return {"total_recipes": len(self.get_all_recipes())}

    def clear(self) -> None:
        """Resets recipes to defaults."""
        for r in self.get_all_recipes():
            self.delete_recipe(r["name"])
        self._ensure_default_recipes()
