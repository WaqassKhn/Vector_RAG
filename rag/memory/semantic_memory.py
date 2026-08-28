"""
rag/memory/semantic_memory.py
─────────────────────────────
Semantic Memory Engine for User Preferences and Domain Fact Knowledge.

Stores:
  1. Persistent User Preferences (e.g. currency formatting, output style, detail depth)
  2. Domain Fact Graph (subject-predicate-object triples with confidence and source)

Backed by SQLite via DatabaseManager. Ensures user constraints and learned domain facts
persist indefinitely across sessions.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from config import MAX_SEMANTIC_FACTS
from database.db_manager import DatabaseManager, get_db

logger = logging.getLogger(__name__)

# Default baseline preferences for financial RAG
_DEFAULT_PREFERENCES = {
    "currency_format": "INR Crores / ₹ (Quote exact source units)",
    "response_style": "Concise, factual, with inline source citations",
    "table_formatting": "Markdown tables for multi-period metrics",
}


class SemanticMemory:
    """
    Persistent semantic memory store for preferences and domain facts.
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        max_facts: int = MAX_SEMANTIC_FACTS,
    ):
        self.db = db or get_db()
        self.max_facts = max_facts
        self._ensure_default_preferences()

    def _ensure_default_preferences(self) -> None:
        """Seeds default user preferences if not already set."""
        existing = self.db.get_all_preferences()
        if not existing:
            for k, v in _DEFAULT_PREFERENCES.items():
                self.db.set_preference(k, v)

    # ─── User Preferences API ─────────────────────────────────────────────────

    def set_preference(self, key: str, value: str) -> None:
        """Sets a persistent user preference."""
        clean_key = key.strip().lower().replace(" ", "_")
        self.db.set_preference(clean_key, value.strip())
        logger.info(f"[SemanticMemory] Set preference '{clean_key}' = '{value.strip()}'")

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Gets a user preference value."""
        clean_key = key.strip().lower().replace(" ", "_")
        return self.db.get_preference(clean_key, default)

    def get_all_preferences(self) -> Dict[str, str]:
        """Returns all user preferences as a dictionary."""
        return self.db.get_all_preferences()

    def delete_preference(self, key: str) -> bool:
        """Deletes a preference."""
        clean_key = key.strip().lower().replace(" ", "_")
        return self.db.delete_preference(clean_key)

    # ─── Domain Facts API ─────────────────────────────────────────────────────

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        source: str = "conversation",
        confidence: float = 1.0,
    ) -> str:
        """Adds a domain fact triple (Subject - Predicate - Object)."""
        fact_id = self.db.save_fact(
            subject=subject.strip(),
            predicate=predicate.strip(),
            object_=object_.strip(),
            source=source.strip(),
            confidence=confidence,
        )
        logger.info(f"[SemanticMemory] Saved fact: ({subject}, {predicate}, {object_})")
        return fact_id

    def get_all_facts(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Returns all stored domain facts."""
        return self.db.get_all_facts(limit=min(limit, self.max_facts))

    def search_facts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches facts matching query tokens across subject, predicate, or object.
        """
        facts = self.get_all_facts()
        if not facts:
            return []

        tokens = set(re.findall(r'\w+', query.lower()))
        if not tokens:
            return facts[:top_k]

        scored_facts = []
        for f in facts:
            text = f"{f['subject']} {f['predicate']} {f['object']}".lower()
            matches = sum(1 for token in tokens if token in text)
            if matches > 0:
                scored_facts.append((matches, f))

        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [f for score, f in scored_facts[:top_k]]

    def delete_fact(self, fact_id: str) -> bool:
        """Deletes a fact by ID."""
        return self.db.delete_fact(fact_id)

    def clear_facts(self) -> None:
        """Clears all domain facts."""
        self.db.clear_facts()

    # ─── Learning & Extraction ────────────────────────────────────────────────

    def extract_and_update(self, user_msg: str, assistant_msg: str = "") -> Dict[str, Any]:
        """
        Rule-based heuristic extractor that auto-detects user preference commands
        and key factual declarations from dialogue turns.
        """
        extracted_prefs = {}
        extracted_facts = []
        text = user_msg.strip()

        # Rule 1a: "Please always use X for Y" / "Use X for Y"
        pref_for_match = re.search(
            r'(?:please always use|always use|prefer|use)\s+([^:.,\n]+)\s+for\s+([^:.,\n]+)',
            text,
            re.IGNORECASE,
        )
        if pref_for_match:
            val = pref_for_match.group(1).strip()
            key = pref_for_match.group(2).strip().lower().replace(" ", "_")
            self.set_preference(key, val)
            extracted_prefs[key] = val

        # Rule 1b: "I prefer X to / as Y" / "set preference for X as/to/: Y"
        pref_as_match = re.search(
            r'(?:i prefer|set preference for|set)\s+([^:.,\n]+)(?::|to|as|=)\s+([^.\n]+)',
            text,
            re.IGNORECASE,
        )
        if pref_as_match:
            key = pref_as_match.group(1).strip().lower().replace(" ", "_")
            val = pref_as_match.group(2).strip()
            self.set_preference(key, val)
            extracted_prefs[key] = val

        # Rule 2: Explicit definition patterns: "Remember that X is/was/has/are Y"
        fact_match = re.search(
            r'(?:remember that|note that|fact:)\s+(.+?)\s+(is|has|are|was)\s+([^.\n]+)',
            text,
            re.IGNORECASE,
        )
        if fact_match:
            subj = fact_match.group(1).strip()
            pred = fact_match.group(2).strip()
            obj = fact_match.group(3).strip()
            self.add_fact(subj, pred, obj, source="user_prompt")
            extracted_facts.append({"subject": subj, "predicate": pred, "object": obj})

        return {"preferences": extracted_prefs, "facts": extracted_facts}

    # ─── Context Formatting ───────────────────────────────────────────────────

    def get_context_string(self, query: Optional[str] = None) -> str:
        """
        Formats active user preferences and relevant domain facts for RAG prompt injection.
        """
        parts = []

        # 1. Preferences
        prefs = self.get_all_preferences()
        if prefs:
            pref_lines = [f"- {k.replace('_', ' ').title()}: {v}" for k, v in prefs.items()]
            parts.append("[USER PREFERENCES & FORMATTING GUIDELINES]:\n" + "\n".join(pref_lines))

        # 2. Relevant Facts (if query supplied)
        if query:
            matched_facts = self.search_facts(query, top_k=4)
            if matched_facts:
                fact_lines = [
                    f"- {f['subject']} — {f['predicate']}: {f['object']} (Source: {f.get('source', 'knowledge')})"
                    for f in matched_facts
                ]
                parts.append("[KNOWN DOMAIN FACTS (Semantic Memory)]:\n" + "\n".join(fact_lines))

        return "\n\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Returns semantic memory statistics."""
        return {
            "total_preferences": len(self.get_all_preferences()),
            "total_facts": len(self.get_all_facts()),
        }

    def clear(self) -> None:
        """Resets all preferences and facts."""
        self.clear_facts()
        for k in list(self.get_all_preferences().keys()):
            self.delete_preference(k)
        self._ensure_default_preferences()
