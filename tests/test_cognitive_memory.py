"""
tests/test_cognitive_memory.py
──────────────────────────────
Unit and integration tests for Multi-Tier Cognitive Memory Engine:
  - EpisodicMemory (time decay, similarity retrieval)
  - SemanticMemory (preferences, domain facts, pattern extraction)
  - ProceduralMemory (task recipes, intent matching)
  - CognitiveMemoryHub (unified coordination)
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from database.db_manager import DatabaseManager
from rag.memory.episodic_memory import EpisodicMemory
from rag.memory.semantic_memory import SemanticMemory
from rag.memory.procedural_memory import ProceduralMemory
from rag.memory.cognitive_hub import CognitiveMemoryHub


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        db = DatabaseManager(db_path=db_path)
        yield db


def test_episodic_memory_time_decay(temp_db: DatabaseManager):
    episodic = EpisodicMemory(db=temp_db, time_decay_lambda=0.01, similarity_threshold=0.50)

    # 1. Record episode
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    eid = episodic.record_episode(
        query="What is the installed capacity?",
        answer="Installed capacity is 76,000 MW.",
        citations=["doc.pdf (Page 1)"],
        query_embedding=q_vec,
    )
    assert eid is not None

    # 2. Search with identical vector (similarity ~ 1.0)
    results = episodic.search_episodes(query_embedding=q_vec, top_k=3)
    assert len(results) == 1
    assert results[0]["raw_similarity"] >= 0.99
    assert results[0]["time_decayed_score"] >= 0.99
    assert results[0]["episode"]["query"] == "What is the installed capacity?"

    # 3. Search with orthogonal vector (similarity ~ 0.0) -> should not match
    orth_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    results_orth = episodic.search_episodes(query_embedding=orth_vec, top_k=3)
    assert len(results_orth) == 0

    # 4. Context string generation
    context_str = episodic.get_context_string(query_embedding=q_vec)
    assert "RELEVANT PAST INTERACTIONS" in context_str
    assert "Installed capacity is 76,000 MW" in context_str


def test_semantic_memory_preferences_and_facts(temp_db: DatabaseManager):
    semantic = SemanticMemory(db=temp_db)

    # 1. Preferences
    semantic.set_preference("output_style", "Executive summary with bullet points")
    assert semantic.get_preference("output_style") == "Executive summary with bullet points"

    # 2. Add Fact
    fid = semantic.add_fact(
        subject="NTPC Green Energy",
        predicate="target_capacity_2032",
        object_="60 GW",
        source="Investor Presentation",
    )
    assert fid is not None

    # 3. Search Fact
    matched = semantic.search_facts("What is the green energy target?")
    assert len(matched) >= 1
    assert matched[0]["subject"] == "NTPC Green Energy"

    # 4. Extraction heuristic
    extracted = semantic.extract_and_update(
        user_msg="Please always use INR Crores for currency. Remember that Coal PLF was 77%.",
    )
    assert "currency" in extracted["preferences"] or semantic.get_preference("currency") is not None
    assert len(extracted["facts"]) >= 1

    # 5. Context string
    ctx = semantic.get_context_string(query="green energy")
    assert "USER PREFERENCES" in ctx
    assert "KNOWN DOMAIN FACTS" in ctx


def test_procedural_memory_recipes(temp_db: DatabaseManager):
    procedural = ProceduralMemory(db=temp_db)

    # 1. Default recipes exist
    recipes = procedural.get_all_recipes()
    assert len(recipes) >= 4

    # 2. Match financial comparison
    match_comp = procedural.match_recipe("Compare Q1 vs Q4 operational generation and revenue")
    assert match_comp is not None
    assert match_comp["name"] == "financial_comparison"

    # 3. Match tabular analysis
    match_tab = procedural.match_recipe("Give me the balance sheet breakdown and schedule table")
    assert match_tab is not None
    assert match_tab["name"] == "tabular_analysis"

    # 4. Custom recipe registration
    procedural.register_recipe(
        name="coal_supply_audit",
        trigger_patterns=["coal supply", "rake movement", "linkage coal"],
        steps=["1. Check ACQ compliance.", "2. Extract domestic vs imported blend ratio."],
    )
    custom_match = procedural.match_recipe("Review the linkage coal and rake movement stats")
    assert custom_match is not None
    assert custom_match["name"] == "coal_supply_audit"


def test_cognitive_memory_hub_coordination(temp_db: DatabaseManager):
    hub = CognitiveMemoryHub(db=temp_db)

    # 1. Post interaction update
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    hub.post_interaction_update(
        query="What is NTPC's total capacity?",
        answer="NTPC's total capacity is 76 GW.",
        citations=["annual_report.pdf (Page 5)"],
        query_embedding=q_vec,
        session_id="test-session-1",
        prompt_tokens=100,
        completion_tokens=50,
    )

    # 2. Build cognitive context for a related follow-up
    cog_context = hub.build_cognitive_context(
        query="Compare total capacity vs renewable capacity",
        query_embedding=q_vec,
        session_id="test-session-1",
    )
    assert len(cog_context) > 0
    # Should include matched financial comparison recipe
    assert "PROCEDURAL REASONING RECIPE" in cog_context or "FINANCIAL COMPARISON" in cog_context

    # 3. Dashboard summary metrics
    summary = hub.get_dashboard_summary()
    assert summary["working_memory"]["total_turns"] == 1
    assert summary["episodic_memory"]["total_episodes"] == 1
    assert summary["semantic_memory"]["total_preferences"] >= 1
    assert summary["procedural_memory"]["total_recipes"] >= 4
