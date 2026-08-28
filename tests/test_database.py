"""
tests/test_database.py
──────────────────────
Unit and integration tests for DatabaseManager SQLite persistence.
"""

import tempfile
from pathlib import Path
import pytest

from database.db_manager import DatabaseManager


@pytest.fixture
def temp_db():
    """Provides an isolated temporary DatabaseManager for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_rag.db"
        db = DatabaseManager(db_path=db_path)
        yield db


def test_session_lifecycle(temp_db: DatabaseManager):
    # 1. Create session
    sid = temp_db.create_session(title="Financial Q1 Review")
    assert sid is not None

    # 2. Get session
    sess = temp_db.get_session(sid)
    assert sess is not None
    assert sess["title"] == "Financial Q1 Review"

    # 3. List sessions
    sessions = temp_db.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == sid

    # 4. Update title
    updated = temp_db.update_session_title(sid, "Financial Q1 & Q2 Review")
    assert updated is True
    sess_updated = temp_db.get_session(sid)
    assert sess_updated["title"] == "Financial Q1 & Q2 Review"

    # 5. Delete session
    deleted = temp_db.delete_session(sid)
    assert deleted is True
    assert temp_db.get_session(sid) is None


def test_message_persistence(temp_db: DatabaseManager):
    sid = temp_db.create_session(title="Test Chat")

    # 1. Save user message
    mid_user = temp_db.save_message(
        session_id=sid,
        role="user",
        content="What was NTPC's generation capacity in FY24?",
    )
    assert mid_user is not None

    # 2. Save assistant message with citations and grounding
    mid_asst = temp_db.save_message(
        session_id=sid,
        role="assistant",
        content="NTPC reached 76 GW commercial capacity [Source: report.pdf, Page 4].",
        citations=["report.pdf (Page 4)"],
        grounding_score=0.95,
        grounding_passed=True,
        tokens_used=420,
    )
    assert mid_asst is not None

    # 3. Retrieve messages
    messages = temp_db.get_session_messages(sid)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["citations"] == ["report.pdf (Page 4)"]
    assert messages[1]["grounding_passed"] is True
    assert messages[1]["tokens_used"] == 420

    # 4. Clear messages
    temp_db.clear_session_messages(sid)
    assert len(temp_db.get_session_messages(sid)) == 0


def test_document_and_chunk_persistence(temp_db: DatabaseManager):
    # 1. Register document
    temp_db.upsert_document(
        filename="ntpc_annual_report_fy24.pdf",
        chunk_count=12,
        file_size=1024000,
    )

    docs = temp_db.get_documents()
    assert "ntpc_annual_report_fy24.pdf" in docs
    assert docs["ntpc_annual_report_fy24.pdf"]["chunk_count"] == 12

    # 2. Save chunks
    chunks = [
        {
            "chunk_id": "ntpc_annual_report_fy24_chunk_0",
            "filename": "ntpc_annual_report_fy24.pdf",
            "text": "NTPC is India's largest power utility.",
            "page_number": 1,
            "has_table": False,
            "numeric_count": 0,
            "header_context": "Introduction",
        },
        {
            "chunk_id": "ntpc_annual_report_fy24_chunk_1",
            "filename": "ntpc_annual_report_fy24.pdf",
            "text": "Total generation reached 422 Billion Units in FY24.",
            "page_number": 2,
            "has_table": True,
            "numeric_count": 2,
            "header_context": "Operational Highlights",
        },
    ]
    temp_db.save_chunk_texts(chunks)

    # 3. Retrieve chunk text
    text0 = temp_db.get_chunk_text("ntpc_annual_report_fy24_chunk_0")
    assert text0 == "NTPC is India's largest power utility."

    all_chunks = temp_db.get_all_chunks()
    assert len(all_chunks) == 2

    # 4. Delete document (cascades to chunks)
    deleted_chunks = temp_db.delete_document("ntpc_annual_report_fy24.pdf")
    assert deleted_chunks == 2
    assert "ntpc_annual_report_fy24.pdf" not in temp_db.get_documents()
    assert temp_db.get_chunk_text("ntpc_annual_report_fy24_chunk_0") is None


def test_user_preferences_and_facts(temp_db: DatabaseManager):
    # Preferences
    temp_db.set_preference("currency", "INR Crore")
    assert temp_db.get_preference("currency") == "INR Crore"
    assert temp_db.get_preference("nonexistent", default="default_val") == "default_val"

    prefs = temp_db.get_all_preferences()
    assert prefs.get("currency") == "INR Crore"

    temp_db.delete_preference("currency")
    assert temp_db.get_preference("currency") is None

    # Semantic Facts
    fid = temp_db.save_fact("NTPC", "capacity_gw", "76", source="annual_report")
    assert fid is not None

    facts = temp_db.get_all_facts()
    assert len(facts) == 1
    assert facts[0]["subject"] == "NTPC"
    assert facts[0]["object"] == "76"

    temp_db.delete_fact(fid)
    assert len(temp_db.get_all_facts()) == 0


def test_token_logging(temp_db: DatabaseManager):
    temp_db.log_token_usage(
        session_id="test-sess",
        prompt_tokens=250,
        completion_tokens=150,
        model="llama-3.3-70b",
        task="answer",
    )

    summary = temp_db.get_token_usage_summary()
    assert summary["total_requests"] == 1
    assert summary["total_prompt_tokens"] == 250
    assert summary["total_completion_tokens"] == 150
    assert summary["total_tokens"] == 400
