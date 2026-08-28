"""
tests/test_pinecone_db.py
─────────────────────────
Integration and unit tests for PineconeDB.
Tests upserting chunks, semantic similarity search, metadata filtering,
document registry tracking, and deletion.
"""

import time
import pytest
import numpy as np
from pathlib import Path
from vectorstore.pinecone_db import PineconeDB
from vectorstore.embeddings import EmbeddingManager
from config import PINECONE_API_KEY


@pytest.fixture(scope="module")
def pinecone_db():
    if not PINECONE_API_KEY:
        pytest.skip("PINECONE_API_KEY not configured in environment.")
    try:
        db = PineconeDB()
        return db
    except Exception as exc:
        pytest.skip(f"Pinecone connection failed: {exc}")


def test_pinecone_upsert_and_search(pinecone_db):
    emb_mgr = EmbeddingManager(use_gemini=False)
    
    test_chunks = [
        {
            "chunk_id": "test_doc.pdf_p1_c0",
            "text": "NTPC generated 400 billion units of thermal and renewable electricity in fiscal year 2024.",
            "filename": "test_doc.pdf",
            "page_number": 1,
            "has_table": False,
            "numeric_count": 2,
            "header_context": "Financial & Operational Performance",
        },
        {
            "chunk_id": "test_doc.pdf_p1_c1",
            "text": "Solar capacity increased by 25% with new photovoltaic installations in Gujarat.",
            "filename": "test_doc.pdf",
            "page_number": 1,
            "has_table": False,
            "numeric_count": 1,
            "header_context": "Renewable Energy Expansion",
        },
    ]

    texts = [c["text"] for c in test_chunks]
    vecs = emb_mgr.embed_texts(texts)
    assert vecs.shape == (2, 384)

    # Upsert
    pinecone_db.upsert_chunks(test_chunks, vecs)
    time.sleep(2)  # Allow cloud index to reflect changes

    # Search with document filter
    query_vec = emb_mgr.embed_query("How much electricity did NTPC generate?")
    results = pinecone_db.search(query_vec, top_k=2, filter_filenames=["test_doc.pdf"])

    assert len(results) > 0
    chunks_text = " ".join([c["text"] for c, _ in results])
    assert "electricity" in chunks_text.lower() or "400" in chunks_text


def test_pinecone_metadata_filtering(pinecone_db):
    emb_mgr = EmbeddingManager(use_gemini=False)
    query_vec = emb_mgr.embed_query("solar photovoltaic installations")

    # Search with matching filter
    filtered_results = pinecone_db.search(
        query_vec, top_k=2, filter_filenames=["test_doc.pdf"]
    )
    for chunk, _ in filtered_results:
        assert chunk["filename"] == "test_doc.pdf"

    # Search with non-matching filter
    empty_results = pinecone_db.search(
        query_vec, top_k=2, filter_filenames=["non_existent_doc.pdf"]
    )
    assert len(empty_results) == 0


def test_pinecone_document_registry_and_deletion(pinecone_db):
    registry = pinecone_db.list_documents()
    assert "test_doc.pdf" in registry
    assert registry["test_doc.pdf"]["chunk_count"] >= 2

    # Delete test document
    deleted_count = pinecone_db.delete_by_filename("test_doc.pdf")
    assert deleted_count >= 2

    # Verify registry updated
    updated_registry = pinecone_db.list_documents()
    assert "test_doc.pdf" not in updated_registry
