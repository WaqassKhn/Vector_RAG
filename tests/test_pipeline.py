import pytest
import numpy as np
from pathlib import Path
from pipeline.cleaner import DocumentCleaner
from pipeline.parser import DocumentParser
from pipeline.chunker import DocumentChunker
from vectorstore.embeddings import EmbeddingManager
from vectorstore.vector_db import VectorDatabase
from rag.reranker import HybridReranker
from evaluation.grounding_eval import DocumentGroundingEvaluator

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "dirty_company_report.txt"

def test_cleaner_fixes_hyphenated_line_breaks():
    dirty = "macro-\neconomic headwinds and dis-\nruptions."
    cleaned = DocumentCleaner.clean_document_text(dirty)
    assert "macroeconomic" in cleaned
    assert "disruptions" in cleaned

def test_document_parser():
    parsed = DocumentParser.parse_file(str(SAMPLE_PATH))
    assert parsed["file_type"] == "txt"
    assert "145.8M" in parsed["full_text"]
    assert len(parsed["pages"]) > 0

def test_document_chunker():
    parsed = DocumentParser.parse_file(str(SAMPLE_PATH))
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_parsed_document(parsed)
    assert len(chunks) > 0
    # Check that chunks retain metadata
    first_chunk = chunks[0]
    assert "filename" in first_chunk
    assert "page_number" in first_chunk
    assert "text" in first_chunk

def test_embedding_and_vector_db():
    embeddings_mgr = EmbeddingManager(use_gemini=False)
    vdb = VectorDatabase()
    
    test_chunks = [
        {"chunk_id": "c1", "text": "Net revenue reached $145.8M in Q3 2024.", "filename": "test.txt", "page_number": 1},
        {"chunk_id": "c2", "text": "Total cash stood at $94.5M against debt of $120M.", "filename": "test.txt", "page_number": 1}
    ]
    
    texts = [c["text"] for c in test_chunks]
    vecs = embeddings_mgr.embed_texts(texts)
    assert vecs.shape == (2, 384)

    vdb.add_chunks(test_chunks, vecs)
    
    q_vec = embeddings_mgr.embed_query("What was the net revenue in Q3?")
    results = vdb.search(q_vec, top_k=2)
    assert len(results) == 2
    top_doc, score = results[0]
    assert "145.8M" in top_doc["text"]

def test_hybrid_reranker():
    reranker = HybridReranker()
    dense_results = [
        ({"chunk_id": "c1", "text": "Acme revenue was $145.8M in Q3 2024.", "filename": "doc1", "page_number": 1}, 0.85),
        ({"chunk_id": "c2", "text": "Employees enjoyed the summer retreat.", "filename": "doc1", "page_number": 2}, 0.70)
    ]
    reranked = reranker.rerank("Q3 revenue $145.8M", dense_results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0][0]["chunk_id"] == "c1"

def test_grounding_evaluator_numerical_audit():
    evaluator = DocumentGroundingEvaluator()
    context_chunks = [{"filename": "doc1", "page_number": 1, "text": "Net revenue reached $145.8M in Q3 2024."}]
    
    # Fully grounded answer
    good_answer = "According to the report, net revenue reached $145.8M in Q3 2024."
    res_good = evaluator.evaluate_grounding("What was net revenue?", good_answer, context_chunks)
    assert res_good["numerical_accuracy_score"] == 1.0
    assert len(res_good["unsupported_numbers"]) == 0

    # Hallucinated answer with fake number $500M
    bad_answer = "Net revenue was $500.0M in Q3 2024."
    res_bad = evaluator.evaluate_grounding("What was net revenue?", bad_answer, context_chunks)
    assert "$500.0M" in res_bad["unsupported_numbers"] or "500.0" in [n.replace("$","").replace("M","") for n in res_bad["unsupported_numbers"]]
    assert res_bad["numerical_accuracy_score"] < 1.0
