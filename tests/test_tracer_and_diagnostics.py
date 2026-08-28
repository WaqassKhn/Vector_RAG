"""
tests/test_tracer_and_diagnostics.py
────────────────────────────────────
Unit tests for ExecutionTracer and ErrorDiagnosticManager.
"""

import pytest
from rag.tracer import ExecutionTracer
from rag.error_handler import ErrorDiagnosticManager, DiagnosticError


def test_execution_tracer_lifecycle():
    tracer = ExecutionTracer(query_id="test_q1")
    span1 = tracer.start_span("embed_query", component="embedding", inputs={"query": "test"})
    tracer.finish_span(span1, outputs={"dim": 384})

    span2 = tracer.start_span("llm_call", component="llm", inputs={"model": "test-model"})
    tracer.finish_span(span2, outputs={"tokens": 50}, metadata={"model": "test-model", "prompt_tokens": 10, "completion_tokens": 40})

    tracer.finish()
    summary = tracer.get_summary()

    assert summary["query_id"] == "test_q1"
    assert summary["span_count"] == 2
    assert summary["llm_calls_count"] == 1
    assert summary["total_tokens_estimated"] == 50
    assert "test-model" in summary["models_used"]
    assert tracer.total_duration_ms >= 0


def test_error_diagnostic_auth():
    exc = Exception("HTTP 401 Unauthorized: Invalid API Key")
    diag = ErrorDiagnosticManager.diagnose(exc)
    assert diag.category == "LLM_AUTHENTICATION_ERROR"
    assert "OPENROUTER_API_KEY" in diag.root_cause
    assert len(diag.remedy_steps) > 0


def test_error_diagnostic_rate_limit():
    exc = Exception("HTTP 429 Too Many Requests: Rate limit exceeded")
    diag = ErrorDiagnosticManager.diagnose(exc)
    assert diag.category == "LLM_RATE_LIMIT_ERROR"
    assert "429" in diag.title


def test_error_diagnostic_dimension_mismatch():
    exc = Exception("Vector dimension mismatch: expected 1536, got 384")
    diag = ErrorDiagnosticManager.diagnose(exc)
    assert diag.category == "PINECONE_DIMENSION_MISMATCH"
    assert "384" in diag.root_cause
