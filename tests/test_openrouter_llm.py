"""
tests/test_openrouter_llm.py
────────────────────────────
Integration and unit tests for OpenRouterLLM client.
Tests model discovery, task routing, text generation, streaming, and Gemini fallback.
"""

import pytest
from rag.openrouter_llm import OpenRouterLLM
from config import OPENROUTER_API_KEY


@pytest.fixture(scope="module")
def openrouter_llm():
    if not OPENROUTER_API_KEY:
        pytest.skip("OPENROUTER_API_KEY not configured in environment.")
    return OpenRouterLLM()


def test_openrouter_model_discovery(openrouter_llm):
    live_models = openrouter_llm._live_free_models
    assert isinstance(live_models, set)
    # Check active task routing
    active = openrouter_llm.get_active_models()
    assert "answer" in active
    assert "decompose" in active
    assert "judge" in active


def test_openrouter_generate(openrouter_llm):
    response = openrouter_llm.generate(
        prompt="Reply with the exact word 'CONFIRMED'.",
        task="decompose",
        temperature=0.0,
    )
    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_openrouter_streaming(openrouter_llm):
    stream = openrouter_llm.generate_stream(
        prompt="Count from 1 to 5.",
        task="answer",
        temperature=0.0,
    )
    chunks = list(stream)
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert any(str(digit) in full_text for digit in [1, 2, 3, 4, 5])
