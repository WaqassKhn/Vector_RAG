"""
rag/openrouter_llm.py
─────────────────────
Primary LLM client for the RAG system.

Routes each task to the best available free OpenRouter model using a
priority-ordered fallback chain. Falls back to GeminiLLM (last resort) if
all OpenRouter models are exhausted or the API is unreachable.

Task types:
    "answer"    — main grounded answer generation
    "decompose" — query planning / decomposition
    "judge"     — LLM-as-judge grounding evaluation
    "compress"  — conversation memory summarisation
    "triage"    — document scope selection

Usage:
    llm = OpenRouterLLM()
    text = llm.generate("What is NTPC's revenue?", task="answer")

    # Streaming (for Streamlit st.write_stream):
    for chunk in llm.generate_stream("What is NTPC's revenue?", task="answer"):
        print(chunk, end="", flush=True)
"""

import json
import time
import logging
from typing import Generator, Optional

import httpx

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME,
    OPENROUTER_MODELS,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)


class OpenRouterLLM:
    """
    Task-aware OpenRouter LLM client with automatic free-model routing,
    per-model fallback, SSE streaming, and Gemini as final fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        gemini_fallback=None,   # GeminiLLM instance — imported lazily to avoid circular deps
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self._gemini_fallback = gemini_fallback  # set lazily on first need

        self.last_model_used = "openrouter"

        # Fetch live free models once at startup; cached for the lifetime of this object.
        self._live_free_models: set[str] = self._fetch_live_free_models()

        if not self._live_free_models:
            logger.warning(
                "[OpenRouterLLM] No live free models detected. "
                "Check OPENROUTER_API_KEY and network connectivity. Will fall back to Gemini."
            )
        else:
            logger.info(
                f"[OpenRouterLLM] {len(self._live_free_models)} free models available."
            )

    # ─── Availability ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True if the API key is set and at least one live free model exists."""
        return bool(self.api_key and self._live_free_models)

    # ─── Internal: model discovery ───────────────────────────────────────────

    def _fetch_live_free_models(self) -> set[str]:
        """
        Fetches the live model list from OpenRouter and returns the set of
        model IDs that are currently free (id ends with ':free' AND prompt price == '0').
        Returns an empty set on any failure.
        """
        if not self.api_key:
            return set()
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    f"{OPENROUTER_BASE_URL}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()

            models = data.get("data", [])
            free_ids: set[str] = set()
            for m in models:
                mid = m.get("id", "")
                pricing = m.get("pricing", {})
                prompt_price = str(pricing.get("prompt", "1"))
                comp_price = str(pricing.get("completion", "1"))
                if (
                    mid.endswith(":free")
                    or mid == "openrouter/free"
                    or (prompt_price == "0" and comp_price == "0")
                ):
                    free_ids.add(mid)

            if self.api_key:
                free_ids.add("openrouter/free")

            return free_ids

        except Exception as exc:
            logger.warning(f"[OpenRouterLLM] Failed to fetch model list: {exc}")
            return set()

    def _get_model_for_task(self, task: str) -> Optional[str]:
        """Returns the highest-priority live model for the given task, or None."""
        priority_list = OPENROUTER_MODELS.get(task, OPENROUTER_MODELS["answer"])
        for model_id in priority_list:
            if model_id in self._live_free_models:
                return model_id
        # If none in live list, try them anyway (model list may be stale)
        if priority_list:
            return priority_list[0]
        return None

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str],
    ) -> list[dict]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _default_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
            "Content-Type": "application/json",
        }

    # ─── Internal: single model call ─────────────────────────────────────────

    def _call_model(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        stream: bool = False,
    ):
        """
        Makes a single call to OpenRouter for the given model.
        - stream=False  → returns full response string
        - stream=True   → returns httpx.Response in streaming mode (caller must iterate)
        Raises httpx.HTTPStatusError on 4xx/5xx so the caller can try the next model.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if stream:
            # Return a context-managed streaming response — caller iterates SSE lines
            return httpx.stream(
                "POST",
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=self._default_headers(),
                json=payload,
                timeout=120,
            )
        else:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=self._default_headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

    # ─── Internal: Gemini fallback ────────────────────────────────────────────

    def _get_gemini(self):
        """Lazily imports and instantiates GeminiLLM to avoid circular imports."""
        if self._gemini_fallback is None:
            if not GEMINI_API_KEY:
                return None
            try:
                from rag.llm import GeminiLLM
                self._gemini_fallback = GeminiLLM()
            except Exception as exc:
                logger.error(f"[OpenRouterLLM] Failed to init Gemini fallback: {exc}")
                return None
        return self._gemini_fallback

    def _gemini_generate(self, prompt: str, system_instruction: Optional[str], temperature: float) -> str:
        gemini = self._get_gemini()
        if gemini and gemini.is_available():
            logger.info("[OpenRouterLLM] Falling back to Gemini API.")
            return gemini.generate(prompt=prompt, system_instruction=system_instruction, temperature=temperature)
        return "[Error] All LLM backends unavailable. Check OPENROUTER_API_KEY and GEMINI_API_KEY."

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        task: str = "answer",
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """
        Generates a full text response for the given prompt.

        Tries models in priority order for the task. Falls back to the next model
        on 429 (rate limit) or 5xx errors. Falls back to Gemini if all fail.
        """
        messages = self._build_messages(prompt, system_instruction)
        priority_list = OPENROUTER_MODELS.get(task, OPENROUTER_MODELS["answer"])

        tried: list[str] = []
        for model_id in priority_list:
            tried.append(model_id)
            try:
                logger.debug(f"[OpenRouterLLM] task={task} → trying model: {model_id}")
                result = self._call_model(model_id, messages, temperature, stream=False)
                self.last_model_used = model_id
                logger.info(f"[OpenRouterLLM] task={task} → success with: {model_id}")
                return result

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429:
                    logger.warning(f"[OpenRouterLLM] 429 rate limit on {model_id}, trying next...")
                    time.sleep(2)
                    continue
                elif status in (402, 403):
                    logger.warning(f"[OpenRouterLLM] {status} on {model_id} (quota/billing), trying next...")
                    continue
                elif 500 <= status < 600:
                    logger.warning(f"[OpenRouterLLM] {status} server error on {model_id}, trying next...")
                    continue
                else:
                    logger.error(f"[OpenRouterLLM] Unexpected {status} on {model_id}: {exc}")
                    continue

            except Exception as exc:
                logger.warning(f"[OpenRouterLLM] Error with {model_id}: {exc}, trying next...")
                continue

        # All OpenRouter models exhausted — fall back to Gemini
        logger.warning(f"[OpenRouterLLM] All models tried ({tried}). Falling back to Gemini.")
        return self._gemini_generate(prompt, system_instruction, temperature)

    def generate_stream(
        self,
        prompt: str,
        task: str = "answer",
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Generator[str, None, None]:
        """
        Streams the response token-by-token as a generator of string chunks.

        Designed for use with Streamlit's st.write_stream():
            st.write_stream(llm.generate_stream(prompt, task="answer"))

        Falls back to a non-streaming Gemini call (yielded as one chunk) if all
        OpenRouter models fail.
        """
        messages = self._build_messages(prompt, system_instruction)
        priority_list = OPENROUTER_MODELS.get(task, OPENROUTER_MODELS["answer"])

        for model_id in priority_list:
            try:
                logger.debug(f"[OpenRouterLLM] stream task={task} → trying: {model_id}")
                payload = {
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": True,
                }
                with httpx.Client(timeout=120) as client:
                    with client.stream(
                        "POST",
                        f"{OPENROUTER_BASE_URL}/chat/completions",
                        headers=self._default_headers(),
                        json=payload,
                    ) as resp:
                        if resp.status_code == 429:
                            logger.warning(f"[OpenRouterLLM] stream 429 on {model_id}, trying next...")
                            time.sleep(2)
                            continue
                        if resp.status_code in (402, 403, 500, 502, 503):
                            logger.warning(f"[OpenRouterLLM] stream {resp.status_code} on {model_id}, trying next...")
                            continue

                        resp.raise_for_status()
                        self.last_model_used = model_id
                        logger.info(f"[OpenRouterLLM] streaming from: {model_id}")

                        # Parse SSE lines
                        for line in resp.iter_lines():
                            line = line.strip()
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                json_str = line[6:]
                                try:
                                    data = json.loads(json_str)
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    continue
                        return  # successful stream complete

            except Exception as exc:
                logger.warning(f"[OpenRouterLLM] Stream error on {model_id}: {exc}, trying next...")
                continue

        # All streaming models failed — fall back to Gemini (non-streaming, yield as one chunk)
        logger.warning("[OpenRouterLLM] All stream models failed. Falling back to Gemini (non-streaming).")
        result = self._gemini_generate(prompt, system_instruction, temperature)
        yield result

    def get_active_models(self) -> dict[str, Optional[str]]:
        """
        Returns the currently-selected model for each task type.
        Useful for displaying model status in the UI Settings tab.
        """
        return {task: self._get_model_for_task(task) for task in OPENROUTER_MODELS}
