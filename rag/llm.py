import os
import time
from typing import Optional
from config import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False


class GeminiLLM:
    """
    Wrapper for Google Gemini API via official google-genai SDK.
    Supports system instructions, temperature tuning, and fallback execution.
    """

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
        self.client = None

        if HAS_GEMINI_SDK and self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def is_available(self) -> bool:
        """Check if Gemini client is properly initialized with API key."""
        return self.client is not None

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1
    ) -> str:
        """
        Generates text completion using Gemini API.
        """
        if not HAS_GEMINI_SDK:
            raise ImportError("The 'google-genai' package is required. Install via pip install google-genai.")
        
        if not self.client:
            raise ValueError(
                "Gemini API key is missing. Please provide a valid GEMINI_API_KEY in environment or UI."
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction if system_instruction else None
        )

        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                return response.text if response.text else ""
            except Exception as e:
                err_str = str(e)
                if ("NOT_FOUND" in err_str or "404" in err_str) and self.model_name != "gemini-flash-lite-latest":
                    print(f"[Model 404] Model '{self.model_name}' unavailable. Auto-switching to 'gemini-flash-lite-latest'...", flush=True)
                    self.model_name = "gemini-flash-lite-latest"
                    continue
                elif ("RESOURCE_EXHAUSTED" in err_str or "429" in err_str) and attempt < 4:
                    wait_time = 20 * (attempt + 1)
                    print(f"[Rate Limit 429] Waiting {wait_time}s before retry (Attempt {attempt+1}/5)...", flush=True)
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Gemini API generation error: {err_str}")
