import hashlib
import os
import time
from collections import OrderedDict
from threading import Lock

from google import genai
from google.genai import types

from app.core.config import GEMINI_API_KEY


class GeminiClient:
    """
    Gemini client for ACOT with:
    - configurable primary model
    - multiple fallback models
    - short retries for temporary 503/timeout errors
    - immediate failure for quota/permission errors
    - bounded output tokens
    - in-memory response cache
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Primary model.
        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash",
        )

        # Fallback chain.
        # Flash-Lite is intentionally early because Google positions it
        # for high-throughput / low-latency workloads.
        configured_fallbacks = os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-2.5-flash",
        )

        fallback_models = [
            item.strip()
            for item in configured_fallbacks.split(",")
            if item.strip()
        ]

        self.models = []
        for model in [self.model] + fallback_models:
            if model and model not in self.models:
                self.models.append(model)

        # Keep retries short. We want to fail over to another model quickly
        # instead of repeatedly waiting on a model that is under load.
        self.max_retries = 2
        self.retry_delays = (1.0,)

        # Bound output size.
        self.max_output_tokens = int(
            os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "900")
        )

        # Response cache.
        self.cache_enabled = True
        self.cache_ttl_seconds = 300
        self.cache_max_entries = 100

        self._cache = OrderedDict()
        self._cache_lock = Lock()

    def _is_retryable_error(self, error: Exception) -> bool:
        message = str(error).lower()

        # These should not be retried in a tight loop.
        permanent_terms = (
            "quota exceeded",
            "resource exhausted",
            "project has been denied access",
            "permission_denied",
            "403",
            "401",
            "invalid api key",
            "api key not valid",
            "not found",
            "invalid model",
        )

        if any(term in message for term in permanent_terms):
            return False

        # Temporary service/capacity errors.
        retryable_terms = (
            "503",
            "service unavailable",
            "temporarily unavailable",
            "high demand",
            "overloaded",
            "timeout",
            "timed out",
            "504",
            "deadline exceeded",
        )

        return any(term in message for term in retryable_terms)

    def _cache_key(self, prompt: str, model: str) -> str:
        raw = f"{model}\n{prompt}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _get_cached(self, key: str):
        if not self.cache_enabled:
            return None

        now = time.time()

        with self._cache_lock:
            item = self._cache.get(key)

            if item is None:
                return None

            created_at, value = item

            if now - created_at > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None

            self._cache.move_to_end(key)
            return value

    def _set_cached(self, key: str, value: str):
        if not self.cache_enabled:
            return

        with self._cache_lock:
            self._cache[key] = (time.time(), value)
            self._cache.move_to_end(key)

            while len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()

    def _generate_with_model(self, model: str, prompt: str, max_output_tokens=None):
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=(
                    max_output_tokens
                    if max_output_tokens is not None
                    else self.max_output_tokens
                ),
            ),
        )

        if not response:
            raise RuntimeError("Gemini returned an empty response.")

        if not response.text:
            raise RuntimeError("Gemini returned no text content.")

        return response.text.strip()

    def _format_error(self, model: str, error: Exception) -> str:
        message = str(error)

        lowered = message.lower()

        if "quota" in lowered or "resource exhausted" in lowered:
            return (
                "Gemini quota/rate limit reached for the current project. "
                "Check the active project/tier and AI Studio rate limits."
            )

        if "403" in lowered or "permission_denied" in lowered:
            return f"Gemini project access denied for model '{model}': {message}"

        return f"Gemini generation failed on model '{model}': {message}"

    def generate(self, prompt: str, max_output_tokens=None):
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        last_error = None

        for model_index, model in enumerate(self.models):
            cache_key = self._cache_key(prompt, model)
            cached = self._get_cached(cache_key)

            if cached is not None:
                print(f"Gemini cache hit: {model}")
                return cached

            for attempt in range(self.max_retries):
                try:
                    if attempt > 0:
                        delay = self.retry_delays[
                            min(
                                attempt - 1,
                                len(self.retry_delays) - 1,
                            )
                        ]

                        print(
                            f"Gemini temporary failure on {model}. "
                            f"Retrying in {delay:.1f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})..."
                        )

                        time.sleep(delay)

                    response = self._generate_with_model(
                        model=model,
                        prompt=prompt,
                        max_output_tokens=max_output_tokens,
                    )

                    self._set_cached(cache_key, response)
                    return response

                except Exception as error:
                    last_error = error

                    # Permanent problems should stop immediately.
                    if not self._is_retryable_error(error):
                        raise RuntimeError(
                            self._format_error(model, error)
                        ) from error

                    # For temporary 503/high-demand errors, retry this model
                    # only once, then fail over.
                    if attempt == 0:
                        continue

                    break

            # Move quickly to the next model after temporary failures.
            if model_index < len(self.models) - 1:
                next_model = self.models[model_index + 1]

                print(
                    f"Gemini model '{model}' is temporarily unavailable. "
                    f"Falling back to '{next_model}'."
                )

        raise RuntimeError(
            "Gemini generation failed on all configured models. "
            f"Last error: {last_error}"
        ) from last_error
