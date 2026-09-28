import hashlib
import os
import time
from collections import OrderedDict
from threading import Lock

from openai import OpenAI

from app.core.config import AICREDITS_API_KEY, AICREDITS_BASE_URL


class AICreditsClient:
    """
    ACOT LLM client using AICredits and openai/gpt-4o-mini.

    AICredits is an OpenAI-compatible gateway:
        https://api.aicredits.in/v1
    """

    def __init__(self):
        if not AICREDITS_API_KEY:
            raise ValueError(
                "AICREDITS_API_KEY is not configured."
            )

        self.client = OpenAI(
            base_url=AICREDITS_BASE_URL,
            api_key=AICREDITS_API_KEY,
        )

        self.model = os.getenv(
            "AICREDITS_MODEL",
            "openai/gpt-4o-mini",
        )

        # One retry of the same model. A failure does not switch models.
        self.max_retries = 2
        self.retry_delays = (1.0,)

        # Output token limit
        self.max_output_tokens = int(
            os.getenv("AICREDITS_MAX_OUTPUT_TOKENS", "900")
        )

        # Response cache
        self.cache_enabled = True
        self.cache_ttl_seconds = 300
        self.cache_max_entries = 100

        self._cache = OrderedDict()
        self._cache_lock = Lock()

    def _is_retryable_error(self, error: Exception) -> bool:
        message = str(error).lower()

        # Permanent errors
        permanent_terms = (
            "401",
            "403",
            "404",
            "invalid api key",
            "authentication",
            "unauthorized",
            "forbidden",
            "invalid model",
            "model not found",
            "insufficient credits",
        )

        if any(term in message for term in permanent_terms):
            return False

        # Temporary errors
        retryable_terms = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "rate limit",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "overloaded",
            "capacity",
            "no text content",
            "empty response",
            "empty message",
            "no choices",
        )

        return any(term in message for term in retryable_terms)

    def _cache_key(
        self,
        prompt: str,
        model: str,
    ) -> str:
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

    def _set_cached(
        self,
        key: str,
        value: str,
    ):
        if not self.cache_enabled:
            return

        with self._cache_lock:
            self._cache[key] = (
                time.time(),
                value,
            )

            self._cache.move_to_end(key)

            while len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()

    def _generate_with_model(
        self,
        model: str,
        prompt: str,
        max_output_tokens=None,
    ):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self.max_output_tokens
            ),
        )

        if not response:
            raise RuntimeError(
                "AICredits returned an empty response."
            )

        if not response.choices:
            raise RuntimeError(
                "AICredits returned no choices."
            )

        message = response.choices[0].message

        if not message:
            raise RuntimeError(
                "AICredits returned an empty message."
            )

        content = self._message_text(message)

        if not content:
            raise RuntimeError(
                "AICredits returned no text content."
            )

        return content.strip()

    def _stream_with_model(
        self,
        model: str,
        prompt: str,
        max_output_tokens=None,
    ):
        stream = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else self.max_output_tokens
            ),
            stream=True,
        )

        if not stream:
            raise RuntimeError(
                "AICredits returned an empty response."
            )

        for chunk in stream:
            if not chunk or not getattr(chunk, "choices", None):
                continue

            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue

            text = self._message_text(delta)
            if text:
                yield text

    @staticmethod
    def _message_text(message) -> str:
        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            return "\n".join(parts)

        return ""

    def _format_error(
        self,
        model: str,
        error: Exception,
    ) -> str:
        message = str(error)
        lowered = message.lower()

        if (
            "429" in lowered
            or "rate limit" in lowered
        ):
            return (
                f"AICredits rate limit reached "
                f"for model '{model}': {message}"
            )

        if (
            "401" in lowered
            or "403" in lowered
            or "authentication" in lowered
            or "unauthorized" in lowered
        ):
            return (
                f"AICredits authentication/access "
                f"error for model '{model}': {message}"
            )

        return (
            f"AICredits generation failed on "
            f"model '{model}': {message}"
        )

    def generate(
        self,
        prompt: str,
        max_output_tokens=None,
    ):
        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        model = self.model
        cache_key = self._cache_key(
            prompt,
            model,
        )

        cached = self._get_cached(
            cache_key
        )

        if cached is not None:
            print(
                f"AICredits cache hit: {model}"
            )
            return cached

        last_error = None

        for attempt in range(
            self.max_retries
        ):
            try:

                if attempt > 0:
                    delay = self.retry_delays[
                        min(
                            attempt - 1,
                            len(self.retry_delays) - 1,
                        )
                    ]

                    print(
                        f"AICredits temporary "
                        f"failure on {model}. "
                        f"Retrying in "
                        f"{delay:.1f}s "
                        f"(attempt "
                        f"{attempt + 1}/"
                        f"{self.max_retries})..."
                    )

                    time.sleep(delay)

                response = (
                    self._generate_with_model(
                        model=model,
                        prompt=prompt,
                        max_output_tokens=(
                            max_output_tokens
                        ),
                    )
                )

                self._set_cached(
                    cache_key,
                    response,
                )

                return response

            except Exception as error:
                last_error = error

                # Permanent errors:
                # do not retry.
                if not self._is_retryable_error(
                    error
                ):
                    raise RuntimeError(
                        self._format_error(
                            model,
                            error,
                        )
                    ) from error

                # Retry the same model once.
                if attempt == 0:
                    continue

                break

        raise RuntimeError(
            "AICredits generation failed on "
            f"model '{model}'. "
            f"Last error: {last_error}"
        ) from last_error

    def stream(
        self,
        prompt: str,
        max_output_tokens=None,
    ):
        """Yield answer text as the model produces it.

        Fallback models are used only before the first token. The finished
        text is cached after a complete stream, matching generate().
        """

        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        last_error = None

        for model_index, model in enumerate(
            self.models
        ):
            cache_key = self._cache_key(
                prompt,
                model,
            )

            cached = self._get_cached(
                cache_key
            )

            if cached is not None:
                print(
                    f"AICredits cache hit: {model}"
                )
                yield cached
                return

            for attempt in range(
                self.max_retries
            ):
                started = False

                try:
                    if attempt > 0:
                        delay = self.retry_delays[
                            min(
                                attempt - 1,
                                len(self.retry_delays) - 1,
                            )
                        ]

                        print(
                            f"AICredits temporary "
                            f"failure on {model}. "
                            f"Retrying in "
                            f"{delay:.1f}s "
                            f"(attempt "
                            f"{attempt + 1}/"
                            f"{self.max_retries})..."
                        )

                        time.sleep(delay)

                    pieces = []

                    for delta in self._stream_with_model(
                        model=model,
                        prompt=prompt,
                        max_output_tokens=(
                            max_output_tokens
                        ),
                    ):
                        if not delta:
                            continue

                        started = True
                        pieces.append(delta)
                        yield delta

                    response = "".join(pieces).strip()

                    if not response:
                        raise RuntimeError(
                            "AICredits returned no text content."
                        )

                    self._set_cached(
                        cache_key,
                        response,
                    )

                    return

                except Exception as error:
                    last_error = error

                    if started:
                        raise RuntimeError(
                            self._format_error(
                                model,
                                error,
                            )
                        ) from error

                    if not self._is_retryable_error(
                        error
                    ):
                        raise RuntimeError(
                            self._format_error(
                                model,
                                error,
                            )
                        ) from error

                    if attempt == 0:
                        continue

                    break

            if (
                model_index
                < len(self.models) - 1
            ):
                next_model = self.models[
                    model_index + 1
                ]

                print(
                    f"AICredits model "
                    f"'{model}' is temporarily "
                    f"unavailable. Falling back "
                    f"to '{next_model}'."
                )

        raise RuntimeError(
            "AICredits generation failed on "
            "all configured models. "
            f"Last error: {last_error}"
        ) from last_error