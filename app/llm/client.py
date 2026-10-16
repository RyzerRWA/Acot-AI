import asyncio
import hashlib
import os
import time
from collections import OrderedDict
from threading import Lock

from openai import AsyncOpenAI

from app.core.config import AICREDITS_API_KEY, AICREDITS_BASE_URL
from app.subscription.token_usage import (
    add_tokens,
    get_tokens,
)

# Limits concurrent in-flight LLM calls to AICredits. The provider's
# concurrency cap is ~5, so anything above that risks HTTP 429s.
_LLM_CONCURRENCY = int(os.getenv("AICREDITS_MAX_CONCURRENCY", "10"))
_llm_semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)

# How long a request may wait for a free LLM slot before we give
# up and tell the user we are busy. Without this, overload shows up
# as clients hanging until their own timeout (e.g. 120s).

_LLM_QUEUE_TIMEOUT_SECONDS = float(
    os.getenv("AICREDITS_QUEUE_TIMEOUT_SECONDS", "15")
)

# Whether a cached LLM response counts against the caller's usage quota.
# Default off: a cache hit performs no LLM call and costs nothing, so
# billing it in full charges users for tokens the provider was never asked
# for. Flip to 1 if you would rather meter repeated questions.

BILL_CACHE_HITS = os.getenv("ACOT_BILL_CACHE_HITS", "0") not in (
    "0",
    "false",
    "False",
    "",
)


class LLMQueueTimeoutError(RuntimeError):
    """No LLM slot freed up within the queue budget."""


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

        self.client = AsyncOpenAI(
            base_url=AICREDITS_BASE_URL,
            api_key=AICREDITS_API_KEY,
        )

        self.model = os.getenv(
            "AICREDITS_MODEL",
            "openai/gpt-4o-mini",
        )

        # One retry of the same model.
        self.max_retries = 2
        self.retry_delays = (1.0,)

        # Output token limit.
        self.max_output_tokens = int(
            os.getenv(
                "AICREDITS_MAX_OUTPUT_TOKENS",
                "900",
            )
        )

        # Response cache.
        self.cache_enabled = True
        self.cache_ttl_seconds = 300
        self.cache_max_entries = 100

        self._cache = OrderedDict()
        self._cache_lock = Lock()

    # =========================================================
    # RETRY HANDLING
    # =========================================================

    def _is_retryable_error(
        self,
        error: Exception,
    ) -> bool:

        message = str(error).lower()

        permanent_terms = (
            ["401","403","404","invalid api key","authentication","unauthorized","forbidden",
            "invalid model","model not found","insufficient credits",]
        )

        if any(
            term in message
            for term in permanent_terms
        ):
            return False

        retryable_terms = (
            "429","500","502","503","504","rate limit","timeout","timed out","temporarily unavailable",
            "overloaded","capacity","no text content","empty response","empty message","no choices",
        )

        return any(
            term in message
            for term in retryable_terms
        )

    # =========================================================
    # CONCURRENCY (bounded queue wait)
    # =========================================================

    async def _acquire_slot(self):
        """Wait for a free LLM slot, but never forever.

        Raises LLMQueueTimeoutError when no slot frees up within
        _LLM_QUEUE_TIMEOUT_SECONDS so overload surfaces as a fast,
        friendly "busy" response instead of a 120s client hang.
        """
        try:
            await asyncio.wait_for(
                _llm_semaphore.acquire(),
                timeout=_LLM_QUEUE_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise LLMQueueTimeoutError(
                "ACOT is busy right now, please try again in a moment."
            ) from error

    def _release_slot(self):
        _llm_semaphore.release()

    # =========================================================
    # CACHE
    # =========================================================

    def _cache_key(
        self,
        prompt: str,
        model: str,
    ) -> str:

        raw = (
            f"{model}\n{prompt}"
        ).encode("utf-8")

        return hashlib.sha256(
            raw
        ).hexdigest()

    def _get_cached(
        self,
        key: str,
    ):

        """
        Return (text, tokens) for a live cache entry, else None.

        The token count is stored alongside the text so a cache hit can
        replay it. Without this a cached answer reports 0 tokens and the
        caller's usage total silently under-counts.
        """

        if not self.cache_enabled:
            return None

        now = time.time()

        with self._cache_lock:

            item = self._cache.get(key)

            if item is None:
                return None

            created_at, value, tokens = item

            if (
                now - created_at
                > self.cache_ttl_seconds
            ):
                self._cache.pop(
                    key,
                    None,
                )
                return None

            self._cache.move_to_end(
                key
            )

            return value, tokens

    def _set_cached(
        self,
        key: str,
        value: str,
        tokens: int = 0,
    ):

        if not self.cache_enabled:
            return

        with self._cache_lock:

            self._cache[key] = (
                time.time(),
                value,
                tokens,
            )

            self._cache.move_to_end(
                key
            )

            while (
                len(self._cache)
                > self.cache_max_entries
            ):
                self._cache.popitem(
                    last=False
                )

    def clear_cache(self):

        with self._cache_lock:
            self._cache.clear()

    # =========================================================
    # NON-STREAMING LLM CALL
    # =========================================================

    async def _generate_with_model(
        self,
        model: str,
        prompt: str,
        max_output_tokens=None,
    ):

        response = await self.client.chat.completions.create(
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

        # -----------------------------------------------------
        # Track actual token usage.
        # -----------------------------------------------------

        usage = getattr(
            response,
            "usage",
            None,
        )

        if usage:
            add_tokens(
                getattr(
                    usage,
                    "total_tokens",
                    0,
                )
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

        content = self._message_text(
            message
        )

        if not content:
            raise RuntimeError(
                "AICredits returned no text content."
            )

        return content.strip()

    # =========================================================
    # STREAMING LLM CALL
    # =========================================================

    async def _stream_with_model(
        self,
        model: str,
        prompt: str,
        max_output_tokens=None,
    ):

        stream = await self.client.chat.completions.create(
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
            stream_options={
                "include_usage": True
            },
        )

        if not stream:
            raise RuntimeError(
                "AICredits returned an empty response."
            )

        async for chunk in stream:

            if not chunk:
                continue

            # -------------------------------------------------
            # Final streaming chunk may contain usage.
            # -------------------------------------------------

            usage = getattr(
                chunk,
                "usage",
                None,
            )

            if usage:
                add_tokens(
                    getattr(
                        usage,
                        "total_tokens",
                        0,
                    )
                )

            if not getattr(
                chunk,
                "choices",
                None,
            ):
                continue

            delta = getattr(
                chunk.choices[0],
                "delta",
                None,
            )

            if delta is None:
                continue

            text = self._message_text(
                delta
            )

            if text:
                yield text

    # =========================================================
    # MESSAGE TEXT
    # =========================================================

    @staticmethod
    def _message_text(
        message,
    ) -> str:

        content = getattr(
            message,
            "content",
            None,
        )

        if isinstance(
            content,
            str,
        ):
            return content

        if isinstance(
            content,
            list,
        ):

            parts = []

            for item in content:

                if isinstance(
                    item,
                    str,
                ):
                    parts.append(item)
                    continue

                text = getattr(
                    item,
                    "text",
                    None,
                )

                if (
                    text is None
                    and isinstance(
                        item,
                        dict,
                    )
                ):
                    text = (
                        item.get("text")
                        or item.get("content")
                    )

                if text:
                    parts.append(
                        str(text)
                    )

            return "\n".join(parts)

        return ""

    # =========================================================
    # ERROR FORMAT
    # =========================================================

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

    # =========================================================
    # GENERATE
    # =========================================================

    async def generate(
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

            cached_text, cached_tokens = cached

            # A cache hit makes no LLM call, so no tokens are spent and
            # none are billed. Previously these were replayed in full, which
            # charged users for work that never happened: an account with
            # 19 questions recorded 98833 tokens where asking each question
            # once cost 24894. Set ACOT_BILL_CACHE_HITS=1 to bill them.
            if cached_tokens and BILL_CACHE_HITS:
                add_tokens(cached_tokens)

            return cached_text

        last_error = None

        tokens_before = get_tokens()

        for attempt in range(
            self.max_retries
        ):

            try:

                if attempt > 0:

                    delay = self.retry_delays[
                        min(
                            attempt - 1,
                            len(
                                self.retry_delays
                            ) - 1,
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

                    await asyncio.sleep(delay)

                await self._acquire_slot()
                try:
                    response = (
                        await self._generate_with_model(
                            model=model,
                            prompt=prompt,
                            max_output_tokens=(
                                max_output_tokens
                            ),
                        )
                    )
                finally:
                    self._release_slot()

                self._set_cached(
                    cache_key,
                    response,
                    get_tokens() - tokens_before,
                )

                return response

            except LLMQueueTimeoutError:

                # Busy: retrying would just re-queue behind the
                # same full semaphore. Fail fast with the message.
                raise

            except Exception as error:

                last_error = error

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

        raise RuntimeError(
            "AICredits generation failed on "
            f"model '{model}'. "
            f"Last error: {last_error}"
        ) from last_error

    # =========================================================
    # STREAM
    # =========================================================

    async def stream(
        self,
        prompt: str,
        max_output_tokens=None,
    ):
        """
        Yield answer text as the model produces it.

        Uses the same single model as generate().
        A failure before the first token is retried once.
        The finished text is cached only after a complete stream.
        """

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

            cached_text, cached_tokens = cached

            if cached_tokens and BILL_CACHE_HITS:
                add_tokens(cached_tokens)

            yield cached_text
            return

        last_error = None

        tokens_before = get_tokens()

        for attempt in range(
            self.max_retries
        ):

            started = False

            try:

                if attempt > 0:

                    delay = self.retry_delays[
                        min(
                            attempt - 1,
                            len(
                                self.retry_delays
                            ) - 1,
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

                    await asyncio.sleep(delay)

                pieces = []

                await self._acquire_slot()
                try:
                    async for delta in self._stream_with_model(
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
                finally:
                    self._release_slot()

                response = "".join(
                    pieces
                ).strip()

                if not response:

                    raise RuntimeError(
                        "AICredits returned no text content."
                    )

                self._set_cached(
                    cache_key,
                    response,
                    get_tokens() - tokens_before,
                )

                return

            except LLMQueueTimeoutError:

                # Busy: no slot freed up in time. Fail fast with
                # the friendly message instead of re-queueing.
                raise

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

        raise RuntimeError(
            "AICredits generation failed on "
            f"model '{model}'. "
            f"Last error: {last_error}"
        ) from last_error