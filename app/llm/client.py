import time

from google import genai

from app.core.config import GEMINI_API_KEY


class GeminiClient:
    """Small Gemini wrapper with retry handling for temporary API failures."""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Keep the model configurable so it can be changed from one place.
        self.model = "gemini-3.6-flash"

        # A second model can be supplied through the environment later without
        # changing the rest of the ACOT codebase.
        self.fallback_model = None

        self.max_retries = 3
        self.retry_delays = (1.5, 3.0, 5.0)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Return True only for failures that are normally temporary."""
        message = str(error).lower()

        retryable_terms = (
            "503",
            "service unavailable",
            "temporarily unavailable",
            "high demand",
            "overloaded",
            "timeout",
            "timed out",
            "429",
            "rate limit",
            "resource exhausted",
        )

        return any(term in message for term in retryable_terms)

    def _generate_with_model(self, model: str, prompt: str):
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
        )

        if not response:
            raise RuntimeError("Gemini returned an empty response.")

        if not response.text:
            raise RuntimeError("Gemini returned no text content.")

        return response.text

    def generate(self, prompt: str):
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        models_to_try = [self.model]

        if self.fallback_model and self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)

        last_error = None

        for model_index, model in enumerate(models_to_try):
            for attempt in range(self.max_retries):
                try:
                    if attempt > 0:
                        delay = self.retry_delays[min(
                            attempt - 1,
                            len(self.retry_delays) - 1,
                        )]
                        print(
                            f"Gemini temporary failure. Retrying in "
                            f"{delay:.1f}s... (attempt {attempt + 1}/"
                            f"{self.max_retries})"
                        )
                        time.sleep(delay)

                    return self._generate_with_model(model, prompt)

                except Exception as error:
                    last_error = error

                    # Do not waste demo time retrying authentication,
                    # permission, invalid-model, or other permanent failures.
                    if not self._is_retryable_error(error):
                        raise RuntimeError(
                            f"Gemini generation failed: {error}"
                        ) from error

            # A configured fallback model is tried only after the primary
            # model has exhausted its transient retries.
            if model_index < len(models_to_try) - 1:
                print(
                    f"Primary Gemini model '{model}' is temporarily "
                    "unavailable. Trying the fallback model."
                )

        raise RuntimeError(
            f"Gemini generation failed after {self.max_retries} retries: "
            f"{last_error}"
        ) from last_error
