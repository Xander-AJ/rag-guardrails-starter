"""Google Gemini provider.

Concrete implementation of :class:`~rag_guardrails.providers.base.BaseProvider`
backed by the Google Generative AI SDK (``google-generativeai``).

The shared role/content message list is flattened into a single prompt string
(Gemini does not use the OpenAI message schema). The API key is applied process
-wide via :func:`google.generativeai.configure`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from rag_guardrails.config import Settings, get_config
from rag_guardrails.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderResponse,
)


def _to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten role/content messages into a single prompt string."""
    return "\n\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
    )


# TODO: google-generativeai is deprecated. Migrate to google-genai package
# when ready. See: https://ai.google.dev/gemini-api/docs/migrate
class GeminiProvider(BaseProvider):
    """LLM provider backed by Google Gemini.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. When omitted,
            the shared singleton from
            :func:`~rag_guardrails.config.get_config` is used.

    Raises:
        ProviderError: If no Gemini API key is configured.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()

        if self._config.gemini_api_key is None:
            raise ProviderError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in the "
                "environment or .env file."
            )

        genai.configure(api_key=self._config.gemini_api_key.get_secret_value())
        self._model: str = self._config.model_name or "gemini-1.5-flash"

    def get_model_name(self) -> str:
        """Return the Gemini model identifier this provider will use."""
        return self._model

    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate a single buffered completion via Gemini.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``generate_content_async``.
                A ``model`` override is honored.

        Returns:
            A :class:`ProviderResponse` with the generated content and usage.

        Raises:
            ProviderError: If the Gemini API call fails.
        """
        model_name = kwargs.pop("model", self._model)
        model: Any = genai.GenerativeModel(model_name)
        try:
            response = await model.generate_content_async(
                _to_prompt(messages), **kwargs
            )
        except google_exceptions.GoogleAPIError as exc:
            raise ProviderError(f"Gemini API error: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            content=response.text or "",
            model=model_name,
            input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            output_tokens=(
                getattr(usage, "candidates_token_count", 0) if usage else 0
            ),
            raw_response={"text": response.text or ""},
        )

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a completion via Gemini, yielding text chunks.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``generate_content_async``.
                A ``model`` override is honored.

        Yields:
            Successive string chunks of the generated content.

        Raises:
            ProviderError: If the Gemini API call fails before or during
                streaming.
        """
        model_name = kwargs.pop("model", self._model)
        model: Any = genai.GenerativeModel(model_name)
        try:
            stream = await model.generate_content_async(
                _to_prompt(messages), stream=True, **kwargs
            )
            async for chunk in stream:
                text = getattr(chunk, "text", "")
                if text:
                    yield text
        except google_exceptions.GoogleAPIError as exc:
            raise ProviderError(f"Gemini API error: {exc}") from exc
