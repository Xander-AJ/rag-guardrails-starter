"""Cohere provider.

Concrete implementation of :class:`~rag_guardrails.providers.base.BaseProvider`
backed by the Cohere v2 Chat API via the async ``cohere.AsyncClientV2`` client.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import cohere
from cohere.core.api_error import ApiError as CohereApiError

from rag_guardrails.config import Settings, get_config
from rag_guardrails.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderResponse,
)


class CohereProvider(BaseProvider):
    """LLM provider backed by the Cohere v2 Chat API.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. When omitted,
            the shared singleton from
            :func:`~rag_guardrails.config.get_config` is used.

    Raises:
        ProviderError: If no Cohere API key is configured.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()

        if self._config.cohere_api_key is None:
            raise ProviderError(
                "Cohere API key is not configured. Set COHERE_API_KEY in the "
                "environment or .env file."
            )

        self._model: str = self._config.model_name or "command-r-plus"
        self._client: Any = cohere.AsyncClientV2(
            api_key=self._config.cohere_api_key.get_secret_value()
        )

    def get_model_name(self) -> str:
        """Return the Cohere model identifier this provider will use."""
        return self._model

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Pull the text out of a Cohere v2 chat message's content blocks."""
        content = getattr(message, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                return text
        return ""

    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate a single buffered completion via Cohere.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``chat``. A ``model``
                override is honored.

        Returns:
            A :class:`ProviderResponse` with the generated content and usage.

        Raises:
            ProviderError: If the Cohere API call fails.
        """
        model = kwargs.pop("model", self._model)
        try:
            response = await self._client.chat(
                model=model, messages=messages, **kwargs
            )
        except CohereApiError as exc:
            raise ProviderError(f"Cohere API error: {exc}") from exc

        tokens = getattr(getattr(response, "usage", None), "tokens", None)
        return ProviderResponse(
            content=self._extract_text(response.message),
            model=model,
            input_tokens=int(getattr(tokens, "input_tokens", 0) or 0),
            output_tokens=int(getattr(tokens, "output_tokens", 0) or 0),
            raw_response=response.dict(),
        )

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a completion via Cohere, yielding text chunks.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``chat_stream``. A ``model``
                override is honored.

        Yields:
            Successive string chunks of the generated content.

        Raises:
            ProviderError: If the Cohere API call fails before or during
                streaming.
        """
        model = kwargs.pop("model", self._model)
        try:
            async for event in self._client.chat_stream(
                model=model, messages=messages, **kwargs
            ):
                if getattr(event, "type", None) != "content-delta":
                    continue
                delta = event.delta.message.content.text
                if delta:
                    yield delta
        except CohereApiError as exc:
            raise ProviderError(f"Cohere API error: {exc}") from exc
