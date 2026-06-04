"""Mistral provider.

Concrete implementation of :class:`~rag_guardrails.providers.base.BaseProvider`
backed by the Mistral API via the async methods of the ``mistralai.Mistral``
client (``chat.complete_async`` / ``chat.stream_async``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from mistralai import Mistral
from mistralai.models import SDKError

from rag_guardrails.config import Settings, get_config
from rag_guardrails.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderResponse,
)


class MistralProvider(BaseProvider):
    """LLM provider backed by the Mistral chat API.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. When omitted,
            the shared singleton from
            :func:`~rag_guardrails.config.get_config` is used.

    Raises:
        ProviderError: If no Mistral API key is configured.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()

        if self._config.mistral_api_key is None:
            raise ProviderError(
                "Mistral API key is not configured. Set MISTRAL_API_KEY in the "
                "environment or .env file."
            )

        self._model: str = self._config.model_name or "mistral-small-latest"
        self._client: Any = Mistral(
            api_key=self._config.mistral_api_key.get_secret_value()
        )

    def get_model_name(self) -> str:
        """Return the Mistral model identifier this provider will use."""
        return self._model

    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate a single buffered completion via Mistral.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``chat.complete_async``. A
                ``model`` override is honored.

        Returns:
            A :class:`ProviderResponse` with the generated content and usage.

        Raises:
            ProviderError: If the Mistral API call fails.
        """
        model = kwargs.pop("model", self._model)
        try:
            response = await self._client.chat.complete_async(
                model=model, messages=messages, **kwargs
            )
        except SDKError as exc:
            raise ProviderError(f"Mistral API error: {exc}") from exc

        content = response.choices[0].message.content or ""
        usage = response.usage
        return ProviderResponse(
            content=content if isinstance(content, str) else str(content),
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw_response=response.model_dump(),
        )

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a completion via Mistral, yielding text chunks.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``chat.stream_async``. A
                ``model`` override is honored.

        Yields:
            Successive string chunks of the generated content.

        Raises:
            ProviderError: If the Mistral API call fails before or during
                streaming.
        """
        model = kwargs.pop("model", self._model)
        try:
            stream = await self._client.chat.stream_async(
                model=model, messages=messages, **kwargs
            )
            async for event in stream:
                delta = event.data.choices[0].delta.content
                if delta:
                    yield delta
        except SDKError as exc:
            raise ProviderError(f"Mistral API error: {exc}") from exc
