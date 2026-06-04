"""Ollama provider.

Concrete implementation of :class:`~rag_guardrails.providers.base.BaseProvider`
backed by a local `Ollama <https://github.com/ollama/ollama>`_ server via the
``ollama.AsyncClient``. No API key is required; the server location is read from
``ollama_base_url`` in configuration.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import ollama
from ollama import AsyncClient

from rag_guardrails.config import Settings, get_config
from rag_guardrails.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderResponse,
)

_OLLAMA_DOCS = "https://github.com/ollama/ollama"

# Connection/protocol errors normalized into ProviderError.
_OLLAMA_ERRORS = (
    ollama.ResponseError,
    ollama.RequestError,
    ConnectionError,
    httpx.HTTPError,
)


class OllamaProvider(BaseProvider):
    """LLM provider backed by a local Ollama server.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. When omitted,
            the shared singleton from
            :func:`~rag_guardrails.config.get_config` is used. ``ollama_base_url``
            locates the server.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()
        self._model: str = self._config.model_name or "llama3.2"
        self._client: Any = AsyncClient(host=self._config.ollama_base_url)

    def get_model_name(self) -> str:
        """Return the Ollama model identifier this provider will use."""
        return self._model

    def _connection_error(self, exc: Exception) -> ProviderError:
        """Wrap an Ollama transport error with a docs pointer."""
        return ProviderError(
            f"Ollama request to {self._config.ollama_base_url} failed: {exc}. "
            f"Ensure the Ollama server is running and the model is pulled "
            f"(see {_OLLAMA_DOCS})."
        )

    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate a single buffered completion via Ollama.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``chat``. A ``model``
                override is honored.

        Returns:
            A :class:`ProviderResponse` with the generated content and usage.

        Raises:
            ProviderError: If the Ollama request fails.
        """
        model = kwargs.pop("model", self._model)
        try:
            response = await self._client.chat(
                model=model, messages=messages, **kwargs
            )
        except _OLLAMA_ERRORS as exc:
            raise self._connection_error(exc) from exc

        content = response.message.content or ""
        return ProviderResponse(
            content=content,
            model=response.model,
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            raw_response=response.model_dump(),
        )

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a completion via Ollama, yielding text chunks.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``chat``. A ``model``
                override is honored.

        Yields:
            Successive string chunks of the generated content.

        Raises:
            ProviderError: If the Ollama request fails before or during
                streaming.
        """
        model = kwargs.pop("model", self._model)
        try:
            stream = await self._client.chat(
                model=model, messages=messages, stream=True, **kwargs
            )
            async for part in stream:
                delta = part.message.content
                if delta:
                    yield delta
        except _OLLAMA_ERRORS as exc:
            raise self._connection_error(exc) from exc
