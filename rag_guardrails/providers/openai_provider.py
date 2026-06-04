"""OpenAI provider (default).

Concrete implementation of :class:`~rag_guardrails.providers.base.BaseProvider`
backed by the OpenAI API via the official ``AsyncOpenAI`` client.

Responsibilities:

- Implement buffered (:meth:`OpenAIProvider.complete`) and streaming
  (:meth:`OpenAIProvider.stream`) chat generation against OpenAI
  chat-completion models.
- Read credentials and the default model from
  :class:`~rag_guardrails.config.Settings` (falling back to
  :func:`~rag_guardrails.config.get_config`).
- Map OpenAI responses onto :class:`~rag_guardrails.providers.base.ProviderResponse`
  and wrap OpenAI errors in
  :class:`~rag_guardrails.providers.base.ProviderError`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import openai
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
)

from rag_guardrails.config import Settings, get_config
from rag_guardrails.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderResponse,
)


class OpenAIProvider(BaseProvider):
    """LLM provider backed by the OpenAI chat-completions API.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings` instance. When
            omitted, the shared singleton from
            :func:`~rag_guardrails.config.get_config` is used.

    Raises:
        ProviderError: If no OpenAI API key is configured.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()

        if self._config.openai_api_key is None:
            raise ProviderError(
                "OpenAI API key is not configured. Set OPENAI_API_KEY in the "
                "environment or .env file."
            )

        self._model: str = self._config.model_name
        self._client = AsyncOpenAI(
            api_key=self._config.openai_api_key.get_secret_value()
        )

    def get_model_name(self) -> str:
        """Return the OpenAI model identifier this provider will use."""
        return self._model

    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate a single buffered completion via OpenAI.

        Args:
            messages: Chat messages in OpenAI role/content form.
            **kwargs: Extra parameters forwarded to
                ``chat.completions.create`` (e.g. ``temperature``,
                ``max_tokens``). A ``model`` override is honored if supplied.

        Returns:
            A :class:`ProviderResponse` with the generated content and usage.

        Raises:
            ProviderError: If the OpenAI API call fails.
        """
        model = kwargs.pop("model", self._model)
        try:
            response = cast(
                ChatCompletion,
                await self._client.chat.completions.create(
                    model=model,
                    messages=cast("list[ChatCompletionMessageParam]", messages),
                    stream=False,
                    **kwargs,
                ),
            )
        except openai.AuthenticationError as exc:
            raise ProviderError(
                f"OpenAI authentication failed: {exc}"
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderError(f"OpenAI rate limit exceeded: {exc}") from exc
        except openai.APIError as exc:
            raise ProviderError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        return ProviderResponse(
            content=content,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw_response=response.model_dump(),
        )

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a completion via OpenAI, yielding text chunks.

        Args:
            messages: Chat messages in OpenAI role/content form.
            **kwargs: Extra parameters forwarded to
                ``chat.completions.create``. A ``model`` override is honored if
                supplied.

        Yields:
            Successive string chunks of the generated content.

        Raises:
            ProviderError: If the OpenAI API call fails before or during
                streaming.
        """
        model = kwargs.pop("model", self._model)
        try:
            stream = cast(
                "AsyncStream[ChatCompletionChunk]",
                await self._client.chat.completions.create(
                    model=model,
                    messages=cast("list[ChatCompletionMessageParam]", messages),
                    stream=True,
                    **kwargs,
                ),
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except openai.AuthenticationError as exc:
            raise ProviderError(
                f"OpenAI authentication failed: {exc}"
            ) from exc
        except openai.RateLimitError as exc:
            raise ProviderError(f"OpenAI rate limit exceeded: {exc}") from exc
        except openai.APIError as exc:
            raise ProviderError(f"OpenAI API error: {exc}") from exc
