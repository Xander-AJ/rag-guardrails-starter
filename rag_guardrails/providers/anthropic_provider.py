"""Anthropic provider.

Concrete implementation of :class:`~rag_guardrails.providers.base.BaseProvider`
backed by the Anthropic Messages API via the official ``AsyncAnthropic`` client.

The shared role/content message format is adapted to Anthropic's API: any
``system`` messages are collapsed into the separate ``system`` parameter, and
the remaining user/assistant turns are passed as ``messages``. Anthropic
requires ``max_tokens``; it defaults to 1024 and can be overridden per call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from rag_guardrails.config import Settings, get_config
from rag_guardrails.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderResponse,
)

_DEFAULT_MAX_TOKENS = 1024


def _split_system(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split a message list into a system string and chat turns.

    Args:
        messages: Chat messages in role/content form.

    Returns:
        A ``(system, chat)`` tuple where ``system`` is the concatenated text of
        any system messages and ``chat`` is the remaining user/assistant turns.
    """
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    chat = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") != "system"
    ]
    return "\n".join(system_parts), chat


class AnthropicProvider(BaseProvider):
    """LLM provider backed by the Anthropic Messages API.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings`. When omitted,
            the shared singleton from
            :func:`~rag_guardrails.config.get_config` is used.

    Raises:
        ProviderError: If no Anthropic API key is configured.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self._config: Settings = config or get_config()

        if self._config.anthropic_api_key is None:
            raise ProviderError(
                "Anthropic API key is not configured. Set ANTHROPIC_API_KEY in "
                "the environment or .env file."
            )

        self._model: str = self._config.model_name or "claude-3-5-haiku-20241022"
        self._client: Any = AsyncAnthropic(
            api_key=self._config.anthropic_api_key.get_secret_value()
        )

    def get_model_name(self) -> str:
        """Return the Anthropic model identifier this provider will use."""
        return self._model

    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate a single buffered completion via Anthropic.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``messages.create`` (e.g.
                ``temperature``, ``max_tokens``). A ``model`` override is honored.

        Returns:
            A :class:`ProviderResponse` with the generated content and usage.

        Raises:
            ProviderError: If the Anthropic API call fails.
        """
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS)
        system, chat = _split_system(messages)
        if system:
            kwargs["system"] = system
        try:
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=chat,
                **kwargs,
            )
        except anthropic.APIError as exc:
            raise ProviderError(f"Anthropic API error: {exc}") from exc

        content = response.content[0].text if response.content else ""
        usage = response.usage
        return ProviderResponse(
            content=content,
            model=response.model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            raw_response=response.model_dump(),
        )

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a completion via Anthropic, yielding text chunks.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to ``messages.stream``. A
                ``model`` override is honored.

        Yields:
            Successive string chunks of the generated content.

        Raises:
            ProviderError: If the Anthropic API call fails before or during
                streaming.
        """
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", _DEFAULT_MAX_TOKENS)
        system, chat = _split_system(messages)
        if system:
            kwargs["system"] = system
        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=chat,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except anthropic.APIError as exc:
            raise ProviderError(f"Anthropic API error: {exc}") from exc
