"""LangChain provider wrapper.

Adapter that lets any LangChain chat model be used through the shared provider
contract (:class:`~rag_guardrails.providers.base.BaseProvider`).

Responsibilities:

- Wrap a caller-supplied LangChain ``BaseChatModel`` instance, or construct a
  default :class:`~langchain_openai.ChatOpenAI` from
  :class:`~rag_guardrails.config.Settings` when none is given.
- Translate the system's role/content message format into LangChain message
  objects, and map LangChain responses back onto
  :class:`~rag_guardrails.providers.base.ProviderResponse`.
- Wrap LangChain and unexpected errors in
  :class:`~rag_guardrails.providers.base.ProviderError`.

The caller-supplied ``llm`` is the escape hatch: any model LangChain supports
(Anthropic, Google, local models, etc.) can be passed in directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.exceptions import LangChainException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from rag_guardrails.config import Settings, get_config
from rag_guardrails.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderResponse,
)


def _to_langchain_messages(
    messages: list[dict[str, Any]],
) -> list[BaseMessage]:
    """Convert role/content message dicts into LangChain message objects.

    ``"system"`` maps to :class:`SystemMessage`, ``"assistant"``/``"ai"`` map to
    :class:`AIMessage`, and every other role (``"user"``, ``"human"``, or an
    unknown value) maps to :class:`HumanMessage`.

    Args:
        messages: Chat messages in role/content form.

    Returns:
        The equivalent list of LangChain :class:`BaseMessage` instances.
    """
    converted: list[BaseMessage] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role in ("assistant", "ai"):
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


class LangChainProvider(BaseProvider):
    """LLM provider backed by an arbitrary LangChain chat model.

    Args:
        config: Optional :class:`~rag_guardrails.config.Settings` instance. When
            omitted, the shared singleton from
            :func:`~rag_guardrails.config.get_config` is used. Only consulted
            when ``llm`` is not supplied.
        llm: Optional pre-configured LangChain ``BaseChatModel``. When provided,
            it is used directly (the escape hatch for any LangChain-supported
            model). When omitted, a :class:`~langchain_openai.ChatOpenAI`
            instance is constructed from ``config``.

    Raises:
        ProviderError: If ``llm`` is omitted and no OpenAI API key is configured
            for the default ``ChatOpenAI`` backend.
    """

    def __init__(
        self,
        config: Settings | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self._config: Settings = config or get_config()

        if llm is not None:
            self._llm: BaseChatModel = llm
        else:
            if self._config.openai_api_key is None:
                raise ProviderError(
                    "No LangChain model was provided and no OpenAI API key is "
                    "configured for the default ChatOpenAI backend. Pass an "
                    "`llm` instance or set OPENAI_API_KEY."
                )
            # Imported lazily so the optional ChatOpenAI dependency is only
            # required when the default backend is actually constructed.
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                api_key=self._config.openai_api_key,
                model=self._config.model_name,
            )

        self._model: str = getattr(
            self._llm, "model_name", None
        ) or self._config.model_name

    def get_model_name(self) -> str:
        """Return the identifier of the wrapped LangChain model."""
        return self._model

    async def complete(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate a single buffered completion via the LangChain model.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to the model's ``ainvoke``
                (e.g. ``temperature``, ``max_tokens``).

        Returns:
            A :class:`ProviderResponse` with the generated content and usage.
            Token counts are read from ``response.usage_metadata`` when the
            backend reports them, defaulting to ``0`` otherwise.

        Raises:
            ProviderError: If the LangChain model fails to produce a response.
        """
        lc_messages = _to_langchain_messages(messages)
        try:
            response = await self._llm.ainvoke(lc_messages, **kwargs)
        except LangChainException as exc:
            raise ProviderError(f"LangChain error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - normalize to ProviderError
            raise ProviderError(
                f"Unexpected LangChain provider error: {exc}"
            ) from exc

        content = self._content_to_str(response.content)

        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        response_model = (
            response.response_metadata.get("model_name")
            if hasattr(response, "response_metadata")
            else None
        ) or self._model

        return ProviderResponse(
            content=content,
            model=response_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response={
                "content": content,
                "usage_metadata": usage,
                "response_metadata": getattr(response, "response_metadata", {}),
            },
        )

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a completion via the LangChain model, yielding text chunks.

        Args:
            messages: Chat messages in role/content form.
            **kwargs: Extra parameters forwarded to the model's ``astream``.

        Yields:
            Successive non-empty string chunks of the generated content.

        Raises:
            ProviderError: If the LangChain model fails before or during
                streaming.
        """
        lc_messages = _to_langchain_messages(messages)
        try:
            async for chunk in self._llm.astream(lc_messages, **kwargs):
                text = self._content_to_str(chunk.content)
                if text:
                    yield text
        except LangChainException as exc:
            raise ProviderError(f"LangChain error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - normalize to ProviderError
            raise ProviderError(
                f"Unexpected LangChain provider error: {exc}"
            ) from exc

    @staticmethod
    def _content_to_str(content: Any) -> str:
        """Flatten LangChain message content into a plain string.

        LangChain message content may be a string or a list of content blocks
        (e.g. for multimodal models). This extracts the textual portion.

        Args:
            content: The ``content`` attribute of a LangChain message or chunk.

        Returns:
            The content rendered as a string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
            return "".join(parts)
        return str(content)
